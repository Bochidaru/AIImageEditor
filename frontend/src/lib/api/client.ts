/**
 * Real implementation of the backend contract — talks to the FastAPI app in
 * src/inpaint_core/api/ (run via `python scripts/run_api.py --fake` or
 * `--config configs/default.yaml`; see frontend/.env.local.example).
 *
 * Every exported function here has the exact same signature as its
 * counterpart in mock.ts, so lib/api/index.ts can swap between them without
 * any calling component changing.
 */

import {
  BoxPrompt,
  EditResult,
  GenerationOptions,
  GenerationResult,
  MaskOptions,
  OutpaintMargins,
  SegmentationResult,
  SelectionPrompt,
  UpscaleOptions,
} from "@/lib/types";
import { ApiError } from "@/lib/api/errors";

const BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

// ---------------------------------------------------------------------------
// Wire-format helpers — camelCase (frontend) <-> snake_case (Pydantic)
// ---------------------------------------------------------------------------

function selectionBody(selection: SelectionPrompt) {
  return selection.kind === "point"
    ? { kind: "point", points: selection.points, labels: selection.labels }
    : { kind: "box", x1: selection.x1, y1: selection.y1, x2: selection.x2, y2: selection.y2 };
}

function boxBody(box: BoxPrompt) {
  return { kind: "box", x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 };
}

function maskOptionsBody(options: MaskOptions) {
  return {
    threshold: options.threshold,
    dilate: options.dilate,
    erode: options.erode,
    candidate_index: options.candidateIndex ?? null,
  };
}

function generationOptionsBody(options: GenerationOptions) {
  return {
    seed: options.seed,
    num_inference_steps: options.numInferenceSteps,
    guidance_scale: options.guidanceScale ?? null,
  };
}

function upscaleOptionsBody(options: UpscaleOptions) {
  return {
    scale: options.scale,
    face_enhance: options.faceEnhance,
    tile: options.tile,
    tile_pad: options.tilePad,
    pre_pad: options.prePad,
  };
}

function marginsBody(margins: OutpaintMargins) {
  return { left: margins.left, top: margins.top, right: margins.right, bottom: margins.bottom };
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

async function post<T>(path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      `Could not reach the API at ${BASE_URL}. Is scripts/run_api.py running?`,
      "NETWORK_ERROR",
    );
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const payload = await response.json();
      if (typeof payload?.detail === "string") detail = payload.detail;
      else if (Array.isArray(payload?.detail)) {
        detail = payload.detail.map((item: { msg?: string }) => item.msg).join("; ");
      }
    } catch {
      // response body wasn't JSON; keep the generic message
    }
    // 422 = FastAPI's own Pydantic schema-validation failure (missing/malformed
    // field). 400 = app.py's domain-error handler (ValueError/IndexError from
    // business-rule checks, e.g. "provide exactly one of selection or mask").
    // These are different failure classes — keep their codes distinct.
    const code =
      response.status === 422 ? "VALIDATION_ERROR" : response.status === 400 ? "BAD_REQUEST" : "HTTP_ERROR";
    throw new ApiError(detail, code);
  }

  return response.json() as Promise<T>;
}

interface SegmentationResponseBody {
  masks: string[];
  scores: number[];
  best_index: number;
  metadata: Record<string, unknown>;
}

interface GenerationResponseBody {
  image: string;
  seed: number;
  metadata: Record<string, unknown>;
}

interface EditResponseBody {
  image: string;
  mask: string;
  generated_image: string | null;
  metadata: EditResult["metadata"];
}

function toSegmentationResult(body: SegmentationResponseBody): SegmentationResult {
  return { masks: body.masks, scores: body.scores, bestIndex: body.best_index, metadata: body.metadata };
}

function toGenerationResult(body: GenerationResponseBody): GenerationResult {
  return { image: body.image, seed: body.seed, metadata: body.metadata };
}

function toEditResult(body: EditResponseBody): EditResult {
  return {
    image: body.image,
    mask: body.mask,
    generatedImage: body.generated_image ?? undefined,
    metadata: body.metadata,
  };
}

// ---------------------------------------------------------------------------
// Image registration — upload once, reuse by id (see api/image_cache.py).
// Avoids re-sending the full image on every /api/segment call while a user
// tries several points/boxes against the same unchanged source image.
// ---------------------------------------------------------------------------

export async function registerImage(image: string): Promise<string> {
  const body = await post<{ image_id: string }>("/api/images", { image });
  return body.image_id;
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

export async function segment(
  image: string,
  selection: SelectionPrompt,
  imageId?: string,
): Promise<SegmentationResult> {
  const body = await post<SegmentationResponseBody>("/api/segment", {
    ...(imageId ? { image_id: imageId } : { image }),
    selection: selectionBody(selection),
  });
  return toSegmentationResult(body);
}

// ---------------------------------------------------------------------------
// Masked edit operations
// ---------------------------------------------------------------------------

interface MaskedEditArgs {
  image: string;
  selection?: SelectionPrompt;
  mask?: string;
  maskOptions?: MaskOptions;
  generationOptions?: GenerationOptions;
}

export async function removeObject(args: MaskedEditArgs): Promise<EditResult> {
  const body = await post<EditResponseBody>("/api/remove-object", {
    image: args.image,
    selection: args.selection ? selectionBody(args.selection) : null,
    mask: args.mask ?? null,
    mask_options: args.maskOptions ? maskOptionsBody(args.maskOptions) : null,
    generation_options: args.generationOptions ? generationOptionsBody(args.generationOptions) : null,
  });
  return toEditResult(body);
}

export async function replaceObject(args: MaskedEditArgs & { prompt: string }): Promise<EditResult> {
  const body = await post<EditResponseBody>("/api/replace-object", {
    image: args.image,
    prompt: args.prompt,
    selection: args.selection ? selectionBody(args.selection) : null,
    mask: args.mask ?? null,
    mask_options: args.maskOptions ? maskOptionsBody(args.maskOptions) : null,
    generation_options: args.generationOptions ? generationOptionsBody(args.generationOptions) : null,
  });
  return toEditResult(body);
}

export async function replaceBackground(args: {
  image: string;
  foregroundSelection?: SelectionPrompt;
  foregroundMask?: string;
  maskOptions?: MaskOptions;
  generationOptions?: GenerationOptions;
  prompt: string;
}): Promise<EditResult> {
  const body = await post<EditResponseBody>("/api/replace-background", {
    image: args.image,
    prompt: args.prompt,
    foreground_selection: args.foregroundSelection ? selectionBody(args.foregroundSelection) : null,
    foreground_mask: args.foregroundMask ?? null,
    mask_options: args.maskOptions ? maskOptionsBody(args.maskOptions) : null,
    generation_options: args.generationOptions ? generationOptionsBody(args.generationOptions) : null,
  });
  return toEditResult(body);
}

export async function addObjectByPrompt(args: {
  image: string;
  placement?: BoxPrompt;
  mask?: string;
  prompt: string;
  maskOptions?: MaskOptions;
  generationOptions?: GenerationOptions;
}): Promise<EditResult> {
  const body = await post<EditResponseBody>("/api/add-object/prompt", {
    image: args.image,
    prompt: args.prompt,
    placement: args.placement ? boxBody(args.placement) : null,
    mask: args.mask ?? null,
    mask_options: args.maskOptions ? maskOptionsBody(args.maskOptions) : null,
    generation_options: args.generationOptions ? generationOptionsBody(args.generationOptions) : null,
  });
  return toEditResult(body);
}

export async function addObjectByReference(args: {
  image: string;
  reference: string;
  placement?: BoxPrompt;
  mask?: string;
  referenceMask?: string;
  maskOptions?: MaskOptions;
  generationOptions?: GenerationOptions;
}): Promise<EditResult> {
  const body = await post<EditResponseBody>("/api/add-object/reference", {
    image: args.image,
    reference: args.reference,
    placement: args.placement ? boxBody(args.placement) : null,
    mask: args.mask ?? null,
    reference_mask: args.referenceMask ?? null,
    mask_options: args.maskOptions ? maskOptionsBody(args.maskOptions) : null,
    generation_options: args.generationOptions ? generationOptionsBody(args.generationOptions) : null,
  });
  return toEditResult(body);
}

export async function promptEdit(
  image: string,
  prompt: string,
  generationOptions?: GenerationOptions,
): Promise<GenerationResult> {
  const body = await post<GenerationResponseBody>("/api/prompt-edit", {
    image,
    prompt,
    generation_options: generationOptions ? generationOptionsBody(generationOptions) : null,
  });
  return toGenerationResult(body);
}

export async function generateImage(
  prompt: string,
  width: number,
  height: number,
  generationOptions?: GenerationOptions,
): Promise<GenerationResult> {
  const body = await post<GenerationResponseBody>("/api/generate", {
    prompt,
    width,
    height,
    generation_options: generationOptions ? generationOptionsBody(generationOptions) : null,
  });
  return toGenerationResult(body);
}

export async function outpaint(
  image: string,
  prompt: string,
  margins: OutpaintMargins,
  generationOptions?: GenerationOptions,
): Promise<EditResult> {
  const body = await post<EditResponseBody>("/api/outpaint", {
    image,
    prompt,
    margins: marginsBody(margins),
    generation_options: generationOptions ? generationOptionsBody(generationOptions) : null,
  });
  return toEditResult(body);
}

export async function upscale(image: string, options?: UpscaleOptions): Promise<GenerationResult> {
  const body = await post<GenerationResponseBody>("/api/upscale", {
    image,
    options: options ? upscaleOptionsBody(options) : null,
  });
  return toGenerationResult(body);
}
