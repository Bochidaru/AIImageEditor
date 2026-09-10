/**
 * Mock implementation of the backend contract.
 *
 * IMPORTANT: `inpaint_core` (src/inpaint_core/) is a plain Python library
 * today — there is no HTTP/REST layer yet. This file stands in for that
 * future API. Every exported function's signature mirrors a real
 * ImageProcessor method 1:1 (see src/inpaint_core/processor.py); when a
 * real endpoint exists, only the function bodies below need to change —
 * no component should need to change its call shape.
 */

import {
  BoxPrompt,
  EditResult,
  GenerationOptions,
  GenerationResult,
  MaskOptions,
  OutpaintMargins,
  PointPrompt,
  ProcessingStage,
  SegmentationResult,
  SelectionPrompt,
  UpscaleOptions,
} from "@/lib/types";
import { ApiError } from "@/lib/api/errors";

// ---------------------------------------------------------------------------
// Timing / fake latency — tuned to feel like real model inference so the
// processing UI (progress + stage labels) can be designed against something
// realistic instead of an instant resolve.
// ---------------------------------------------------------------------------

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function stage(elapsedMs: number): ProcessingStage {
  return {
    elapsedMs,
    peakAllocatedMb: 4200 + Math.round(Math.random() * 3000),
    peakReservedMb: 5100 + Math.round(Math.random() * 3000),
  };
}

/** Simulated failure for the error-state UI; ~1 in 12 calls. */
function maybeFail(op: string) {
  if (Math.random() < 1 / 12) {
    throw new ApiError(
      `${op} failed: the model ran out of memory while processing this request.`,
      "OUT_OF_MEMORY",
    );
  }
}

export { ApiError };

// ---------------------------------------------------------------------------
// Placeholder pixels: a tinted 2x2 PNG data URL so results are visibly
// distinct from the source without shipping binary fixtures.
// ---------------------------------------------------------------------------

function tintedPlaceholder(seed: number): string {
  const hue = seed % 360;
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const gradient = ctx.createLinearGradient(0, 0, 512, 512);
  gradient.addColorStop(0, `hsl(${hue} 35% 18%)`);
  gradient.addColorStop(1, `hsl(${(hue + 40) % 360} 35% 28%)`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 512, 512);
  return canvas.toDataURL("image/png");
}

function maskPlaceholder(): string {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 512, 512);
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.ellipse(256, 256, 140, 170, 0, 0, Math.PI * 2);
  ctx.fill();
  return canvas.toDataURL("image/png");
}

function seedFromOptions(options: GenerationOptions): number {
  return options.seed;
}

function buildResult(mode: string, options: GenerationOptions, ms: number): EditResult {
  const seed = seedFromOptions(options);
  return {
    image: tintedPlaceholder(seed + mode.length),
    mask: maskPlaceholder(),
    generatedImage: tintedPlaceholder(seed + mode.length),
    metadata: {
      mode,
      seed,
      stages: { [mode]: stage(ms) },
      compositing: false,
    },
  };
}

// Mirrors buildResult for the prompt-only-edit operations (replace-background,
// add-object/prompt) which edit directly from a prompt instruction and never
// resolve a mask server-side — see operations/background_replacement.py and
// object_insertion.py's by_prompt.
function buildGenerationResult(mode: string, options: GenerationOptions, ms: number): GenerationResult {
  const seed = seedFromOptions(options);
  return {
    image: tintedPlaceholder(seed + mode.length),
    seed,
    metadata: { mode, stages: { [mode]: stage(ms) } },
  };
}

// ---------------------------------------------------------------------------
// Image registration (mock: instant, no real caching needed)
// ---------------------------------------------------------------------------

export async function registerImage(_image: string): Promise<string> {
  await wait(50);
  return `mock-${Math.random().toString(36).slice(2)}`;
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

export async function segment(
  _image: string,
  _selection: SelectionPrompt,
  _imageId?: string,
): Promise<SegmentationResult> {
  await wait(650);
  maybeFail("Segmentation");
  const masks = [maskPlaceholder(), maskPlaceholder(), maskPlaceholder()];
  return {
    masks,
    scores: [0.94, 0.81, 0.67],
    bestIndex: 0,
    metadata: { backend: "sam2", multimaskOutput: true },
  };
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
  requireExactlyOne(args.selection, args.mask, "selection", "mask");
  await wait(2600);
  maybeFail("Object removal");
  return buildResult("object_removal", args.generationOptions ?? defaultGen(), 2600);
}

export async function replaceObject(
  args: MaskedEditArgs & { prompt: string },
): Promise<EditResult> {
  requireExactlyOne(args.selection, args.mask, "selection", "mask");
  requirePrompt(args.prompt);
  await wait(3400);
  maybeFail("Object replacement");
  return buildResult("object_replacement", args.generationOptions ?? defaultGen(), 3400);
}

export async function replaceBackground(
  args: {
    image: string;
    prompt: string;
    generationOptions?: GenerationOptions;
  },
): Promise<GenerationResult> {
  requirePrompt(args.prompt);
  await wait(3800);
  maybeFail("Background replacement");
  return buildGenerationResult("background_replacement", args.generationOptions ?? defaultGen(), 3800);
}

export async function addObjectByPrompt(
  args: {
    image: string;
    placement?: BoxPrompt;
    prompt: string;
    generationOptions?: GenerationOptions;
  },
): Promise<GenerationResult> {
  requirePrompt(args.prompt);
  await wait(3100);
  maybeFail("Add object");
  return buildGenerationResult("object_insertion_prompt", args.generationOptions ?? defaultGen(), 3100);
}

export async function addObjectByReference(
  args: {
    image: string;
    reference: string;
    placement?: BoxPrompt;
    mask?: string;
    referenceMask?: string;
    maskOptions?: MaskOptions;
    generationOptions?: GenerationOptions;
  },
): Promise<EditResult> {
  requireExactlyOne(args.placement, args.mask, "placement", "mask");
  await wait(3300);
  maybeFail("Add object (reference)");
  return buildResult("object_insertion_reference", args.generationOptions ?? defaultGen(), 3300);
}

export async function promptEdit(
  image: string,
  prompt: string,
  generationOptions?: GenerationOptions,
): Promise<GenerationResult> {
  requirePrompt(prompt);
  await wait(2900);
  maybeFail("Prompt edit");
  const seed = seedFromOptions(generationOptions ?? defaultGen());
  return { image: tintedPlaceholder(seed + 11), seed, metadata: { backend: "flux2_klein" } };
}

export async function generateImage(
  prompt: string,
  width: number,
  height: number,
  generationOptions?: GenerationOptions,
): Promise<GenerationResult> {
  requirePrompt(prompt);
  await wait(3600);
  maybeFail("Image generation");
  const seed = seedFromOptions(generationOptions ?? defaultGen());
  return {
    image: tintedPlaceholder(seed + width + height),
    seed,
    metadata: { backend: "flux2_klein", width, height },
  };
}

export async function outpaint(
  image: string,
  prompt: string,
  margins: OutpaintMargins,
  generationOptions?: GenerationOptions,
): Promise<EditResult> {
  requirePrompt(prompt);
  if (margins.left + margins.top + margins.right + margins.bottom <= 0) {
    throw new ApiError("At least one outpaint margin must be positive.", "INVALID_MARGINS");
  }
  await wait(3700);
  maybeFail("Outpainting");
  return buildResult("outpainting", generationOptions ?? defaultGen(), 3700);
}

export async function upscale(
  image: string,
  options?: UpscaleOptions,
): Promise<GenerationResult> {
  await wait(2200);
  maybeFail("Upscaling");
  const scale = options?.scale ?? 4;
  return {
    image: tintedPlaceholder(scale * 97),
    seed: 0,
    metadata: { backend: "realesrgan", scale },
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function defaultGen(): GenerationOptions {
  return { seed: 42, numInferenceSteps: 28 };
}

function requirePrompt(prompt: string) {
  if (!prompt || !prompt.trim()) {
    throw new ApiError("A prompt is required.", "EMPTY_PROMPT");
  }
}

function requireExactlyOne(a: unknown, b: unknown, nameA: string, nameB: string) {
  const hasA = a !== undefined && a !== null;
  const hasB = b !== undefined && b !== null;
  if (hasA === hasB) {
    throw new ApiError(`Provide exactly one of ${nameA} or ${nameB}.`, "INVALID_SELECTION");
  }
}

export type { PointPrompt };
