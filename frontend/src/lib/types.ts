/**
 * Mirrors src/inpaint_core/types.py and the operation signatures on
 * ImageProcessor (src/inpaint_core/processor.py). Field names are
 * camelCased for TS/JS convention; the mock API layer in `lib/api/`
 * is the only place that would need to change shape when a real
 * HTTP backend is introduced.
 */

// ---------------------------------------------------------------------------
// Selection prompts (segmentation input)
// ---------------------------------------------------------------------------

export interface PointPrompt {
  kind: "point";
  points: [x: number, y: number][];
  /** 1 = positive (include), 0 = negative (exclude). */
  labels: (0 | 1)[];
}

export interface BoxPrompt {
  kind: "box";
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export type SelectionPrompt = PointPrompt | BoxPrompt;

// ---------------------------------------------------------------------------
// Options (mirrors MaskOptions / GenerationOptions / OutpaintMargins / UpscaleOptions)
// ---------------------------------------------------------------------------

export interface MaskOptions {
  threshold: number; // default 127
  dilate: number; // default 0
  erode: number; // default 0
  candidateIndex?: number;
}

export const DEFAULT_MASK_OPTIONS: MaskOptions = {
  threshold: 127,
  dilate: 0,
  erode: 0,
};

export interface GenerationOptions {
  seed: number; // default 42
  numInferenceSteps: number; // default 28
  guidanceScale?: number;
}

export const DEFAULT_GENERATION_OPTIONS: GenerationOptions = {
  seed: 42,
  numInferenceSteps: 28,
};

export interface OutpaintMargins {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface UpscaleOptions {
  scale: number; // default 4
  faceEnhance: boolean; // default false
  tile: number; // default 0
  tilePad: number; // default 10
  prePad: number; // default 0
}

export const DEFAULT_UPSCALE_OPTIONS: UpscaleOptions = {
  scale: 4,
  faceEnhance: false,
  tile: 0,
  tilePad: 10,
  prePad: 0,
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

export interface ProcessingStage {
  elapsedMs: number;
  peakAllocatedMb: number | null;
  peakReservedMb: number | null;
}

export interface SegmentationResult {
  /** Object URLs / data URLs for each candidate mask, uint8 0/255 PNG. */
  masks: string[];
  scores: number[];
  bestIndex: number;
  metadata: Record<string, unknown>;
}

export interface GenerationResult {
  image: string;
  seed: number;
  metadata: Record<string, unknown>;
}

export interface EditResult {
  image: string;
  mask: string;
  generatedImage?: string;
  metadata: {
    mode: string;
    seed: number;
    stages: Record<string, ProcessingStage>;
    [key: string]: unknown;
  };
}

// ---------------------------------------------------------------------------
// Operations — the 10 capabilities verified on ImageProcessor
// ---------------------------------------------------------------------------

export type OperationId =
  | "segment"
  | "remove_object"
  | "replace_object"
  | "replace_background"
  | "add_object_by_prompt"
  | "add_object_by_reference"
  | "prompt_edit"
  | "generate_image"
  | "outpaint"
  | "upscale";

export interface OperationMeta {
  id: OperationId;
  label: string;
  shortLabel: string;
  description: string;
  model: string;
  /** Whether the tool needs a source image loaded on the canvas first. */
  requiresImage: boolean;
  /** Whether the tool needs a mask/selection (point, box, or drawn). */
  requiresMask: boolean;
  /** Whether the tool needs a free-text prompt. */
  requiresPrompt: boolean;
  /** Whether the tool needs a second reference image upload. */
  requiresReference: boolean;
}

export const OPERATIONS: OperationMeta[] = [
  {
    id: "remove_object",
    label: "Remove Object",
    shortLabel: "Remove",
    description: "Erase a selected object and reconstruct the background.",
    model: "OmniPaint",
    requiresImage: true,
    requiresMask: true,
    requiresPrompt: false,
    requiresReference: false,
  },
  {
    id: "replace_object",
    label: "Replace Object",
    shortLabel: "Replace",
    description: "Swap a selected object for something described by a prompt.",
    model: "FLUX Fill",
    requiresImage: true,
    requiresMask: true,
    requiresPrompt: true,
    requiresReference: false,
  },
  {
    id: "replace_background",
    label: "Replace Background",
    shortLabel: "Background",
    description: "Keep the selected foreground, regenerate everything else.",
    model: "FLUX Fill",
    requiresImage: true,
    requiresMask: true,
    requiresPrompt: true,
    requiresReference: false,
  },
  {
    id: "add_object_by_prompt",
    label: "Add Object (Prompt)",
    shortLabel: "Add · Prompt",
    description: "Place a new object described by text into a chosen region.",
    model: "FLUX Fill",
    requiresImage: true,
    requiresMask: true,
    requiresPrompt: true,
    requiresReference: false,
  },
  {
    id: "add_object_by_reference",
    label: "Add Object (Reference)",
    shortLabel: "Add · Reference",
    description: "Place a subject from a second image into a chosen region.",
    model: "OmniPaint",
    requiresImage: true,
    requiresMask: true,
    requiresPrompt: false,
    requiresReference: true,
  },
  {
    id: "prompt_edit",
    label: "Prompt Edit",
    shortLabel: "Edit",
    description: "Edit the whole image by describing the change, no mask needed.",
    model: "FLUX Kontext",
    requiresImage: true,
    requiresMask: false,
    requiresPrompt: true,
    requiresReference: false,
  },
  {
    id: "outpaint",
    label: "Outpaint",
    shortLabel: "Outpaint",
    description: "Extend the canvas beyond its original edges.",
    model: "FLUX Fill",
    requiresImage: true,
    requiresMask: false,
    requiresPrompt: true,
    requiresReference: false,
  },
  {
    id: "upscale",
    label: "Upscale",
    shortLabel: "Upscale",
    description: "Increase resolution and optionally enhance faces.",
    model: "Real-ESRGAN",
    requiresImage: true,
    requiresMask: false,
    requiresPrompt: false,
    requiresReference: false,
  },
  {
    id: "generate_image",
    label: "Generate Image",
    shortLabel: "Generate",
    description: "Create a new image from a text prompt.",
    model: "FLUX.1-dev",
    requiresImage: false,
    requiresMask: false,
    requiresPrompt: true,
    requiresReference: false,
  },
];

export type ProcessingStatus = "idle" | "segmenting" | "processing" | "success" | "error";
