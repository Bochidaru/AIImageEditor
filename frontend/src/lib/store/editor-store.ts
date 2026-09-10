"use client";

import { create } from "zustand";
import {
  BoxPrompt,
  EditResult,
  GenerationOptions,
  GenerationResult,
  ImageSize,
  MaskOptions,
  OperationId,
  OPERATIONS_BY_ID,
  OutpaintMargins,
  ProcessingStatus,
  SelectionPrompt,
  UpscaleOptions,
  DEFAULT_GENERATION_OPTIONS,
  DEFAULT_MASK_OPTIONS,
  DEFAULT_UPSCALE_OPTIONS,
} from "@/lib/types";

export interface HistoryEntry {
  id: string;
  operation: OperationId;
  image: string;
  createdAt: number;
  metadata: Record<string, unknown>;
}

/** Caps the in-memory history list — each entry retains a full-resolution
 * base64 image, so an unbounded list grows without limit over a long
 * session. Mirrors the backend's ImageCache max_entries cap. */
const MAX_HISTORY_ENTRIES = 30;

/** Each tool's backend is tuned to a different step count (Flux2 Klein: 4,
 * Flux Fill/OmniPaint: 28 — see OperationMeta.defaultSteps). Single shared
 * lookup used by every action that needs to apply or re-apply a tool's tuned
 * default, so the "which default does this tool get" question has exactly
 * one answer instead of being independently re-derived per call site. */
function defaultStepsFor(tool: OperationId | null): number | undefined {
  return tool !== null ? OPERATIONS_BY_ID[tool].defaultSteps : undefined;
}

/** Records or clears `tool`'s entry in customStepsByTool for a newly-set
 * numInferenceSteps value — the single place setGenerationOptions' per-tool
 * memory logic lives, so it reads as one step there instead of being
 * inlined among the rest of that action's option-merging. */
function withTrackedSteps(
  customStepsByTool: Partial<Record<OperationId, number>>,
  tool: OperationId,
  numInferenceSteps: number,
): Partial<Record<OperationId, number>> {
  const next = { ...customStepsByTool };
  if (numInferenceSteps === defaultStepsFor(tool)) {
    // Back at this tool's own default — nothing to remember.
    delete next[tool];
  } else {
    next[tool] = numInferenceSteps;
  }
  return next;
}

interface EditorState {
  // Canvas
  sourceImage: string | null;
  sourceFileName: string | null;
  activeImage: string | null; // current working image (source, or latest result)
  /** Backend image_id for `activeImage`, once registered via POST /api/images.
   * Null means "not registered yet" — reset to null every time activeImage
   * changes so a stale id is never reused against different pixels. */
  activeImageId: string | null;
  /** Natural pixel dimensions of `activeImage`, set once the <img> element
   * loads — used to convert normalized [0,1] selection/placement coordinates
   * to the pixel coordinates the backend expects. Reset to null whenever
   * activeImage changes, same as activeImageId. */
  activeImageSize: ImageSize | null;
  referenceImage: string | null;
  /** Backend image_id for `referenceImage`, once registered via POST
   * /api/images — same caching purpose as activeImageId, so trying several
   * points against the same reference photo doesn't re-upload it each time.
   * Reset to null whenever referenceImage changes. */
  referenceImageId: string | null;
  /** Point selection made on referenceImage (Add Object · Reference), used
   * to isolate one subject out of a reference photo that has several. */
  referenceSelection: SelectionPrompt | null;
  /** Segmentation-result mask (base64) for referenceSelection — sent as
   * reference_mask so the backend only lifts the selected subject instead
   * of the whole reference image. */
  referenceMaskPreview: string | null;

  // Selection / mask
  activeTool: OperationId | null;
  selection: SelectionPrompt | null;
  maskPreview: string | null;
  candidateMasks: string[];
  selectedMaskIndex: number;

  // Options (shared across tools that use them)
  maskOptions: MaskOptions;
  generationOptions: GenerationOptions;
  /** Per-tool manual Steps overrides, keyed by OperationId. A tool present
   * here (with a value different from its own tuned default) keeps that
   * value across any number of tool switches; a tool absent from this map
   * always falls back to its own tuned default (OperationMeta.defaultSteps)
   * — so customizing Steps for one tool never leaks into a different tool
   * the user hasn't touched, including one selected for the first time.
   * Cleared entirely by setSourceImage (a fresh editing session starts
   * every tool back at its own tuned default). */
  customStepsByTool: Partial<Record<OperationId, number>>;
  upscaleOptions: UpscaleOptions;
  outpaintMargins: OutpaintMargins;
  placement: BoxPrompt | null;
  prompt: string;

  // Processing
  status: ProcessingStatus;
  errorMessage: string | null;
  progressLabel: string | null;

  // History
  history: HistoryEntry[];
  historyIndex: number; // -1 = at source

  // Actions
  setSourceImage: (dataUrl: string, fileName: string) => void;
  setActiveImageId: (imageId: string | null) => void;
  setActiveImageSize: (size: ImageSize) => void;
  setReferenceImage: (dataUrl: string | null) => void;
  setReferenceImageId: (imageId: string | null) => void;
  setReferenceSelection: (selection: SelectionPrompt | null) => void;
  setReferenceMaskPreview: (mask: string | null) => void;
  setActiveTool: (tool: OperationId | null) => void;
  setSelection: (selection: SelectionPrompt | null) => void;
  setPlacement: (box: BoxPrompt | null) => void;
  setPrompt: (prompt: string) => void;
  setMaskOptions: (options: Partial<MaskOptions>) => void;
  setGenerationOptions: (options: Partial<GenerationOptions>) => void;
  setUpscaleOptions: (options: Partial<UpscaleOptions>) => void;
  setOutpaintMargins: (margins: Partial<OutpaintMargins>) => void;
  setCandidateMasks: (masks: string[], bestIndex: number) => void;
  selectCandidateMask: (index: number) => void;
  beginProcessing: (label: string) => void;
  resolveEdit: (operation: OperationId, result: EditResult) => void;
  resolveGeneration: (operation: OperationId, result: GenerationResult) => void;
  fail: (message: string) => void;
  clearError: () => void;
  reset: () => void;
  undo: () => void;
  redo: () => void;
  jumpToHistory: (index: number) => void;
}

const initial = {
  sourceImage: null,
  sourceFileName: null,
  activeImage: null,
  activeImageId: null,
  activeImageSize: null,
  referenceImage: null,
  referenceImageId: null,
  referenceSelection: null,
  referenceMaskPreview: null,
  activeTool: null,
  selection: null,
  maskPreview: null,
  candidateMasks: [],
  selectedMaskIndex: 0,
  maskOptions: DEFAULT_MASK_OPTIONS,
  generationOptions: DEFAULT_GENERATION_OPTIONS,
  customStepsByTool: {},
  upscaleOptions: DEFAULT_UPSCALE_OPTIONS,
  outpaintMargins: { left: 0, top: 0, right: 0, bottom: 0 },
  placement: null,
  prompt: "",
  status: "idle" as ProcessingStatus,
  errorMessage: null,
  progressLabel: null,
  history: [] as HistoryEntry[],
  historyIndex: -1,
};

export const useEditorStore = create<EditorState>((set, get) => {
  /**
   * Wraps zustand's `set`: whenever a patch touches `activeImage`, this also
   * resets `activeImageId` to null in the same update. Every action below
   * that changes the active image goes through this instead of the raw
   * `set`, so a future call site that changes `activeImage` gets the
   * "clear the stale id" invariant automatically instead of needing to
   * remember to add `activeImageId: null` by hand.
   */
  function setWithImageReset(
    partial: Partial<EditorState> | ((state: EditorState) => Partial<EditorState>),
  ) {
    set((state) => {
      const patch = typeof partial === "function" ? partial(state) : partial;
      return "activeImage" in patch ? { ...patch, activeImageId: null, activeImageSize: null } : patch;
    });
  }

  return {
    ...initial,

    setSourceImage: (dataUrl, fileName) =>
      setWithImageReset((state) => ({
        sourceImage: dataUrl,
        sourceFileName: fileName,
        activeImage: dataUrl,
        history: [],
        historyIndex: -1,
        status: "idle",
        errorMessage: null,
        selection: null,
        maskPreview: null,
        candidateMasks: [],
        // A fresh editing session starts every tool back at its own tuned
        // default — without this, the Steps slider would keep showing a
        // value customized for the previous image.
        customStepsByTool: {},
        generationOptions: {
          ...state.generationOptions,
          numInferenceSteps: defaultStepsFor(state.activeTool) ?? state.generationOptions.numInferenceSteps,
        },
      })),

    setActiveImageId: (imageId) => set({ activeImageId: imageId }),
    setActiveImageSize: (size) => set({ activeImageSize: size }),

    setReferenceImage: (dataUrl) =>
      set({
        referenceImage: dataUrl,
        referenceImageId: null,
        referenceSelection: null,
        referenceMaskPreview: null,
      }),
    setReferenceImageId: (imageId) => set({ referenceImageId: imageId }),
    setReferenceSelection: (selection) => set({ referenceSelection: selection }),
    setReferenceMaskPreview: (mask) => set({ referenceMaskPreview: mask }),

  setActiveTool: (tool) =>
    set((state) => {
      // Use this tool's own remembered override if it has one, else its
      // tuned default — keyed per-tool (customStepsByTool) so a manual
      // Steps edit for one tool never leaks into a different tool the user
      // hasn't touched, including one being selected for the first time.
      const steps = (tool !== null ? state.customStepsByTool[tool] : undefined) ?? defaultStepsFor(tool);
      return {
        activeTool: tool,
        selection: null,
        maskPreview: null,
        candidateMasks: [],
        placement: null,
        // Clear a stale error overlay from the previous tool, but don't
        // stomp an in-flight "processing" run just because the user switched
        // tools while it's still going.
        status: state.status === "error" ? "idle" : state.status,
        errorMessage: null,
        generationOptions:
          steps !== undefined
            ? { ...state.generationOptions, numInferenceSteps: steps }
            : state.generationOptions,
      };
    }),

  setSelection: (selection) => set({ selection }),
  setPlacement: (box) => set({ placement: box }),
  setPrompt: (prompt) => set({ prompt }),

  setMaskOptions: (options) =>
    set((state) => ({ maskOptions: { ...state.maskOptions, ...options } })),
  setGenerationOptions: (options) =>
    set((state) => {
      const generationOptions = { ...state.generationOptions, ...options };
      // The single write path for numInferenceSteps (the Steps slider goes
      // through this same action) — record/clear this tool's override here
      // so no other call site can change the value without the tracking
      // that setActiveTool depends on going stale.
      if (options.numInferenceSteps === undefined || state.activeTool === null) {
        return { generationOptions };
      }
      const customStepsByTool = withTrackedSteps(
        state.customStepsByTool,
        state.activeTool,
        options.numInferenceSteps,
      );
      return { generationOptions, customStepsByTool };
    }),
  setUpscaleOptions: (options) =>
    set((state) => ({ upscaleOptions: { ...state.upscaleOptions, ...options } })),
  setOutpaintMargins: (margins) =>
    set((state) => ({ outpaintMargins: { ...state.outpaintMargins, ...margins } })),

  setCandidateMasks: (masks, bestIndex) =>
    set({
      candidateMasks: masks,
      selectedMaskIndex: bestIndex,
      maskPreview: masks[bestIndex] ?? null,
    }),
  selectCandidateMask: (index) =>
    set((state) => ({
      selectedMaskIndex: index,
      maskPreview: state.candidateMasks[index] ?? null,
    })),

  beginProcessing: (label) =>
    set({ status: "processing", progressLabel: label, errorMessage: null }),

  resolveEdit: (operation, result) => {
    const entry: HistoryEntry = {
      id: crypto.randomUUID(),
      operation,
      image: result.image,
      createdAt: Date.now(),
      metadata: result.metadata,
    };
    setWithImageReset((state) => {
      const truncated = state.history.slice(0, state.historyIndex + 1);
      const history = [...truncated, entry].slice(-MAX_HISTORY_ENTRIES);
      return {
        status: "success",
        activeImage: result.image,
        history,
        historyIndex: history.length - 1,
        progressLabel: null,
      };
    });
  },

  resolveGeneration: (operation, result) => {
    const entry: HistoryEntry = {
      id: crypto.randomUUID(),
      operation,
      image: result.image,
      createdAt: Date.now(),
      metadata: result.metadata,
    };
    setWithImageReset((state) => {
      const truncated = state.history.slice(0, state.historyIndex + 1);
      const history = [...truncated, entry].slice(-MAX_HISTORY_ENTRIES);
      return {
        status: "success",
        activeImage: result.image,
        sourceImage: state.sourceImage ?? result.image,
        history,
        historyIndex: history.length - 1,
        progressLabel: null,
      };
    });
  },

  fail: (message) => set({ status: "error", errorMessage: message, progressLabel: null }),
  clearError: () => set({ status: "idle", errorMessage: null }),

  reset: () => set({ ...initial, history: [] }),

  undo: () => {
    const { historyIndex, history, sourceImage } = get();
    if (historyIndex < 0) return;
    const nextIndex = historyIndex - 1;
    setWithImageReset({
      historyIndex: nextIndex,
      activeImage: nextIndex < 0 ? sourceImage : history[nextIndex].image,
      status: "idle",
    });
  },

  redo: () => {
    const { historyIndex, history } = get();
    if (historyIndex >= history.length - 1) return;
    const nextIndex = historyIndex + 1;
    setWithImageReset({
      historyIndex: nextIndex,
      activeImage: history[nextIndex].image,
      status: "idle",
    });
  },

  jumpToHistory: (index) => {
    const { history, sourceImage } = get();
    if (index < -1 || index >= history.length) return;
    setWithImageReset({
      historyIndex: index,
      activeImage: index < 0 ? sourceImage : history[index].image,
      status: "idle",
    });
  },
  };
});
