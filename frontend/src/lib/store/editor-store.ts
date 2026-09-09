"use client";

import { create } from "zustand";
import {
  BoxPrompt,
  EditResult,
  GenerationOptions,
  GenerationResult,
  MaskOptions,
  OperationId,
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

interface EditorState {
  // Canvas
  sourceImage: string | null;
  sourceFileName: string | null;
  activeImage: string | null; // current working image (source, or latest result)
  /** Backend image_id for `activeImage`, once registered via POST /api/images.
   * Null means "not registered yet" — reset to null every time activeImage
   * changes so a stale id is never reused against different pixels. */
  activeImageId: string | null;
  referenceImage: string | null;

  // Selection / mask
  activeTool: OperationId | null;
  selection: SelectionPrompt | null;
  maskPreview: string | null;
  candidateMasks: string[];
  selectedMaskIndex: number;

  // Options (shared across tools that use them)
  maskOptions: MaskOptions;
  generationOptions: GenerationOptions;
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
  setActiveImageId: (imageId: string) => void;
  setReferenceImage: (dataUrl: string | null) => void;
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
  referenceImage: null,
  activeTool: null,
  selection: null,
  maskPreview: null,
  candidateMasks: [],
  selectedMaskIndex: 0,
  maskOptions: DEFAULT_MASK_OPTIONS,
  generationOptions: DEFAULT_GENERATION_OPTIONS,
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
      return "activeImage" in patch ? { ...patch, activeImageId: null } : patch;
    });
  }

  return {
    ...initial,

    setSourceImage: (dataUrl, fileName) =>
      setWithImageReset({
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
      }),

    setActiveImageId: (imageId) => set({ activeImageId: imageId }),

    setReferenceImage: (dataUrl) => set({ referenceImage: dataUrl }),

  setActiveTool: (tool) =>
    set({
      activeTool: tool,
      selection: null,
      maskPreview: null,
      candidateMasks: [],
      placement: null,
      errorMessage: null,
    }),

  setSelection: (selection) => set({ selection }),
  setPlacement: (box) => set({ placement: box }),
  setPrompt: (prompt) => set({ prompt }),

  setMaskOptions: (options) =>
    set((state) => ({ maskOptions: { ...state.maskOptions, ...options } })),
  setGenerationOptions: (options) =>
    set((state) => ({ generationOptions: { ...state.generationOptions, ...options } })),
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
      const history = [...truncated, entry];
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
      const history = [...truncated, entry];
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
