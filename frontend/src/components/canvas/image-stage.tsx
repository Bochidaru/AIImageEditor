"use client";

import * as React from "react";
import { toast } from "sonner";
import { useEditorStore } from "@/lib/store/editor-store";
import { OPERATIONS } from "@/lib/types";
import * as api from "@/lib/api";
import { CompareSlider } from "@/components/canvas/compare-slider";

const DRAG_THRESHOLD = 6;

interface RenderedImageRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Where an `object-contain` image actually renders inside its (usually
 * larger, different-aspect-ratio) container — the letterbox offset plus the
 * scaled-down width/height. Both click-coordinate math (toImageSpace) and
 * overlay positioning (points/box) must agree on this same rect, or the
 * overlay drifts away from the image by exactly the letterbox gap whenever
 * the image's aspect ratio doesn't match the container's.
 */
function computeRenderedImageRect(
  container: HTMLDivElement,
  imageEl: HTMLImageElement,
): RenderedImageRect | null {
  if (!imageEl.naturalWidth || !imageEl.naturalHeight) return null;
  const containerRect = container.getBoundingClientRect();
  const scale = Math.min(
    containerRect.width / imageEl.naturalWidth,
    containerRect.height / imageEl.naturalHeight,
  );
  const width = imageEl.naturalWidth * scale;
  const height = imageEl.naturalHeight * scale;
  return {
    width,
    height,
    left: (containerRect.width - width) / 2,
    top: (containerRect.height - height) / 2,
  };
}

/** Normalized (0-1) image-space point derived from a client mouse event. */
function toImageSpace(
  event: React.MouseEvent,
  container: HTMLDivElement,
  imageEl: HTMLImageElement,
) {
  const containerRect = container.getBoundingClientRect();
  const rect = computeRenderedImageRect(container, imageEl);
  const { left: offsetX, top: offsetY, width: renderedWidth, height: renderedHeight } = rect ?? {
    left: 0,
    top: 0,
    width: containerRect.width,
    height: containerRect.height,
  };

  const localX = event.clientX - containerRect.left - offsetX;
  const localY = event.clientY - containerRect.top - offsetY;

  return {
    x: Math.min(1, Math.max(0, localX / renderedWidth)),
    y: Math.min(1, Math.max(0, localY / renderedHeight)),
    withinImage:
      localX >= 0 && localY >= 0 && localX <= renderedWidth && localY <= renderedHeight,
  };
}

export function ImageStage() {
  const activeImage = useEditorStore((s) => s.activeImage);
  const sourceImage = useEditorStore((s) => s.sourceImage);
  const activeTool = useEditorStore((s) => s.activeTool);
  const selection = useEditorStore((s) => s.selection);
  const setSelection = useEditorStore((s) => s.setSelection);
  const placement = useEditorStore((s) => s.placement);
  const setPlacement = useEditorStore((s) => s.setPlacement);
  const maskPreview = useEditorStore((s) => s.maskPreview);
  const setCandidateMasks = useEditorStore((s) => s.setCandidateMasks);
  const setActiveImageId = useEditorStore((s) => s.setActiveImageId);
  const beginProcessing = useEditorStore((s) => s.beginProcessing);
  const clearError = useEditorStore((s) => s.clearError);
  const status = useEditorStore((s) => s.status);

  const containerRef = React.useRef<HTMLDivElement>(null);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const [dragStart, setDragStart] = React.useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = React.useState<{ x: number; y: number } | null>(null);
  // The image's actual on-screen rect (see computeRenderedImageRect) — point
  // markers, the mask preview, and the selection box are positioned against
  // this, not the container, so they land exactly on the image regardless of
  // letterboxing.
  const [renderedRect, setRenderedRect] = React.useState<RenderedImageRect | null>(null);
  // In-flight /api/images registration, keyed by image, so two rapid clicks
  // against the same unregistered image share one request instead of each
  // firing its own upload.
  const pendingRegistration = React.useRef<{ image: string; promise: Promise<string | undefined> } | null>(
    null,
  );

  const meta = OPERATIONS.find((op) => op.id === activeTool);
  const usesPlacement =
    activeTool === "add_object_by_prompt" || activeTool === "add_object_by_reference";
  const interactive = Boolean(meta?.requiresMask) && status !== "processing";
  const showingResult = useEditorStore((s) => s.historyIndex) >= 0;

  const recomputeRenderedRect = React.useCallback(() => {
    const container = containerRef.current;
    const img = imageRef.current;
    if (!container || !img) return;
    setRenderedRect(computeRenderedImageRect(container, img));
  }, []);

  // Recompute whenever the container is resized (window resize, sidebar
  // toggle, etc.) and whenever a different image becomes active — an image
  // swap can change aspect ratio without the container itself resizing.
  React.useEffect(() => {
    if (showingResult) return;
    recomputeRenderedRect();
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(recomputeRenderedRect);
    observer.observe(container);
    return () => observer.disconnect();
  }, [activeImage, showingResult, recomputeRenderedRect]);

  /**
   * Segmentation runs once per point/box a user tries while hunting for the
   * right selection, all against the same unchanged image — register it
   * once (POST /api/images) and reuse the id instead of re-sending the full
   * base64 payload on every single click. Falls back to sending the image
   * inline if registration fails or is skipped (e.g. mock mode).
   */
  async function ensureImageId(image: string): Promise<string | undefined> {
    const cached = useEditorStore.getState().activeImageId;
    if (cached) return cached;

    // A registration for this exact image is already in flight (e.g. a
    // second click landed before the first one's request resolved) — share
    // it instead of firing a duplicate upload.
    if (pendingRegistration.current?.image === image) {
      return pendingRegistration.current.promise;
    }

    const promise = (async () => {
      try {
        const imageId = await api.registerImage(image);
        // Only persist if the active image hasn't changed while this was in
        // flight (e.g. the user loaded a different image mid-request).
        if (useEditorStore.getState().activeImage === image) {
          setActiveImageId(imageId);
        }
        return imageId;
      } catch {
        return undefined;
      } finally {
        if (pendingRegistration.current?.image === image) {
          pendingRegistration.current = null;
        }
      }
    })();

    pendingRegistration.current = { image, promise };
    return promise;
  }

  /** True once `image` is no longer the active image — e.g. the user loaded
   * a different one while a segmentation request for it was in flight. */
  function isStaleImage(image: string): boolean {
    return useEditorStore.getState().activeImage !== image;
  }

  async function resolveSegmentation(x: number, y: number, negative: boolean) {
    if (!activeImage) return;
    const nextSelection = { kind: "point" as const, points: [[x, y]] as [number, number][], labels: [negative ? 0 : 1] as (0 | 1)[] };
    setSelection(nextSelection);
    beginProcessing("Segmenting selection…");
    try {
      const imageId = await ensureImageId(activeImage);
      const result = await api.segment(activeImage, nextSelection, imageId);
      // The user may have loaded a different image while this was in
      // flight — discard results computed for the now-abandoned one.
      if (isStaleImage(activeImage)) return;
      setCandidateMasks(result.masks, result.bestIndex);
      clearError();
    } catch {
      toast.error("Segmentation failed", { description: "Try clicking a different point." });
    }
  }

  function handleClick(event: React.MouseEvent) {
    if (!interactive || usesPlacement || !containerRef.current || !imageRef.current) return;
    const { x, y, withinImage } = toImageSpace(event, containerRef.current, imageRef.current);
    if (!withinImage) return;
    void resolveSegmentation(x, y, event.shiftKey);
  }

  function handlePointerDown(event: React.PointerEvent) {
    if (!interactive || !containerRef.current || !imageRef.current) return;
    const { x, y, withinImage } = toImageSpace(event, containerRef.current, imageRef.current);
    if (!withinImage) return;
    setDragStart({ x, y });
    setDragCurrent({ x, y });
  }

  function handlePointerMove(event: React.PointerEvent) {
    if (!dragStart || !containerRef.current || !imageRef.current) return;
    const { x, y } = toImageSpace(event, containerRef.current, imageRef.current);
    setDragCurrent({ x, y });
  }

  async function handlePointerUp(event: React.PointerEvent) {
    if (!dragStart || !dragCurrent || !containerRef.current) return;
    const containerRect = containerRef.current.getBoundingClientRect();
    const distancePx =
      Math.abs(dragCurrent.x - dragStart.x) * containerRect.width +
      Math.abs(dragCurrent.y - dragStart.y) * containerRect.height;

    if (distancePx < DRAG_THRESHOLD) {
      setDragStart(null);
      setDragCurrent(null);
      if (usesPlacement) return; // treated as a no-op tap in placement mode
      handleClick(event);
      return;
    }

    const box = {
      kind: "box" as const,
      x1: Math.min(dragStart.x, dragCurrent.x),
      y1: Math.min(dragStart.y, dragCurrent.y),
      x2: Math.max(dragStart.x, dragCurrent.x),
      y2: Math.max(dragStart.y, dragCurrent.y),
    };
    setDragStart(null);
    setDragCurrent(null);

    if (usesPlacement) {
      setPlacement(box);
      return;
    }

    if (!activeImage) return;
    setSelection(box);
    beginProcessing("Segmenting selection…");
    try {
      const imageId = await ensureImageId(activeImage);
      const result = await api.segment(activeImage, box, imageId);
      // The user may have loaded a different image while this was in
      // flight — discard results computed for the now-abandoned one.
      if (isStaleImage(activeImage)) return;
      setCandidateMasks(result.masks, result.bestIndex);
      clearError();
    } catch {
      toast.error("Segmentation failed", { description: "Try a different box." });
    }
  }

  const box =
    dragStart && dragCurrent
      ? {
          x1: Math.min(dragStart.x, dragCurrent.x),
          y1: Math.min(dragStart.y, dragCurrent.y),
          x2: Math.max(dragStart.x, dragCurrent.x),
          y2: Math.max(dragStart.y, dragCurrent.y),
        }
      : null;

  const persistedBox = usesPlacement ? placement : selection?.kind === "box" ? selection : null;
  const points = !usesPlacement && selection?.kind === "point" ? selection.points : [];

  if (!activeImage) return null;

  return (
    <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-canvas p-6">
      <div
        ref={containerRef}
        onClick={handleClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        className="relative flex h-full max-h-full w-full max-w-full items-center justify-center"
        style={{ cursor: interactive ? (usesPlacement ? "crosshair" : "pointer") : "default" }}
      >
        {showingResult && sourceImage ? (
          <CompareSlider before={sourceImage} after={activeImage} />
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            ref={imageRef}
            src={activeImage}
            alt="Loaded canvas"
            className="max-h-full max-w-full select-none object-contain"
            draggable={false}
            onLoad={recomputeRenderedRect}
          />
        )}

        {/* Positioned to exactly match the rendered image (not the
            surrounding container) so overlays never drift off by the
            letterbox gap — see computeRenderedImageRect. */}
        {!showingResult && renderedRect && (
          <div
            className="pointer-events-none absolute"
            style={{
              left: renderedRect.left,
              top: renderedRect.top,
              width: renderedRect.width,
              height: renderedRect.height,
            }}
          >
            {maskPreview && interactive && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={maskPreview}
                alt=""
                aria-hidden="true"
                className="h-full w-full object-contain opacity-40 mix-blend-screen"
                style={{ filter: "sepia(1) saturate(6) hue-rotate(280deg)" }}
              />
            )}

            {points.map(([x, y], index) => (
              <span
                key={index}
                className="absolute size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-brand shadow"
                style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
              />
            ))}

            {(box ?? persistedBox) && (
              <div
                className="absolute border-2 border-brand bg-brand/10"
                style={{
                  left: `${(box ?? persistedBox)!.x1 * 100}%`,
                  top: `${(box ?? persistedBox)!.y1 * 100}%`,
                  width: `${((box ?? persistedBox)!.x2 - (box ?? persistedBox)!.x1) * 100}%`,
                  height: `${((box ?? persistedBox)!.y2 - (box ?? persistedBox)!.y1) * 100}%`,
                }}
              />
            )}
          </div>
        )}
      </div>

      {interactive && !showingResult && (
        <p className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-background/80 px-3 py-1 text-xs text-muted-foreground backdrop-blur-sm">
          {usesPlacement
            ? "Drag a box to place the object"
            : "Click a point, or drag a box · Shift+click to exclude"}
        </p>
      )}
    </div>
  );
}
