"use client";

import * as React from "react";
import { toast } from "sonner";
import { UploadSimple } from "@phosphor-icons/react";
import { useEditorStore } from "@/lib/store/editor-store";
import * as api from "@/lib/api";
import { toPixelSelection, type ImageSize } from "@/lib/types";
import { computeRenderedImageRect, type RenderedImageRect } from "@/lib/image-geometry";
import { readFileAsDataUrl, validateImageFile } from "@/lib/image-file";

/**
 * Reference image for Add Object (Reference): shows the whole image
 * (object-contain, not cropped) and lets the user click the specific
 * subject to lift when the reference photo has more than one — mirrors
 * the point-segmentation flow on the main canvas, scoped to this image.
 * The resulting mask is sent as reference_mask (see object_insertion.py's
 * by_reference) so the backend only lifts the clicked subject.
 */
export function ReferenceImagePicker() {
  const referenceImage = useEditorStore((s) => s.referenceImage);
  const setReferenceImage = useEditorStore((s) => s.setReferenceImage);
  const referenceSelection = useEditorStore((s) => s.referenceSelection);
  const setReferenceSelection = useEditorStore((s) => s.setReferenceSelection);
  const referenceMaskPreview = useEditorStore((s) => s.referenceMaskPreview);
  const setReferenceMaskPreview = useEditorStore((s) => s.setReferenceMaskPreview);

  const containerRef = React.useRef<HTMLDivElement>(null);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const [renderedRect, setRenderedRect] = React.useState<RenderedImageRect | null>(null);
  const [imageSize, setImageSize] = React.useState<ImageSize | null>(null);
  const [segmenting, setSegmenting] = React.useState(false);

  const recomputeRenderedRect = React.useCallback(() => {
    const container = containerRef.current;
    const img = imageRef.current;
    if (!container || !img) return;
    setRenderedRect(computeRenderedImageRect(container, img));
  }, []);

  const handleImageLoad = React.useCallback(() => {
    recomputeRenderedRect();
    const img = imageRef.current;
    if (img && img.naturalWidth && img.naturalHeight) {
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    }
  }, [recomputeRenderedRect]);

  React.useEffect(() => {
    recomputeRenderedRect();
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(recomputeRenderedRect);
    observer.observe(container);
    return () => observer.disconnect();
  }, [referenceImage, recomputeRenderedRect]);

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const error = validateImageFile(file);
    if (error) {
      toast.error("Reference image rejected", { description: error });
      return;
    }
    setImageSize(null);
    setReferenceImage(await readFileAsDataUrl(file));
  }

  async function handleClick(event: React.MouseEvent) {
    const container = containerRef.current;
    const img = imageRef.current;
    if (!container || !img || !referenceImage || !imageSize || segmenting) return;
    const rect = computeRenderedImageRect(container, img);
    if (!rect) return;
    const containerRect = container.getBoundingClientRect();
    const localX = event.clientX - containerRect.left - rect.left;
    const localY = event.clientY - containerRect.top - rect.top;
    if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) return;

    const selection = {
      kind: "point" as const,
      points: [[localX / rect.width, localY / rect.height]] as [number, number][],
      labels: [1] as (0 | 1)[],
    };
    setReferenceSelection(selection);
    setSegmenting(true);
    try {
      const pixelSelection = toPixelSelection(selection, imageSize);
      const result = await api.segment(referenceImage, pixelSelection);
      setReferenceMaskPreview(result.masks[result.bestIndex] ?? null);
    } catch {
      toast.error("Could not select subject", { description: "Try clicking a different point." });
    } finally {
      setSegmenting(false);
    }
  }

  function clearSelection() {
    setReferenceSelection(null);
    setReferenceMaskPreview(null);
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-muted-foreground">Reference image</span>
        {referenceSelection && (
          <button
            type="button"
            onClick={clearSelection}
            className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          >
            Use whole image
          </button>
        )}
      </div>

      {referenceImage && (
        <div
          ref={containerRef}
          onClick={handleClick}
          className="relative flex h-32 w-full items-center justify-center overflow-hidden rounded-md border border-border bg-muted/20 cursor-crosshair"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imageRef}
            src={referenceImage}
            alt="Reference subject to insert"
            className="max-h-full max-w-full object-contain"
            onLoad={handleImageLoad}
          />
          {renderedRect && referenceMaskPreview && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={referenceMaskPreview}
              alt=""
              aria-hidden="true"
              className="pointer-events-none absolute opacity-40 mix-blend-screen"
              style={{
                left: renderedRect.left,
                top: renderedRect.top,
                width: renderedRect.width,
                height: renderedRect.height,
                filter: "sepia(1) saturate(6) hue-rotate(280deg)",
              }}
            />
          )}
          {segmenting && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/50 text-xs text-muted-foreground">
              Selecting…
            </div>
          )}
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        {referenceImage
          ? "Click the subject to lift if the photo has more than one — otherwise the whole image is used."
          : "Upload a photo containing the object you want to place."}
      </p>

      <label className="flex cursor-pointer items-center justify-center gap-1.5 rounded-md border border-dashed border-border py-2 text-xs text-muted-foreground hover:border-brand/50 hover:text-foreground">
        <UploadSimple className="size-3.5" />
        {referenceImage ? "Replace reference" : "Upload reference image"}
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="sr-only"
          onChange={handleUpload}
        />
      </label>
    </div>
  );
}
