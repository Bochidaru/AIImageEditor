"use client";

import * as React from "react";
import { toast } from "sonner";
import { UploadSimple, CursorClick } from "@phosphor-icons/react";
import { useEditorStore } from "@/lib/store/editor-store";
import * as api from "@/lib/api";
import { toPixelSelection, type ImageSize } from "@/lib/types";
import { computeRenderedImageRect, type RenderedImageRect } from "@/lib/image-geometry";
import { readFileAsDataUrl, validateImageFile } from "@/lib/image-file";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";

/** Renders `image` plus its segmentation-mask overlay (if any), tracking the
 * letterboxed rect itself so overlay + click math stay correct at any size —
 * shared between the small thumbnail and the larger picker dialog. */
function ReferenceCanvas({
  image,
  mask,
  onImageSize,
  onPointClick,
  className,
  imgClassName,
}: {
  image: string;
  mask: string | null;
  onImageSize?: (size: ImageSize) => void;
  onPointClick?: (point: [number, number]) => void;
  className?: string;
  imgClassName?: string;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const [rect, setRect] = React.useState<RenderedImageRect | null>(null);

  const recomputeRect = React.useCallback(() => {
    const container = containerRef.current;
    const img = imageRef.current;
    if (!container || !img) return;
    setRect(computeRenderedImageRect(container, img));
  }, []);

  const handleLoad = React.useCallback(() => {
    recomputeRect();
    const img = imageRef.current;
    if (img && img.naturalWidth && img.naturalHeight) {
      onImageSize?.({ width: img.naturalWidth, height: img.naturalHeight });
    }
  }, [recomputeRect, onImageSize]);

  React.useEffect(() => {
    recomputeRect();
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(recomputeRect);
    observer.observe(container);
    return () => observer.disconnect();
  }, [image, recomputeRect]);

  function handleClick(event: React.MouseEvent) {
    if (!onPointClick) return;
    const container = containerRef.current;
    const img = imageRef.current;
    if (!container || !img) return;
    const liveRect = computeRenderedImageRect(container, img);
    if (!liveRect) return;
    const containerRect = container.getBoundingClientRect();
    const localX = event.clientX - containerRect.left - liveRect.left;
    const localY = event.clientY - containerRect.top - liveRect.top;
    if (localX < 0 || localY < 0 || localX > liveRect.width || localY > liveRect.height) return;
    onPointClick([localX / liveRect.width, localY / liveRect.height]);
  }

  return (
    <div
      ref={containerRef}
      onClick={onPointClick ? handleClick : undefined}
      className={className}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imageRef}
        src={image}
        alt="Reference subject to insert"
        className={imgClassName}
        onLoad={handleLoad}
      />
      {rect && mask && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={mask}
          alt=""
          aria-hidden="true"
          className="pointer-events-none absolute opacity-40 mix-blend-screen"
          style={{
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height,
            filter: "sepia(1) saturate(6) hue-rotate(280deg)",
          }}
        />
      )}
    </div>
  );
}

/**
 * Reference image for Add Object (Reference): the thumbnail always shows the
 * whole image (object-contain, never cropped). Clicking it opens a larger
 * dialog to click the specific subject to lift when the reference photo has
 * more than one — the small thumbnail is too tight a target to click
 * precisely. The resulting mask is sent as reference_mask (see
 * object_insertion.py's by_reference) so the backend only lifts the clicked
 * subject.
 */
export function ReferenceImagePicker() {
  const referenceImage = useEditorStore((s) => s.referenceImage);
  const setReferenceImage = useEditorStore((s) => s.setReferenceImage);
  const setReferenceImageId = useEditorStore((s) => s.setReferenceImageId);
  const referenceSelection = useEditorStore((s) => s.referenceSelection);
  const setReferenceSelection = useEditorStore((s) => s.setReferenceSelection);
  const referenceMaskPreview = useEditorStore((s) => s.referenceMaskPreview);
  const setReferenceMaskPreview = useEditorStore((s) => s.setReferenceMaskPreview);

  const [imageSize, setImageSize] = React.useState<ImageSize | null>(null);
  const [segmenting, setSegmenting] = React.useState(false);
  const [pickerOpen, setPickerOpen] = React.useState(false);
  // In-flight /api/images registration, keyed by image, so trying several
  // points against the same reference photo shares one upload instead of
  // each click re-sending the full base64 payload — mirrors image-stage.tsx's
  // ensureImageId for the main canvas image.
  const pendingRegistration = React.useRef<{ image: string; promise: Promise<string | undefined> } | null>(
    null,
  );

  async function ensureReferenceImageId(image: string): Promise<string | undefined> {
    const cached = useEditorStore.getState().referenceImageId;
    if (cached) return cached;

    if (pendingRegistration.current?.image === image) {
      return pendingRegistration.current.promise;
    }

    const promise = (async () => {
      try {
        const imageId = await api.registerImage(image);
        if (useEditorStore.getState().referenceImage === image) {
          setReferenceImageId(imageId);
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

  async function handlePointClick(point: [number, number]) {
    if (!referenceImage || !imageSize || segmenting) return;
    const selection = {
      kind: "point" as const,
      points: [point] as [number, number][],
      labels: [1] as (0 | 1)[],
    };
    setReferenceSelection(selection);
    setSegmenting(true);
    const pixelSelection = toPixelSelection(selection, imageSize);
    try {
      const imageId = await ensureReferenceImageId(referenceImage);
      let result;
      try {
        result = await api.segment(referenceImage, pixelSelection, imageId);
      } catch (error) {
        if (!imageId) throw error;
        // A cached image_id can go stale server-side (LRU eviction, a
        // dev-server restart) — retry once with the image sent inline,
        // mirroring image-stage.tsx's runSegmentation.
        setReferenceImageId(null);
        result = await api.segment(referenceImage, pixelSelection);
      }
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
        <button
          type="button"
          onClick={() => setPickerOpen(true)}
          className="group relative flex h-32 w-full items-center justify-center overflow-hidden rounded-md border border-border bg-muted/20"
        >
          <ReferenceCanvas
            image={referenceImage}
            mask={referenceMaskPreview}
            onImageSize={setImageSize}
            className="pointer-events-none relative flex h-full w-full items-center justify-center"
            imgClassName="max-h-full max-w-full object-contain"
          />
          <div className="absolute inset-0 flex items-center justify-center gap-1.5 bg-background/0 text-xs font-medium text-transparent transition-colors group-hover:bg-background/60 group-hover:text-foreground">
            <CursorClick className="size-3.5" />
            {referenceSelection ? "Change subject" : "Click to select subject"}
          </div>
        </button>
      )}

      <p className="text-xs text-muted-foreground">
        {referenceImage
          ? "Click the image to select the subject to lift if the photo has more than one — otherwise the whole image is used."
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

      <Dialog open={pickerOpen} onOpenChange={setPickerOpen}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Select the subject</DialogTitle>
            <DialogDescription>
              Click the object you want to place. Click again to try a different point.
            </DialogDescription>
          </DialogHeader>

          {referenceImage && (
            <div className="relative flex h-[50vh] w-full items-center justify-center overflow-hidden rounded-md border border-border bg-muted/20">
              <ReferenceCanvas
                image={referenceImage}
                mask={referenceMaskPreview}
                onImageSize={setImageSize}
                onPointClick={handlePointClick}
                className="relative flex h-full w-full cursor-crosshair items-center justify-center"
                imgClassName="max-h-full max-w-full select-none object-contain"
              />
              {segmenting && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/50 text-xs text-muted-foreground">
                  Selecting…
                </div>
              )}
            </div>
          )}

          <DialogFooter className="items-center sm:justify-between">
            <button
              type="button"
              onClick={clearSelection}
              disabled={!referenceSelection}
              className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline disabled:pointer-events-none disabled:opacity-40"
            >
              Use whole image
            </button>
            <DialogClose asChild>
              <Button size="sm">Done</Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
