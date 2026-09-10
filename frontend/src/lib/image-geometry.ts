export interface RenderedImageRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Where an `object-contain` image actually renders inside its (usually
 * larger, different-aspect-ratio) container — the letterbox offset plus the
 * scaled-down width/height. Anything positioning overlays or computing
 * click/drag coordinates against the image (not the surrounding container)
 * needs this same rect, or it drifts off the image by exactly the letterbox
 * gap whenever the image's aspect ratio doesn't match the container's.
 */
export function computeRenderedImageRect(
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
