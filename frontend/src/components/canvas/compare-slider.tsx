"use client";

import * as React from "react";
import { ArrowsLeftRight } from "@phosphor-icons/react";
import { computeRenderedImageRect, type RenderedImageRect } from "@/lib/image-geometry";

export function CompareSlider({ before, after }: { before: string; after: string }) {
  const [position, setPosition] = React.useState(50);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const draggingRef = React.useRef(false);
  // Where the (letterboxed) image actually renders inside the container —
  // the clip seam and the handle must be positioned against this, not the
  // raw container box, or they drift off the image whenever its aspect
  // ratio doesn't match the container's (see computeRenderedImageRect).
  const [renderedRect, setRenderedRect] = React.useState<RenderedImageRect | null>(null);

  const recomputeRenderedRect = React.useCallback(() => {
    const container = containerRef.current;
    const img = imageRef.current;
    if (!container || !img) return;
    setRenderedRect(computeRenderedImageRect(container, img));
  }, []);

  React.useEffect(() => {
    recomputeRenderedRect();
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(recomputeRenderedRect);
    observer.observe(container);
    return () => observer.disconnect();
  }, [after, recomputeRenderedRect]);

  function updateFromClientX(clientX: number) {
    const containerRect = containerRef.current?.getBoundingClientRect();
    if (!containerRect || !renderedRect) return;
    const localX = clientX - containerRect.left - renderedRect.left;
    const percent = (localX / renderedRect.width) * 100;
    setPosition(Math.min(100, Math.max(0, percent)));
  }

  return (
    <div
      ref={containerRef}
      className="relative h-full max-h-full w-full max-w-full select-none"
      onPointerMove={(event) => {
        if (draggingRef.current) updateFromClientX(event.clientX);
      }}
      onPointerUp={() => {
        draggingRef.current = false;
      }}
      onPointerLeave={() => {
        draggingRef.current = false;
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imageRef}
        src={after}
        alt="Result"
        className="pointer-events-none mx-auto h-full max-h-full w-full max-w-full object-contain"
        draggable={false}
        onLoad={recomputeRenderedRect}
      />

      {renderedRect && (
        <div
          className="absolute"
          style={{
            left: renderedRect.left,
            top: renderedRect.top,
            width: renderedRect.width,
            height: renderedRect.height,
          }}
        >
          <div
            className="pointer-events-none absolute inset-0 overflow-hidden"
            style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={before}
              alt="Original"
              className="h-full w-full object-contain"
              draggable={false}
            />
          </div>

          <div
            role="slider"
            aria-label="Comparison position"
            aria-valuenow={Math.round(position)}
            aria-valuemin={0}
            aria-valuemax={100}
            tabIndex={0}
            onPointerDown={(event) => {
              draggingRef.current = true;
              updateFromClientX(event.clientX);
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowLeft") setPosition((p) => Math.max(0, p - 5));
              if (event.key === "ArrowRight") setPosition((p) => Math.min(100, p + 5));
            }}
            className="absolute inset-y-0 flex w-6 -translate-x-1/2 cursor-ew-resize items-center justify-center focus-visible:outline-none"
            style={{ left: `${position}%` }}
          >
            <span className="absolute inset-y-0 w-px bg-white/80" />
            <span className="relative flex size-8 items-center justify-center rounded-full bg-white text-black shadow-lg">
              <ArrowsLeftRight className="size-4" />
            </span>
          </div>
        </div>
      )}

      <span className="pointer-events-none absolute left-3 top-3 rounded-full bg-background/80 px-2 py-0.5 text-[11px] text-muted-foreground backdrop-blur-sm">
        Before
      </span>
      <span className="pointer-events-none absolute right-3 top-3 rounded-full bg-background/80 px-2 py-0.5 text-[11px] text-muted-foreground backdrop-blur-sm">
        After
      </span>
    </div>
  );
}
