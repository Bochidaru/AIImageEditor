"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { ImageSquare, UploadSimple } from "@phosphor-icons/react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { readFileAsDataUrl, validateImageFile } from "@/lib/image-file";

const AmbientScene = dynamic(
  () => import("@/components/scene/ambient-scene").then((m) => m.AmbientScene),
  { ssr: false },
);

export function EmptyState({ onLoad }: { onLoad: (dataUrl: string, name: string) => void }) {
  const [isDragging, setDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  async function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    const error = validateImageFile(file);
    if (error) {
      toast.error("Upload rejected", { description: error });
      return;
    }
    onLoad(await readFileAsDataUrl(file), file.name);
  }

  return (
    <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-canvas">
      <AmbientScene className="pointer-events-none absolute inset-0 opacity-40" />

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void handleFiles(event.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
        aria-label="Upload an image to start editing"
        className={cn(
          "relative z-10 flex w-full max-w-md cursor-pointer flex-col items-center gap-3 rounded-xl border border-dashed border-border/70 bg-background/70 px-10 py-14 text-center backdrop-blur-sm transition-colors",
          "hover:border-brand/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          isDragging && "border-brand bg-brand/5",
        )}
      >
        <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
          {isDragging ? (
            <UploadSimple className="size-6" />
          ) : (
            <ImageSquare className="size-6" />
          )}
        </span>
        <div>
          <p className="text-sm font-medium">Drop an image here, or click to browse</p>
          <p className="mt-1 text-xs text-muted-foreground">
            PNG, JPEG, or WebP — up to 25MB
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="sr-only"
          onChange={(event) => void handleFiles(event.target.files)}
        />
      </div>
    </div>
  );
}
