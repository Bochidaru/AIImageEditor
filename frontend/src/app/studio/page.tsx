"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { toast } from "sonner";
import { TopBar } from "@/components/layout/top-bar";
import { ToolSidebar } from "@/components/layout/tool-sidebar";
import { EmptyState } from "@/components/canvas/empty-state";
import { useEditorStore } from "@/lib/store/editor-store";
import { readFileAsDataUrl, validateImageFile } from "@/lib/image-file";

// Dynamic imports with lazy loading skeletons
const ImageStage = dynamic(
  () => import("@/components/canvas/image-stage").then((m) => m.ImageStage),
  {
    loading: () => (
      <div className="flex flex-1 items-center justify-center bg-canvas">
        <div className="flex flex-col items-center gap-3">
          <div className="size-10 rounded-full border-2 border-brand/30 border-t-brand animate-spin" />
          <span className="text-xs text-muted-foreground animate-pulse">Loading canvas...</span>
        </div>
      </div>
    ),
  },
);

const PropertiesPanel = dynamic(
  () => import("@/components/layout/properties-panel").then((m) => m.PropertiesPanel),
  {
    loading: () => (
      <aside className="w-80 shrink-0 border-l border-border/50 bg-background/50 p-4 animate-pulse hidden md:flex flex-col gap-4">
        <div className="h-6 w-32 rounded bg-muted" />
        <div className="h-24 rounded-lg bg-muted/50" />
        <div className="space-y-2">
          <div className="h-4 w-20 rounded bg-muted" />
          <div className="h-10 rounded bg-muted/40" />
        </div>
        <div className="space-y-2">
          <div className="h-4 w-28 rounded bg-muted" />
          <div className="h-10 rounded bg-muted/40" />
        </div>
      </aside>
    ),
  },
);

const HistoryStrip = dynamic(
  () => import("@/components/canvas/history-strip").then((m) => m.HistoryStrip),
  { ssr: false },
);

const ProcessingOverlay = dynamic(
  () => import("@/components/canvas/processing-overlay").then((m) => m.ProcessingOverlay),
  { ssr: false },
);

export default function StudioPage() {
  const sourceImage = useEditorStore((s) => s.sourceImage);
  const setSourceImage = useEditorStore((s) => s.setSourceImage);
  const inputRef = React.useRef<HTMLInputElement>(null);

  async function handleUploadInput(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const error = validateImageFile(file);
    if (error) {
      toast.error("Upload rejected", { description: error });
      return;
    }
    setSourceImage(await readFileAsDataUrl(file), file.name);
  }

  return (
    <div className="flex h-screen flex-col">
      <TopBar onUploadClick={() => inputRef.current?.click()} />
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="sr-only"
        onChange={handleUploadInput}
      />

      <div className="flex flex-1 overflow-hidden">
        <ToolSidebar />

        <main className="relative flex flex-1 flex-col overflow-hidden">
          {sourceImage ? (
            <>
              <div className="relative flex flex-1 overflow-hidden">
                <ImageStage />
                <ProcessingOverlay />
              </div>
              <HistoryStrip />
            </>
          ) : (
            <EmptyState onLoad={setSourceImage} />
          )}
        </main>

        <PropertiesPanel />
      </div>
    </div>
  );
}
