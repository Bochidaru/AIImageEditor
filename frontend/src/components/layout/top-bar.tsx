"use client";

import * as React from "react";
import { useTheme } from "next-themes";
import {
  ArrowCounterClockwise,
  ArrowClockwise,
  DownloadSimple,
  MoonStars,
  SunDim,
  Sparkle,
  UploadSimple,
} from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useEditorStore } from "@/lib/store/editor-store";
import { useMounted } from "@/hooks/use-mounted";
import { toast } from "sonner";

export function TopBar({ onUploadClick }: { onUploadClick: () => void }) {
  const { theme, setTheme } = useTheme();
  const mounted = useMounted();
  const sourceFileName = useEditorStore((s) => s.sourceFileName);
  const activeImage = useEditorStore((s) => s.activeImage);
  const history = useEditorStore((s) => s.history);
  const historyIndex = useEditorStore((s) => s.historyIndex);
  const undo = useEditorStore((s) => s.undo);
  const redo = useEditorStore((s) => s.redo);

  const canUndo = historyIndex >= 0;
  const canRedo = historyIndex < history.length - 1;

  function handleExport() {
    if (!activeImage) return;
    const link = document.createElement("a");
    link.href = activeImage;
    link.download = `${sourceFileName?.replace(/\.[^.]+$/, "") ?? "edited-image"}.png`;
    link.click();
    toast.success("Export ready", {
      description: "Downloaded the current canvas as PNG.",
    });
  }

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-background px-4">
      <div className="flex items-center gap-2 font-semibold tracking-tight">
        <span className="flex size-7 items-center justify-center rounded-md bg-brand text-brand-foreground">
          <Sparkle weight="fill" className="size-4" />
        </span>
        <span className="hidden sm:inline">Aperture</span>
      </div>

      <Separator orientation="vertical" className="h-6" />

      <span className="truncate text-sm text-muted-foreground max-w-48">
        {sourceFileName ?? "No image loaded"}
      </span>

      <div className="ml-auto flex items-center gap-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              disabled={!canUndo}
              onClick={undo}
              aria-label="Undo"
            >
              <ArrowCounterClockwise className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Undo</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              disabled={!canRedo}
              onClick={redo}
              aria-label="Redo"
            >
              <ArrowClockwise className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Redo</TooltipContent>
        </Tooltip>

        <Separator orientation="vertical" className="mx-1 h-6" />

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Toggle theme"
            >
              {mounted && theme === "light" ? (
                <MoonStars className="size-4" />
              ) : (
                <SunDim className="size-4" />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent>Toggle theme</TooltipContent>
        </Tooltip>

        <Button variant="outline" size="sm" onClick={onUploadClick} className="gap-1.5">
          <UploadSimple className="size-4" />
          <span className="hidden sm:inline">Upload</span>
        </Button>

        <Button
          size="sm"
          className="gap-1.5 bg-brand text-brand-foreground hover:bg-brand/90"
          disabled={!activeImage}
          onClick={handleExport}
        >
          <DownloadSimple className="size-4" />
          <span className="hidden sm:inline">Export</span>
        </Button>
      </div>
    </header>
  );
}
