"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { useEditorStore } from "@/lib/store/editor-store";
import { OPERATIONS_BY_ID } from "@/lib/types";

export function HistoryStrip() {
  const sourceImage = useEditorStore((s) => s.sourceImage);
  const history = useEditorStore((s) => s.history);
  const historyIndex = useEditorStore((s) => s.historyIndex);
  const jumpToHistory = useEditorStore((s) => s.jumpToHistory);

  if (!sourceImage || history.length === 0) return null;

  return (
    <div
      role="tablist"
      aria-label="Edit history"
      className="flex shrink-0 items-center gap-2 overflow-x-auto border-t border-border bg-background px-4 py-2"
    >
      <Thumbnail
        image={sourceImage}
        label="Original"
        active={historyIndex === -1}
        onClick={() => jumpToHistory(-1)}
      />
      {history.map((entry, index) => (
        <Thumbnail
          key={entry.id}
          image={entry.image}
          label={OPERATIONS_BY_ID[entry.operation].shortLabel}
          active={historyIndex === index}
          onClick={() => jumpToHistory(index)}
        />
      ))}
    </div>
  );
}

function Thumbnail({
  image,
  label,
  active,
  onClick,
}: {
  image: string;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "flex shrink-0 flex-col items-center gap-1 rounded-md p-1 transition-colors",
        active ? "bg-brand/15 ring-1 ring-brand/50" : "hover:bg-accent",
      )}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={image}
        alt={label}
        className="size-12 rounded object-cover"
        draggable={false}
      />
      <span className="max-w-14 truncate text-[10px] text-muted-foreground">{label}</span>
    </button>
  );
}
