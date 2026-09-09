"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { OPERATIONS } from "@/lib/types";
import { OPERATION_ICONS } from "@/lib/operation-icons";
import { useEditorStore } from "@/lib/store/editor-store";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function ToolSidebar() {
  const activeTool = useEditorStore((s) => s.activeTool);
  const setActiveTool = useEditorStore((s) => s.setActiveTool);
  const hasImage = Boolean(useEditorStore((s) => s.sourceImage));

  return (
    <nav
      aria-label="Editing tools"
      className="flex w-16 shrink-0 flex-col items-center gap-1 border-r border-border bg-background py-3 sm:w-56 sm:items-stretch sm:px-2"
    >
      {OPERATIONS.map((op) => {
        const Icon = OPERATION_ICONS[op.id];
        const isActive = activeTool === op.id;
        const disabled = op.requiresImage && !hasImage;
        return (
          <Tooltip key={op.id}>
            <TooltipTrigger asChild>
              <button
                type="button"
                disabled={disabled}
                aria-pressed={isActive}
                aria-label={op.label}
                onClick={() => setActiveTool(op.id)}
                className={cn(
                  "group flex items-center gap-2.5 rounded-md px-2.5 py-2.5 text-left text-sm transition-colors sm:px-3",
                  "disabled:pointer-events-none disabled:opacity-40",
                  isActive
                    ? "bg-brand/15 text-foreground ring-1 ring-inset ring-brand/40"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <Icon
                  weight={isActive ? "fill" : "regular"}
                  className={cn(
                    "size-5 shrink-0",
                    isActive ? "text-brand" : "text-muted-foreground group-hover:text-foreground",
                  )}
                />
                <span className="hidden truncate sm:inline">{op.label}</span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="right" className="sm:hidden">
              {op.label}
            </TooltipContent>
          </Tooltip>
        );
      })}
    </nav>
  );
}
