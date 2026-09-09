"use client";

import * as React from "react";
import { CircleNotch, WarningCircle } from "@phosphor-icons/react";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { useEditorStore } from "@/lib/store/editor-store";

/**
 * Indeterminate-feeling but bounded progress: model calls in the real
 * backend report elapsed time only after completion (see
 * MemoryTracker.measure in inpaint_core), so there is no true percentage
 * to show mid-flight. We animate toward ~92% and let the actual resolve
 * snap to 100%, rather than faking false precision.
 */
export function ProcessingOverlay() {
  const status = useEditorStore((s) => s.status);
  const progressLabel = useEditorStore((s) => s.progressLabel);
  const errorMessage = useEditorStore((s) => s.errorMessage);
  const clearError = useEditorStore((s) => s.clearError);
  const [progress, setProgress] = React.useState(8);

  // Reset the bar the moment a new run starts, computed during render
  // (not an effect) so it never lags a status change by a frame.
  const [trackedStatus, setTrackedStatus] = React.useState(status);
  if (status !== trackedStatus) {
    setTrackedStatus(status);
    if (status === "processing") setProgress(8);
  }

  React.useEffect(() => {
    if (status !== "processing") return;
    const id = window.setInterval(() => {
      setProgress((value) => (value >= 92 ? value : value + (92 - value) * 0.08));
    }, 200);
    return () => window.clearInterval(id);
  }, [status]);

  if (status === "processing") {
    return (
      <div
        role="status"
        aria-live="polite"
        aria-busy="true"
        className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-background/75 backdrop-blur-sm"
      >
        <CircleNotch className="size-8 animate-spin text-brand" />
        <div className="w-56 space-y-2 text-center">
          <p className="text-sm font-medium">{progressLabel ?? "Processing…"}</p>
          <Progress value={progress} />
        </div>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div
        role="alert"
        className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-background/90 px-6 text-center backdrop-blur-sm"
      >
        <WarningCircle className="size-8 text-destructive" />
        <p className="max-w-sm text-sm font-medium">{errorMessage ?? "Something went wrong."}</p>
        <Button variant="outline" size="sm" onClick={clearError}>
          Dismiss
        </Button>
      </div>
    );
  }

  return null;
}
