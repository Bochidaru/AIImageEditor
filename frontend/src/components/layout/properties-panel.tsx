"use client";

import * as React from "react";
import { toast } from "sonner";
import { CircleNotch, Sparkle, SlidersHorizontal, X } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useEditorStore } from "@/lib/store/editor-store";
import { OPERATIONS, toPixelSelection, toPixelBox } from "@/lib/types";
import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import { ReferenceImagePicker } from "@/components/layout/reference-image-picker";

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <Label className="text-xs font-medium text-muted-foreground">{label}</Label>
        {hint ? <span className="text-xs text-muted-foreground">{hint}</span> : null}
      </div>
      {children}
    </div>
  );
}

export function PropertiesPanel() {
  const activeTool = useEditorStore((s) => s.activeTool);
  const activeImage = useEditorStore((s) => s.activeImage);
  const selection = useEditorStore((s) => s.selection);
  const placement = useEditorStore((s) => s.placement);
  const activeImageSize = useEditorStore((s) => s.activeImageSize);
  const referenceImage = useEditorStore((s) => s.referenceImage);
  const referenceMaskPreview = useEditorStore((s) => s.referenceMaskPreview);
  const prompt = useEditorStore((s) => s.prompt);
  const setPrompt = useEditorStore((s) => s.setPrompt);
  const maskOptions = useEditorStore((s) => s.maskOptions);
  const setMaskOptions = useEditorStore((s) => s.setMaskOptions);
  const generationOptions = useEditorStore((s) => s.generationOptions);
  const setGenerationOptions = useEditorStore((s) => s.setGenerationOptions);
  const setNumInferenceSteps = useEditorStore((s) => s.setNumInferenceSteps);
  const upscaleOptions = useEditorStore((s) => s.upscaleOptions);
  const setUpscaleOptions = useEditorStore((s) => s.setUpscaleOptions);
  const outpaintMargins = useEditorStore((s) => s.outpaintMargins);
  const setOutpaintMargins = useEditorStore((s) => s.setOutpaintMargins);
  const status = useEditorStore((s) => s.status);
  const beginProcessing = useEditorStore((s) => s.beginProcessing);
  const resolveEdit = useEditorStore((s) => s.resolveEdit);
  const resolveGeneration = useEditorStore((s) => s.resolveGeneration);
  const fail = useEditorStore((s) => s.fail);
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const meta = OPERATIONS.find((op) => op.id === activeTool);
  const isProcessing = status === "processing" || status === "segmenting";

  if (!activeTool || !meta) {
    return (
      <aside className="hidden w-80 shrink-0 flex-col border-l border-border bg-background p-4 lg:flex">
        <p className="text-sm text-muted-foreground">
          Choose a tool from the left to see its options here.
        </p>
      </aside>
    );
  }

  const hasSelectionOrMask = Boolean(selection);
  const hasPlacement = Boolean(placement) || Boolean(selection);
  const missingMask = meta.requiresMask && !hasPlacement && !hasSelectionOrMask;
  const missingPrompt = meta.requiresPrompt && !prompt.trim();
  const missingReference = meta.requiresReference && !referenceImage;
  const missingImage = meta.requiresImage && !activeImage;
  const canRun = !missingMask && !missingPrompt && !missingReference && !missingImage && !isProcessing;

  async function run() {
    if (!activeImage || !meta) return;
    beginProcessing(`Running ${meta.label.toLowerCase()}…`);
    // selection/placement are tracked normalized ([0,1]) for overlay
    // rendering on the canvas — the backend expects pixel coordinates in
    // the original image's dimensions (see toPixelSelection/toPixelBox).
    const pixelSelection =
      selection && activeImageSize ? toPixelSelection(selection, activeImageSize) : undefined;
    const pixelPlacement =
      placement && activeImageSize ? toPixelBox(placement, activeImageSize) : undefined;
    try {
      switch (meta.id) {
        case "remove_object": {
          const result = await api.removeObject({
            image: activeImage,
            selection: pixelSelection,
            maskOptions,
            generationOptions,
          });
          resolveEdit(meta.id, result);
          break;
        }
        case "replace_object": {
          const result = await api.replaceObject({
            image: activeImage,
            selection: pixelSelection,
            prompt,
            maskOptions,
            generationOptions,
          });
          resolveEdit(meta.id, result);
          break;
        }
        case "replace_background": {
          // No mask/selection: Flux2 Klein edits directly from the prompt
          // instruction (see operations/background_replacement.py).
          const result = await api.replaceBackground({
            image: activeImage,
            prompt,
            generationOptions,
          });
          resolveGeneration(meta.id, result);
          break;
        }
        case "add_object_by_prompt": {
          // No mask: placement (if any) only steers the instruction's
          // wording (see object_insertion.py's by_prompt), it isn't
          // resolved into a mask.
          const result = await api.addObjectByPrompt({
            image: activeImage,
            placement: pixelPlacement,
            prompt,
            generationOptions,
          });
          resolveGeneration(meta.id, result);
          break;
        }
        case "add_object_by_reference": {
          if (!referenceImage) return;
          const result = await api.addObjectByReference({
            image: activeImage,
            reference: referenceImage,
            placement: pixelPlacement,
            referenceMask: referenceMaskPreview ?? undefined,
            maskOptions,
            generationOptions,
          });
          resolveEdit(meta.id, result);
          break;
        }
        case "prompt_edit": {
          const result = await api.promptEdit(activeImage, prompt, generationOptions);
          resolveGeneration(meta.id, result);
          break;
        }
        case "generate_image": {
          const result = await api.generateImage(prompt, 1024, 1024, generationOptions);
          resolveGeneration(meta.id, result);
          break;
        }
        case "outpaint": {
          const result = await api.outpaint(activeImage, prompt, outpaintMargins, generationOptions);
          resolveEdit(meta.id, result);
          break;
        }
        case "upscale": {
          const result = await api.upscale(activeImage, upscaleOptions);
          resolveGeneration(meta.id, result);
          break;
        }
      }
      toast.success(`${meta.label} complete`);
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Something went wrong. Please try again.";
      fail(message);
      toast.error(`${meta.label} failed`, { description: message });
    }
  }

  return (
    <>
      {/* Mobile trigger: the sidebar below is a slide-up sheet on small screens. */}
      <Button
        size="icon"
        onClick={() => setMobileOpen(true)}
        aria-label="Open tool options"
        className="fixed right-4 bottom-20 z-30 size-12 rounded-full bg-brand text-brand-foreground shadow-lg hover:bg-brand/90 lg:hidden"
      >
        <SlidersHorizontal className="size-5" />
      </Button>

      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          "fixed inset-x-0 bottom-0 z-40 flex max-h-[80vh] w-full flex-col gap-5 overflow-y-auto rounded-t-2xl border-t border-border bg-background p-4 shadow-2xl transition-transform duration-300 ease-out",
          "lg:static lg:z-auto lg:w-80 lg:max-h-none lg:translate-y-0 lg:rounded-none lg:border-l lg:border-t-0 lg:shadow-none lg:transition-none",
          mobileOpen ? "translate-y-0" : "translate-y-full lg:translate-y-0",
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">{meta.label}</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">{meta.description}</p>
            <span className="mt-2 inline-flex items-center rounded-full border border-border px-2 py-0.5 text-[11px] text-muted-foreground">
              {meta.model}
            </span>
          </div>
          <Button
            size="icon"
            variant="ghost"
            aria-label="Close tool options"
            onClick={() => setMobileOpen(false)}
            className="-mr-2 -mt-1 lg:hidden"
          >
            <X className="size-4" />
          </Button>
        </div>

      <Separator />

      {meta.allowsMask && (
        <div className="space-y-1.5">
          <Label className="text-xs font-medium text-muted-foreground">
            Selection{!meta.requiresMask && " (optional)"}
          </Label>
          <p className="text-xs text-muted-foreground">
            {hasPlacement
              ? "Region selected on canvas."
              : meta.requiresMask
                ? "Click a point, or drag a box on the canvas to select a region."
                : "Optionally drag a box on the canvas to steer where the object is placed."}
          </p>
        </div>
      )}

      {meta.requiresReference && <ReferenceImagePicker />}

      {meta.requiresPrompt && (
        <div className="space-y-1.5">
          <Label htmlFor="prompt" className="text-xs font-medium text-muted-foreground">
            Prompt
          </Label>
          <Textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={
              meta.id === "generate_image"
                ? "A cinematic mountain lake at sunrise, realistic photography"
                : "Describe the change you want…"
            }
            rows={3}
            className="resize-none text-sm"
          />
        </div>
      )}

      {meta.id === "outpaint" && (
        <div className="space-y-2">
          <Label className="text-xs font-medium text-muted-foreground">Extend margins (px)</Label>
          <div className="grid grid-cols-2 gap-2">
            {(["left", "top", "right", "bottom"] as const).map((side) => (
              <div key={side} className="space-y-1">
                <span className="text-[11px] capitalize text-muted-foreground">{side}</span>
                <input
                  type="number"
                  min={0}
                  value={outpaintMargins[side]}
                  onChange={(event) =>
                    setOutpaintMargins({ [side]: Math.max(0, Number(event.target.value)) })
                  }
                  className="h-8 w-full rounded-md border border-input bg-transparent px-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {meta.id === "upscale" && (
        <div className="space-y-4">
          <Field label="Scale" hint={`${upscaleOptions.scale}x`}>
            <Slider
              value={[upscaleOptions.scale]}
              min={2}
              max={8}
              step={1}
              onValueChange={([value]) => setUpscaleOptions({ scale: value })}
            />
          </Field>
          <div className="flex items-center justify-between">
            <Label htmlFor="face-enhance" className="text-xs font-medium text-muted-foreground">
              Face enhance
            </Label>
            <Switch
              id="face-enhance"
              checked={upscaleOptions.faceEnhance}
              onCheckedChange={(checked) => setUpscaleOptions({ faceEnhance: checked })}
            />
          </div>
        </div>
      )}

      {meta.requiresMask && (
        <details className="group rounded-md border border-border">
          <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-muted-foreground marker:hidden">
            Mask refinement
          </summary>
          <div className="space-y-4 border-t border-border p-3">
            <Field label="Threshold" hint={String(maskOptions.threshold)}>
              <Slider
                value={[maskOptions.threshold]}
                min={0}
                max={255}
                step={1}
                onValueChange={([value]) => setMaskOptions({ threshold: value })}
              />
            </Field>
            <Field label="Dilate" hint={`${maskOptions.dilate}px`}>
              <Slider
                value={[maskOptions.dilate]}
                min={0}
                max={40}
                step={1}
                onValueChange={([value]) => setMaskOptions({ dilate: value })}
              />
            </Field>
            <Field label="Erode" hint={`${maskOptions.erode}px`}>
              <Slider
                value={[maskOptions.erode]}
                min={0}
                max={40}
                step={1}
                onValueChange={([value]) => setMaskOptions({ erode: value })}
              />
            </Field>
          </div>
        </details>
      )}

      <details className="group rounded-md border border-border">
        <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-muted-foreground marker:hidden">
          Generation settings
        </summary>
        <div className="space-y-4 border-t border-border p-3">
          <Field label="Steps" hint={String(generationOptions.numInferenceSteps)}>
            <Slider
              value={[generationOptions.numInferenceSteps]}
              min={4}
              max={50}
              step={1}
              onValueChange={([value]) => setNumInferenceSteps(value)}
            />
          </Field>
          <Field label="Seed">
            <input
              type="number"
              value={generationOptions.seed}
              onChange={(event) => setGenerationOptions({ seed: Number(event.target.value) })}
              className="h-8 w-full rounded-md border border-input bg-transparent px-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </Field>
        </div>
      </details>

      <div className="mt-auto sticky bottom-0 -mx-4 border-t border-border bg-background px-4 pt-4">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="block">
              <Button
                className="w-full gap-1.5 bg-brand text-brand-foreground hover:bg-brand/90"
                disabled={!canRun}
                onClick={run}
              >
                {isProcessing ? (
                  <CircleNotch className="size-4 animate-spin" />
                ) : (
                  <Sparkle className="size-4" weight="fill" />
                )}
                {isProcessing ? "Processing…" : `Run ${meta.shortLabel}`}
              </Button>
            </span>
          </TooltipTrigger>
          {!canRun && !isProcessing ? (
            <TooltipContent>
              {missingImage
                ? "Upload an image first."
                : missingMask
                  ? "Select a region on the canvas first."
                  : missingReference
                    ? "Upload a reference image first."
                    : missingPrompt
                      ? "Write a prompt first."
                      : "Complete the required fields."}
            </TooltipContent>
          ) : null}
        </Tooltip>
      </div>
      </aside>
    </>
  );
}
