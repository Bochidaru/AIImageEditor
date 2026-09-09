"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowRight, Play, Camera, Stack, Sparkle } from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";
import { SceneSkeleton } from "@/components/ui/section-skeleton";
import { PhotoLayersAnimated } from "@/components/landing/photo-layers-animated";
import { Reveal } from "@/components/ui/reveal";

const HeroScene = dynamic(
  () => import("@/components/scene/hero-scene").then((m) => m.HeroScene),
  {
    ssr: false,
    loading: () => <SceneSkeleton className="h-full w-full" />,
  },
);

export function LandingHero() {
  const [activeVisual, setActiveVisual] = React.useState<"layers" | "camera">("layers");

  return (
    <section className="relative overflow-hidden bg-canvas">
      {/* Night-gradient backdrop */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(80% 60% at 78% 20%, color-mix(in oklch, var(--brand), transparent 60%) 0%, transparent 60%), linear-gradient(180deg, #0c0a14 0%, #150f24 55%, #0c0a14 100%)",
        }}
      />

      <div className="relative mx-auto grid max-w-6xl grid-cols-1 items-center gap-10 px-6 py-20 md:grid-cols-2 md:py-28">
        <div className="relative z-10">
          <Reveal delay={0}>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-canvas-foreground/80">
              <Sparkle className="size-3.5 text-brand" weight="fill" />
              Segmentation-guided AI editing
            </span>
          </Reveal>

          <Reveal delay={100}>
            <h1 className="mt-5 text-4xl font-bold tracking-tight text-canvas-foreground sm:text-5xl lg:text-6xl">
              Chỉnh sửa ảnh.
              <br />
              Không cần vẽ tay.
            </h1>
          </Reveal>

          <Reveal delay={200}>
            <p className="mt-5 max-w-md text-base text-canvas-foreground/70">
              Chọn một điểm trên ảnh, mô tả điều bạn muốn — Aperture tự động
              phân vùng, xoá, thay thế hoặc mở rộng ảnh bằng các model AI
              chuyên biệt, không cần thao tác mask thủ công.
            </p>
          </Reveal>

          <Reveal delay={300}>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Button
                asChild
                size="lg"
                className="h-11 gap-2 bg-brand px-6 text-base text-brand-foreground hover:bg-brand/90 shadow-[0_0_20px_rgba(236,72,153,0.3)] transition-shadow"
              >
                <Link href="/studio">
                  Bắt đầu chỉnh sửa
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="h-11 gap-2 border-white/15 bg-white/5 px-6 text-base text-canvas-foreground hover:bg-white/10 hover:text-canvas-foreground"
              >
                <a href="#workflow">
                  <Play className="size-4" weight="fill" />
                  Xem quy trình
                </a>
              </Button>
            </div>
          </Reveal>

          <Reveal delay={380}>
            <p className="mt-6 text-xs text-canvas-foreground/50">
              SAM2 · FLUX Fill · FLUX Kontext · OmniPaint · Real-ESRGAN
            </p>
          </Reveal>
        </div>

        {/* 3D Showcase Container */}
        <div className="relative z-10 flex flex-col items-center justify-self-center md:justify-self-end w-full max-w-lg">
          <Reveal delay={150} yOffset={30} className="w-full">
            {/* Visual Mode Selector Tabs */}
            <div className="mb-3 flex items-center justify-end">
              <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/[0.04] p-1 backdrop-blur-md">
                <button
                  onClick={() => setActiveVisual("layers")}
                  className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-all ${
                    activeVisual === "layers"
                      ? "bg-brand text-brand-foreground shadow-sm"
                      : "text-canvas-foreground/60 hover:text-canvas-foreground"
                  }`}
                >
                  <Stack className="size-3.5" weight="bold" />
                  3D Photo Layers
                </button>
                <button
                  onClick={() => setActiveVisual("camera")}
                  className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-all ${
                    activeVisual === "camera"
                      ? "bg-brand text-brand-foreground shadow-sm"
                      : "text-canvas-foreground/60 hover:text-canvas-foreground"
                  }`}
                >
                  <Camera className="size-3.5" weight="bold" />
                  3D Camera
                </button>
              </div>
            </div>

            {/* Visual Display Box */}
            <div className="relative aspect-square w-full rounded-2xl border border-white/10 bg-white/[0.02] p-2 backdrop-blur-sm shadow-[0_0_50px_rgba(236,72,153,0.12)] overflow-hidden">
              {activeVisual === "layers" ? (
                <PhotoLayersAnimated />
              ) : (
                <HeroScene className="absolute inset-0" />
              )}
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
