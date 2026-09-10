"use client";

import * as React from "react";

export function SceneSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={`flex items-center justify-center rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-sm ${className ?? ""}`}
      aria-hidden="true"
    >
      <div className="relative flex flex-col items-center gap-3">
        {/* Glowing pulsing camera aperture ring placeholder */}
        <div className="relative flex size-28 items-center justify-center rounded-full border border-brand/30 bg-brand/5 shadow-[0_0_30px_rgba(236,72,153,0.15)] animate-pulse">
          <div className="size-16 rounded-full border border-brand/40 bg-brand/10" />
          <div className="absolute size-6 rounded-full bg-brand/30 shadow-[0_0_15px_rgba(236,72,153,0.6)]" />
        </div>
        <div className="h-2 w-24 rounded-full bg-white/10 animate-pulse" />
      </div>
    </div>
  );
}

export function SectionSkeleton({
  height = "min-h-[380px]",
  titleWidth = "w-48",
}: {
  height?: string;
  titleWidth?: string;
}) {
  return (
    <div className={`mx-auto max-w-6xl px-6 py-20 ${height} animate-pulse`} aria-hidden="true">
      <div className={`h-8 ${titleWidth} rounded-lg bg-white/10 mb-4`} />
      <div className="h-4 w-72 max-w-full rounded bg-white/5 mb-12" />
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-44 rounded-xl border border-white/5 bg-white/[0.02] p-5"
          >
            <div className="size-10 rounded-lg bg-white/10 mb-4" />
            <div className="h-5 w-3/4 rounded bg-white/10 mb-2" />
            <div className="h-3.5 w-full rounded bg-white/5 mb-1.5" />
            <div className="h-3.5 w-4/5 rounded bg-white/5" />
          </div>
        ))}
      </div>
    </div>
  );
}
