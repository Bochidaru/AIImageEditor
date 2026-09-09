"use client";

import { OPERATIONS } from "@/lib/types";
import { OPERATION_ICONS } from "@/lib/operation-icons";
import { Reveal } from "@/components/ui/reveal";

export function FeatureGrid() {
  return (
    <section id="features" className="bg-canvas py-20">
      <div className="mx-auto max-w-6xl px-6">
        <div className="max-w-xl">
          <Reveal delay={0}>
            <h2 className="text-3xl font-bold tracking-tight text-canvas-foreground sm:text-4xl">
              Một studio, mười cách chỉnh sửa
            </h2>
          </Reveal>
          <Reveal delay={120}>
            <p className="mt-3 text-canvas-foreground/70">
              Mỗi công cụ gọi đúng model chuyên biệt cho việc đó — không có
              model vạn năng làm mọi thứ tệ như nhau.
            </p>
          </Reveal>
        </div>

        <div className="mt-12 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {OPERATIONS.map((op, index) => {
            const Icon = OPERATION_ICONS[op.id];
            return (
              <Reveal key={op.id} delay={Math.min(index * 80, 480)} yOffset={30}>
                <div className="group h-full rounded-xl border border-white/10 bg-white/[0.03] p-5 transition-all hover:border-brand/30 hover:bg-white/[0.05] hover:-translate-y-1">
                  <span className="flex size-10 items-center justify-center rounded-lg bg-brand/15 text-brand">
                    <Icon className="size-5" />
                  </span>
                  <h3 className="mt-4 text-sm font-semibold text-canvas-foreground">{op.label}</h3>
                  <p className="mt-1.5 text-sm text-canvas-foreground/60">{op.description}</p>
                  <span className="mt-3 inline-block rounded-full border border-white/10 px-2 py-0.5 text-[11px] text-canvas-foreground/50">
                    {op.model}
                  </span>
                </div>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
