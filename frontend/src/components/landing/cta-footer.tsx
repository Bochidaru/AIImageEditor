"use client";

import Link from "next/link";
import { ArrowRight, Sparkle } from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";
import { Reveal } from "@/components/ui/reveal";

export function CtaFooter() {
  return (
    <>
      <section className="border-t border-white/5 bg-canvas py-20">
        <div className="mx-auto max-w-3xl px-6 text-center">
          <Reveal delay={0}>
            <h2 className="text-3xl font-bold tracking-tight text-canvas-foreground sm:text-4xl">
              Sẵn sàng chỉnh sửa tấm ảnh đầu tiên?
            </h2>
          </Reveal>
          <Reveal delay={120}>
            <p className="mx-auto mt-3 max-w-md text-canvas-foreground/70">
              Không cần tài khoản để thử giao diện — mở studio và tải ảnh lên ngay.
            </p>
          </Reveal>
          <Reveal delay={240}>
            <Button
              asChild
              size="lg"
              className="mt-8 h-11 gap-2 bg-brand px-6 text-base text-brand-foreground hover:bg-brand/90 shadow-[0_0_20px_rgba(236,72,153,0.3)] transition-shadow"
            >
              <Link href="/studio">
                Mở Studio
                <ArrowRight className="size-4" />
              </Link>
            </Button>
          </Reveal>
        </div>
      </section>

      <footer className="border-t border-white/5 bg-canvas py-10">
        <div className="mx-auto flex max-w-6xl flex-col items-center gap-3 px-6 text-sm text-canvas-foreground/50 sm:flex-row sm:justify-between">
          <div className="flex items-center gap-2 text-canvas-foreground/80">
            <span className="flex size-6 items-center justify-center rounded-md bg-brand text-brand-foreground">
              <Sparkle weight="fill" className="size-3.5" />
            </span>
            Aperture
          </div>
          <p>Frontend demo — chưa kết nối backend.</p>
        </div>
      </footer>
    </>
  );
}
