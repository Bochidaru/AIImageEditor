"use client";

import Link from "next/link";
import { Sparkle } from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";

const LINKS = [
  { label: "Tính năng", href: "#features" },
  { label: "Quy trình", href: "#workflow" },
];

export function LandingNav() {
  return (
    <header className="sticky top-0 z-30 border-b border-white/5 bg-canvas/70 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-3 px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight text-canvas-foreground">
          <span className="flex size-7 items-center justify-center rounded-md bg-brand text-brand-foreground">
            <Sparkle weight="fill" className="size-4" />
          </span>
          Aperture
        </Link>

        <nav className="ml-6 hidden items-center gap-6 text-sm text-canvas-foreground/70 md:flex">
          {LINKS.map((link) => (
            <a key={link.href} href={link.href} className="transition-colors hover:text-canvas-foreground">
              {link.label}
            </a>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <Button asChild variant="ghost" className="text-canvas-foreground hover:bg-white/10 hover:text-canvas-foreground">
            <Link href="/studio">Đăng nhập</Link>
          </Button>
          <Button asChild className="bg-brand text-brand-foreground hover:bg-brand/90">
            <Link href="/studio">Mở Studio</Link>
          </Button>
        </div>
      </div>
    </header>
  );
}
