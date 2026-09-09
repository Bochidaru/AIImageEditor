"use client";

import * as React from "react";
import { useScrollReveal } from "@/hooks/use-scroll-reveal";

interface RevealProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  delay?: number; // Milliseconds delay (e.g. 0, 120, 240)
  yOffset?: number; // Slide distance in px (default 40px)
  duration?: number; // Duration in ms (default 750ms)
  threshold?: number;
  rootMargin?: string;
  className?: string;
}

/**
 * Reusable scroll-triggered reveal wrapper.
 * Uses GPU-accelerated transforms (translateY, opacity) with Apple/Linear.app style easing.
 */
export function Reveal({
  children,
  delay = 0,
  yOffset = 40,
  duration = 750,
  threshold = 0.15,
  rootMargin = "0px 0px -40px 0px",
  className = "",
  style,
  ...props
}: RevealProps) {
  const { ref, isRevealed } = useScrollReveal<HTMLDivElement>({
    threshold,
    rootMargin,
    once: true,
  });

  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: isRevealed ? 1 : 0,
        transform: isRevealed ? "translateY(0)" : `translateY(${yOffset}px)`,
        transition: `opacity ${duration}ms cubic-bezier(0.16, 1, 0.3, 1) ${delay}ms, transform ${duration}ms cubic-bezier(0.16, 1, 0.3, 1) ${delay}ms`,
        willChange: "transform, opacity",
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}
