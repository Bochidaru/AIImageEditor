"use client";

import * as React from "react";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

interface ScrollRevealOptions {
  threshold?: number;
  rootMargin?: string;
  once?: boolean;
}

/**
 * Custom hook providing IntersectionObserver scroll-triggered reveal state.
 * - Unobserves element after reveal to prevent memory leaks and redundant triggers.
 * - Automatically respects prefers-reduced-motion.
 */
export function useScrollReveal<T extends HTMLElement = HTMLDivElement>({
  threshold = 0.15,
  rootMargin = "0px 0px -40px 0px",
  once = true,
}: ScrollRevealOptions = {}) {
  const ref = React.useRef<T>(null);
  const reduced = usePrefersReducedMotion();
  // SSR-safe: `reduced`'s server snapshot is always false (see
  // use-reduced-motion.ts), so this matches what the server rendered on the
  // client's first (hydration) render. Do NOT also fold in a
  // `typeof IntersectionObserver` check here — that's true during SSR (no
  // DOM) but false in every real browser, so branching on it inside the
  // initial state would make the state itself diverge between server and
  // client and produce a hydration mismatch on every Reveal on the page.
  const [isRevealed, setIsRevealed] = React.useState(() => reduced);

  React.useEffect(() => {
    if (reduced) {
      // Syncs to the OS-level reduced-motion preference, which can change
      // live while mounted (a no-op on mount, since the initial state above
      // already matches `reduced` at that point).
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setIsRevealed(true);
      return;
    }

    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      // IntersectionObserver support can only be checked client-side — a
      // one-time environment-capability fallback, not derivable at render.
      setIsRevealed(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsRevealed(true);
          if (once) observer.unobserve(el);
        } else if (!once) {
          setIsRevealed(false);
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [threshold, rootMargin, once, reduced]);

  return { ref, isRevealed };
}
