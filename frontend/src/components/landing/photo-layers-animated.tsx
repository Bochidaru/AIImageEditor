"use client";

import * as React from "react";
import Image from "next/image";
import { motion } from "motion/react";
import { CaretLeft, CaretRight } from "@phosphor-icons/react";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

interface PhotoCard {
  id: string;
  src: string;
  title: string;
  tag: string;
}

const CARDS: PhotoCard[] = [
  {
    id: "landscape",
    src: "/cards/landscape.jpg",
    title: "Alpine Lake Reflection",
    tag: "Landscape · FLUX Fill",
  },
  {
    id: "portrait",
    src: "/cards/portrait.jpg",
    title: "Editorial Freckle Portrait",
    tag: "Portrait · Retouch Core",
  },
  {
    id: "lake",
    src: "/cards/lake.png",
    title: "Serene Emerald Lake",
    tag: "Nature · Outpaint",
  },
  {
    id: "cat",
    src: "/cards/cat.jpg",
    title: "Golden Feline Focus",
    tag: "Subject · SAM2 Segment",
  },
  {
    id: "culinary",
    src: "/cards/culinary.jpg",
    title: "Gourmet Culinary Bowl",
    tag: "Still Life · OmniPaint",
  },
  {
    id: "horse",
    src: "/cards/horse.jpg",
    title: "Wild Stallion Meadow",
    tag: "Wildlife · FLUX Kontext",
  },
  {
    id: "cat2",
    src: "/cards/cat2.jpg",
    title: "Curious Kitten Macro",
    tag: "Detail · Real-ESRGAN",
  },
];

export function PhotoLayersAnimated() {
  const reduced = usePrefersReducedMotion();
  const [activeIdx, setActiveIdx] = React.useState(0);

  // Drag interaction state
  const [dragOffset, setDragOffset] = React.useState(0);
  const [isDragging, setIsDragging] = React.useState(false);
  const [dragStartX, setDragStartX] = React.useState(0);

  // Auto demo state
  const [isAutoDemo, setIsAutoDemo] = React.useState(false);
  const [autoDemoProgress, setAutoDemoProgress] = React.useState(0);
  // Starts at 0 (not Date.now(), which would be an impure render call) and
  // is set to the real mount time in the effect right below — auto-demo's
  // 4s idle check only needs this to be roughly "now" shortly after mount.
  const lastInteractionRef = React.useRef<number>(0);

  React.useEffect(() => {
    lastInteractionRef.current = Date.now();
  }, []);

  // Handle switching to next / previous card
  const handleNext = React.useCallback(() => {
    setActiveIdx((prev) => (prev + 1) % CARDS.length);
    setDragOffset(0);
  }, []);

  const handlePrev = React.useCallback(() => {
    setActiveIdx((prev) => (prev - 1 + CARDS.length) % CARDS.length);
    setDragOffset(0);
  }, []);

  // Pointer event handlers for rock-solid mouse & touch drag
  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    lastInteractionRef.current = Date.now();
    setIsDragging(true);
    setDragStartX(e.clientX);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!isDragging) return;
    lastInteractionRef.current = Date.now();
    const diff = e.clientX - dragStartX;
    // Allow dragging left or slightly right
    setDragOffset(diff);
  }

  function handlePointerUp(e: React.PointerEvent<HTMLDivElement>) {
    if (!isDragging) return;
    setIsDragging(false);
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture already released
    }

    // Threshold to switch card
    if (dragOffset < -60) {
      handleNext();
    } else if (dragOffset > 60) {
      handlePrev();
    } else {
      setDragOffset(0);
    }
  }

  // Automatic demo cycle when user is idle
  React.useEffect(() => {
    if (reduced) return;

    const interval = setInterval(() => {
      // If user interacted recently (last 4 seconds), pause auto-demo
      if (Date.now() - lastInteractionRef.current < 4000 || isDragging) {
        setIsAutoDemo(false);
        setAutoDemoProgress(0);
        return;
      }

      // Start auto-demo swipe
      setIsAutoDemo(true);
      const startTime = Date.now();
      const duration = 1200;

      const step = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        setAutoDemoProgress(progress);

        if (progress < 1) {
          requestAnimationFrame(step);
        } else {
          setActiveIdx((prev) => (prev + 1) % CARDS.length);
          setIsAutoDemo(false);
          setAutoDemoProgress(0);
        }
      };
      requestAnimationFrame(step);
    }, 4500);

    return () => clearInterval(interval);
  }, [reduced, isDragging]);

  // Current active cards in stack
  const topCard = CARDS[activeIdx];
  const nextCard = CARDS[(activeIdx + 1) % CARDS.length];
  const backCard = CARDS[(activeIdx + 2) % CARDS.length];

  // Combined progress (either from manual drag or auto-demo)
  const currentDragProgress = isDragging
    ? Math.min(Math.max(-dragOffset / 280, -0.3), 1)
    : isAutoDemo
    ? autoDemoProgress
    : 0;

  return (
    <div
      className="relative flex h-full w-full select-none items-center justify-center overflow-hidden rounded-xl bg-gradient-to-b from-[#130d24] via-[#0e091a] to-[#07050e]"
      style={{ perspective: 1200 }}
      onPointerDown={() => {
        lastInteractionRef.current = Date.now();
      }}
    >
      {/* Background ambient drifting bokeh motes */}
      {!reduced && (
        <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
          {[
            { top: "18%", left: "12%", size: 6, dur: 5.2 },
            { top: "72%", left: "82%", size: 8, dur: 6.0 },
            { top: "28%", left: "78%", size: 5, dur: 4.8 },
            { top: "82%", left: "22%", size: 7, dur: 5.5 },
          ].map((mote, i) => (
            <motion.span
              key={i}
              className="absolute rounded-full bg-brand/25 shadow-[0_0_15px_rgba(236,72,153,0.4)] blur-[1px]"
              style={{
                top: mote.top,
                left: mote.left,
                width: mote.size,
                height: mote.size,
              }}
              animate={{
                y: [-6, 6, -6],
                opacity: [0.2, 0.6, 0.2],
              }}
              transition={{
                duration: mote.dur,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          ))}
        </div>
      )}

      {/* 3D Angled Stack Scene */}
      <div
        className="relative flex h-[310px] w-[330px] sm:h-[350px] sm:w-[370px] items-center justify-center cursor-grab active:cursor-grabbing"
        style={{
          transformStyle: "preserve-3d",
          transform: "rotateY(-16deg) rotateX(10deg) rotateZ(-4deg)",
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        {/* ================= CARD 3 (Back Layer) ================= */}
        <div
          onClick={(e) => {
            e.stopPropagation();
            handleNext();
          }}
          className="absolute inset-0 rounded-xl border-4 border-white/80 bg-white shadow-[0_12px_30px_rgba(0,0,0,0.7)] transition-all duration-500 ease-out overflow-hidden cursor-pointer"
          style={{
            transform: `translate3d(${56 - currentDragProgress * 28}px, ${
              -24 + currentDragProgress * 12
            }px, ${-60 + currentDragProgress * 40}px) scale(${0.86 + currentDragProgress * 0.06})`,
            filter: `blur(${Math.max(2.5 - currentDragProgress * 1.5, 0.8)}px) brightness(${
              0.8 + currentDragProgress * 0.1
            })`,
            opacity: 0.85 + currentDragProgress * 0.15,
          }}
        >
          <div className="pointer-events-none relative h-full w-full">
            <Image
              src={backCard.src}
              alt={backCard.title}
              fill
              draggable={false}
              sizes="370px"
              className="object-cover pointer-events-none"
            />
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-transparent via-white/10 to-transparent" />
          </div>
        </div>

        {/* ================= CARD 2 (Next Layer - Springs forward) ================= */}
        <div
          onClick={(e) => {
            e.stopPropagation();
            handleNext();
          }}
          className="absolute inset-0 rounded-xl border-4 border-white bg-white shadow-[0_16px_40px_rgba(0,0,0,0.65)] transition-all duration-500 ease-out overflow-hidden cursor-pointer"
          style={{
            transform: `translate3d(${28 - currentDragProgress * 28}px, ${
              -12 + currentDragProgress * 12
            }px, ${-20 + currentDragProgress * 35}px) scale(${0.93 + currentDragProgress * 0.07})`,
            filter: `blur(${Math.max(1.2 - currentDragProgress * 1.2, 0)}px) brightness(${
              0.92 + currentDragProgress * 0.08
            })`,
            zIndex: 10,
          }}
        >
          <div className="pointer-events-none relative h-full w-full">
            <Image
              src={nextCard.src}
              alt={nextCard.title}
              fill
              draggable={false}
              sizes="370px"
              className="object-cover pointer-events-none"
            />
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-transparent via-white/15 to-transparent" />
          </div>
        </div>

        {/* ================= CARD 1 (Top Layer - Direct drag / flick away) ================= */}
        <div
          key={topCard.id}
          className="absolute inset-0 rounded-xl border-4 border-white bg-white shadow-[0_24px_50px_rgba(0,0,0,0.7)] overflow-hidden"
          style={{
            zIndex: 20,
            transformStyle: "preserve-3d",
            transform:
              currentDragProgress !== 0
                ? `translate3d(${currentDragProgress * -320}px, ${
                    currentDragProgress * -15
                  }px, ${currentDragProgress * 20}px) rotateY(${
                    currentDragProgress * -42
                  }deg) rotateZ(${currentDragProgress * -14}deg)`
                : "translate3d(0px, 0px, 15px)",
            opacity: currentDragProgress > 0 ? Math.max(1 - currentDragProgress * 1.25, 0) : 1,
            transition: isDragging ? "none" : "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        >
          <div className="pointer-events-none relative h-full w-full">
            <Image
              src={topCard.src}
              alt={topCard.title}
              fill
              priority
              draggable={false}
              sizes="370px"
              className="object-cover pointer-events-none"
            />

            {/* Glowing Crop corner markers */}
            <div className="pointer-events-none absolute top-3 left-3 size-5 border-t-2 border-l-2 border-pink-400 drop-shadow-[0_0_8px_rgba(244,114,182,0.8)]" />
            <div className="pointer-events-none absolute bottom-3 right-3 size-5 border-b-2 border-r-2 border-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.8)]" />

            {/* Specular gloss sheen */}
            <div
              className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-transparent via-white/20 to-transparent"
              style={{
                transform: `translateX(${currentDragProgress * 100}%)`,
                transition: "transform 0.3s ease-out",
              }}
            />
          </div>
        </div>

        {/* ================= DEMO CURSOR GESTURE ================= */}
        {isAutoDemo && (
          <div
            className="pointer-events-none absolute z-30"
            style={{
              left: "60%",
              top: "52%",
              transform: `translate3d(${autoDemoProgress * -250}px, ${
                autoDemoProgress * -10
              }px, 60px)`,
            }}
          >
            <div className="relative flex items-center justify-center">
              <div className="size-12 rounded-full border border-brand/60 bg-brand/20 shadow-[0_0_20px_rgba(236,72,153,0.8)] animate-ping" />
              <div className="absolute size-9 rounded-full border-2 border-white/80 bg-white/30 backdrop-blur-sm shadow-[0_0_15px_rgba(255,255,255,0.7)]" />
              <div className="absolute size-3 rounded-full bg-white shadow-[0_0_10px_#ffffff]" />
              <div
                className="absolute right-[-24px] h-1.5 rounded-full bg-gradient-to-r from-transparent to-brand/80"
                style={{ width: `${Math.min(autoDemoProgress * 60, 45)}px` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Interactive Navigation Chevron Buttons */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          lastInteractionRef.current = Date.now();
          handlePrev();
        }}
        aria-label="Previous photo"
        className="absolute left-3 top-1/2 -translate-y-1/2 z-30 flex size-9 items-center justify-center rounded-full border border-white/15 bg-black/50 text-white/80 backdrop-blur-md transition-all hover:bg-brand hover:text-white hover:border-brand/40 shadow-lg"
      >
        <CaretLeft className="size-5" weight="bold" />
      </button>

      <button
        onClick={(e) => {
          e.stopPropagation();
          lastInteractionRef.current = Date.now();
          handleNext();
        }}
        aria-label="Next photo"
        className="absolute right-3 top-1/2 -translate-y-1/2 z-30 flex size-9 items-center justify-center rounded-full border border-white/15 bg-black/50 text-white/80 backdrop-blur-md transition-all hover:bg-brand hover:text-white hover:border-brand/40 shadow-lg"
      >
        <CaretRight className="size-5" weight="bold" />
      </button>

      {/* Floating Info & Interactive Dots Badge */}
      <div className="absolute bottom-3 inset-x-3 z-30 flex items-center justify-between pointer-events-none">
        <div className="rounded-lg border border-white/15 bg-black/65 px-3 py-1.5 backdrop-blur-md text-[11px] font-medium text-white/90 shadow-lg flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-brand animate-pulse" />
          {topCard.tag} · {activeIdx + 1}/{CARDS.length}
        </div>

        {/* Interactive dots */}
        <div className="pointer-events-auto flex items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-2.5 py-1 backdrop-blur-md">
          {CARDS.map((card, idx) => (
            <button
              key={card.id}
              onClick={() => {
                lastInteractionRef.current = Date.now();
                setActiveIdx(idx);
                setDragOffset(0);
              }}
              aria-label={`Go to photo ${idx + 1}`}
              className={`size-2 rounded-full transition-all ${
                activeIdx === idx
                  ? "bg-brand scale-125 shadow-[0_0_8px_rgba(236,72,153,0.8)]"
                  : "bg-white/40 hover:bg-white/70"
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
