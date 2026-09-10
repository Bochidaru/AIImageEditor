"use client";

import { CursorClick, TextAa, MagicWand, DownloadSimple } from "@phosphor-icons/react";
import { Reveal } from "@/components/ui/reveal";

const STEPS = [
  {
    icon: MagicWand,
    title: "Tải ảnh & chọn vùng",
    description: "Click một điểm hoặc kéo khung — SAM2 tự tách vùng cần chỉnh.",
  },
  {
    icon: CursorClick,
    title: "Chọn thao tác",
    description: "Xoá vật thể, thay thế, đổi nền, chèn vật thể mới, hoặc mở rộng khung ảnh.",
  },
  {
    icon: TextAa,
    title: "Mô tả bằng lời",
    description: "Viết prompt mô tả kết quả mong muốn cho các thao tác cần sinh ảnh mới.",
  },
  {
    icon: DownloadSimple,
    title: "So sánh & xuất ảnh",
    description: "Kéo thanh so sánh trước/sau, hoàn tác nếu cần, rồi tải ảnh kết quả về.",
  },
];

export function WorkflowSection() {
  return (
    <section id="workflow" className="border-t border-white/5 bg-canvas py-20">
      <div className="mx-auto max-w-6xl px-6">
        <Reveal delay={0}>
          <h2 className="max-w-xl text-3xl font-bold tracking-tight text-canvas-foreground sm:text-4xl">
            Bốn bước, không cần kỹ năng vẽ mask
          </h2>
        </Reveal>

        <ol className="mt-12 grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((step, index) => (
            <li key={step.title} className="relative">
              <Reveal delay={index * 130} yOffset={35}>
                <span className="text-sm font-mono text-brand/70">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="mt-3 flex size-10 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] text-canvas-foreground">
                  <step.icon className="size-5" />
                </span>
                <h3 className="mt-4 text-sm font-semibold text-canvas-foreground">{step.title}</h3>
                <p className="mt-1.5 text-sm text-canvas-foreground/60">{step.description}</p>
              </Reveal>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
