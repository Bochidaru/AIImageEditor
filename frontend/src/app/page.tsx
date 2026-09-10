import dynamic from "next/dynamic";
import { LandingNav } from "@/components/landing/landing-nav";
import { LandingHero } from "@/components/landing/landing-hero";
import { SectionSkeleton } from "@/components/ui/section-skeleton";

const FeatureGrid = dynamic(
  () => import("@/components/landing/feature-grid").then((m) => m.FeatureGrid),
  {
    loading: () => <SectionSkeleton titleWidth="w-56" height="min-h-[420px]" />,
  },
);

const WorkflowSection = dynamic(
  () => import("@/components/landing/workflow-section").then((m) => m.WorkflowSection),
  {
    loading: () => <SectionSkeleton titleWidth="w-72" height="min-h-[360px]" />,
  },
);

const CtaFooter = dynamic(
  () => import("@/components/landing/cta-footer").then((m) => m.CtaFooter),
  {
    loading: () => (
      <div className="mx-auto max-w-6xl px-6 py-12 animate-pulse" aria-hidden="true">
        <div className="h-40 rounded-2xl border border-white/5 bg-white/[0.02]" />
      </div>
    ),
  },
);

export default function LandingPage() {
  return (
    <div className="min-h-dvh bg-canvas">
      <LandingNav />
      <LandingHero />
      <FeatureGrid />
      <WorkflowSection />
      <CtaFooter />
    </div>
  );
}
