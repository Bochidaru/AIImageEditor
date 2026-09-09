"use client";

import * as React from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, MeshDistortMaterial } from "@react-three/drei";
import * as THREE from "three";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

/**
 * Purely decorative ambient background for the empty/upload state. Kept out
 * of the main image canvas so it never competes with the actual editing
 * surface (Phase 5: "avoid animation that distracts from image editing").
 * Respects prefers-reduced-motion by freezing rotation.
 */

function DriftingShape({
  position,
  color,
  scale,
  speed,
  reduced,
}: {
  position: [number, number, number];
  color: string;
  scale: number;
  speed: number;
  reduced: boolean;
}) {
  const ref = React.useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (reduced || !ref.current) return;
    ref.current.rotation.x += delta * speed * 0.15;
    ref.current.rotation.y += delta * speed * 0.2;
  });

  return (
    <Float
      speed={reduced ? 0 : speed}
      rotationIntensity={reduced ? 0 : 0.4}
      floatIntensity={reduced ? 0 : 0.6}
    >
      <mesh ref={ref} position={position} scale={scale}>
        <icosahedronGeometry args={[1, 4]} />
        <MeshDistortMaterial
          color={color}
          roughness={0.25}
          metalness={0.1}
          distort={0.35}
          speed={reduced ? 0 : 1.4}
        />
      </mesh>
    </Float>
  );
}

export function AmbientScene({ className }: { className?: string }) {
  const reduced = usePrefersReducedMotion();

  return (
    <div className={className} aria-hidden="true" style={{ width: "100%", height: "100%" }}>
      <Canvas
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true }}
        camera={{ position: [0, 0, 7], fov: 42 }}
        resize={{ scroll: false, debounce: 0 }}
        style={{ width: "100%", height: "100%" }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[4, 4, 4]} intensity={1.4} />
        <directionalLight position={[-4, -2, 2]} intensity={0.5} />
        <DriftingShape
          position={[-3.4, 1, -2]}
          color="#ec4899"
          scale={0.7}
          speed={0.8}
          reduced={reduced}
        />
        <DriftingShape
          position={[3.6, -0.8, -3]}
          color="#a78bfa"
          scale={0.5}
          speed={1.1}
          reduced={reduced}
        />
        <DriftingShape
          position={[1, 2.2, -4]}
          color="#38bdf8"
          scale={0.35}
          speed={1.4}
          reduced={reduced}
        />
      </Canvas>
    </div>
  );
}
