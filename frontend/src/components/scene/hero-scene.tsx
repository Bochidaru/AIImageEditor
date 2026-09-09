"use client";

import * as React from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Sparkles, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion";

/**
 * Realistic and iconic 3D Camera for the Aperture hero banner.
 * Featuring:
 * - Rectangular camera body with ergonomic handgrip and dual-tone metallic finish
 * - Multistage professional lens barrel with knurled zoom ring & glowing aperture trim
 * - Optical convex front lens element with high-gloss clearcoat
 * - Pentaprism viewfinder hump with hot shoe mount
 * - Textured dials: Mode dial, shutter button assembly, control wheel
 * - Brand emblem badge, AF assist lamp, and rear LCD display outline
 * - Mouse parallax tilt + smooth continuous turntable float
 */
function CameraModel({ reduced }: { reduced: boolean }) {
  const group = React.useRef<THREE.Group>(null);
  const pointerTarget = React.useRef({ x: 0, y: 0 });

  useFrame((state, delta) => {
    if (!group.current) return;

    if (!reduced) {
      // Continuous slow turntable spin
      group.current.rotation.y += delta * 0.18;

      // Mouse parallax tilt: lerp towards pointer position
      pointerTarget.current.x = THREE.MathUtils.lerp(
        pointerTarget.current.x,
        state.pointer.x * 0.25,
        0.05
      );
      pointerTarget.current.y = THREE.MathUtils.lerp(
        pointerTarget.current.y,
        -state.pointer.y * 0.2,
        0.05
      );

      group.current.rotation.x = 0.12 + pointerTarget.current.y;
      group.current.rotation.z = -pointerTarget.current.x * 0.4;
    }
  });

  // Materials
  const chassisMaterial = (
    <meshStandardMaterial
      color="#231a38"
      roughness={0.28}
      metalness={0.75}
      envMapIntensity={1.2}
    />
  );

  const gripMaterial = (
    <meshStandardMaterial
      color="#120d20"
      roughness={0.8}
      metalness={0.2}
    />
  );

  const chromeMaterial = (
    <meshStandardMaterial
      color="#d4d4d8"
      roughness={0.15}
      metalness={0.9}
    />
  );

  const darkMetalMaterial = (
    <meshStandardMaterial
      color="#181326"
      roughness={0.35}
      metalness={0.85}
    />
  );

  const neonBrandMaterial = (
    <meshStandardMaterial
      color="#f472b6"
      emissive="#ec4899"
      emissiveIntensity={2.5}
      roughness={0.2}
      metalness={0.1}
    />
  );

  const lensGlassMaterial = (
    <meshPhysicalMaterial
      color="#030208"
      roughness={0.04}
      metalness={0.25}
      clearcoat={1}
      clearcoatRoughness={0.05}
      transmission={0.3}
      reflectivity={0.95}
      ior={1.52}
    />
  );

  return (
    <Float
      speed={reduced ? 0 : 1.2}
      rotationIntensity={0}
      floatIntensity={reduced ? 0 : 0.4}
    >
      <group ref={group} rotation={[0.12, -0.4, 0]}>
        {/* ================= 1. MAIN CAMERA BODY ================= */}
        {/* Core rectangular body */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[1.56, 0.96, 0.58]} />
          {chassisMaterial}
        </mesh>

        {/* Ergonomic Handgrip (Front-Right) */}
        <mesh position={[0.66, -0.01, 0.14]}>
          <boxGeometry args={[0.26, 0.94, 0.36]} />
          {gripMaterial}
        </mesh>
        {/* Grip front curvature fillet */}
        <mesh position={[0.66, -0.01, 0.32]} rotation={[0, 0, 0]}>
          <cylinderGeometry args={[0.12, 0.12, 0.92, 24]} />
          {gripMaterial}
        </mesh>

        {/* Top-plate beveled chamfer strip */}
        <mesh position={[0, 0.48, 0]}>
          <boxGeometry args={[1.52, 0.04, 0.56]} />
          {darkMetalMaterial}
        </mesh>

        {/* ================= 2. TOP PLATE & CONTROLS ================= */}
        {/* Pentaprism / EVF Viewfinder Hump */}
        <mesh position={[-0.06, 0.58, 0]}>
          <boxGeometry args={[0.52, 0.24, 0.48]} />
          {chassisMaterial}
        </mesh>
        {/* Viewfinder front slope */}
        <mesh position={[-0.06, 0.62, 0.16]} rotation={[0.5, 0, 0]}>
          <boxGeometry args={[0.48, 0.16, 0.22]} />
          {chassisMaterial}
        </mesh>
        {/* Hot Shoe Mount on top of prism */}
        <mesh position={[-0.06, 0.72, 0]}>
          <boxGeometry args={[0.2, 0.04, 0.22]} />
          {chromeMaterial}
        </mesh>

        {/* Mode Dial (Top Left) */}
        <mesh position={[-0.52, 0.53, 0.04]}>
          <cylinderGeometry args={[0.16, 0.16, 0.1, 32]} />
          {darkMetalMaterial}
        </mesh>
        {/* Mode dial textured knurl ring */}
        <mesh position={[-0.52, 0.54, 0.04]}>
          <torusGeometry args={[0.16, 0.012, 8, 32]} />
          {chromeMaterial}
        </mesh>

        {/* Shutter Button Assembly (Top Right, above grip) */}
        <mesh position={[0.64, 0.52, 0.14]}>
          <cylinderGeometry args={[0.12, 0.14, 0.08, 24]} />
          {chromeMaterial}
        </mesh>
        {/* Shutter release button (accent color) */}
        <mesh position={[0.64, 0.57, 0.14]}>
          <cylinderGeometry args={[0.08, 0.08, 0.04, 24]} />
          {neonBrandMaterial}
        </mesh>

        {/* Secondary Control Wheel (Top Right rear) */}
        <mesh position={[0.34, 0.52, -0.08]}>
          <cylinderGeometry args={[0.13, 0.13, 0.09, 24]} />
          {darkMetalMaterial}
        </mesh>

        {/* Strap Lugs (Left & Right sides) */}
        <mesh position={[-0.8, 0.24, 0]} rotation={[0, 0, Math.PI / 2]}>
          <torusGeometry args={[0.045, 0.015, 8, 16]} />
          {chromeMaterial}
        </mesh>
        <mesh position={[0.8, 0.24, 0]} rotation={[0, 0, Math.PI / 2]}>
          <torusGeometry args={[0.045, 0.015, 8, 16]} />
          {chromeMaterial}
        </mesh>

        {/* ================= 3. LENS ASSEMBLY ================= */}
        {/* Lens Mount Flange (Chrome collar on body) */}
        <mesh position={[-0.06, -0.02, 0.32]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.47, 0.49, 0.08, 48]} />
          {chromeMaterial}
        </mesh>

        {/* Lens Main Barrel */}
        <mesh position={[-0.06, -0.02, 0.52]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.43, 0.45, 0.34, 48]} />
          {darkMetalMaterial}
        </mesh>

        {/* Knurled Focus / Zoom Ring */}
        <mesh position={[-0.06, -0.02, 0.52]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.445, 0.445, 0.16, 48]} />
          {gripMaterial}
        </mesh>

        {/* Front Lens Barrel Extension */}
        <mesh position={[-0.06, -0.02, 0.74]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.41, 0.43, 0.28, 48]} />
          {darkMetalMaterial}
        </mesh>

        {/* Outer Lens Filter Rim / Bevel */}
        <mesh position={[-0.06, -0.02, 0.88]} rotation={[0, 0, 0]}>
          <torusGeometry args={[0.41, 0.026, 16, 48]} />
          {chromeMaterial}
        </mesh>

        {/* Glowing Brand Aperture Ring */}
        <mesh position={[-0.06, -0.02, 0.84]} rotation={[0, 0, 0]}>
          <torusGeometry args={[0.34, 0.024, 16, 48]} />
          {neonBrandMaterial}
        </mesh>

        {/* Front Lens Glass Element (Convex curvature) */}
        <mesh position={[-0.06, -0.02, 0.78]} rotation={[Math.PI / 2, 0, 0]}>
          <sphereGeometry args={[0.34, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.48]} />
          {lensGlassMaterial}
        </mesh>

        {/* Inner Aperture Diaphragm Ring */}
        <mesh position={[-0.06, -0.02, 0.7]} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.14, 0.32, 32]} />
          {chassisMaterial}
        </mesh>

        {/* ================= 4. FRONT DETAILS & ACCENTS ================= */}
        {/* Brand Red/Magenta Dot Badge (Leica/Hasselblad style) */}
        <mesh position={[-0.56, 0.26, 0.3]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.075, 0.075, 0.02, 24]} />
          {neonBrandMaterial}
        </mesh>

        {/* AF Assist Illuminator Lamp */}
        <mesh position={[-0.38, 0.3, 0.3]}>
          <sphereGeometry args={[0.038, 16, 16]} />
          <meshStandardMaterial
            color="#fbbf24"
            emissive="#f59e0b"
            emissiveIntensity={1.8}
            roughness={0.2}
          />
        </mesh>

        {/* Lens Release Button */}
        <mesh position={[0.36, -0.22, 0.3]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.045, 0.045, 0.025, 16]} />
          {chromeMaterial}
        </mesh>

        {/* Front Cyber Accent Light Strip */}
        <mesh position={[-0.72, -0.05, 0.29]} rotation={[0, 0, 0]}>
          <boxGeometry args={[0.025, 0.6, 0.02]} />
          {neonBrandMaterial}
        </mesh>

        {/* ================= 5. REAR ELEMENTS ================= */}
        {/* Back LCD Screen frame & display */}
        <mesh position={[-0.1, 0, -0.3]}>
          <boxGeometry args={[1.08, 0.68, 0.02]} />
          <meshStandardMaterial
            color="#080610"
            roughness={0.1}
            metalness={0.4}
          />
        </mesh>
        {/* Rear Viewfinder Eyepiece */}
        <mesh position={[-0.06, 0.58, -0.26]}>
          <boxGeometry args={[0.22, 0.16, 0.08]} />
          {gripMaterial}
        </mesh>
      </group>
    </Float>
  );
}

export function HeroScene({ className }: { className?: string }) {
  const reduced = usePrefersReducedMotion();

  return (
    <div
      className={`cursor-grab active:cursor-grabbing ${className ?? ""}`}
      style={{ width: "100%", height: "100%" }}
    >
      <Canvas
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true }}
        camera={{ position: [0, 0.1, 4.2], fov: 32 }}
        resize={{ scroll: false, debounce: 0 }}
        style={{ width: "100%", height: "100%" }}
      >
        <ambientLight intensity={1.1} />
        {/* Key light */}
        <directionalLight position={[4, 5, 5]} intensity={2.6} />
        {/* Cool rim light */}
        <directionalLight position={[-4, -1, 3]} intensity={1.5} color="#c4b5fd" />
        {/* Brand pink point light emphasizing camera lens & aperture */}
        <pointLight position={[1.5, -0.8, 3]} intensity={14} color="#ec4899" distance={10} />
        {/* Subtle violet fill */}
        <pointLight position={[-2, 2.5, 2]} intensity={8} color="#d8b4fe" distance={10} />

        <CameraModel reduced={reduced} />

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate={!reduced}
          autoRotateSpeed={1.5}
          enableDamping
          dampingFactor={0.06}
          maxPolarAngle={Math.PI / 2 + 0.35}
          minPolarAngle={Math.PI / 2 - 0.45}
        />

        {!reduced && (
          <Sparkles
            count={45}
            scale={5.5}
            size={2.2}
            speed={0.25}
            color="#f0abfc"
            opacity={0.65}
          />
        )}
      </Canvas>
    </div>
  );
}
