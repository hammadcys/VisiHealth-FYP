"use client";

import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Text3D, Center } from "@react-three/drei";
import * as THREE from "three";

/**
 * Animated 3D Confidence Gauge
 */
function ConfidenceRing({ value }: { value: number }) {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.5;
    }
  });

  // Calculate color based on confidence
  const color = value >= 70 ? "#10b981" : value >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <group>
      {/* Background ring */}
      <mesh rotation={[0, 0, 0]}>
        <torusGeometry args={[2, 0.2, 16, 100]} />
        <meshStandardMaterial color="#e5e7eb" />
      </mesh>

      {/* Progress ring */}
      <mesh ref={ringRef} rotation={[0, 0, -Math.PI / 2]}>
        <torusGeometry args={[2, 0.25, 16, 100, (value / 100) * Math.PI * 2]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
      </mesh>

      {/* Center text */}
      <Center>
        <Text3D
          font="/fonts/helvetiker_regular.typeface.json"
          size={0.8}
          height={0.2}
          curveSegments={12}
        >
          {value.toFixed(0)}%
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
        </Text3D>
      </Center>
    </group>
  );
}

/**
 * ConfidenceGauge3D Component
 */
export function ConfidenceGauge3D({
  value,
  className = "",
}: {
  value: number;
  className?: string;
}) {
  return (
    <div className={className} style={{ width: "100%", height: "300px" }}>
      <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <spotLight position={[10, 10, 10]} angle={0.3} penumbra={1} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />

        {/* Confidence Ring */}
        <ConfidenceRing value={value} />
      </Canvas>
    </div>
  );
}
