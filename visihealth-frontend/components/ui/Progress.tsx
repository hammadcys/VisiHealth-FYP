"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface ProgressProps {
  value: number; // 0-100
  className?: string;
  color?: "blue" | "green" | "yellow" | "red";
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
}

export function Progress({
  value,
  className,
  color = "blue",
  showLabel = false,
  size = "md",
}: ProgressProps) {
  const safeValue = Math.min(100, Math.max(0, value));

  const colors = {
    blue: "bg-blue-500",
    green: "bg-green-500",
    yellow: "bg-yellow-500",
    red: "bg-red-500",
  };

  const sizes = {
    sm: "h-1",
    md: "h-2",
    lg: "h-3",
  };

  return (
    <div className={cn("w-full", className)}>
      <div className={cn("w-full bg-gray-200 rounded-full overflow-hidden", sizes[size])}>
        <div
          className={cn("h-full transition-all duration-500 ease-out", colors[color])}
          style={{ width: `${safeValue}%` }}
        />
      </div>
      {showLabel && (
        <div className="text-sm text-gray-600 mt-1 text-right">{safeValue.toFixed(1)}%</div>
      )}
    </div>
  );
}
