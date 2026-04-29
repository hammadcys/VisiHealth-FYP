"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover3d?: boolean;
}

export function Card({ children, className, hover3d = false }: CardProps) {
  return (
    <div
      className={cn(
        "bg-white rounded-xl shadow-md border border-gray-200 transition-all duration-300",
        hover3d && "hover:shadow-2xl hover:-translate-y-1 hover:scale-[1.02]",
        className
      )}
      style={
        hover3d
          ? {
              transform: "translateZ(0)",
              transition: "transform 0.3s ease, box-shadow 0.3s ease",
            }
          : undefined
      }
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-6 py-4 border-b border-gray-200", className)}>{children}</div>;
}

export function CardContent({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-6 py-4", className)}>{children}</div>;
}

export function CardFooter({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-6 py-4 border-t border-gray-200", className)}>{children}</div>;
}
