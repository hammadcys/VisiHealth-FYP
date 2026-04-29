"use client";

import React from "react";
import { motion } from "framer-motion";
import { Upload, MessageSquare, Sparkles } from "lucide-react";

const steps = [
  {
    icon: Upload,
    title: "Upload Medical Image",
    description: "Drag and drop or select your medical scan (X-ray, CT, MRI, etc.)",
    color: "blue",
  },
  {
    icon: MessageSquare,
    title: "Ask Your Question",
    description: "Type any question about the medical image in natural language",
    color: "purple",
  },
  {
    icon: Sparkles,
    title: "Get AI Analysis",
    description: "Receive instant answer with confidence score, attention map, and explanation",
    color: "green",
  },
];

export function HowItWorks() {
  return (
    <section className="py-20 bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold text-gray-900 mb-4">How It Works</h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Three simple steps to get AI-powered medical image analysis
          </p>
        </motion.div>

        {/* Steps */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                className="relative"
              >
                {/* Connector line */}
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-16 left-full w-full h-0.5 bg-gray-300 -z-10" />
                )}

                <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-shadow duration-300">
                  {/* Step number */}
                  <div className="flex items-center justify-between mb-4">
                    <div
                      className={`flex items-center justify-center w-16 h-16 rounded-full ${
                        step.color === "blue"
                          ? "bg-blue-100 text-blue-600"
                          : step.color === "purple"
                          ? "bg-purple-100 text-purple-600"
                          : "bg-green-100 text-green-600"
                      }`}
                    >
                      <Icon className="h-8 w-8" />
                    </div>
                    <div className="text-4xl font-bold text-gray-200">0{index + 1}</div>
                  </div>

                  <h3 className="text-xl font-bold text-gray-900 mb-2">{step.title}</h3>
                  <p className="text-gray-600">{step.description}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
