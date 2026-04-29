"use client";

import React from "react";
import { motion } from "framer-motion";
import { Brain, Target, Lightbulb, Zap, Shield, BarChart3 } from "lucide-react";
import { Card, CardContent } from "../ui/Card";

const features = [
  {
    icon: Brain,
    title: "Cross-Attention Fusion",
    description: "ResNet50 spatial patches attend to every BioLinkBERT token — question words guide which image regions matter",
    color: "blue",
  },
  {
    icon: BarChart3,
    title: "High Accuracy",
    description: "74% validation accuracy on SLAKE 1.0 — 202 normalized answer classes across radiology, MRI and CT",
    color: "green",
  },
  {
    icon: Target,
    title: "ROI Detection",
    description: "Identifies 39 anatomical regions of interest with spatial confidence scoring per region",
    color: "purple",
  },
  {
    icon: Lightbulb,
    title: "Explainable AI",
    description: "Knowledge graph with 4,444 medical triplets generates human-readable rationale for every prediction",
    color: "yellow",
  },
  {
    icon: Zap,
    title: "Real-time Analysis",
    description: "Dual-head predictor routes CLOSED (yes/no) and OPEN (descriptive) questions to specialized classifiers",
    color: "red",
  },
  {
    icon: Shield,
    title: "Privacy Focused",
    description: "Images are processed in-memory and never written to disk — your scans stay private",
    color: "indigo",
  },
];

const colorClasses = {
  blue: "bg-blue-100 text-blue-600",
  green: "bg-green-100 text-green-600",
  purple: "bg-purple-100 text-purple-600",
  yellow: "bg-yellow-100 text-yellow-600",
  red: "bg-red-100 text-red-600",
  indigo: "bg-indigo-100 text-indigo-600",
};

export function Features() {
  return (
    <section className="py-20 bg-white">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Why VisiHealth AI?</h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            State-of-the-art medical visual question answering powered by deep learning
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card hover3d className="h-full">
                  <CardContent className="pt-6">
                    <div
                      className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${
                        colorClasses[feature.color as keyof typeof colorClasses]
                      }`}
                    >
                      <Icon className="h-6 w-6" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
