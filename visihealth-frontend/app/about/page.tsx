"use client";

import { ArrowLeft, Brain, Cpu, Database, Layers, Zap, CheckCircle } from "lucide-react";
import { motion } from "framer-motion";
import Link from "next/link";

const techStack = {
  frontend: [
    { name: "Next.js 16", description: "React framework with App Router" },
    { name: "TypeScript", description: "Type-safe development" },
    { name: "Tailwind CSS", description: "Utility-first styling" },
    { name: "Framer Motion", description: "Smooth animations" },
  ],
  backend: [
    { name: "Flask", description: "Python web framework" },
    { name: "PyTorch", description: "Deep learning framework" },
    { name: "BERT", description: "Language understanding model" },
    { name: "ResNet", description: "Image feature extraction" },
  ],
  model: [
    { name: "Fusion Architecture", description: "Multi-modal learning" },
    { name: "Knowledge Graph", description: "Medical knowledge integration" },
    { name: "Attention Mechanism", description: "Interpretable predictions" },
    { name: "Transfer Learning", description: "Pre-trained models" },
  ],
};

const modelSpecs = {
  accuracy: "74.36%",
  parameters: "~120M",
  imageEncoder: "ResNet-50",
  textEncoder: "BioLinkBERT",
  fusionMethod: "Cross-Attention",
  trainingData: "SLAKE 1.0 Dataset",
};

const features = [
  {
    icon: Brain,
    title: "Multi-Modal Fusion",
    description:
      "Combines visual and textual information for comprehensive medical analysis",
  },
  {
    icon: Zap,
    title: "Attention Mechanism",
    description:
      "Visualizes which image regions the AI focuses on for interpretability",
  },
  {
    icon: Database,
    title: "Knowledge Graph",
    description: "Incorporates medical knowledge for enhanced reasoning",
  },
  {
    icon: Cpu,
    title: "Transfer Learning",
    description: "Leverages pre-trained models for better performance",
  },
];

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-12">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              About VisiHealth AI
            </h1>
            <p className="text-xl text-gray-600 mt-4 max-w-3xl">
              An advanced medical Visual Question Answering system powered by
              multi-modal deep learning and knowledge graphs
            </p>
          </motion.div>
        </div>

        {/* Model Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-2xl p-8 mb-8 shadow-xl"
        >
          <div className="flex items-center gap-4 mb-6">
            <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
              <Brain className="w-8 h-8" />
            </div>
            <div>
              <h2 className="text-3xl font-bold">Model Architecture</h2>
              <p className="text-blue-100">Multi-modal Fusion with Attention</p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
            {Object.entries(modelSpecs).map(([key, value]) => (
              <div key={key} className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                <p className="text-blue-100 text-sm mb-1">
                  {key
                    .replace(/([A-Z])/g, " $1")
                    .replace(/^./, (str) => str.toUpperCase())}
                </p>
                <p className="text-xl font-bold">{value}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-12"
        >
          <h2 className="text-3xl font-bold text-gray-800 mb-6">
            Key Features
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className="bg-white rounded-xl p-6 border-2 border-gray-200 hover:border-blue-300 transition-colors shadow-md"
              >
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-blue-100 rounded-xl">
                    <feature.icon className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600">{feature.description}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-12"
        >
          <h2 className="text-3xl font-bold text-gray-800 mb-6">
            Technology Stack
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {Object.entries(techStack).map(([category, items], catIndex) => (
              <motion.div
                key={category}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + catIndex * 0.1 }}
                className="bg-white rounded-xl p-6 border-2 border-gray-200 shadow-md"
              >
                <h3 className="text-xl font-semibold text-gray-800 mb-4 capitalize flex items-center gap-2">
                  <Layers className="w-5 h-5 text-blue-600" />
                  {category}
                </h3>
                <div className="space-y-3">
                  {items.map((item, index) => (
                    <div key={index} className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="font-semibold text-gray-800">
                          {item.name}
                        </p>
                        <p className="text-sm text-gray-600">
                          {item.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Dataset Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-white rounded-2xl p-8 border-2 border-gray-200 shadow-md"
        >
          <div className="flex items-center gap-3 mb-6">
            <Database className="w-8 h-8 text-blue-600" />
            <h2 className="text-3xl font-bold text-gray-800">
              Training Dataset
            </h2>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                SLAKE 1.0 Dataset
              </h3>
              <div className="space-y-3 text-gray-700">
                <p className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>
                    <strong>642 radiology images</strong> (chest X-rays, CT scans, MRI)
                  </span>
                </p>
                <p className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>
                    <strong>9,835 English QA pairs</strong> (from 14,028 bilingual total)
                  </span>
                </p>
                <p className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>
                    <strong>202 answer classes</strong> (normalized, typo-cleaned)
                  </span>
                </p>
                <p className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span>
                    <strong>4,444 knowledge graph triplets</strong> for medical reasoning
                  </span>
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Question Types
              </h3>
              <div className="space-y-2">
                {[
                  "Abnormality detection",
                  "Organ identification",
                  "Disease classification",
                  "Modality recognition",
                  "Plane orientation",
                  "Location queries",
                ].map((type, index) => (
                  <div
                    key={index}
                    className="bg-blue-50 text-blue-800 px-4 py-2 rounded-lg font-medium"
                  >
                    {type}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-8 p-6 bg-yellow-50 border-2 border-yellow-200 rounded-xl">
            <div className="flex items-start gap-3">
              <div className="text-yellow-600 mt-1">
                <svg
                  className="w-6 h-6"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div>
                <h4 className="font-semibold text-yellow-800 mb-1">
                  Medical Disclaimer
                </h4>
                <p className="text-yellow-700">
                  VisiHealth AI is a research prototype and should not be used
                  for actual medical diagnosis. Always consult qualified
                  healthcare professionals for medical advice.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* How It Works */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-12 bg-gradient-to-br from-indigo-50 to-blue-50 rounded-2xl p-8 border-2 border-blue-200"
        >
          <h2 className="text-3xl font-bold text-gray-800 mb-6">
            How It Works
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: "1",
                title: "Image Processing",
                description:
                  "ResNet-50 extracts visual features from the medical image, identifying key anatomical structures and patterns",
              },
              {
                step: "2",
                title: "Question Understanding",
                description:
                  "BioBERT processes the medical question, understanding clinical terminology and intent",
              },
              {
                step: "3",
                title: "Multi-Modal Fusion",
                description:
                  "Attention mechanism combines visual and textual features, enhanced with medical knowledge graph for accurate predictions",
              },
            ].map((item, index) => (
              <div key={index} className="relative">
                <div className="absolute -top-4 -left-4 w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-bold shadow-lg">
                  {item.step}
                </div>
                <div className="bg-white rounded-xl p-6 pt-8 h-full border-2 border-gray-200">
                  <h3 className="text-xl font-semibold text-gray-800 mb-3">
                    {item.title}
                  </h3>
                  <p className="text-gray-600">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
