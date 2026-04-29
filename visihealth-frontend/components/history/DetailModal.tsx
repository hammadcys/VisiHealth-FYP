"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, Brain, TrendingUp, MapPin, FileText, Calendar, Award } from "lucide-react";
import type { AnalysisHistory } from "@/types/api";

interface DetailModalProps {
  item: AnalysisHistory | null;
  onClose: () => void;
}

export function DetailModal({ item, onClose }: DetailModalProps) {
  if (!item) return null;

  const date = new Date(item.timestamp);
  const formattedDate = date.toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  const formattedTime = date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });

  const confidenceColor =
    item.confidence >= 0.7
      ? "text-green-600"
      : item.confidence >= 0.4
      ? "text-yellow-600"
      : "text-red-600";

  const confidenceLabel =
    item.confidence >= 0.7
      ? "High Confidence"
      : item.confidence >= 0.4
      ? "Medium Confidence"
      : "Low Confidence";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
        >
          {/* Header */}
          <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Brain className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-800">
                  Analysis Details
                </h2>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <Calendar className="w-4 h-4" />
                  {formattedDate} • {formattedTime}
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X className="w-6 h-6 text-gray-500" />
            </button>
          </div>

          <div className="p-6 space-y-6">
            {/* Image */}
            {item.imageUrl && (
              <div className="rounded-xl overflow-hidden border-2 border-gray-200">
                <img
                  src={item.imageUrl}
                  alt="Medical scan"
                  className="w-full max-h-96 object-contain bg-gray-50"
                />
              </div>
            )}

            {/* Question */}
            <div className="bg-blue-50 rounded-xl p-4 border-2 border-blue-100">
              <p className="text-sm font-medium text-blue-800 mb-2">Question</p>
              <p className="text-gray-800">{item.question}</p>
            </div>

            {/* Answer */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border-2 border-blue-200">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-blue-600 text-white rounded-xl">
                  <Award className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-gray-600 mb-2">
                    AI Diagnosis
                  </h3>
                  <p className="text-2xl font-bold text-gray-800 mb-4">
                    {item.answer}
                  </p>
                  <div className="flex items-center gap-2 mb-2">
                    <div className="flex-1 h-3 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        style={{ width: `${item.confidence * 100}%` }}
                        className={`h-full ${
                          item.confidence >= 0.7
                            ? "bg-green-500"
                            : item.confidence >= 0.4
                            ? "bg-yellow-500"
                            : "bg-red-500"
                        }`}
                      />
                    </div>
                    <span className={`text-sm font-semibold ${confidenceColor}`}>
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className={`text-sm ${confidenceColor} font-medium`}>
                    {confidenceLabel}
                  </p>
                </div>
              </div>
            </div>

            {/* Top Predictions */}
            {item.topPredictions && item.topPredictions.length > 1 && (
              <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-gray-800">
                    Alternative Predictions
                  </h3>
                </div>
                <div className="space-y-3">
                  {item.topPredictions.slice(1, 5).map((pred, index) => (
                    <div key={index} className="flex items-center gap-3">
                      <span className="text-sm font-medium text-gray-500 w-6">
                        #{index + 2}
                      </span>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">
                            {pred.answer}
                          </span>
                          <span className="text-sm text-gray-600">
                            {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            style={{ width: `${pred.confidence * 100}%` }}
                            className="h-full bg-blue-400"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ROI */}
            {item.roi && (
              <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
                <div className="flex items-center gap-2 mb-4">
                  <MapPin className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-gray-800">
                    Region of Interest
                  </h3>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Top Region</p>
                    <p className="text-lg font-semibold text-gray-800">
                      {item.roi.region_name || `Region ${item.roi.top_region}`}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">Confidence</p>
                    <p className="text-lg font-semibold text-blue-600">
                      {(item.roi.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Rationale */}
            {item.rationale && (
              <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-gray-800">
                    Analysis Rationale
                  </h3>
                </div>
                <p className="text-gray-700 leading-relaxed">{item.rationale}</p>
              </div>
            )}

            {/* Attention Map */}
            {item.attentionMap && (
              <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
                <div className="flex items-center gap-2 mb-4">
                  <Brain className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-gray-800">Attention Map</h3>
                </div>
                <img
                  src={`data:image/png;base64,${item.attentionMap}`}
                  alt="Attention visualization"
                  className="w-full rounded-lg"
                />
                <p className="text-sm text-gray-600 mt-3">
                  This heatmap shows which regions the AI focused on when making
                  its prediction.
                </p>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
