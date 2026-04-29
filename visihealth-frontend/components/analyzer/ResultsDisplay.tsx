"use client";

import { motion } from "framer-motion";
import {
  Brain,
  TrendingUp,
  MapPin,
  FileText,
} from "lucide-react";
import type { PredictionResponse } from "@/types/api";

interface ResultsDisplayProps {
  result: PredictionResponse["data"];
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const confidenceColor =
    result.confidence >= 0.7
      ? "text-green-600"
      : result.confidence >= 0.4
      ? "text-yellow-600"
      : "text-red-600";

  const confidenceLabel =
    result.confidence >= 0.7
      ? "High Confidence"
      : result.confidence >= 0.4
      ? "Medium Confidence"
      : "Low Confidence";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Main Answer */}
      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-6 border-2 border-blue-200">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-blue-600 text-white rounded-xl">
            <Brain className="w-6 h-6" />
          </div>
          <div className="flex-1">
            <h3 className="text-sm font-medium text-gray-600 mb-2">
              AI Diagnosis
            </h3>
            <p className="text-2xl font-bold text-gray-800">{result.answer}</p>
            <div className="mt-4 flex items-center gap-2">
              <div className="flex-1 h-3 bg-gray-200 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.confidence * 100}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                  className={`h-full ${
                    result.confidence >= 0.7
                      ? "bg-green-500"
                      : result.confidence >= 0.4
                      ? "bg-yellow-500"
                      : "bg-red-500"
                  }`}
                />
              </div>
              <span className={`text-sm font-semibold ${confidenceColor}`}>
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <p className={`text-sm mt-2 ${confidenceColor} font-medium`}>
              {confidenceLabel}
            </p>
          </div>
        </div>
      </div>

      {/* Top Predictions */}
      {result.top_predictions && result.top_predictions.length > 1 && (
        <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <h3 className="font-semibold text-gray-800">
              Alternative Predictions
            </h3>
          </div>
          <div className="space-y-3">
            {result.top_predictions.slice(1, 4).map((pred, index) => (
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

      {/* ROI Information */}
      {result.roi && (
        <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <MapPin className="w-5 h-5 text-blue-600" />
            <h3 className="font-semibold text-gray-800">Region of Interest</h3>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Top Region</p>
              <p className="text-lg font-semibold text-gray-800">
                {result.roi.region_name || `Region ${result.roi.top_region}`}
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-600">Confidence</p>
              <p className="text-lg font-semibold text-blue-600">
                {(result.roi.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Rationale */}
      {result.rationale && (
        <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="w-5 h-5 text-blue-600" />
            <h3 className="font-semibold text-gray-800">Analysis Rationale</h3>
          </div>
          <p className="text-gray-700 leading-relaxed">{result.rationale}</p>
        </div>
      )}

      {/* Attention Map */}
      {result.attention_map && (
        <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-5 h-5 text-blue-600" />
            <h3 className="font-semibold text-gray-800">Attention Map</h3>
          </div>
          <img
            src={`data:image/png;base64,${result.attention_map}`}
            alt="Attention visualization"
            className="w-full rounded-lg"
          />
          <p className="text-sm text-gray-600 mt-3">
            This heatmap shows which regions the AI focused on when making its
            prediction.
          </p>
        </div>
      )}
    </motion.div>
  );
}
