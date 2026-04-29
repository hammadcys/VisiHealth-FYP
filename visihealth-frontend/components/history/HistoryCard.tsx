"use client";

import { motion } from "framer-motion";
import { Clock, Brain, TrendingUp } from "lucide-react";
import type { AnalysisHistory } from "@/types/api";

interface HistoryCardProps {
  item: AnalysisHistory;
  onClick: () => void;
}

export function HistoryCard({ item, onClick }: HistoryCardProps) {
  const confidenceColor =
    item.confidence >= 0.7
      ? "text-green-600 bg-green-50"
      : item.confidence >= 0.4
      ? "text-yellow-600 bg-yellow-50"
      : "text-red-600 bg-red-50";

  const date = new Date(item.timestamp);
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  const formattedTime = date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <motion.div
      whileHover={{ scale: 1.02, y: -4 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all cursor-pointer border border-gray-100 overflow-hidden"
    >
      {/* Thumbnail or Placeholder */}
      <div className="relative h-48 bg-gradient-to-br from-blue-50 to-indigo-100 overflow-hidden">
        {(item as { thumbnail?: string }).thumbnail ? (
          <>
            <img
              src={(item as { thumbnail?: string }).thumbnail}
              alt="Medical scan thumbnail"
              className="w-full h-full object-cover"
              style={{ imageRendering: "auto", filter: "blur(0)" }}
            />
            {/* subtle overlay so confidence badge reads clearly */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-2">
            <Brain className="w-10 h-10 text-blue-300" />
            <p className="text-xs text-blue-400 font-medium px-4 text-center truncate max-w-full">
              {(item as { imageName?: string }).imageName ?? "Medical Image"}
            </p>
          </div>
        )}
        <div className={`absolute top-3 right-3 px-3 py-1 rounded-full text-sm font-semibold ${confidenceColor}`}>
          {(item.confidence * 100).toFixed(0)}%
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-3">
        {/* Question */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Question</p>
          <p className="text-sm text-gray-700 line-clamp-2 font-medium">
            {item.question}
          </p>
        </div>

        {/* Answer */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Answer</p>
          <p className="text-lg font-bold text-gray-800 line-clamp-1">
            {item.answer}
          </p>
        </div>

        {/* Top Predictions */}
        {item.topPredictions && item.topPredictions.length > 1 && (
          <div className="pt-2 border-t border-gray-100">
            <div className="flex items-center gap-1 text-xs text-gray-500 mb-2">
              <TrendingUp className="w-3 h-3" />
              <span>Alternatives</span>
            </div>
            <div className="flex gap-2">
              {item.topPredictions.slice(1, 3).map((pred, idx) => (
                <div
                  key={idx}
                  className="flex-1 text-xs bg-gray-50 rounded px-2 py-1"
                >
                  <p className="font-medium text-gray-700 truncate">
                    {pred.answer}
                  </p>
                  <p className="text-gray-500">{(pred.confidence * 100).toFixed(0)}%</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timestamp */}
        <div className="flex items-center gap-2 text-xs text-gray-500 pt-2 border-t border-gray-100">
          <Clock className="w-3 h-3" />
          <span>
            {formattedDate} • {formattedTime}
          </span>
        </div>
      </div>
    </motion.div>
  );
}
