"use client";

import { MessageSquare, Sparkles } from "lucide-react";
import { motion } from "framer-motion";

interface QuestionInputProps {
  question: string;
  onQuestionChange: (question: string) => void;
  disabled?: boolean;
}

const SUGGESTED_QUESTIONS = [
  "What abnormality is shown in this image?",
  "Is there any sign of disease?",
  "What is the diagnosis?",
  "What organ is visible in this scan?",
  "Is this image normal or abnormal?",
  "What medical condition is present?",
];

export function QuestionInput({
  question,
  onQuestionChange,
  disabled = false,
}: QuestionInputProps) {
  return (
    <div className="space-y-4">
      <div className="relative">
        <div className="absolute left-4 top-4 text-blue-600">
          <MessageSquare className="w-5 h-5" />
        </div>
        <textarea
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          disabled={disabled}
          placeholder="Ask a question about the medical image..."
          rows={3}
          className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-gray-900 disabled:bg-gray-50 disabled:text-gray-500 resize-none"
        />
      </div>

      {/* Suggested Questions */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <Sparkles className="w-4 h-4" />
          <span className="font-medium">Suggested questions:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {SUGGESTED_QUESTIONS.map((suggested, index) => (
            <motion.button
              key={index}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onQuestionChange(suggested)}
              disabled={disabled}
              className="px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {suggested}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}
