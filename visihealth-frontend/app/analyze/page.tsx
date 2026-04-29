"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader, Send, ArrowLeft, AlertTriangle } from "lucide-react";
import { ImageUploader } from "@/components/analyzer/ImageUploader";
import { QuestionInput } from "@/components/analyzer/QuestionInput";
import { ResultsDisplay } from "@/components/analyzer/ResultsDisplay";
import { api } from "@/lib/api";
import type { PredictionResponse } from "@/types/api";
import toast from "react-hot-toast";
import Link from "next/link";

export default function AnalyzePage() {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResponse["data"] | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);

  const handleImageSelect = useCallback((file: File) => {
    setImage(file);
    setResult(null);
    setImageError(null); // Clear any previous error

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleImageRemove = useCallback(() => {
    setImage(null);
    setImagePreview(null);
    setResult(null);
    setImageError(null); // Clear error when image is removed
  }, []);

  const handleAnalyze = async () => {
    if (!image || !question.trim()) {
      toast.error("Please upload an image and enter a question");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const response = await api.predict(image, question.trim());
      
      if (response.success && response.data) {
        let finalResult = response.data;

        // Fallback: if /predict does not include a renderable base64 attention map,
        // fetch it from the dedicated visualization endpoint.
        if (
          !finalResult.attention_map ||
          typeof finalResult.attention_map !== "string"
        ) {
          try {
            const attentionResponse = await api.visualizeAttention(image, question.trim());
            if (attentionResponse.success && attentionResponse.data?.visualization) {
              finalResult = {
                ...finalResult,
                attention_map: attentionResponse.data.visualization,
              };
            }
          } catch (attentionError) {
            console.warn("Attention visualization unavailable:", attentionError);
          }
        }

        setResult(finalResult);
        toast.success("Analysis complete!");
        
        // Save to history with compressed thumbnail
        await saveToHistory(finalResult);
      } else {
        const errMsg = response.error || "Analysis failed";
        // Surface medical-image rejection as an inline banner
        if (
          errMsg.toLowerCase().includes("medical") ||
          errMsg.toLowerCase().includes("does not appear")
        ) {
          setImageError(errMsg);
        } else {
          toast.error(errMsg);
        }
      }
    } catch (error) {
      // Only real network/server errors reach here (422s are returned as structured responses)
      const errMsg = error instanceof Error ? error.message : "Failed to analyze image";
      toast.error(errMsg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ─── Canvas thumbnail compressor ──────────────────────────────────────────
  // Shrinks the image to 80×80 px at 30 % JPEG quality.
  // Raw base64 preview ≈ 400 KB → thumbnail ≈ 3–5 KB — safe for localStorage.
  const compressThumbnail = (dataUrl: string): Promise<string> =>
    new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const SIZE = 80;
        const canvas = document.createElement("canvas");
        canvas.width = SIZE;
        canvas.height = SIZE;
        const ctx = canvas.getContext("2d");
        if (!ctx) { resolve(""); return; }
        // Centre-crop to square then scale
        const min = Math.min(img.width, img.height);
        const sx = (img.width - min) / 2;
        const sy = (img.height - min) / 2;
        ctx.drawImage(img, sx, sy, min, min, 0, 0, SIZE, SIZE);
        resolve(canvas.toDataURL("image/jpeg", 0.30));
      };
      img.onerror = () => resolve("");
      img.src = dataUrl;
    });

  const saveToHistory = async (data: PredictionResponse["data"]) => {
    // Build the lightweight entry first (no image)
    const newEntry: Record<string, unknown> = {
      id: Date.now().toString(),
      imageName: image?.name ?? "Medical Image",
      question,
      answer: data.answer,
      confidence: data.confidence,
      timestamp: new Date().toISOString(),
      topPredictions: data.top_predictions,
      roi: data.roi,
      rationale: data.rationale,
    };

    // Compress a thumbnail only if we have a preview
    if (imagePreview) {
      try {
        const thumb = await compressThumbnail(imagePreview);
        if (thumb) newEntry.thumbnail = thumb;  // ≈ 3–5 KB
      } catch {
        // thumbnail generation failed — no image stored, that's fine
      }
    }

    const persistHistory = (entries: object[]) =>
      localStorage.setItem("visihealth_history", JSON.stringify(entries));

    try {
      const existing: Record<string, unknown>[] = JSON.parse(
        localStorage.getItem("visihealth_history") || "[]"
      );
      // Prepend new entry; keep max 20 items
      const updated = [newEntry, ...existing].slice(0, 20);
      // Only the 5 most recent entries keep their thumbnail to save space
      const withThumbs = updated.map((entry, idx) =>
        idx >= 5 ? { ...entry, thumbnail: undefined } : entry
      );
      persistHistory(withThumbs);
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "QuotaExceededError") {
        try {
          // Fallback: save only entry without thumbnail + trim to 10
          const existing: Record<string, unknown>[] = JSON.parse(
            localStorage.getItem("visihealth_history") || "[]"
          );
          const stripped = [{ ...newEntry, thumbnail: undefined }, ...existing]
            .slice(0, 10)
            .map((e) => ({ ...e, thumbnail: undefined }));
          try {
            persistHistory(stripped);
          } catch {
            localStorage.removeItem("visihealth_history");
            persistHistory([{ ...newEntry, thumbnail: undefined }]);
          }
          toast("History trimmed — storage was full", { icon: "ℹ️" });
        } catch {
          console.warn("Could not save history — storage unavailable");
        }
      } else {
        console.error("Failed to save to history:", error);
      }
    }
  };

  const handleReset = () => {
    setImage(null);
    setImagePreview(null);
    setQuestion("");
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
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
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Medical Image Analysis
            </h1>
            <p className="text-gray-600 mt-2">
              Upload a medical image and ask questions to get AI-powered insights
            </p>
          </motion.div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Input */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Step 1: Upload Image
              </h2>
              <ImageUploader
                image={image}
                imagePreview={imagePreview}
                onImageSelect={handleImageSelect}
                onImageRemove={handleImageRemove}
                disabled={isAnalyzing}
              />

              {/* Non-medical image warning banner */}
              <AnimatePresence>
                {imageError && (
                  <motion.div
                    key="img-error"
                    initial={{ opacity: 0, y: -8, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -8, scale: 0.97 }}
                    transition={{ duration: 0.25 }}
                    className="flex items-start gap-3 p-4 rounded-xl border border-amber-300 bg-amber-50 text-amber-800"
                  >
                    <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0 text-amber-500" />
                    <div>
                      <p className="font-semibold text-sm">Non-medical image detected</p>
                      <p className="text-sm mt-0.5">{imageError}</p>
                      <p className="text-xs mt-2 text-amber-600">
                        Remove the current image and upload an X-ray, MRI, CT scan, ultrasound, or histology slide.
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Step 2: Ask Question
              </h2>
              <QuestionInput
                question={question}
                onQuestionChange={setQuestion}
                disabled={isAnalyzing}
              />
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleAnalyze}
                disabled={!image || !question.trim() || isAnalyzing}
                className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-xl font-semibold hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Analyze Image
                  </>
                )}
              </motion.button>

              {result && (
                <motion.button
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleReset}
                  className="px-6 py-4 rounded-xl font-semibold border-2 border-gray-300 hover:border-gray-400 transition-all"
                >
                  New Analysis
                </motion.button>
              )}
            </div>
          </motion.div>

          {/* Right Column - Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:sticky lg:top-8"
          >
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 min-h-[400px]">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Analysis Results
              </h2>

              {!result && !isAnalyzing && (
                <div className="flex flex-col items-center justify-center h-64 text-gray-400">
                  <div className="w-16 h-16 border-4 border-gray-200 border-dashed rounded-full mb-4" />
                  <p className="text-center">
                    Results will appear here after analysis
                  </p>
                </div>
              )}

              {isAnalyzing && (
                <div className="flex flex-col items-center justify-center h-64">
                  <Loader className="w-12 h-12 text-blue-600 animate-spin mb-4" />
                  <p className="text-gray-600 font-medium">
                    AI is analyzing your image...
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    This may take a few seconds
                  </p>
                </div>
              )}

              {result && <ResultsDisplay result={result} />}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
