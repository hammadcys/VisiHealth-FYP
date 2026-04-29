"use client";

import { useCallback } from "react";
import { Upload, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ImageUploaderProps {
  image: File | null;
  imagePreview: string | null;
  onImageSelect: (file: File) => void;
  onImageRemove: () => void;
  disabled?: boolean;
}

export function ImageUploader({
  image,
  imagePreview,
  onImageSelect,
  onImageRemove,
  disabled = false,
}: ImageUploaderProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (disabled) return;

      const files = Array.from(e.dataTransfer.files);
      const imageFile = files.find((file) => file.type.startsWith("image/"));
      if (imageFile) {
        onImageSelect(imageFile);
      }
    },
    [onImageSelect, disabled]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  return (
    <div className="space-y-4">
      <AnimatePresence mode="wait">
        {!imagePreview ? (
          <motion.div
            key="uploader"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className={`
              relative border-2 border-dashed rounded-xl p-12
              transition-all duration-300 cursor-pointer
              ${
                disabled
                  ? "border-gray-300 bg-gray-50 cursor-not-allowed"
                  : "border-blue-300 bg-blue-50/30 hover:bg-blue-50/50 hover:border-blue-400"
              }
            `}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              disabled={disabled}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
            />
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="p-4 bg-blue-100 rounded-full">
                <Upload className="w-8 h-8 text-blue-600" />
              </div>
              <div>
                <p className="text-lg font-semibold text-gray-700">
                  Drop your medical image here
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  or click to browse • PNG, JPG up to 10MB
                </p>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative rounded-xl overflow-hidden border-2 border-blue-200 bg-gray-50"
          >
            <img
              src={imagePreview}
              alt="Preview"
              className="w-full h-auto max-h-96 object-contain"
            />
            <button
              onClick={onImageRemove}
              disabled={disabled}
              className="absolute top-4 right-4 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors disabled:opacity-50"
            >
              <X className="w-5 h-5" />
            </button>
            {image && (
              <div className="absolute bottom-4 left-4 px-4 py-2 bg-black/60 text-white rounded-lg text-sm backdrop-blur-sm">
                <p className="font-medium">{image.name}</p>
                <p className="text-xs text-gray-300">
                  {(image.size / 1024).toFixed(1)} KB
                </p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
