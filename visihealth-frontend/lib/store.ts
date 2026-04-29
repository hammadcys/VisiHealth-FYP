/**
 * Zustand State Management Store
 * Global state for VisiHealth AI
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AnalysisHistory, AnalyzerState } from "@/types/api";

interface AppState extends AnalyzerState {
  // History
  history: AnalysisHistory[];

  // Analyzer actions
  setImage: (file: File | null, preview: string | null) => void;
  setQuestion: (question: string) => void;
  setAnalyzing: (isAnalyzing: boolean) => void;
  setResult: (result: AnalyzerState["result"]) => void;
  setError: (error: string | null) => void;
  clearAnalyzer: () => void;

  // History actions
  addToHistory: (analysis: AnalysisHistory) => void;
  removeFromHistory: (id: string) => void;
  clearHistory: () => void;
}

// Initial state
const initialState: AnalyzerState & { history: AnalysisHistory[] } = {
  image: null,
  imagePreview: null,
  question: "",
  isAnalyzing: false,
  result: null,
  error: null,
  history: [],
};

/**
 * Main application store
 */
export const useStore = create<AppState>()(
  persist(
    (set) => ({
      ...initialState,

      // Analyzer actions
      setImage: (file, preview) =>
        set({
          image: file,
          imagePreview: preview,
          error: null,
        }),

      setQuestion: (question) =>
        set({
          question,
          error: null,
        }),

      setAnalyzing: (isAnalyzing) =>
        set({
          isAnalyzing,
          error: null,
        }),

      setResult: (result) =>
        set({
          result,
          isAnalyzing: false,
          error: null,
        }),

      setError: (error) =>
        set({
          error,
          isAnalyzing: false,
        }),

      clearAnalyzer: () =>
        set({
          image: null,
          imagePreview: null,
          question: "",
          isAnalyzing: false,
          result: null,
          error: null,
        }),

      // History actions
      addToHistory: (analysis) =>
        set((state) => ({
          history: [analysis, ...state.history],
        })),

      removeFromHistory: (id) =>
        set((state) => ({
          history: state.history.filter((item) => item.id !== id),
        })),

      clearHistory: () =>
        set({
          history: [],
        }),
    }),
    {
      name: "visihealth-storage", // localStorage key
      partialize: (state) => ({
        history: state.history, // Only persist history
      }),
    }
  )
);
