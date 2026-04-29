/**
 * API Client for VisiHealth AI Backend
 * Handles all communication with Flask server
 */

import axios, { AxiosError } from "axios";
import type {
  PredictionResponse,
  VocabularyResponse,
  ModelInfoResponse,
  HealthResponse,
} from "@/types/api";

// Backend API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api";

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds (model inference can take time)
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * API methods
 */
export const api = {
  /**
   * Health check
   */
  health: async (): Promise<HealthResponse> => {
    try {
      const response = await apiClient.get<HealthResponse>("/health");
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  /**
   * Get model information
   */
  modelInfo: async (): Promise<ModelInfoResponse> => {
    try {
      const response = await apiClient.get<ModelInfoResponse>("/model/info");
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  /**
   * Get answer vocabulary
   */
  getVocabulary: async (): Promise<VocabularyResponse> => {
    try {
      const response = await apiClient.get<VocabularyResponse>("/answers/vocabulary");
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  /**
   * Single image prediction
   */
  predict: async (image: File, question: string): Promise<PredictionResponse> => {
    try {
      const formData = new FormData();
      formData.append("image", image);
      formData.append("question", question);

      const response = await apiClient.post<PredictionResponse>("/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      return response.data;
    } catch (error) {
      // 422 = non-medical image rejection — return as a structured response
      // so the caller can display the banner without a noisy console error.
      if (axios.isAxiosError(error) && error.response?.status === 422) {
        const msg =
          (error.response.data as { error?: string })?.error ||
          "This does not appear to be a medical image. Please upload a medical scan.";
        return { success: false, error: msg } as unknown as PredictionResponse;
      }
      throw handleApiError(error);
    }
  },

  /**
   * Get attention map visualization
   */
  visualizeAttention: async (
    image: File,
    question: string
  ): Promise<{ success: boolean; data: { visualization: string } }> => {
    try {
      const formData = new FormData();
      formData.append("image", image);
      formData.append("question", question);

      const response = await apiClient.post("/visualize/attention", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      return response.data;
    } catch (error) {
      // Silently handle 422 (non-medical) — caller already handled it via /predict
      if (axios.isAxiosError(error) && error.response?.status === 422) {
        return { success: false, data: { visualization: "" } };
      }
      throw handleApiError(error);
    }
  },

  /**
   * Batch predictions
   */
  predictBatch: async (
    images: File[],
    questions: string[]
  ): Promise<{ success: boolean; data: { results: PredictionResponse["data"][] } }> => {
    try {
      const formData = new FormData();

      images.forEach((image) => {
        formData.append("images", image);
      });

      questions.forEach((question) => {
        formData.append("questions", question);
      });

      const response = await apiClient.post("/predict/batch", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },
};

/**
 * Handle API errors
 */
function handleApiError(error: unknown): Error {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ error?: string; message?: string }>;

    if (axiosError.response) {
      // Server responded with error
      const message =
        axiosError.response.data?.error ||
        axiosError.response.data?.message ||
        `Server error: ${axiosError.response.status}`;
      return new Error(message);
    } else if (axiosError.request) {
      // Request made but no response
      return new Error(
        "Cannot connect to backend server. Please ensure the server is running on http://localhost:5000"
      );
    }
  }

  return error instanceof Error ? error : new Error("An unexpected error occurred");
}

export default api;
