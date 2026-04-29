/**
 * TypeScript types for VisiHealth AI Frontend
 */

export interface PredictionResponse {
  success: boolean;
  data: {
    answer: string;
    confidence: number;
    top_predictions: TopPrediction[];
    roi: ROIInfo;
    rationale: string;
    attention_map?: string; // base64 encoded image
  };
  error?: string;
}

export interface TopPrediction {
  answer: string;
  confidence: number;
}

export interface ROIInfo {
  top_region: number;
  confidence: number;
  region_name?: string;
}

export interface VocabularyResponse {
  success: boolean;
  data: {
    total: number;
    vocabulary: string[];
  };
}

export interface ModelInfoResponse {
  model_loaded: boolean;
  device: string;
  num_classes: number;
  kg_enabled: boolean;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface AnalysisHistory {
  id: string;
  imageUrl: string;
  question: string;
  answer: string;
  confidence: number;
  timestamp: Date;
  topPredictions: TopPrediction[];
  roi: ROIInfo;
  rationale: string;
  attentionMap?: string;
}

export interface AnalyzerState {
  image: File | null;
  imagePreview: string | null;
  question: string;
  isAnalyzing: boolean;
  result: PredictionResponse["data"] | null;
  error: string | null;
}
