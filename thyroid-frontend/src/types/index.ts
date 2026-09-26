/* ── TypeScript interfaces mirroring api/schemas.py exactly ── */

export interface ModelInfo {
  id: string;
  label: string;
  seeds: number[];
  type: 'single' | 'seed_ensemble' | 'full_ensemble';
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface PerModelBreakdown {
  arch: string;
  seed: number;
  probability: number;
  weight: number | null;
}

export interface HeatmapEntry {
  arch: string;
  method: string;
  image_base64: string;
}

export interface PredictResponse {
  prediction: 'malignant' | 'benign';
  probability: number;
  raw_probability: number;
  calibrated: boolean;
  temperature: number;
  threshold: number;
  confidence: 'high' | 'low';
  uncertainty_score: number;
  device: string;
  ensemble_weights_used: Record<string, number> | null;
  per_model_breakdown: PerModelBreakdown[];
  heatmaps: HeatmapEntry[];
  localization_overlap: number | null;
}

export interface HealthResponse {
  status: string;
  device: string;
}
