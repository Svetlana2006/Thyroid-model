/* ── API client: fetch wrappers for the backend ── */

import type { HealthResponse, ModelsResponse, PredictResponse } from '../types';

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('Backend unreachable');
  return res.json();
}

export async function getModels(): Promise<ModelsResponse> {
  const res = await fetch('/api/models');
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function predict(file: File, modelId: string): Promise<PredictResponse> {
  const form = new FormData();
  form.append('image', file);
  form.append('model_id', modelId);
  const res = await fetch('/api/predict', { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail ?? 'Prediction failed');
  }
  return res.json();
}
