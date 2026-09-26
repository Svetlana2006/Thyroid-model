"""Pydantic request/response schemas for the thyroid classification API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


# ── /api/models ───────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    label: str
    seeds: List[int] = []
    type: str  # "single" | "seed_ensemble" | "full_ensemble"


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


# ── /api/predict ──────────────────────────────────────────────────────────────

class PerModelBreakdown(BaseModel):
    arch: str
    seed: int
    probability: float
    weight: Optional[float] = None


class HeatmapEntry(BaseModel):
    arch: str
    method: str  # "gradcam++" | "attention_rollout"
    image_base64: str


class PredictResponse(BaseModel):
    prediction: str  # "malignant" | "benign"
    probability: float  # calibrated
    raw_probability: float  # uncalibrated sigmoid
    calibrated: bool
    temperature: float
    threshold: float
    confidence: str  # "high" | "low"
    uncertainty_score: float
    device: str
    ensemble_weights_used: Optional[Dict[str, float]] = None
    per_model_breakdown: List[PerModelBreakdown] = []
    heatmaps: List[HeatmapEntry] = []
    localization_overlap: Optional[float] = None


# ── /api/health ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    device: str
