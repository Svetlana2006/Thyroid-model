"""Core inference logic: load checkpoint(s) → predict, reused across routes."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from api.calibration import CalibrationData, load_calibration, load_ensemble_weights
from api.explain import generate_heatmap, get_method_for_arch
from api.registry import resolve_checkpoints
from api.schemas import HeatmapEntry, PerModelBreakdown, PredictResponse
from src.models import build_model
from src.transforms import get_val_transforms

# Cache loaded models to avoid reloading on every request
_model_cache: Dict[str, Tuple[torch.nn.Module, dict]] = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _load_model(arch: str, seed: int, checkpoint_path: Path) -> torch.nn.Module:
    """Load a model from checkpoint, with caching."""
    cache_key = f"{arch}_seed{seed}"
    if cache_key in _model_cache:
        return _model_cache[cache_key][0]

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(arch, dropout=config.get("dropout", 0.3))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model_cache[cache_key] = (model, config)
    return model


def _mc_dropout_variance(model: torch.nn.Module, image_tensor: torch.Tensor, n_forward: int = 10) -> float:
    """MC-dropout uncertainty estimate for a single model."""
    model.train()  # enable dropout
    probs = []
    with torch.no_grad():
        for _ in range(n_forward):
            logit = model(image_tensor).item()
            probs.append(_sigmoid(logit))
    model.eval()
    return float(np.var(probs))


def predict(image_bytes: bytes, model_id: str) -> PredictResponse:
    """Run full prediction pipeline for a given image and model_id."""
    # 1. Load and preprocess image
    pil_image = Image.open(image_bytes).convert("RGB")
    original_np = np.array(pil_image)

    transform = get_val_transforms()
    transformed = transform(image=original_np)
    image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    # Prepare a display-sized original for heatmap overlay
    # Match inference preprocessing: resize to 256 then center crop to 224
    display_img = cv2.resize(original_np, (256, 256), interpolation=cv2.INTER_LINEAR)
    display_img = display_img[16:240, 16:240]

    # 2. Resolve which checkpoints to load
    checkpoints = resolve_checkpoints(model_id)
    if not checkpoints:
        raise ValueError(f"Model not found: {model_id}")

    # 3. Run inference on each checkpoint
    per_model: List[PerModelBreakdown] = []
    heatmaps: List[HeatmapEntry] = []
    all_probs_raw: List[float] = []
    all_probs_cal: List[float] = []
    all_calibrations: List[CalibrationData] = []
    mc_variances: List[float] = []

    for arch, seed, ckpt_path in checkpoints:
        model = _load_model(arch, seed, ckpt_path)
        cal = load_calibration(arch, seed)
        all_calibrations.append(cal)

        # Forward pass
        with torch.no_grad():
            logit = model(image_tensor).item()

        raw_prob = _sigmoid(logit)
        cal_logit = logit / cal.temperature
        cal_prob = _sigmoid(cal_logit)

        all_probs_raw.append(raw_prob)
        all_probs_cal.append(cal_prob)

        per_model.append(PerModelBreakdown(
            arch=arch,
            seed=seed,
            probability=round(cal_prob, 4),
            weight=None,  # filled in for ensemble below
        ))

        # MC-dropout uncertainty
        mc_var = _mc_dropout_variance(model, image_tensor, n_forward=10)
        mc_variances.append(mc_var)

        # Heatmap (generate for each unique architecture, not every seed)
        # For seed ensembles, only generate one heatmap per arch
        arch_already_has_heatmap = any(h.arch == arch for h in heatmaps)
        if not arch_already_has_heatmap:
            heatmap_bytes = generate_heatmap(model, arch, image_tensor, display_img)
            if heatmap_bytes:
                heatmaps.append(HeatmapEntry(
                    arch=arch,
                    method=get_method_for_arch(arch),
                    image_base64=base64.b64encode(heatmap_bytes).decode("ascii"),
                ))

    # 4. Aggregate predictions
    is_ensemble = len(checkpoints) > 1
    ensemble_weights_used: Optional[Dict[str, float]] = None

    if is_ensemble and model_id == "full_ensemble":
        # Load ensemble weights
        ew = load_ensemble_weights()
        if ew and len(ew.get("weights", [])) == len(checkpoints):
            weights = ew["weights"]
            model_names = ew["models"]
            ensemble_weights_used = {}
            for name, w in zip(model_names, weights):
                ensemble_weights_used[name] = w

            # Apply weights to calibrated probs
            final_prob = sum(w * p for w, p in zip(weights, all_probs_cal))
            final_raw = sum(w * p for w, p in zip(weights, all_probs_raw))

            # Update per_model with weights
            for i, pm in enumerate(per_model):
                key = f"{pm.arch}_s{pm.seed}"
                pm.weight = ensemble_weights_used.get(key, 0.0)
        else:
            # Fall back to equal weighting
            final_prob = float(np.mean(all_probs_cal))
            final_raw = float(np.mean(all_probs_raw))
    elif is_ensemble:
        # Seed ensemble: simple average
        final_prob = float(np.mean(all_probs_cal))
        final_raw = float(np.mean(all_probs_raw))
    else:
        final_prob = all_probs_cal[0]
        final_raw = all_probs_raw[0]

    # 5. Determine calibration status (all models must be calibrated)
    all_calibrated = all(c.calibrated for c in all_calibrations)

    # Use the first model's threshold for single/seed-ensemble,
    # or average thresholds for full ensemble
    if len(all_calibrations) == 1:
        threshold = all_calibrations[0].threshold
        temperature = all_calibrations[0].temperature
    else:
        threshold = float(np.mean([c.threshold for c in all_calibrations]))
        temperature = float(np.mean([c.temperature for c in all_calibrations]))

    # 6. Prediction class
    prediction = "malignant" if final_prob >= threshold else "benign"

    # 7. Uncertainty / confidence
    uncertainty_score = float(np.mean(mc_variances)) if mc_variances else 0.0
    # Threshold: if variance > 0.01, flag as low confidence
    confidence = "low" if uncertainty_score > 0.01 else "high"

    return PredictResponse(
        prediction=prediction,
        probability=round(final_prob, 4),
        raw_probability=round(final_raw, 4),
        calibrated=all_calibrated,
        temperature=round(temperature, 4),
        threshold=round(threshold, 4),
        confidence=confidence,
        uncertainty_score=round(uncertainty_score, 6),
        device=str(DEVICE),
        ensemble_weights_used=ensemble_weights_used,
        per_model_breakdown=per_model,
        heatmaps=heatmaps,
        localization_overlap=None,  # Only if ground-truth bbox exists
    )
