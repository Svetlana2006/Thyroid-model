"""Loads calibration data (temperature, threshold) from metrics JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

RESULTS_DIR = Path("outputs/results")


class CalibrationData(NamedTuple):
    temperature: float
    threshold: float
    calibrated: bool


def load_calibration(arch: str, seed: int) -> CalibrationData:
    """Load temperature + youden_threshold from the metrics JSON for a checkpoint.

    Falls back to temperature=1.0, threshold=0.5 if the file is missing,
    and sets calibrated=False so the frontend can show an explicit note.
    """
    metrics_path = RESULTS_DIR / f"{arch}_seed{seed}_metrics.json"
    if not metrics_path.exists():
        return CalibrationData(temperature=1.0, threshold=0.5, calibrated=False)

    try:
        with open(metrics_path) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        temperature = metrics.get("temperature", 1.0)
        threshold = metrics.get("youden_threshold", 0.5)
        # If youden_threshold key is missing (trained before the fix), still mark uncalibrated
        calibrated = "youden_threshold" in metrics and "temperature" in metrics
        return CalibrationData(temperature=temperature, threshold=threshold, calibrated=calibrated)
    except (json.JSONDecodeError, KeyError):
        return CalibrationData(temperature=1.0, threshold=0.5, calibrated=False)


def load_ensemble_weights() -> Optional[dict]:
    """Load ensemble_weights.json if it exists.

    Returns the full dict {fit_split, models, weights} or None.
    """
    weights_path = RESULTS_DIR / "ensemble_weights.json"
    if not weights_path.exists():
        return None
    try:
        with open(weights_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None
