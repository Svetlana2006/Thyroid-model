"""
Ensemble logic for TN5000.
- Simple soft-vote ensemble (equal weights)
- Weighted ensemble (optimised on val AUC)
- Deep ensemble uncertainty (mean + variance across 9 models: 3 arch × 3 seeds)
"""

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def soft_vote_ensemble(logits_list: List[np.ndarray]) -> np.ndarray:
    """Simple average of sigmoid probabilities, returned as logits."""
    probs = [1.0 / (1.0 + np.exp(-np.clip(l, -500, 500))) for l in logits_list]
    mean_prob = np.mean(probs, axis=0)
    # Convert back to logit space for consistency
    mean_prob = np.clip(mean_prob, 1e-7, 1 - 1e-7)
    return np.log(mean_prob / (1 - mean_prob))


def weighted_ensemble(
    logits_list: List[np.ndarray],
    labels: np.ndarray,
    weight_step: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    """
    Grid search over ensemble weights (sum to 1, step=0.1).
    Optimises val AUC. Returns (best_logits, best_weights).
    """
    n = len(logits_list)
    probs_list = [1.0 / (1.0 + np.exp(-np.clip(l, -500, 500))) for l in logits_list]

    best_auc = -1.0
    best_weights = [1.0 / n] * n

    # Generate all weight combinations summing to 1 (discretised)
    steps = int(1.0 / weight_step) + 1
    candidates = [i * weight_step for i in range(steps)]

    def _search(remaining_budget, current_weights, depth):
        nonlocal best_auc, best_weights
        if depth == n - 1:
            w = current_weights + [remaining_budget]
            if abs(sum(w) - 1.0) > 1e-6:
                return
            mean_prob = sum(wt * p for wt, p in zip(w, probs_list))
            mean_prob_cl = np.clip(mean_prob, 1e-7, 1 - 1e-7)
            logits = np.log(mean_prob_cl / (1 - mean_prob_cl))
            auc = roc_auc_score(labels, logits)
            if auc > best_auc:
                best_auc = auc
                best_weights = w
            return
        for c in candidates:
            if c > remaining_budget + 1e-9:
                break
            _search(round(remaining_budget - c, 10), current_weights + [c], depth + 1)

    _search(1.0, [], 0)

    best_probs = sum(w * p for w, p in zip(best_weights, probs_list))
    best_probs = np.clip(best_probs, 1e-7, 1 - 1e-7)
    best_logits = np.log(best_probs / (1 - best_probs))
    return best_logits, best_weights


def deep_ensemble_uncertainty(
    logits_per_model: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given logits from N models (keys = model IDs), compute:
    - predictive mean (ensemble output) in logit space
    - predictive variance (uncertainty score)

    Returns: (mean_logits, variance_of_probs)
    """
    all_probs = np.array(
        [1.0 / (1.0 + np.exp(-np.clip(l, -500, 500))) for l in logits_per_model.values()]
    )
    mean_probs = all_probs.mean(axis=0)
    var_probs = all_probs.var(axis=0)

    mean_probs = np.clip(mean_probs, 1e-7, 1 - 1e-7)
    mean_logits = np.log(mean_probs / (1 - mean_probs))
    return mean_logits, var_probs


def flag_low_confidence(
    variance: np.ndarray,
    percentile_threshold: float = 90.0,
) -> np.ndarray:
    """
    Flag samples with predictive variance above the given percentile
    as 'low confidence — recommend specialist review'.
    Returns boolean array.
    """
    threshold = np.percentile(variance, percentile_threshold)
    return variance > threshold
