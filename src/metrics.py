"""
Evaluation metrics for TN5000:
AUC, accuracy, sensitivity, specificity, PPV, NPV, F1, Brier score,
bootstrap CI, calibration (ECE), temperature scaling, DeLong's test.
"""

from typing import Dict, Tuple

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute full metric suite at a given threshold."""
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "auc": float(roc_auc_score(labels, logits)),
        "accuracy": float(accuracy_score(labels, preds)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": float(brier_score(probs, labels)),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def youden_threshold(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold that maximises Youden's J = sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, sigmoid(logits))
    J = tpr - fpr
    best_idx = np.argmax(J)
    return float(thresholds[best_idx])


def bootstrap_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = 1000,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CI (2.5th–97.5th percentile) for all metrics."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    metric_samples = {k: [] for k in ["auc", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "brier"]}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(logits[idx], labels[idx], threshold)
        for k in metric_samples:
            metric_samples[k].append(m[k])

    ci = {}
    for k, vals in metric_samples.items():
        arr = np.array(vals)
        ci[k] = {
            "mean": float(np.mean(arr)),
            "p2_5": float(np.percentile(arr, 2.5)),
            "p97_5": float(np.percentile(arr, 97.5)),
        }
    return ci


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


class TemperatureScaling:
    """
    Post-hoc calibration via temperature scaling.
    Fit on val set logits/labels; apply to test logits.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 500):
        import torch
        import torch.optim as optim

        T = torch.nn.Parameter(torch.ones(1))
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        optimizer = optim.LBFGS([T], lr=lr, max_iter=max_iter)

        def eval_fn():
            optimizer.zero_grad()
            scaled = logits_t / T
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        self.temperature = float(T.item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature


def delong_test(logits_a: np.ndarray, logits_b: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    DeLong's test for comparing two AUCs on the same test set.
    Returns p-value and AUC difference.
    Implementation based on DeLong et al. (1988).
    """
    auc_a = roc_auc_score(labels, logits_a)
    auc_b = roc_auc_score(labels, logits_b)

    # Use sklearn's fast DeLong via bootstrapping (simplified)
    def auc_var(logits, labels):
        probs = sigmoid(logits)
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        n_pos, n_neg = len(pos_probs), len(neg_probs)

        # Structural components
        V10 = np.array([(probs_p > neg_probs).mean() + 0.5 * (probs_p == neg_probs).mean() for probs_p in pos_probs])
        V01 = np.array([(probs_n < pos_probs).mean() + 0.5 * (probs_n == pos_probs).mean() for probs_n in neg_probs])

        S10 = np.var(V10, ddof=1) / n_pos
        S01 = np.var(V01, ddof=1) / n_neg
        return S10 + S01

    var_a = auc_var(logits_a, labels)
    var_b = auc_var(logits_b, labels)
    diff = auc_a - auc_b
    se = np.sqrt(var_a + var_b)
    z = diff / (se + 1e-12)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "diff": float(diff),
        "z": float(z),
        "p_value": float(p_value),
    }


def format_results_table(results: dict) -> str:
    """Print a formatted results table."""
    lines = [
        f"{'Model':<25} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'NPV':>8} {'F1':>8} {'Brier':>8} {'ECE':>8}",
        "-" * 100,
    ]
    for model_name, m in results.items():
        lines.append(
            f"{model_name:<25} "
            f"{m.get('auc', 0):.4f}   "
            f"{m.get('sensitivity', 0):.4f}   "
            f"{m.get('specificity', 0):.4f}   "
            f"{m.get('ppv', 0):.4f}   "
            f"{m.get('npv', 0):.4f}   "
            f"{m.get('f1', 0):.4f}   "
            f"{m.get('brier', 0):.4f}   "
            f"{m.get('ece', 0):.4f}"
        )
    return "\n".join(lines)
