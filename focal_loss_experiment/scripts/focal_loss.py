"""
Binary Focal Loss implementation for the focal_loss_experiment.

This is a controlled implementation that preserves the positive-class weighting
and label-smoothing behavior of the existing BCEWithLogitsLoss used in our main model.

Key design decisions:
- pos_weight: direct multiplier on positive-class loss, same as BCEWithLogitsLoss
- label_smooth_eps: same smoothing as main model (0.05)
- gamma=2.0: focal modulation parameter (default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with optional positive-class weighting and label smoothing.

    Formula (standard focal loss with pos_weight):
      FL = alpha * (1 - p_t)^gamma * BCE(logits, targets)

    where alpha = pos_weight for positive targets, 1.0 for negative targets.
    This preserves the same pos_weight semantics as BCEWithLogitsLoss.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0,
                 label_smooth_eps: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "_pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.label_smooth_eps = label_smooth_eps
        self.reduction = reduction

    @property
    def pos_weight(self) -> float:
        return float(self._pos_weight[0])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)

        # Label smoothing — same as BCEWithLogitsLoss path in trainer.py
        targets_smooth = targets * (1.0 - self.label_smooth_eps) + 0.5 * self.label_smooth_eps

        # Sigmoid probability for the positive class
        p = torch.sigmoid(logits)

        # p_t = p if target==1 else 1-p
        p_t = torch.where(targets_smooth == 1, p, 1.0 - p)

        # Focal modulation factor: (1 - p_t)^gamma
        focal_factor = (1.0 - p_t) ** self.gamma

        # Stable per-element BCE (matches trainer.py path)
        bce = F.softplus(-logits) * targets_smooth + F.softplus(logits) * (1.0 - targets_smooth)

        # Positive-class weighting: alpha = pos_weight for pos, 1.0 for neg
        # This matches BCEWithLogitsLoss pos_weight semantics
        pos_mask = (targets_smooth > 0.5).float()
        neg_mask = 1.0 - pos_mask
        alpha = pos_mask * self._pos_weight.to(logits.device) + neg_mask * 1.0

        # Combine: alpha * focal_factor * bce
        loss = alpha * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def test_focal_loss():
    """Unit/sanity tests for BinaryFocalLoss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Binary Focal Loss — Unit Tests")
    print("=" * 60)

    all_pass = True

    def check(condition, msg, expect=None):
        nonlocal all_pass
        ok = bool(condition)
        status = "PASS" if ok else "FAIL"
        if expect is not None:
            print(f"[{status}] {msg} (got={condition:.6f}, expected={expect})")
        else:
            print(f"[{status}] {msg}")
        all_pass = all_pass and ok

    # ── Test 1: forward pass is finite ──────────────────────────
    loss_fn = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    logits = torch.tensor([1.0], device=device)
    targets = torch.tensor([1.0], device=device)
    loss_val = loss_fn(logits, targets)
    check(loss_val.isfinite(), "Forward pass finite", None)

    # ── Test 2: backward pass is finite ─────────────────────────
    logits = torch.tensor([0.5], device=device, requires_grad=True)
    targets = torch.tensor([1.0], device=device)
    loss_val = loss_fn(logits, targets)
    loss_val.backward()
    check(logits.grad.isfinite().all(), "Backward pass finite", None)

    # ── Test 3: batched forward pass finite ─────────────────────
    logits = torch.randn(10, device=device)
    targets = torch.randint(0, 2, (10,), device=device, dtype=torch.float32)
    loss_val = loss_fn(logits, targets)
    check(loss_val.isfinite(), "Batched forward finite", None)

    # ── Test 4: no NaN or Inf in gradients ──────────────────────
    logits = torch.randn(16, device=device, requires_grad=True)
    targets = torch.randint(0, 2, (16,), device=device, dtype=torch.float32)
    loss_val = loss_fn(logits, targets)
    loss_val.backward()
    check(not logits.grad.isnan().any(), "Gradients have no NaN", None)

    # ── Test 5: pos_weight increases positive class loss ───────
    logits = torch.tensor([2.0], device=device)
    targets = torch.tensor([1.0], device=device)
    loss_fn_w = BinaryFocalLoss(gamma=2.0, pos_weight=10.0, label_smooth_eps=0.0)
    loss_fn_u = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    loss_w = loss_fn_w(logits, targets).item()
    loss_u = loss_fn_u(logits, targets).item()
    check(loss_w > loss_u, f"Pos weight effect: weighted={loss_w:.6f} > unweighted={loss_u:.6f}")

    # ── Test 6: label smoothing increases loss for confident preds ──
    loss_fn_s = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.1)
    loss_fn_n = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    logits = torch.tensor([10.0], device=device)
    targets = torch.tensor([1.0], device=device)
    loss_s = loss_fn_s(logits, targets).item()
    loss_n = loss_fn_n(logits, targets).item()
    check(loss_s > loss_n, f"Label smoothing: smooth={loss_s:.6f} > no_smooth={loss_n:.6f}")

    # ── Test 7: gamma modulates loss correctly ─────────────────
    # Hard example: model is wrong (logit negative, target positive)
    hard_logits = torch.tensor([-1.0], device=device)
    hard_target = torch.tensor([1.0], device=device)
    loss_g0 = BinaryFocalLoss(gamma=0.0, pos_weight=1.0, label_smooth_eps=0.0)
    loss_g2 = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    l0 = loss_g0(hard_logits, hard_target).item()
    l2 = loss_g2(hard_logits, hard_target).item()
    check(l2 < l0, f"Gamma=2 reduces hard loss: g2={l2:.6f} < g0={l0:.6f}")

    # ── Test 8: gamma=2 down-weights easy examples ──────────────
    # Easy example: model is correct (logit positive, target positive)
    easy_logits = torch.tensor([5.0], device=device)
    easy_target = torch.tensor([1.0], device=device)
    l0_easy = loss_g0(easy_logits, easy_target).item()
    l2_easy = loss_g2(easy_logits, easy_target).item()
    check(l2_easy < l0_easy, f"Gamma=2 down-weights easy: g2={l2_easy:.6f} < g0={l0_easy:.6f}")

    # ── Test 9: positive examples handled correctly ─────────────
    logits = torch.tensor([100.0], device=device)
    targets = torch.tensor([1.0], device=device)
    loss_val = loss_fn(logits, targets)
    check(loss_val.isfinite() and loss_val.item() >= 0, "Positive example handled correctly")

    # ── Test 10: negative examples handled correctly ────────────
    logits = torch.tensor([-100.0], device=device)
    targets = torch.tensor([0.0], device=device)
    loss_val = loss_fn(logits, targets)
    check(loss_val.isfinite() and loss_val.item() >= 0, "Negative example handled correctly")

    # ── Test 11: gamma=2.0 is actually applied ──────────────────
    logit = torch.tensor([0.5], device=device)
    target = torch.tensor([1.0], device=device)
    fl_g0 = BinaryFocalLoss(gamma=0.0, pos_weight=1.0, label_smooth_eps=0.0)(logit, target).item()
    fl_g2 = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)(logit, target).item()
    check(abs(fl_g2 - fl_g0) > 1e-6, f"Gamma=2 different from gamma=0: g2={fl_g2:.6f} vs g0={fl_g0:.6f}")

    # ── Test 12: loss is non-negative ───────────────────────────
    logits = torch.randn(32, device=device) * 2
    targets = torch.randint(0, 2, (32,), device=device, dtype=torch.float32)
    loss_val = loss_fn(logits, targets)
    check(loss_val.item() >= 0, f"Loss non-negative: {loss_val.item():.6f}")

    # ── Test 13: pos_weight > 1 increases positive loss ─────────
    logits = torch.tensor([0.0], device=device)  # logit=0 => p=0.5
    targets = torch.tensor([1.0], device=device)
    loss_fn_2 = BinaryFocalLoss(gamma=2.0, pos_weight=2.0, label_smooth_eps=0.0)
    loss_fn_1 = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    l2 = loss_fn_2(logits, targets).item()
    l1 = loss_fn_1(logits, targets).item()
    check(l2 > l1, f"pos_weight=2 > pos_weight=1: g2={l2:.6f} > g1={l1:.6f}")

    # ── Test 14: label smoothing has no effect when eps=0 ───────
    loss_fn_e0 = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    loss_fn_e05 = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.05)
    logits = torch.tensor([10.0], device=device)
    targets = torch.tensor([1.0], device=device)
    l0 = loss_fn_e0(logits, targets).item()
    l05 = loss_fn_e05(logits, targets).item()
    check(l05 > l0, f"label_smooth_eps=0.05 > 0: smooth={l05:.6f} > no_smooth={l0:.6f}")

    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    ok = test_focal_loss()
    raise SystemExit(0 if ok else 1)