"""
Corrected Binary Focal Loss implementation for focal_loss_experiment_v2.

This implementation correctly preserves the positive-class weighting and label-smoothing
behavior of the existing BCEWithLogitsLoss used in our main model, while introducing
focal modulation.

Key design decisions:
- pos_weight: applied to the positive class term in BCEWithLogitsLoss, same as BCEWithLogitsLoss
- label_smooth_eps: same smoothing as main model (0.05)
- gamma=2.0: focal modulation parameter (default)
- Correct implementation follows the structure:
  1. Compute per-example BCEWithLogitsLoss with pos_weight and label smoothing
  2. Apply focal modulation to the resulting loss
- This preserves the intended controlled structure:
  - baseline: per-example BCEWithLogitsLoss semantics  
  - focal: same per-example BCE weighting semantics multiplied by focal modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with optional positive-class weighting and label smoothing.
    
    Correct implementation that preserves pos_weight semantics of BCEWithLogitsLoss
    while introducing focal modulation.
    
    Formula (correct focal loss with pos_weight):
      FL = FL_factor * BCE_with_pos_weight(logits, targets)
      
    where:
    - FL_factor = (1 - p_t)^gamma with p_t = p if target==1 else 1-p
    - BCE_with_pos_weight uses pos_weight parameter in BCEWithLogitsLoss
    
    The focal factor is applied AFTER computing BCE with correct pos_weight,
    preserving the baseline BCE class-weighting semantics while introducing focal modulation.
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

        # 1. Label smoothing — same as BCEWithLogitsLoss path in trainer.py:94
        targets_smooth = targets * (1.0 - self.label_smooth_eps) + 0.5 * self.label_smooth_eps

        # 2. Per-example BCEWithLogitsLoss with correct pos_weight semantics
        #    (pos_weight applies specifically to positive class term)
        pos_weight_tensor = self._pos_weight.to(logits.device)
        # This is the CORRECT way: pos_weight applies to positive-class loss term
        # exactly as in PyTorch's BCEWithLogitsLoss
        bce_per_example = F.softplus(-logits) * targets_smooth * pos_weight_tensor + \
                         F.softplus(logits) * (1.0 - targets_smooth)

        # 3. Focal modulation factor: (1 - p_t)^gamma
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1.0 - p)  # use original (not smoothed) targets
        focal_factor = (1.0 - p_t) ** self.gamma

        # 4. Combine: focal_factor * BCE_with_pos_weight
        loss = focal_factor * bce_per_example

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def test_focal_loss():
    """Unit/sanity tests for BinaryFocalLoss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Binary Focal Loss — Unit Tests (CORRECTED)")
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

    # ── Test 9: correct pos_weight handling ────────────────────
    # pos_weight should only affect positive class term
    logits = torch.tensor([0.0], device=device)
    targets = torch.tensor([1.0], device=device)
    loss_fn_no_weight = BinaryFocalLoss(gamma=0.0, pos_weight=1.0, label_smooth_eps=0.0)
    loss_fn_weight = BinaryFocalLoss(gamma=0.0, pos_weight=2.0, label_smooth_eps=0.0)
    loss_no = loss_fn_no_weight(logits, targets).item()
    loss_with = loss_fn_weight(logits, targets).item()
    check(loss_with > loss_no, f"pos_weight=2 > pos_weight=1: {loss_with:.6f} > {loss_no:.6f}")

    # ── Test 10: negative examples not affected by pos_weight ────
    logits = torch.tensor([0.0], device=device)
    targets = torch.tensor([0.0], device=device)
    loss_fn_weight = BinaryFocalLoss(gamma=0.0, pos_weight=10.0, label_smooth_eps=0.0)
    loss_fn_unweighted = BinaryFocalLoss(gamma=0.0, pos_weight=1.0, label_smooth_eps=0.0)
    loss_w = loss_fn_weight(logits, targets).item()
    loss_u = loss_fn_unweighted(logits, targets).item()
    check(abs(loss_w - loss_u) < 1e-6, f"pos_weight should not affect negatives: w={loss_w:.6f}, u={loss_u:.6f}")

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

    # ── Test 13: focal loss specifically down-weights easy examples ──
    # This is the key test: easy examples should have focal/BCE ratio < 1
    # hard examples should have focal/BCE ratio > 1
    easy_logits = torch.tensor([5.0], device=device)  # easy, high confidence
    hard_logits = torch.tensor([-3.0], device=device)  # hard, low confidence
    easy_targets = torch.tensor([1.0], device=device)
    hard_targets = torch.tensor([1.0], device=device)
    
    bce_easy = F.softplus(-easy_logits) * 0.95 + F.softplus(easy_logits) * 0.05
    bce_hard = F.softplus(-hard_logits) * 0.95 + F.softplus(hard_logits) * 0.05
    
    fl_fn = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    focal_easy = fl_fn(easy_logits, easy_targets).item()
    focal_hard = fl_fn(hard_logits, hard_targets).item()
    
    ratio_easy = focal_easy / (bce_easy + 1e-8)
    ratio_hard = focal_hard / (bce_hard + 1e-8)
    
    check(ratio_easy < 0.5, f"Easy example ratio {ratio_easy:.6f} < 0.5 (should be << 1)")
    check(ratio_hard > 1.5, f"Hard example ratio {ratio_hard:.6f} > 1.5 (should be > 1)")

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