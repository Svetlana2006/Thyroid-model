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
import numpy as np


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

        # 1. Apply the same manual label-smoothing transformation used by the main-model training path.
        #    This is NOT done by BCEWithLogitsLoss internally.
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
    # This is the key test: easy examples should have focal/BCE ratio << 1
    # hard examples should have focal/BCE ratio close to 1 (not down-weighted)
    easy_logits = torch.tensor([5.0], device=device)  # easy, high confidence
    hard_logits = torch.tensor([-3.0], device=device)  # hard, low confidence
    easy_targets = torch.tensor([1.0], device=device)
    hard_targets = torch.tensor([1.0], device=device)
    
    # Calculate BCE using the same formula as in the loss function
    def _compute_bce(logits, targets, eps=0.0):
        targets_smooth = targets * (1.0 - eps) + 0.5 * eps
        return float(F.softplus(-logits) * targets_smooth + F.softplus(logits) * (1.0 - targets_smooth))
    
    bce_easy = _compute_bce(easy_logits, easy_targets, 0.0)
    bce_hard = _compute_bce(hard_logits, hard_targets, 0.0)
    
    fl_fn = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    focal_easy = fl_fn(easy_logits, easy_targets).item()
    focal_hard = fl_fn(hard_logits, hard_targets).item()
    
    ratio_easy = focal_easy / (bce_easy + 1e-8)
    ratio_hard = focal_hard / (bce_hard + 1e-8)
    check(ratio_easy < 0.1, f"Easy example ratio {ratio_easy:.6f} < 0.1 (should be << 1)")
    check(ratio_hard > 0.8 and ratio_hard < 1.2, f"Hard example ratio {ratio_hard:.6f} close to 1 (should be ~1)")

    # ── Test 14: label smoothing has no effect when eps=0 ───────
    loss_fn_e0 = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.0)
    loss_fn_e05 = BinaryFocalLoss(gamma=2.0, pos_weight=1.0, label_smooth_eps=0.05)
    logits = torch.tensor([10.0], device=device)
    targets = torch.tensor([1.0], device=device)
    l0 = loss_fn_e0(logits, targets).item()
    l05 = loss_fn_e05(logits, targets).item()
    check(l05 > l0, f"label_smooth_eps=0.05 > 0: smooth={l05:.6f} > no_smooth={l0:.6f}")

    # ── Test 15: gamma=0 EQUIVALENCE with BCEWithLogitsLoss ───────────
    # This is the CRITICAL test: FocalLoss(gamma=0, pos_weight=w, eps=sm)
    # must numerically equal BCEWithLogitsLoss(pos_weight=w) with manually smoothed targets
    print("\n  [GAMMA=0 EQUIVALENCE TEST]")
    pos_w = 2.5
    eps = 0.05
    fl_gamma0 = BinaryFocalLoss(gamma=0.0, pos_weight=pos_w, label_smooth_eps=eps)
    
    test_cases = [
        ("positive easy", torch.tensor([5.0]), torch.tensor([1.0])),
        ("positive hard", torch.tensor([-3.0]), torch.tensor([1.0])),
        ("negative easy", torch.tensor([-5.0]), torch.tensor([0.0])),
        ("negative hard", torch.tensor([3.0]), torch.tensor([0.0])),
        ("logit zero target pos", torch.tensor([0.0]), torch.tensor([1.0])),
        ("logit zero target neg", torch.tensor([0.0]), torch.tensor([0.0])),
        ("large positive", torch.tensor([20.0]), torch.tensor([1.0])),
        ("large negative", torch.tensor([-20.0]), torch.tensor([1.0])),
        ("small pos logit", torch.tensor([0.01]), torch.tensor([1.0])),
        ("small neg logit", torch.tensor([-0.01]), torch.tensor([0.0])),
    ]
    
    all_equiv = True
    for name, logit, target in test_cases:
        fl_loss = fl_gamma0(logit, target).item()
        # Reference: BCEWithLogitsLoss with same pos_weight
        # Labels are manually smoothed (same as src/trainer.py), then passed to BCEWithLogitsLoss
        # pos_weight in BCEWithLogitsLoss applies to positive class term
        targets_smooth = target * (1.0 - eps) + 0.5 * eps
        pw_tensor = torch.tensor([pos_w])
        bce_loss = F.binary_cross_entropy_with_logits(logit, targets_smooth, pos_weight=pw_tensor).item()
        abs_err = abs(fl_loss - bce_loss)
        rel_err = abs_err / (abs(bce_loss) + 1e-8)
        passed = abs_err < 1e-5
        all_equiv = all_equiv and passed
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: FL={fl_loss:.8f}, BCE={bce_loss:.8f}, "
              f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}")
    
    check(all_equiv, f"All gamma=0 equivalence tests passed: {all_equiv}")
    
    # ── Test 16: gamma=0 equivalence with random batch ────────────
    torch.manual_seed(42)
    np.random.seed(42)
    rand_logits = torch.randn(32, device=device) * 4 - 2
    rand_targets = torch.randint(0, 2, (32,), device=device, dtype=torch.float32)
    fl_batch = fl_gamma0(rand_logits, rand_targets).item()
    ts = rand_targets * (1.0 - eps) + 0.5 * eps
    pw_t = torch.tensor([pos_w])
    bce_batch = F.binary_cross_entropy_with_logits(rand_logits, ts, pos_weight=pw_t).item()
    batch_abs_err = abs(fl_batch - bce_batch)
    batch_passed = batch_abs_err < 1e-5
    check(batch_passed, f"Random batch gamma=0 equivalence: abs_err={batch_abs_err:.2e}")
    
    # ── Test 17: gamma=0 GRADIENT EQUIVALENCE ────────────────────
    # For gamma=0, gradients should match BCEWithLogitsLoss gradients exactly
    print("\n  [GAMMA=0 GRADIENT EQUIVALENCE TEST]")
    grad_all_pass = True
    grad_test_cases = [
        ("mixed batch", torch.tensor([2.0, -1.5, 0.5, -3.0, 4.0], device=device, requires_grad=True),
         torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0], device=device)),
        ("confident positive", torch.tensor([5.0], device=device, requires_grad=True),
         torch.tensor([1.0], device=device)),
        ("hard positive", torch.tensor([-2.0], device=device, requires_grad=True),
         torch.tensor([1.0], device=device)),
        ("confident negative", torch.tensor([-5.0], device=device, requires_grad=True),
         torch.tensor([0.0], device=device)),
        ("hard negative", torch.tensor([2.0], device=device, requires_grad=True),
         torch.tensor([0.0], device=device)),
    ]
    
    for name, logits_fl, targets_fl in grad_test_cases:
        # Focal loss with gamma=0
        logits_fl = logits_fl.clone().detach().requires_grad_(True)
        fl_loss = fl_gamma0(logits_fl, targets_fl)
        fl_loss.backward()
        fl_grad = logits_fl.grad.clone()
        
        # BCEWithLogitsLoss with same settings
        logits_bce = logits_fl.clone().detach().requires_grad_(True)
        targets_smooth = targets_fl * (1.0 - eps) + 0.5 * eps
        pw_tensor = torch.tensor([pos_w], device=device)
        bce_loss = F.binary_cross_entropy_with_logits(logits_bce, targets_smooth, pos_weight=pw_tensor)
        bce_loss.backward()
        bce_grad = logits_bce.grad.clone()
        
        # Compare gradients
        grad_diff = torch.abs(fl_grad - bce_grad).max().item()
        grad_passed = grad_diff < 1e-6
        grad_all_pass = grad_all_pass and grad_passed
        status = "PASS" if grad_passed else "FAIL"
        print(f"    [{status}] {name}: max_grad_diff={grad_diff:.2e}")
    
    # Test with different pos_weight values
    for pw in [0.5, 1.0, 2.0, 5.0]:
        fl_gamma0_pw = BinaryFocalLoss(gamma=0.0, pos_weight=pw, label_smooth_eps=eps)
        logits_fl = torch.tensor([1.0, -1.0, 0.5, -0.5], device=device, requires_grad=True)
        targets_fl = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
        
        logits_fl = logits_fl.clone().detach().requires_grad_(True)
        fl_loss = fl_gamma0_pw(logits_fl, targets_fl)
        fl_loss.backward()
        fl_grad = logits_fl.grad.clone()
        
        logits_bce = logits_fl.clone().detach().requires_grad_(True)
        targets_smooth = targets_fl * (1.0 - eps) + 0.5 * eps
        pw_tensor = torch.tensor([pw], device=device)
        bce_loss = F.binary_cross_entropy_with_logits(logits_bce, targets_smooth, pos_weight=pw_tensor)
        bce_loss.backward()
        bce_grad = logits_bce.grad.clone()
        
        grad_diff = torch.abs(fl_grad - bce_grad).max().item()
        grad_passed = grad_diff < 1e-6
        grad_all_pass = grad_all_pass and grad_passed
        status = "PASS" if grad_passed else "FAIL"
        print(f"    [{status}] pos_weight={pw}: max_grad_diff={grad_diff:.2e}")
    
    check(grad_all_pass, f"All gamma=0 gradient equivalence tests passed: {grad_all_pass}")
    
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    ok = test_focal_loss()
    raise SystemExit(0 if ok else 1)