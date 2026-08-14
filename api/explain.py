"""Grad-CAM++ explainability for all architectures, returns PNG bytes.

Fixes over v1:
- Swin-Tiny: replaced broken attention rollout (timm doesn't expose attn weights)
  with Grad-CAM++ on backbone.norm, handling (B,H,W,C) → (B,C,H,W) permutation.
- EfficientNet-B3: target changed from blocks[-1] (poor spatial signal) to
  conv_head (1×1 conv right before global pool — canonical Grad-CAM target).
- Hooks are now removed after every call so they don't accumulate on cached models.
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.interpretability import get_resnet50_target_layer


def _gradcam_pp(
    model: nn.Module,
    target_layer: nn.Module,
    image_tensor: torch.Tensor,
    channels_last: bool = False,
) -> np.ndarray:
    """Compute Grad-CAM++ heatmap.  Returns (h, w) float32 in [0, 1].

    Args:
        channels_last: True when the target layer outputs (B, H, W, C)
                       instead of the standard (B, C, H, W).  Required for
                       Swin Transformer layers.
    """
    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    handles: List[torch.utils.hooks.RemovableHook] = []

    def _fwd(module, inp, out):
        nonlocal activations
        t = out.detach()
        if channels_last and t.dim() == 4:
            t = t.permute(0, 3, 1, 2)          # (B,H,W,C) → (B,C,H,W)
        activations = t

    def _bwd(module, grad_in, grad_out):
        nonlocal gradients
        t = grad_out[0].detach()
        if channels_last and t.dim() == 4:
            t = t.permute(0, 3, 1, 2)
        gradients = t

    handles.append(target_layer.register_forward_hook(_fwd))
    handles.append(target_layer.register_full_backward_hook(_bwd))

    try:
        model.eval()
        inp = image_tensor.clone().requires_grad_(True)
        logit = model(inp).squeeze()
        model.zero_grad()
        logit.backward()

        if activations is None or gradients is None:
            return np.ones((7, 7), dtype=np.float32)

        # ── Grad-CAM++ weighting ──
        g2 = gradients ** 2
        g3 = gradients ** 3
        sum_a = activations.sum(dim=(2, 3), keepdim=True)
        denom = 2 * g2 + g3 * sum_a
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32)
    finally:
        # Always remove hooks so they don't accumulate on cached models
        for h in handles:
            h.remove()


def _overlay(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resize heatmap to match original and blend with JET colormap."""
    h, w = original.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * original + alpha * hm_color).astype(np.uint8)


# ── Target layer selection ────────────────────────────────────────────────────

def _get_target(model: nn.Module, arch: str) -> Tuple[nn.Module, bool]:
    """Return (target_layer, channels_last) for each architecture.

    ResNet-50:       layer4[-1]   — standard last conv block,  channels-first
    EfficientNet-B3: conv_head    — 1×1 conv before global pool, channels-first
    Swin-Tiny:       norm         — LayerNorm after last stage,  channels-last
    """
    if arch == "resnet50":
        return get_resnet50_target_layer(model), False
    if arch == "efficientnet_b3":
        return model.backbone.conv_head, False
    if arch == "swin_tiny":
        return model.backbone.norm, True
    raise ValueError(f"Unknown arch: {arch}")


def get_method_for_arch(_arch: str) -> str:
    """All architectures now use Grad-CAM++."""
    return "gradcam++"


def generate_heatmap(
    model: nn.Module,
    arch: str,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
) -> Optional[bytes]:
    """Generate an overlay heatmap as PNG bytes, or None on failure."""
    try:
        target_layer, channels_last = _get_target(model, arch)
        heatmap = _gradcam_pp(model, target_layer, image_tensor, channels_last=channels_last)
        overlay_img = _overlay(original_image, heatmap)

        buf = io.BytesIO()
        Image.fromarray(overlay_img).save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[explain] Heatmap generation failed for {arch}: {e}")
        return None
