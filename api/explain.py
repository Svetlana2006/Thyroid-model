"""Grad-CAM++ / Attention Rollout explainability, returns PNG bytes for the API."""

from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.interpretability import (
    GradCAMPlusPlus,
    SwinAttentionRollout,
    get_efficientnet_target_layer,
    get_resnet50_target_layer,
)

# Architecture → explainability method mapping
_METHOD_MAP = {
    "resnet50": "gradcam++",
    "efficientnet_b3": "gradcam++",
    "swin_tiny": "attention_rollout",
}


def get_method_for_arch(arch: str) -> str:
    return _METHOD_MAP.get(arch, "gradcam++")


def generate_heatmap(
    model: torch.nn.Module,
    arch: str,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
) -> Optional[bytes]:
    """Generate an overlay heatmap as PNG bytes.

    Args:
        model: loaded model in eval mode
        arch: architecture name
        image_tensor: (1, C, H, W) preprocessed tensor
        original_image: (H, W, 3) uint8 numpy array (original image resized to match)

    Returns:
        PNG bytes of the overlay image, or None on failure
    """
    try:
        method = get_method_for_arch(arch)

        if method == "gradcam++":
            if arch == "resnet50":
                target_layer = get_resnet50_target_layer(model)
            elif arch == "efficientnet_b3":
                target_layer = get_efficientnet_target_layer(model)
            else:
                return None

            cam = GradCAMPlusPlus(model, target_layer)
            heatmap = cam.generate(image_tensor.clone().requires_grad_(True))
            overlay = cam.overlay_on_image(original_image, heatmap, alpha=0.4)

        elif method == "attention_rollout":
            rollout = SwinAttentionRollout(model)
            heatmap = rollout.generate(image_tensor)
            rollout.remove_hooks()

            # Resize heatmap and overlay
            h, w = original_image.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = ((1 - 0.4) * original_image + 0.4 * heatmap_color).astype(np.uint8)
        else:
            return None

        # Convert overlay to PNG bytes
        pil_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        print(f"[explain] Heatmap generation failed for {arch}: {e}")
        return None
