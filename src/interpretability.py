"""
Grad-CAM++ and Attention Rollout interpretability for TN5000.
Per plan §8: Grad-CAM++ for ResNet/EfficientNet, attention rollout for Swin-Tiny.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Grad-CAM++ (ResNet-50 and EfficientNet-B3)
# ─────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation.
    target_layer: last conv block before global pooling.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            image_tensor: (1, C, H, W) tensor, normalised
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        logit = self.model(image_tensor).squeeze()
        self.model.zero_grad()
        logit.backward()

        # Grad-CAM++ weights
        grads = self._gradients  # (1, C, h, w)
        acts = self._activations  # (1, C, h, w)

        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom = 2 * grads_sq + grads_cu * sum_acts
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads_sq / denom

        # ReLU on weights
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay heatmap on original image (H, W, 3) uint8."""
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = (1 - alpha) * original_image + alpha * heatmap_color
        return overlay.astype(np.uint8)


def get_resnet50_target_layer(model) -> nn.Module:
    """Last conv block of ResNet-50 backbone."""
    return model._backbone_raw.layer4[-1]


def get_efficientnet_target_layer(model) -> nn.Module:
    """Last block of EfficientNet-B3."""
    blocks = list(model.backbone.blocks.children())
    return blocks[-1]


# ─────────────────────────────────────────────
# Attention Rollout (Swin-Tiny)
# ─────────────────────────────────────────────

class SwinAttentionRollout:
    """
    Attention rollout for Swin-Tiny (timm implementation).
    Averages attention weights across heads and layers,
    following Abnar & Zuidema (2020).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attention_maps = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def make_hook():
            def hook(module, input, output):
                # timm WindowAttention returns (x, attn_weights) if need_weights
                if isinstance(output, tuple) and len(output) == 2:
                    self._attention_maps.append(output[1].detach().cpu())
            return hook

        if hasattr(self.model.backbone, "layers"):
            for layer in self.model.backbone.layers:
                for block in layer.blocks:
                    if hasattr(block, "attn"):
                        h = block.attn.register_forward_hook(make_hook())
                        self._hooks.append(h)

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Returns a rollout attention map averaged over all layers/heads.
        Output shape: (H, W) normalised to [0, 1]
        """
        self._attention_maps = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_tensor)

        if not self._attention_maps:
            # Fallback: return uniform map
            return np.ones((7, 7))

        # Average across heads for each layer, then rollout
        rollout = None
        for attn in self._attention_maps:
            # attn: (batch, num_heads, seq_len, seq_len) or similar
            if attn.dim() == 4:
                avg_attn = attn.mean(dim=1)[0]  # (seq_len, seq_len)
            else:
                avg_attn = attn[0]

            # Add identity for residual connections
            I = torch.eye(avg_attn.size(-1))
            avg_attn = (avg_attn + I) / 2
            avg_attn = avg_attn / avg_attn.sum(dim=-1, keepdim=True)

            if rollout is None:
                rollout = avg_attn
            else:
                rollout = torch.matmul(avg_attn, rollout)

        # Take relevance from cls token (or mean over seq)
        rollout_np = rollout.numpy()
        if rollout_np.shape[0] > 1:
            mask = rollout_np[0, 1:]  # from [CLS] to patches
        else:
            mask = rollout_np.mean(axis=0)

        size = int(np.sqrt(len(mask)))
        if size * size == len(mask):
            mask = mask.reshape(size, size)
        else:
            mask = mask.reshape(-1, 1)

        if mask.max() > mask.min():
            mask = (mask - mask.min()) / (mask.max() - mask.min())

        return mask.astype(np.float32)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────
# Localization accuracy
# ─────────────────────────────────────────────

def heatmap_localization_accuracy(
    heatmap: np.ndarray,
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    threshold_ratio: float = 0.5,
) -> float:
    """
    Compute % of heatmap 'hot' pixels (above threshold) that fall inside
    the nodule bounding box.

    Args:
        heatmap: (H, W) normalised [0, 1] heatmap at original image resolution
        bbox: (xmin, ymin, xmax, ymax) in pixel coords
        image_size: (width, height) of original image
        threshold_ratio: fraction of max heatmap value to use as threshold

    Returns:
        Fraction of hot pixels inside the nodule bbox.
    """
    w, h = image_size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    threshold = threshold_ratio * heatmap_resized.max()
    hot_mask = heatmap_resized > threshold

    xmin, ymin, xmax, ymax = bbox
    inside_mask = np.zeros((h, w), dtype=bool)
    inside_mask[ymin:ymax, xmin:xmax] = True

    hot_count = hot_mask.sum()
    if hot_count == 0:
        return 0.0
    inside_hot = (hot_mask & inside_mask).sum()
    return float(inside_hot / hot_count)
