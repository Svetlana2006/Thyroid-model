"""
ROI generation for multiscale_roi_context_v1.

Strategy (per §4 and §5 of spec):
─────────────────────────────────

TRAINING (TN5000):
  Ground-truth VOC bounding boxes from TN5000 XML annotations are used directly.
  The box is available for every training image.

INFERENCE on any dataset (TN5000 test / Diveshzz / Thyroid Pretraining):
  No ground-truth boxes are required or used.
  Automatic ROI is generated via GradCAM from the shared backbone's last stage
  (layers.3, 768-ch), thresholded and converted to a bounding box.
  This is computed lazily on the fly at evaluation time.

Dataset annotation status:
  TN5000:                   ✅ Ground-truth VOC XML boxes  (xmin,ymin,xmax,ymax)
  AUITD:                    ❌ No bounding boxes — class labels only
  Diveshzz:                 ❌ No bounding boxes — class labels only
  Thyroid for Pretraining:  ❌ No bounding boxes — class labels only

ROI crop parameters (recorded for reproducibility):
  TIGHT_PAD_RATIO   = 0.10  (10% of box width/height added as padding)
  EXPANDED_SCALE    = 2.00  (box enlarged by 2x in each dimension, centered)
  MIN_CROP_SIZE     = 16    (minimum crop dimension in pixels, safety guard)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# ── ROI parameters ────────────────────────────────────────────────────────────
TIGHT_PAD_RATIO = 0.10    # §3: ~10% padding around ground-truth box
EXPANDED_SCALE  = 2.00    # §3: 2x enlarged box for expanded ROI
MIN_CROP_SIZE   = 16      # safety guard: reject degenerate crops


def compute_tight_roi(
    xmin: int, ymin: int, xmax: int, ymax: int,
    img_w: int, img_h: int,
    pad_ratio: float = TIGHT_PAD_RATIO,
) -> Tuple[int, int, int, int]:
    """
    Tight ROI: ground-truth box + small fixed contextual padding, clamped to image.

    Returns:
        (crop_xmin, crop_ymin, crop_xmax, crop_ymax) in pixel coords
    """
    bw = xmax - xmin
    bh = ymax - ymin
    pad_x = max(1, int(bw * pad_ratio))
    pad_y = max(1, int(bh * pad_ratio))
    cx0 = max(0, xmin - pad_x)
    cy0 = max(0, ymin - pad_y)
    cx1 = min(img_w, xmax + pad_x)
    cy1 = min(img_h, ymax + pad_y)
    return cx0, cy0, cx1, cy1


def compute_expanded_roi(
    xmin: int, ymin: int, xmax: int, ymax: int,
    img_w: int, img_h: int,
    scale: float = EXPANDED_SCALE,
) -> Tuple[int, int, int, int]:
    """
    Expanded ROI: box center stays fixed, width/height multiplied by `scale`,
    clamped to image boundaries. No aspect-ratio distortion.

    Returns:
        (crop_xmin, crop_ymin, crop_xmax, crop_ymax) in pixel coords
    """
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    bw = xmax - xmin
    bh = ymax - ymin
    new_hw = bw * scale / 2.0
    new_hh = bh * scale / 2.0
    ex0 = max(0, int(cx - new_hw))
    ey0 = max(0, int(cy - new_hh))
    ex1 = min(img_w, int(cx + new_hw))
    ey1 = min(img_h, int(cy + new_hh))
    return ex0, ey0, ex1, ey1


def is_valid_crop(x0: int, y0: int, x1: int, y1: int) -> bool:
    """Return True if crop dimensions are above the minimum threshold."""
    return (x1 - x0) >= MIN_CROP_SIZE and (y1 - y0) >= MIN_CROP_SIZE


def crop_and_pad(
    image: np.ndarray,
    x0: int, y0: int, x1: int, y1: int,
) -> np.ndarray:
    """
    Crop image[y0:y1, x0:x1] and return as numpy HWC array.
    Falls back to full image if crop is degenerate.
    """
    if not is_valid_crop(x0, y0, x1, y1):
        return image   # fallback: full image
    return image[y0:y1, x0:x1]


# ── Automatic ROI via GradCAM (for datasets without ground-truth boxes) ───────

class AutoROIExtractor:
    """
    Automatic ROI proposal using GradCAM on the shared backbone's last stage
    (layers.3). Used at inference time for all datasets.

    The model must be in eval mode when calling this.

    Usage:
        extractor = AutoROIExtractor(model)
        box = extractor.get_box(image_tensor)  # (1,3,224,224) normalized tensor
        # box = (x0, y0, x1, y1) in 224x224 coords, or None if degenerate
    """

    def __init__(self, model, cam_threshold: float = 0.4):
        """
        Args:
            model:         MultiScaleROIModel instance
            cam_threshold: fraction of max activation used to binarize CAM (0-1)
        """
        self.model = model
        self.cam_threshold = cam_threshold
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hook_handles = []

    def _register(self):
        # Hook onto layers.3 of the shared backbone
        target = dict(self.model.backbone.named_modules())["layers.3"]

        def fwd(mod, inp, out):
            self._activations = out.detach()   # (1, H, W, C)

        def bwd(mod, gin, gout):
            self._gradients = gout[0].detach() # (1, H, W, C)

        self._hook_handles.append(target.register_forward_hook(fwd))
        self._hook_handles.append(target.register_full_backward_hook(bwd))

    def _remove(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    @torch.no_grad()
    def _cam_no_grad(self, x: torch.Tensor) -> np.ndarray:
        """GradCAM-free approximation: use raw activation magnitudes as saliency."""
        # Register forward hook only
        acts = {}

        def fwd(mod, inp, out):
            acts["feat"] = out.detach()

        target = dict(self.model.backbone.named_modules())["layers.3"]
        h = target.register_forward_hook(fwd)
        # Run full image through backbone
        _ = self.model.backbone(x)
        h.remove()

        feat = acts["feat"]    # (1, H, W, C)
        # Mean over channel dim -> spatial saliency map
        cam = feat.squeeze(0).mean(dim=-1).cpu().numpy()  # (H, W)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def get_box(
        self, x: torch.Tensor, orig_w: int = 224, orig_h: int = 224
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute automatic bounding box from CAM saliency.

        Args:
            x:      (1, 3, 224, 224) normalised tensor
            orig_w: original image width (for rescaling cam coords)
            orig_h: original image height
        Returns:
            (x0, y0, x1, y1) in original image coordinates, or None
        """
        cam = self._cam_no_grad(x)   # (H_cam, W_cam), values in [0,1]
        H_cam, W_cam = cam.shape

        # Threshold
        thresh = self.cam_threshold
        binary = (cam >= thresh).astype(np.uint8)

        # Find bounding box of thresholded region
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if not rows.any() or not cols.any():
            return None   # degenerate: no saliency above threshold

        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        # Rescale to original image coordinates
        x0 = int(c0 / W_cam * orig_w)
        y0 = int(r0 / H_cam * orig_h)
        x1 = int((c1 + 1) / W_cam * orig_w)
        y1 = int((r1 + 1) / H_cam * orig_h)

        if not is_valid_crop(x0, y0, x1, y1):
            return None

        return x0, y0, x1, y1
