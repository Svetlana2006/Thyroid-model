"""Run a saved thyroid classifier checkpoint on one image.

Example:
    python evaluate_one_image.py "C:\\path\\to\\ultrasound.jpg"

    # Occlusion sensitivity heatmap (slow, model-agnostic):
    python evaluate_one_image.py image.jpg --occlusion-heatmap

    # Grad-CAM++ heatmap (fast, gradient-based):
    python evaluate_one_image.py image.jpg --gradcam

    # Both at once:
    python evaluate_one_image.py image.jpg --occlusion-heatmap --gradcam
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics import sigmoid
from src.models import build_model
from src.transforms import get_val_transforms


DEFAULT_CHECKPOINTS = [
    "outputs/checkpoints/swin_tiny_seed0_best.pt",
    "outputs/checkpoints/swin_tiny_seed1_best.pt",
    "outputs/checkpoints/swin_tiny_seed2_best.pt",
]

# ── Architecture → (target attribute path, channels_last) ────────────────────
# These are the canonical Grad-CAM++ target layers for each backbone.
_GRADCAM_TARGETS = {
    "resnet50":        ("backbone.layer4",  False),  # last residual block
    "efficientnet_b3": ("backbone.conv_head", False), # 1×1 conv before global pool
    "swin_tiny":       ("backbone.norm",    True),   # LayerNorm after last stage
}


def _resolve_layer(model: nn.Module, attr_path: str) -> nn.Module:
    """Walk dot-separated attribute path to retrieve a sub-module."""
    obj = model
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def ensemble_probability(models: List[nn.Module], image_batch: torch.Tensor) -> torch.Tensor:
    """Return the mean malignancy probability from all models."""
    with torch.no_grad():
        probabilities = [torch.sigmoid(model(image_batch).squeeze(1)) for model in models]
    return torch.stack(probabilities).mean(dim=0)


# ── Occlusion sensitivity ─────────────────────────────────────────────────────

def occlusion_sensitivity(
    models: List[nn.Module],
    image_tensor: torch.Tensor,
    original_probability: float,
    patch_size: int = 32,
    stride: int = 24,
    batch_size: int = 16,
) -> np.ndarray:
    """Signed probability change after masking each patch with the normalized mean.

    Positive values = the patch was supporting the malignancy prediction.
    Negative values = the patch was suppressing it.
    """
    _, _, height, width = image_tensor.shape
    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))
    if y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    if x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    locations = [(y, x) for y in y_positions for x in x_positions]
    heatmap  = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start:start + batch_size]
        masked = image_tensor.repeat(len(batch_locations), 1, 1, 1)
        for index, (y, x) in enumerate(batch_locations):
            masked[index, :, y:y + patch_size, x:x + patch_size] = 0.0

        occluded_probs = ensemble_probability(models, masked).cpu().numpy()
        prob_drops = original_probability - occluded_probs
        for (y, x), drop in zip(batch_locations, prob_drops):
            heatmap [y:y + patch_size, x:x + patch_size] += drop
            coverage[y:y + patch_size, x:x + patch_size] += 1

    return heatmap / np.maximum(coverage, 1.0)


# ── Grad-CAM++ ────────────────────────────────────────────────────────────────

def gradcam_pp(
    model: nn.Module,
    target_layer: nn.Module,
    image_tensor: torch.Tensor,
    channels_last: bool = False,
) -> np.ndarray:
    """Compute Grad-CAM++ heatmap for a single model.

    Returns a (h, w) float32 array in [0, 1].

    Args:
        channels_last: Set True for Swin layers whose output is (B,H,W,C)
                       instead of the standard (B,C,H,W).
    """
    activations: Optional[torch.Tensor] = None
    gradients:   Optional[torch.Tensor] = None
    handles = []

    def _fwd(_module, _inp, out):
        nonlocal activations
        t = out.detach()
        if channels_last and t.dim() == 4:
            t = t.permute(0, 3, 1, 2)       # (B,H,W,C) → (B,C,H,W)
        activations = t

    def _bwd(_module, _grad_in, grad_out):
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
            return np.zeros((7, 7), dtype=np.float32)

        # Grad-CAM++ alpha weights
        g2 = gradients ** 2
        g3 = gradients ** 3
        sum_a = activations.sum(dim=(2, 3), keepdim=True)
        denom  = 2 * g2 + g3 * sum_a
        denom  = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha  = g2 / denom
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32)
    finally:
        for h in handles:
            h.remove()


def gradcam_ensemble_heatmap(
    models: List[nn.Module],
    archs: List[str],
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
) -> List[dict]:
    """Run Grad-CAM++ on each model, return list of overlay dicts.

    Each dict has keys: arch, overlay (np.ndarray RGB).
    """
    results = []
    for model, arch in zip(models, archs):
        attr_path, channels_last = _GRADCAM_TARGETS.get(arch, ("backbone", False))
        try:
            target_layer = _resolve_layer(model, attr_path)
        except AttributeError:
            print(f"  [warn] Could not resolve layer '{attr_path}' for {arch} — skipping.")
            continue

        cam = gradcam_pp(model, target_layer, image_tensor, channels_last=channels_last)

        h, w = original_image.shape[:2]
        cam_resized  = cv2.resize(cam, (w, h))
        hm_color     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm_color     = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        overlay      = ((0.6 * original_image) + (0.4 * hm_color)).astype(np.uint8)
        results.append({"arch": arch, "overlay": overlay})

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one ultrasound image with a saved checkpoint."
    )
    parser.add_argument("image_path", help="Path to the input image (.jpg, .png, etc.)")
    parser.add_argument(
        "--checkpoint", nargs="+", default=None,
        help="One or more checkpoints. Default: average all three Swin-Tiny seed checkpoints.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for benign/malignant output (default: 0.5)",
    )

    # ── Heatmap flags ──
    parser.add_argument(
        "--occlusion-heatmap", "--occlusion",
        action="store_true",
        help="Save a signed ensemble occlusion-sensitivity heatmap (slower, model-agnostic).",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Save a Grad-CAM++ heatmap per checkpoint (fast, gradient-based).",
    )

    parser.add_argument(
        "--heatmap-output", default=None,
        help=(
            "Output PNG path for the heatmap. For --gradcam with multiple checkpoints "
            "a suffix _<arch>.png is appended automatically. "
            "Default: outputs/heatmaps/<image>_occlusion.png or _gradcam_<arch>.png"
        ),
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    checkpoint_paths = [Path(p) for p in (args.checkpoint or DEFAULT_CHECKPOINTS)]
    missing = [str(p) for p in checkpoint_paths if not p.is_file()]
    if missing:
        parser.error(f"Checkpoint(s) not found: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image        = np.array(Image.open(image_path).convert("RGB"))
    image_tensor = get_val_transforms()(image=image)["image"].unsqueeze(0).to(device)

    # ── Load models ──
    probabilities: List[float] = []
    model_names:   List[str]   = []
    models:        List[nn.Module] = []
    archs:         List[str]   = []

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg  = ckpt["config"]
        arch = ckpt_path.stem.removesuffix("_best").rsplit("_seed", 1)[0]
        model = build_model(arch, dropout=cfg.get("dropout", 0.3)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logit = model(image_tensor).item()
        probabilities.append(float(sigmoid(np.array([logit]))[0]))
        model_names.append(ckpt_path.name)
        models.append(model)
        archs.append(arch)

    probability = float(np.mean(probabilities))
    prediction  = "Malignant" if probability >= args.threshold else "Benign"

    print(f"Device  : {device}")
    print(f"Models  : {', '.join(model_names)}")
    print(f"Probability (malignant) : {probability:.1%}")
    print(f"Prediction at {args.threshold:.2f}: {prediction}")
    for name, prob in zip(model_names, probabilities):
        print(f"  {name}: {prob:.1%}")

    # ── Preprocessing for display (resize-256 → centre-crop-224) ──
    display_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)[16:240, 16:240]
    out_dir = Path("outputs/heatmaps")

    # ── Occlusion heatmap ──────────────────────────────────────────────────────
    if args.occlusion_heatmap:
        print("\nComputing occlusion-sensitivity heatmap…")
        sensitivity   = occlusion_sensitivity(models, image_tensor, probability)
        scale         = max(float(np.abs(sensitivity).max()), 1e-8)
        colour_values = np.clip((sensitivity / scale + 1.0) / 2.0, 0.0, 1.0)
        heatmap       = (colormaps["coolwarm"](colour_values)[..., :3] * 255).astype(np.uint8)
        overlay       = cv2.addWeighted(display_image, 0.55, heatmap, 0.45, 0)

        out_path = (
            Path(args.heatmap_output)
            if args.heatmap_output
            else out_dir / f"{image_path.stem}_occlusion.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out_path)
        print(f"Occlusion heatmap → {out_path}")
        print("  Red = masking lowered malignancy probability (important region).")
        print("  Blue = masking raised it.  White = little effect.")

    # ── Grad-CAM++ heatmap ─────────────────────────────────────────────────────
    if args.gradcam:
        print("\nComputing Grad-CAM++ heatmaps…")
        gc_results = gradcam_ensemble_heatmap(models, archs, image_tensor, display_image)

        for item in gc_results:
            arch_name = item["arch"]
            if args.heatmap_output and len(gc_results) == 1:
                out_path = Path(args.heatmap_output)
            else:
                out_path = out_dir / f"{image_path.stem}_gradcam_{arch_name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(item["overlay"]).save(out_path)
            print(f"  Grad-CAM++ [{arch_name}] → {out_path}")


if __name__ == "__main__":
    main()
