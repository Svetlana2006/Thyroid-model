"""Run a saved thyroid classifier checkpoint on one image.

Example:
    python evaluate_one_image.py "C:\\path\\to\\ultrasound.jpg"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import cv2
from matplotlib import colormaps

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics import sigmoid
from src.models import build_model
from src.transforms import get_val_transforms


DEFAULT_CHECKPOINTS = [
    "outputs/checkpoints/swin_tiny_seed0_best.pt",
    "outputs/checkpoints/swin_tiny_seed1_best.pt",
    "outputs/checkpoints/swin_tiny_seed2_best.pt",
]


def ensemble_probability(models, image_batch):
    """Return the mean malignancy probability from all models for each image."""
    with torch.no_grad():
        probabilities = [torch.sigmoid(model(image_batch).squeeze(1)) for model in models]
    return torch.stack(probabilities).mean(dim=0)


def occlusion_sensitivity(models, image_tensor, original_probability, patch_size=32, stride=24, batch_size=16):
    """Signed probability change after masking each patch with the normalized image mean."""
    _, _, height, width = image_tensor.shape
    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))
    if y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    if x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    locations = [(y, x) for y in y_positions for x in x_positions]
    heatmap = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start:start + batch_size]
        masked = image_tensor.repeat(len(batch_locations), 1, 1, 1)
        for index, (y, x) in enumerate(batch_locations):
            masked[index, :, y:y + patch_size, x:x + patch_size] = 0.0

        occluded_probabilities = ensemble_probability(models, masked).cpu().numpy()
        # Positive = this patch supported the original malignancy probability.
        probability_drops = original_probability - occluded_probabilities
        for (y, x), drop in zip(batch_locations, probability_drops):
            heatmap[y:y + patch_size, x:x + patch_size] += drop
            coverage[y:y + patch_size, x:x + patch_size] += 1

    return heatmap / np.maximum(coverage, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one ultrasound image with a saved checkpoint.")
    parser.add_argument("image_path", help="Path to the input image (.jpg, .png, etc.)")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=None,
        help="One or more checkpoints. Default: average all three Swin-Tiny seed checkpoints.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for benign/malignant output (default: 0.5)",
    )
    parser.add_argument(
        "--occlusion-heatmap", "--heatmap",
        action="store_true",
        help="Save a signed ensemble occlusion-sensitivity heatmap.",
    )
    parser.add_argument(
        "--heatmap-output",
        default=None,
        help="Output PNG path for --occlusion-heatmap (default: outputs/heatmaps/<image>_occlusion.png)",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    checkpoint_paths = [Path(path) for path in (args.checkpoint or DEFAULT_CHECKPOINTS)]
    missing_checkpoints = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing_checkpoints:
        parser.error(f"Checkpoint(s) not found: {', '.join(missing_checkpoints)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = np.array(Image.open(image_path).convert("RGB"))
    image_tensor = get_val_transforms()(image=image)["image"].unsqueeze(0).to(device)

    probabilities = []
    model_names = []
    models = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        architecture = checkpoint_path.stem.removesuffix("_best").rsplit("_seed", 1)[0]
        model = build_model(architecture, dropout=config.get("dropout", 0.3)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logit = model(image_tensor).item()
        probabilities.append(float(sigmoid(np.array([logit]))[0]))
        model_names.append(checkpoint_path.name)
        models.append(model)

    probability = float(np.mean(probabilities))
    prediction = "Malignant" if probability >= args.threshold else "Benign"

    print(f"Device: {device}")
    print(f"Models averaged ({len(model_names)}): {', '.join(model_names)}")
    print(f"Malignancy probability: {probability:.1%}")
    print(f"Prediction at threshold {args.threshold:.2f}: {prediction}")

    if args.occlusion_heatmap:
        sensitivity = occlusion_sensitivity(models, image_tensor, probability)
        scale = max(float(np.abs(sensitivity).max()), 1e-8)
        colour_values = np.clip((sensitivity / scale + 1.0) / 2.0, 0.0, 1.0)
        heatmap = (colormaps["coolwarm"](colour_values)[..., :3] * 255).astype(np.uint8)
        # Match the inference preprocessing: resize to 256 then centre-crop 224.
        display_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)[16:240, 16:240]
        overlay = cv2.addWeighted(display_image, 0.55, heatmap, 0.45, 0)

        output_path = Path(args.heatmap_output) if args.heatmap_output else \
            Path("outputs/heatmaps") / f"{image_path.stem}_occlusion.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(output_path)
        print(f"Ensemble occlusion heatmap saved to: {output_path}")
        print("Red: masking lowered malignancy probability. Blue: masking raised it. White: little effect.")


if __name__ == "__main__":
    main()
