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

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics import sigmoid
from src.models import build_model
from src.transforms import get_val_transforms


DEFAULT_CHECKPOINTS = [
    "outputs/checkpoints/swin_tiny_seed0_best.pt",
    "outputs/checkpoints/swin_tiny_seed1_best.pt",
    "outputs/checkpoints/swin_tiny_seed2_best.pt",
]


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

    probability = float(np.mean(probabilities))
    prediction = "Malignant" if probability >= args.threshold else "Benign"

    print(f"Device: {device}")
    print(f"Models averaged ({len(model_names)}): {', '.join(model_names)}")
    print(f"Malignancy probability: {probability:.1%}")
    print(f"Prediction at threshold {args.threshold:.2f}: {prediction}")


if __name__ == "__main__":
    main()
