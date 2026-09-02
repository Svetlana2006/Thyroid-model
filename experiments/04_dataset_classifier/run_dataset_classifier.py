"""
Experiment 04: Dataset classifier.

Tests whether a classifier can distinguish TN5000 vs AUITD images from:
  A) Simple image statistics (width, height, mean intensity, etc.)
  B) Frozen deep features from trained backbones

If dataset identity is highly predictable, this is evidence of domain shift
in the representation space.

Usage:
    python experiments/04_dataset_classifier/run_dataset_classifier.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUTPUT_DIR = Path("experiments/04_dataset_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image_stats():
    """Load pre-computed stats from domain shift experiment."""
    import csv
    stats_path = Path("experiments/03_domain_shift/image_stats.csv")
    if not stats_path.exists():
        raise FileNotFoundError(
            "Run experiments/03_domain_shift/run_domain_shift.py first")

    with open(stats_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    features = ["width", "height", "aspect_ratio_hw", "mean_intensity",
                 "std_intensity", "entropy", "edge_density", "hf_energy"]

    X = []
    y = []
    dataset_names = []
    for row in rows:
        ds = row["dataset"]
        if ds in ("TN5000", "AUITD"):  # Only compare training datasets
            feat_vec = [float(row[f]) for f in features]
            X.append(feat_vec)
            y.append(0 if ds == "TN5000" else 1)
            dataset_names.append(ds)

    return np.array(X), np.array(y), features, dataset_names


def run_classification(X, y, feature_names, classifier_name, clf):
    """Run 5-fold CV classification and report results."""
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accs = []
    fold_f1s = []
    all_preds = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        all_preds[test_idx] = preds

        fold_accs.append(accuracy_score(y_test, preds))
        fold_f1s.append(f1_score(y_test, preds, average="macro"))

    result = {
        "classifier": classifier_name,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
        "per_fold_accuracy": [float(a) for a in fold_accs],
        "chance_level": float(max(np.mean(y == 0), np.mean(y == 1))),
    }

    # Feature importances for RF
    if hasattr(clf, "feature_importances_"):
        scaler.fit(X)
        clf.fit(scaler.transform(X), y)
        importances = clf.feature_importances_
        result["feature_importances"] = {
            fn: float(imp) for fn, imp in
            sorted(zip(feature_names, importances), key=lambda x: -x[1])
        }

    return result


def main():
    print("=" * 60)
    print("STEP 6: Dataset Classification Experiment")
    print("=" * 60)

    # Part A: Simple image statistics
    print("\nPart A: Classifying datasets from image statistics...")
    X, y, features, _ = load_image_stats()
    print(f"  Samples: {len(y)} (TN5000={np.sum(y==0)}, AUITD={np.sum(y==1)})")
    print(f"  Features: {features}")

    results = {}

    # Logistic Regression
    lr_result = run_classification(
        X, y, features, "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42)
    )
    results["image_stats_logistic_regression"] = lr_result
    print(f"  LR: accuracy={lr_result['mean_accuracy']:.3f} +/- {lr_result['std_accuracy']:.3f}")

    # Random Forest
    rf_result = run_classification(
        X, y, features, "RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    results["image_stats_random_forest"] = rf_result
    print(f"  RF: accuracy={rf_result['mean_accuracy']:.3f} +/- {rf_result['std_accuracy']:.3f}")

    if "feature_importances" in rf_result:
        print("  Feature importances (RF):")
        for fn, imp in rf_result["feature_importances"].items():
            print(f"    {fn}: {imp:.3f}")

    # Part B: Deep features (if checkpoints exist)
    print("\nPart B: Classifying datasets from frozen deep features...")
    try:
        import torch
        from src.models import build_model
        from src.transforms import get_val_transforms
        from PIL import Image as PILImage

        device = torch.device("cpu")  # Use CPU for this experiment
        transform = get_val_transforms()

        archs = ["resnet50", "efficientnet_b3", "swin_tiny"]
        for arch in archs:
            ckpt_path = f"outputs/checkpoints/{arch}_seed0_best.pt"
            if not os.path.exists(ckpt_path):
                print(f"  Skipping {arch} (no checkpoint)")
                continue

            print(f"  Extracting features from {arch}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            config = ckpt.get("config", {})
            model = build_model(arch, dropout=config.get("dropout", 0.3)).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # Get backbone features (before head)
            features_list = []
            labels_list = []

            # Sample from each dataset
            tn5000_paths = []
            for fn in sorted(os.listdir("data_raw/TN5000_forReview/JPEGImages"))[:500]:
                if fn.endswith(".jpg"):
                    tn5000_paths.append(("data_raw/TN5000_forReview/JPEGImages/" + fn, 0))

            auitd_paths = []
            for root, _, files in os.walk("data_raw/auitd_dataset"):
                for fn in files:
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        auitd_paths.append((os.path.join(root, fn), 1))
            np.random.seed(42)
            if len(auitd_paths) > 500:
                indices = np.random.choice(len(auitd_paths), 500, replace=False)
                auitd_paths = [auitd_paths[i] for i in indices]

            all_paths = tn5000_paths + auitd_paths
            np.random.shuffle(all_paths)

            with torch.no_grad():
                for img_path, label in all_paths:
                    try:
                        img = np.array(PILImage.open(img_path).convert("RGB"))
                        tensor = transform(image=img)["image"].unsqueeze(0).to(device)
                        feat = model.backbone(tensor)
                        if feat.dim() > 2:
                            feat = feat.mean(dim=[2, 3])  # Global average pool if spatial
                        feat = feat.squeeze().flatten().numpy()
                        features_list.append(feat)
                        labels_list.append(label)
                    except Exception:
                        pass

            if len(features_list) < 100:
                print(f"  Too few samples for {arch}, skipping")
                continue

            X_deep = np.array(features_list)
            y_deep = np.array(labels_list)
            print(f"    Extracted {len(X_deep)} feature vectors (dim={X_deep.shape[1]})")

            # Classify
            lr_deep = run_classification(
                X_deep, y_deep,
                [f"feat_{i}" for i in range(X_deep.shape[1])],
                f"{arch}_LogisticRegression",
                LogisticRegression(max_iter=1000, random_state=42, C=0.1)
            )
            results[f"deep_{arch}_logistic_regression"] = lr_deep
            print(f"    LR: accuracy={lr_deep['mean_accuracy']:.3f}")

    except Exception as e:
        print(f"  Deep feature extraction failed: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    with open(OUTPUT_DIR / "dataset_classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("DATASET SEPARABILITY SUMMARY")
    print(f"{'='*60}")
    for key, res in results.items():
        print(f"  {key}: accuracy={res['mean_accuracy']:.3f}, "
              f"chance={res['chance_level']:.3f}, "
              f"above_chance={res['mean_accuracy']-res['chance_level']:.3f}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
