import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kagglehub
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import build_model
from src.transforms import get_val_transforms
from src.metrics import compute_metrics

class DiveshDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        
        # Traverse dataset directory
        dataset_dir = os.path.join(data_root, "Thyroid Data")
        
        # Benign = 0
        class_0_dir = os.path.join(dataset_dir, "0")
        if os.path.exists(class_0_dir):
            for img_file in glob.glob(os.path.join(class_0_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        "img_path": img_file,
                        "label": 0
                    })
                    
        # Malignant = 1
        class_1_dir = os.path.join(dataset_dir, "1")
        if os.path.exists(class_1_dir):
            for img_file in glob.glob(os.path.join(class_1_dir, "*.*")):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        "img_path": img_file,
                        "label": 1
                    })
                    
        print(f"Loaded {len(self.samples)} valid images from dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        image = np.array(image)
        
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        return image, torch.tensor(sample["label"], dtype=torch.float32)

def evaluate_models_on_dataset(dataset, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    checkpoint_dir = "outputs/checkpoints"
    archs = ["resnet50", "efficientnet_b3", "swin_tiny"]
    
    all_test_logits = {}
    labels = []
    
    print("Collecting labels...")
    for _, batch_labels in loader:
        labels.append(batch_labels.numpy())
    labels = np.concatenate(labels)
    
    for arch in archs:
        for seed in range(3):
            ckpt_path = os.path.join(checkpoint_dir, f"{arch}_seed{seed}_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_path} (not found)")
                continue
                
            print(f"Evaluating {arch} seed {seed} on external dataset...")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            config = checkpoint.get("config", {})
            model = build_model(arch, dropout=config.get("dropout", 0.3)).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            logits = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    outputs = model(images).squeeze(-1)
                    logits.append(outputs.cpu().numpy())
                    
            all_test_logits[f"{arch}_seed{seed}"] = np.concatenate(logits)
            
    # Simple ensemble (soft voting)
    all_logits_list = list(all_test_logits.values())
    if not all_logits_list:
        print("No models evaluated.")
        return
        
    ensemble_logits = np.mean(all_logits_list, axis=0)
    
    # Calculate metrics
    results = {}
    results["Ensemble"] = compute_metrics(ensemble_logits, labels)
    
    for model_name, model_logits in all_test_logits.items():
        results[model_name] = compute_metrics(model_logits, labels)
        
    print("\nExternal Validation Results:")
    for name, metric in results.items():
        print(f"\n{name}:")
        print(f"  AUC: {metric['auc']:.4f}")
        print(f"  Accuracy: {metric['accuracy']:.4f}")
        print(f"  F1 Score: {metric['f1']:.4f}")
        print(f"  Sensitivity: {metric['sensitivity']:.4f}")
        print(f"  Specificity: {metric['specificity']:.4f}")
        
    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/diveshzz_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    print("Downloading dataset...")
    data_root = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')
    print(f"Dataset path: {data_root}")
    
    dataset = DiveshDataset(data_root, transform=get_val_transforms())
    evaluate_models_on_dataset(dataset)
