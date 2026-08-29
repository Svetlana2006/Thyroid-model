import os
import glob
import json
import numpy as np
import xml.etree.ElementTree as ET
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

class DDTIDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        
        # DDTI Dataset has XMLs and JPEGs. We recursively find XMLs.
        xml_files = glob.glob(os.path.join(data_root, "**", "*.xml"), recursive=True)
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                tirads_elem = root.find("tirads")
                if tirads_elem is None or tirads_elem.text is None:
                    continue
                tirads = tirads_elem.text.strip().lower()
                
                # Map TIRADS to binary classification
                if tirads in ['2', '3']:
                    label = 0
                elif tirads in ['4a', '4b', '4c', '5']:
                    label = 1
                else:
                    continue
                
                # Get associated images (DDTI convention: caseID_1.jpg, caseID_2.jpg)
                case_dir = os.path.dirname(xml_file)
                case_name = os.path.splitext(os.path.basename(xml_file))[0]
                img_files = glob.glob(os.path.join(case_dir, f"{case_name}_*.jpg"))
                
                for img_file in img_files:
                    self.samples.append({
                        "img_path": img_file,
                        "label": label
                    })
            except Exception as e:
                pass
                
        print(f"Loaded {len(self.samples)} valid images from DDTI dataset.")

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

def evaluate_models_on_ddti(dataset, batch_size=32):
    if len(dataset) == 0:
        print("No valid images found in the dataset.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    checkpoint_dir = "outputs/checkpoints"
    archs = ["resnet50", "efficientnet_b3", "swin_tiny"]
    
    all_test_logits = {}
    labels = []
    
    # Collect all labels
    for _, batch_labels in loader:
        labels.append(batch_labels.numpy())
    labels = np.concatenate(labels)
    
    for arch in archs:
        for seed in range(3):
            ckpt_path = os.path.join(checkpoint_dir, f"{arch}_seed{seed}_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_path} (not found)")
                continue
                
            print(f"Evaluating {arch} seed {seed} on DDTI...")
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
        
    print("\nExternal Validation Results (DDTI Dataset):")
    for name, metric in results.items():
        print(f"\n{name}:")
        print(f"  AUC: {metric['auc']:.4f}")
        print(f"  Accuracy: {metric['accuracy']:.4f}")
        print(f"  F1 Score: {metric['f1']:.4f}")
        print(f"  Sensitivity: {metric['sensitivity']:.4f}")
        print(f"  Specificity: {metric['specificity']:.4f}")
        
    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/ddti_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    print("Downloading DDTI dataset...")
    # We do NOT use KaggleDatasetAdapter.PANDAS here because this dataset 
    # uses XML files and JPGs, not standard CSV files. 
    data_root = kagglehub.dataset_download('dasmehdixtr/ddti-thyroid-ultrasound-images')
    print(f"Dataset path: {data_root}")
    
    dataset = DDTIDataset(data_root, transform=get_val_transforms())
    evaluate_models_on_ddti(dataset)
