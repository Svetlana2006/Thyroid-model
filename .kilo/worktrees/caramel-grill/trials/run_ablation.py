import argparse
import os
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import kagglehub

from core import (
    MultiLevelSwin, TN5000TestDataset, DiveshDataset, KaggleThyroidDataset,
    calculate_metrics, TTA_SCALES_DICT, THRESHOLD
)

from sklearn.metrics import roc_auc_score

def expit(x): return 1 / (1 + np.exp(-x))

@torch.no_grad()
def get_predictions(model, loader, device, seed_idx):
    model.eval()
    from collections import defaultdict
    preds = defaultdict(lambda: {"label": None, "logits": []}) # logits: [num_tta] per image
    
    desc_str = f"Evaluating Seed {seed_idx}"
    for tensors, labels, ids in tqdm(loader, desc=desc_str, leave=False):
        tensors = tensors.to(device)
        B, num_tta, C, H, W = tensors.shape
        tensors = tensors.view(B * num_tta, C, H, W)
        
        logits = model(tensors).squeeze(-1).view(B, num_tta).cpu().float().numpy()
        labels = labels.numpy()
        
        for i in range(B):
            obj_id = ids[i]
            preds[obj_id]["label"] = int(labels[i])
            if len(preds[obj_id]["logits"]) == 0:
                preds[obj_id]["logits"] = [[] for _ in range(num_tta)]
            for s_idx in range(num_tta):
                preds[obj_id]["logits"][s_idx].append(logits[i, s_idx])
                
    final_preds = {}
    for obj_id, data in preds.items():
        final_preds[obj_id] = {
            "label": data["label"],
            "scale_logits": [np.mean(img_logits) for img_logits in data["logits"]] # Average over images if patient-level
        }
    return final_preds

def evaluate_dataset(dataset, device):
    loader = DataLoader(dataset, batch_size=8, num_workers=4 if torch.cuda.is_available() else 0)
    
    from collections import defaultdict
    all_preds = defaultdict(lambda: {"label": None, "seeds": []})
    
    for seed in range(5):
        ckpt_path = os.path.join("outputs", "final_model", f"seed{seed}", "best.pt")
        model = MultiLevelSwin(dropout=0.0).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
        preds = get_predictions(model, loader, device, seed)
        
        seed_y_true = []
        seed_y_scores = []
        for obj_id, data in preds.items():
            all_preds[obj_id]["label"] = data["label"]
            all_preds[obj_id]["seeds"].append(data["scale_logits"]) # list of scale logits for this seed
            seed_y_true.append(data["label"])
            seed_y_scores.append(np.mean(data["scale_logits"]))
            
        print(f"  Seed {seed} | TTA AUROC: {roc_auc_score(seed_y_true, seed_y_scores):.4f}")
            
    # Now aggregate over seeds and scales
    ids = sorted(list(all_preds.keys()))
    y_true = np.array([all_preds[i]["label"] for i in ids])
    
    seed_tta_logits = []
    for s_idx in range(5):
        s_logits = [np.mean(all_preds[i]["seeds"][s_idx]) for i in ids] # mean across TTA scales for this seed
        seed_tta_logits.append(np.array(s_logits))
        
    ensemble_logits = np.mean(seed_tta_logits, axis=0) # mean across seeds
    y_pred_prob = expit(ensemble_logits)
    
    metrics = calculate_metrics(y_true, y_pred_prob)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar", type=str, required=True, choices=["A1", "A2", "A3", "A4"])
    parser.add_argument("--s", type=str, required=True, choices=["S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--v", type=str, required=True, choices=["V1", "V2", "V3"])
    args = parser.parse_args()

    sv_key = f"{args.s}{args.v}"
    scales = TTA_SCALES_DICT[sv_key]
    
    print(f"Running Factorial Experiment: {args.ar} {args.s} {args.v}")
    print(f"Preprocessing: {args.ar}")
    print(f"TTA Scales: {scales}")
    print(f"Number of views: len({scales}) = {len(scales)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load paths
    tn5000_path = "data_raw/TN5000_forReview"
    divesh_path = kagglehub.dataset_download('diveshzz/thyroid-cancer-classification-ultrasound-dataset')
    pretrain_path = kagglehub.dataset_download('tingzen/thyroid-for-pretraining')
    
    datasets = {
        "Internal": TN5000TestDataset(tn5000_path, args.ar, scales),
        "Diveshzz": DiveshDataset(divesh_path, args.ar, scales),
        "ThyroidPretrain": KaggleThyroidDataset(pretrain_path, args.ar, scales)
    }
    
    results = {
        "Config": f"{args.ar}{args.s}{args.v}",
        "AR": args.ar,
        "TTA_Scales": str(scales),
        "TTA_Views": len(scales),
        "Timestamp": datetime.now().isoformat()
    }
    
    for name, ds in datasets.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_dataset(ds, device)
        for k, v in metrics.items():
            results[f"{name}_{k}"] = v
            print(f"  {k}: {v:.4f}")
            
    # Save to master CSV
    os.makedirs("trials/results", exist_ok=True)
    csv_path = "trials/results/master_results.csv"
    file_exists = os.path.isfile(csv_path)
    
    fields = [
        "Config", "AR", "TTA_Scales", "TTA_Views",
        "Internal_AUC", "Internal_Accuracy", "Internal_F1", "Internal_Sensitivity", "Internal_Specificity",
        "Diveshzz_AUC", "Diveshzz_Accuracy", "Diveshzz_F1", "Diveshzz_Sensitivity", "Diveshzz_Specificity",
        "ThyroidPretrain_AUC", "ThyroidPretrain_Accuracy", "ThyroidPretrain_F1", "ThyroidPretrain_Sensitivity", "ThyroidPretrain_Specificity",
        "Timestamp"
    ]
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
        
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
