"""Independent, restartable evaluation for Stage 2 checkpoints.

Run this in a fresh GPU session after ``train_stage2.py --config ...``. It
loads exactly one seed model at a time and releases CUDA memory before the next
seed. Each dataset can be evaluated in an independent GPU session.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import kagglehub
import numpy as np
import torch

from train_stage2 import (CONFIGS, OUT, SEEDS, EvalDataset, divesh_samples,
                          metrics, pretrain_samples, seed_logits,
                          tn5000_test_samples, update_status, write_json)

DATASETS = ("Internal", "Diveshzz", "ThyroidPretrain")


def source_for(name: str):
    if name == "Internal":
        return tn5000_test_samples(), False
    if name == "Diveshzz":
        return divesh_samples(kagglehub.dataset_download("diveshzz/thyroid-cancer-classification-ultrasound-dataset")), False
    if name == "ThyroidPretrain":
        return pretrain_samples(kagglehub.dataset_download("tingzen/thyroid-for-pretraining")), True
    raise ValueError(name)


def patient_average(ids, labels, values, scales=None):
    groups = defaultdict(lambda: {"label": None, "values": [], "scales": []})
    for item_id, label, value in zip(ids, labels, values):
        groups[item_id]["label"] = label
        groups[item_id]["values"].append(value)
    if scales is not None:
        for item_id, scale_values in zip(ids, scales):
            groups[item_id]["scales"].append(scale_values)
    out_labels = [value["label"] for value in groups.values()]
    out_values = [float(np.mean(value["values"])) for value in groups.values()]
    out_scales = [np.mean(value["scales"], axis=0) for value in groups.values()] if scales is not None else None
    return out_labels, out_values, out_scales


def evaluate_dataset(config: str, name: str, batch_size: int):
    ar, scales = CONFIGS[config]
    config_dir = OUT / config
    missing = [seed for seed in SEEDS if not (config_dir / f"seed{seed}" / "best.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"{config} is missing best.pt for seed(s): {missing}")
    samples, patient_level = source_for(name)
    print(f"[EVALUATION] {config} | {name}: {len(samples)} images, {len(scales)} views, 5 seeds.", flush=True)
    dataset = EvalDataset(samples, ar, scales)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    predictions = [seed_logits(config_dir / f"seed{seed}" / "best.pt", dataset, device,
                               f"{config} | {name} | seed {seed}/4", batch_size)
                   for seed in SEEDS]
    ids = sorted(predictions[0])
    labels = [predictions[0][item_id][0] for item_id in ids]
    scale_logits = np.asarray([np.mean([pred[item_id][1] for pred in predictions], axis=0) for item_id in ids])
    logits = [float(np.mean(value)) for value in scale_logits]
    seed_aucs = []
    for pred in predictions:
        values = [float(np.mean(pred[item_id][1])) for item_id in ids]
        one_labels, one_values, _ = patient_average(ids, labels, values) if patient_level else (labels, values, None)
        seed_aucs.append(float(metrics(one_labels, one_values)["AUC"]))
    if patient_level:
        labels, logits, scale_logits = patient_average(ids, labels, logits, scale_logits)
        scale_logits = np.asarray(scale_logits)
    result = metrics(labels, logits)
    result["PatientLevel"] = patient_level
    result["PatientCount"] = len(labels) if patient_level else None
    result["PerScaleAUC"] = [float(metrics(labels, scale_logits[:, index])["AUC"]) for index in range(len(scales))]
    artifact = {"config": config, "dataset": name, "ar": ar, "tta_scales": scales,
                "tta_views": len(scales), "per_seed_auc": seed_aucs, "ensemble": result,
                "inference_time_sec": time.time() - started,
                "timestamp": datetime.now(timezone.utc).isoformat()}
    path = config_dir / "evaluation" / f"{name.lower()}.json"
    write_json(path, artifact)
    print(f"[EVALUATION COMPLETE] {config} | {name} | AUC={result['AUC']:.4f}", flush=True)
    return artifact


def upsert_master(row):
    path = OUT / "results" / "master_training_results.csv"
    old_rows, fieldnames = [], []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            old_rows = [item for item in reader if item.get("Config") != row["Config"]]
    fieldnames = list(dict.fromkeys([*fieldnames, *row.keys()]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)
    temporary.replace(path)


def assemble_summary(config: str):
    config_dir = OUT / config
    training_path = config_dir / "training_summary.json"
    if training_path.exists():
        summary = json.loads(training_path.read_text(encoding="utf-8"))
    else:
        seeds = [json.loads((config_dir / f"seed{seed}" / "config.json").read_text(encoding="utf-8")) for seed in SEEDS]
        ar, scales = CONFIGS[config]
        summary = {"config": config, "ar": ar, "tta_scales": scales, "tta_views": len(scales),
                   "seeds": seeds, "total_training_time_sec": sum(item["training_time_sec"] for item in seeds)}
    evaluations = {}
    for name in DATASETS:
        path = config_dir / "evaluation" / f"{name.lower()}.json"
        if path.exists(): evaluations[name] = json.loads(path.read_text(encoding="utf-8"))
    summary["evaluations"] = evaluations
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    write_json(config_dir / "summary.json", summary)
    if set(evaluations) != set(DATASETS): return summary
    row = {"Config": config, "AR": summary["ar"], "TTA_Scales": json.dumps(summary["tta_scales"]),
           "TTA_Views": summary["tta_views"]}
    for seed in summary["seeds"]:
        prefix = f"Seed{seed['seed']}"
        row[f"{prefix}_BestEpoch"] = seed["best_epoch"]
        row[f"{prefix}_BestValAUC"] = seed["best_val_auc"]
        row[f"{prefix}_TestAUC_Internal"] = evaluations["Internal"]["per_seed_auc"][seed["seed"]]
    for name, artifact in evaluations.items():
        for metric, value in artifact["ensemble"].items(): row[f"Ensemble_{name}_{metric}"] = value
    row["TotalTrainingTime"] = summary["total_training_time_sec"]
    row["TotalInferenceTime"] = sum(item["inference_time_sec"] for item in evaluations.values())
    row["Timestamp"] = summary["timestamp"]
    upsert_master(row)
    update_status(OUT, config, state="complete", active_seed=None, completed_seeds=list(SEEDS),
                  summary=str((config_dir / "summary.json").resolve()))
    print(f"[CONFIG COMPLETE] {config}. Master CSV updated.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all",
                        help="Use one dataset per GPU session when quota is tight.")
    parser.add_argument("--batch-size", type=int, default=4, help="Lower after a CUDA out-of-memory error.")
    args = parser.parse_args()
    if args.batch_size < 1: parser.error("--batch-size must be positive")
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    update_status(OUT, args.config, state="evaluating", active_seed=None, dataset=list(selected))
    for name in selected: evaluate_dataset(args.config, name, args.batch_size)
    summary = assemble_summary(args.config)
    remaining = sorted(set(DATASETS) - set(summary.get("evaluations", {})))
    if remaining:
        update_status(OUT, args.config, state="evaluation_partial", active_seed=None, remaining_datasets=remaining)
        print(f"[EVALUATION PARTIAL] Remaining datasets: {', '.join(remaining)}", flush=True)


if __name__ == "__main__":
    main()
