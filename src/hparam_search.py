"""
Optuna-based hyperparameter search for TN5000.
Per plan §7: TPE sampler, 40 trials, MedianPruner, 3 seeds per trial.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from .dataset import TN5000Dataset
from .models import build_model
from .trainer import train_model
from .transforms import get_train_transforms, get_val_transforms

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

DATA_ROOT = None  # set at runtime
SPLIT_DIR = None  # set at runtime


def make_objective(
    arch: str,
    data_root: str,
    train_txt: str,
    val_txt: str,
    n_seeds: int = 3,
    n_epochs_for_trial: int = 20,  # max epochs per trial (pruning kicks in at 15)
    device=None,
    output_dir: Optional[str] = None,
):
    """Create an Optuna objective function for the given architecture."""

    def objective(trial: optuna.Trial) -> float:
        lr_head = trial.suggest_float("lr_head", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        pos_weight_scale = trial.suggest_float("pos_weight_scale", 0.8, 1.5)

        if arch in ("resnet50", "efficientnet_b3"):
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        else:
            batch_size = 16  # Swin fixed for memory

        auc_scores = []
        for seed in range(n_seeds):
            import torch
            torch.manual_seed(seed)

            model = build_model(arch, dropout=dropout)

            train_dataset = TN5000Dataset(
                data_root, train_txt, transform=get_train_transforms()
            )
            val_dataset = TN5000Dataset(
                data_root, val_txt, transform=get_val_transforms()
            )
            pos_weight = train_dataset.get_class_weights()

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size * 2, shuffle=False,
                num_workers=4, pin_memory=True
            )

            config = {
                "lr_head": lr_head,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "pos_weight": float(pos_weight),
                "pos_weight_scale": pos_weight_scale,
                "max_epochs": n_epochs_for_trial,
                "patience": n_epochs_for_trial + 1,  # no early stopping during search
                "min_delta": 0.0,
                "T_0": 10,
                "T_mult": 2,
                "label_smooth_eps": 0.05,
                "grad_clip_norm": 1.0,
            }

            history = train_model(
                model, train_loader, val_loader, config,
                checkpoint_dir=None,
                run_name=f"{arch}_trial{trial.number}_seed{seed}",
                device=device,
            )

            # Report intermediate val AUC for pruning
            for ep, auc in enumerate(history["val_auc"]):
                trial.report(auc, ep)
                if trial.should_prune() and ep >= 14:  # prune at epoch 15
                    raise optuna.exceptions.TrialPruned()

            auc_scores.append(history["best_val_auc"])

        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        # Objective: mean - 0.5 * std (penalise instability)
        objective_value = mean_auc - 0.5 * std_auc

        trial.set_user_attr("mean_auc", float(mean_auc))
        trial.set_user_attr("std_auc", float(std_auc))
        trial.set_user_attr("auc_scores", auc_scores)
        return objective_value

    return objective


def run_hparam_search(
    arch: str,
    data_root: str,
    train_txt: str,
    val_txt: str,
    n_trials: int = 40,
    n_seeds: int = 3,
    n_epochs_for_trial: int = 20,
    output_dir: str = "outputs/optuna",
    device=None,
    study_name: Optional[str] = None,
) -> optuna.Study:
    """Run Optuna hyperparameter search for one architecture."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if study_name is None:
        study_name = f"tn5000_{arch}"

    db_path = output_dir / f"{study_name}.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=14),
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(
        arch=arch,
        data_root=data_root,
        train_txt=train_txt,
        val_txt=val_txt,
        n_seeds=n_seeds,
        n_epochs_for_trial=n_epochs_for_trial,
        device=device,
        output_dir=str(output_dir),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save best config
    best_params = study.best_params
    best_params["arch"] = arch
    config_path = output_dir / f"{study_name}_best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best config saved to {config_path}")

    # Save trial history CSV
    trials_df = study.trials_dataframe()
    csv_path = output_dir / f"{study_name}_trials.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"Trial history saved to {csv_path}")

    return study
