"""Registry: scans outputs/checkpoints/ and outputs/results/ to build the model list."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from api.schemas import ModelInfo

CHECKPOINT_DIR = Path("outputs/checkpoints")
RESULTS_DIR = Path("outputs/results")

# Regex for checkpoint filenames: {arch}_seed{N}_best.pt
_CKPT_RE = re.compile(r"^(?P<arch>.+?)_seed(?P<seed>\d+)_best\.pt$")

# Human-readable labels
_LABELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b3": "EfficientNet-B3",
    "swin_tiny": "Swin-Tiny",
}


def scan_checkpoints() -> Dict[str, List[int]]:
    """Return {arch: [seed0, seed1, ...]} from checkpoint files on disk."""
    result: Dict[str, List[int]] = {}
    if not CHECKPOINT_DIR.exists():
        return result
    for path in sorted(CHECKPOINT_DIR.glob("*_seed*_best.pt")):
        m = _CKPT_RE.match(path.name)
        if m:
            arch = m.group("arch")
            seed = int(m.group("seed"))
            result.setdefault(arch, []).append(seed)
    return result


def build_model_list() -> List[ModelInfo]:
    """Build the full list of selectable models (singles + ensembles)."""
    arch_seeds = scan_checkpoints()
    models: List[ModelInfo] = []

    # Single-seed models
    for arch, seeds in sorted(arch_seeds.items()):
        for seed in sorted(seeds):
            models.append(ModelInfo(
                id=f"{arch}_seed{seed}",
                label=f"{_LABELS.get(arch, arch)} (seed {seed})",
                seeds=[seed],
                type="single",
            ))

    # Per-architecture seed ensembles (only if >1 seed)
    for arch, seeds in sorted(arch_seeds.items()):
        if len(seeds) > 1:
            models.append(ModelInfo(
                id=f"{arch}_ensemble",
                label=f"{_LABELS.get(arch, arch)} (seed-ensemble)",
                seeds=sorted(seeds),
                type="seed_ensemble",
            ))

    # Full ensemble across all architectures (only if >1 arch)
    if len(arch_seeds) > 1:
        all_seeds: List[int] = []
        for seeds in arch_seeds.values():
            all_seeds.extend(seeds)
        models.append(ModelInfo(
            id="full_ensemble",
            label="Ensemble (all architectures)",
            seeds=sorted(set(all_seeds)),
            type="full_ensemble",
        ))

    return models


def resolve_checkpoints(model_id: str) -> List[Tuple[str, int, Path]]:
    """Given a model_id, return [(arch, seed, checkpoint_path), ...]."""
    arch_seeds = scan_checkpoints()

    if model_id == "full_ensemble":
        result = []
        for arch, seeds in sorted(arch_seeds.items()):
            for seed in sorted(seeds):
                result.append((arch, seed, CHECKPOINT_DIR / f"{arch}_seed{seed}_best.pt"))
        return result

    # Seed ensemble: {arch}_ensemble
    if model_id.endswith("_ensemble"):
        arch = model_id.removesuffix("_ensemble")
        seeds = arch_seeds.get(arch, [])
        return [(arch, s, CHECKPOINT_DIR / f"{arch}_seed{s}_best.pt") for s in sorted(seeds)]

    # Single model: {arch}_seed{N}
    m = re.match(r"^(?P<arch>.+?)_seed(?P<seed>\d+)$", model_id)
    if m:
        arch = m.group("arch")
        seed = int(m.group("seed"))
        path = CHECKPOINT_DIR / f"{arch}_seed{seed}_best.pt"
        if path.exists():
            return [(arch, seed, path)]

    return []
