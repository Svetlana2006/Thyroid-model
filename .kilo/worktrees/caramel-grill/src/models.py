"""
TN5000 Model Architectures
==========================
Model A: ResNet-50
Model B: EfficientNet-B3 (via timm)
Model C: Swin-Tiny (via timm)

All use BCEWithLogitsLoss with a single logit output.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ─────────────────────────────────────────────
# Model A — ResNet-50
# ─────────────────────────────────────────────

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 with custom head:
      Linear(2048→256) → ReLU → Dropout(p) → Linear(256→1)

    Freeze schedule (caller drives this via .freeze_epoch(epoch)):
      epochs 1–3:  head only
      epoch 4:     + layer4
      epoch 8:     + layer3
      epoch 12+:   full unfreeze
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the original fully-connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

        # Store references for staged unfreezing
        self._backbone_raw = backbone
        self._layer4 = backbone.layer4
        self._layer3 = backbone.layer3

        # Start fully frozen
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_epoch(self, epoch: int):
        """Apply the staged freeze schedule."""
        if epoch <= 3:
            # Freeze entire backbone, train head only
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif epoch <= 7:
            # Unfreeze layer4
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self._backbone_raw.layer4.parameters():
                param.requires_grad = True
        elif epoch <= 11:
            # Unfreeze layer4 + layer3
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self._backbone_raw.layer4.parameters():
                param.requires_grad = True
            for param in self._backbone_raw.layer3.parameters():
                param.requires_grad = True
        else:
            # Full unfreeze
            for param in self.backbone.parameters():
                param.requires_grad = True

    def get_param_groups(self, lr_head: float, lr_backbone: float):
        """Return param groups with discriminative LRs."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups


# ─────────────────────────────────────────────
# Model B — EfficientNet-B3
# ─────────────────────────────────────────────

class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 (timm) with custom head:
      Linear(1536→256) → ReLU → Dropout(p) → Linear(256→1)

    Staged unfreeze: last 2 blocks at epoch 4, next 2 at epoch 8, rest at epoch 12.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        assert TIMM_AVAILABLE, "timm is required for EfficientNet-B3"

        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=True, num_classes=0
        )
        feature_dim = self.backbone.num_features  # 1536

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

        # Freeze all backbone params initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Identify block groups (timm EfficientNet has `blocks` attribute)
        self._blocks = list(self.backbone.blocks.children())  # list of Sequential blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_epoch(self, epoch: int):
        """Apply staged unfreeze for EfficientNet block groups."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        n_blocks = len(self._blocks)
        if epoch >= 12:
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif epoch >= 8:
            # Unfreeze last 4 blocks
            for block in self._blocks[max(0, n_blocks - 4):]:
                for param in block.parameters():
                    param.requires_grad = True
        elif epoch >= 4:
            # Unfreeze last 2 blocks
            for block in self._blocks[max(0, n_blocks - 2):]:
                for param in block.parameters():
                    param.requires_grad = True

    def get_param_groups(self, lr_head: float, lr_backbone: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups


# ─────────────────────────────────────────────
# Model C — Swin-Tiny
# ─────────────────────────────────────────────

class SwinTinyClassifier(nn.Module):
    """
    Swin-Tiny (timm) with custom head:
      Linear(768→256) → ReLU → Dropout(p) → Linear(256→1)

    Freeze schedule:
      epochs 1–5:  freeze all but head + last stage (stage 3)
      epoch 10+:   full unfreeze
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        assert TIMM_AVAILABLE, "timm is required for Swin-Tiny"

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )
        feature_dim = self.backbone.num_features  # 768

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

        # Freeze everything initially
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_epoch(self, epoch: int):
        if epoch >= 10:
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif epoch >= 6:
            # Unfreeze last stage
            for param in self.backbone.parameters():
                param.requires_grad = False
            # timm Swin has `layers` attribute
            if hasattr(self.backbone, "layers"):
                for param in self.backbone.layers[-1].parameters():
                    param.requires_grad = True
            # Also unfreeze norm
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
        else:
            # epochs 1-5: freeze all backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Only head gets gradient (handled externally)

    def get_param_groups(self, lr_head: float, lr_backbone: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_model(arch: str, dropout: float = 0.3) -> nn.Module:
    """
    Build a model by architecture name.
    arch: 'resnet50' | 'efficientnet_b3' | 'swin_tiny'
    """
    arch = arch.lower()
    if arch == "resnet50":
        return ResNet50Classifier(dropout=dropout)
    elif arch in ("efficientnet_b3", "efficientnet-b3"):
        return EfficientNetB3Classifier(dropout=dropout)
    elif arch in ("swin_tiny", "swin-tiny"):
        return SwinTinyClassifier(dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
