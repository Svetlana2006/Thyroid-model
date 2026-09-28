"""
Multi-Scale ROI + Context Fusion Model (multiscale_roi_context_v1)

Architecture:
- ONE shared Swin-Tiny backbone (same pretrained weights, not tripled)
- Three views processed per image: full image, tight ROI, expanded ROI
- Per-view: multilevel features from layers.1/2/3, GAP, LayerNorm, Linear(ch->128, bias=False)
- Gated fusion of all 9 x 128 = 1152 dims -> classifier

ROI generation:
- Training (TN5000): ground-truth VOC bounding boxes
- Inference on any dataset: GradCAM-based automatic ROI proposal from the shared backbone
  (no external ground-truth boxes required at test time)

Backbone initialised identically to main model MultiLevelSwin.
Backbone parameters are SHARED across all three views (not copied).
"""

import torch
import torch.nn as nn
import timm


# Stage channels — identical to main model
STAGE_CHANNELS = {"layers.1": 192, "layers.2": 384, "layers.3": 768}
PROJ_DIM = 128          # per-stage projection dimension
N_STAGES = 3            # layers.1, layers.2, layers.3
N_VIEWS = 3             # full, tight ROI, expanded ROI
FUSED_DIM = PROJ_DIM * N_STAGES * N_VIEWS   # 128 * 3 * 3 = 1152


class MultiScaleROIModel(nn.Module):
    """
    Multi-Scale ROI + Context Fusion model.

    One shared Swin-Tiny backbone processes all three views in a single
    batched forward pass (views are concatenated along the batch dimension).

    Per-view feature extraction mirrors the main model's MultiLevelSwin:
      - Hook into layers.1 (192-ch), layers.2 (384-ch), layers.3 (768-ch)
      - Spatial GAP: (B, H, W, C) -> (B, C)
      - LayerNorm(C)
      - Linear(C -> 128, bias=False)

    Three views -> 9 x 128 = 1152 dim concatenated representation.

    Gated fusion:
      gate   = sigmoid(Linear(1152 -> 1152))
      fused  = gate * representation
      logit  = Linear(1152 -> 256) -> GELU -> Dropout(0.3) -> Linear(256 -> 1)

    Staged freezing (same schedule as main model):
      epochs < 6:   backbone frozen, only head/projections trainable
      epochs 6-9:   last Swin stage (layers.3) + final norm trainable
      epoch >= 10:  full backbone trainable
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # ── Shared backbone (ONE instance, used for all three views) ──────────
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── Per-stage per-view norms + projections ────────────────────────────
        # Structure: stage_norms[view_key][stage_key], same for stage_projs.
        # Each view has its OWN norm and projection weights so the model can
        # learn view-specific representations after the shared backbone.
        self.view_keys = ["full", "tight", "expanded"]
        self.stage_keys = {name: name.replace(".", "_") for name in STAGE_CHANNELS}

        self.stage_norms = nn.ModuleDict()
        self.stage_projs = nn.ModuleDict()
        for view_key in self.view_keys:
            for stage_name, ch in STAGE_CHANNELS.items():
                sk = self.stage_keys[stage_name]
                combined_key = f"{view_key}__{sk}"
                self.stage_norms[combined_key] = nn.LayerNorm(ch)
                self.stage_projs[combined_key] = nn.Linear(ch, PROJ_DIM, bias=False)

        # ── Gated fusion ──────────────────────────────────────────────────────
        # Simple element-wise gate learned from the full representation
        self.gate = nn.Linear(FUSED_DIM, FUSED_DIM, bias=True)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(FUSED_DIM, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

        # Internal hook storage (populated during forward, cleared after)
        self._stage_feats: dict = {}
        self._hooks: list = []

    # ── Hook management ───────────────────────────────────────────────────────

    def _register_hooks(self):
        for stage_name in STAGE_CHANNELS:
            module = dict(self.backbone.named_modules())[stage_name]
            handle = module.register_forward_hook(
                lambda mod, inp, out, n=stage_name: self._stage_feats.update({n: out})
            )
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Feature extraction for one batch (any view) ───────────────────────────

    def _extract_features(self, x: torch.Tensor, view_key: str) -> torch.Tensor:
        """
        Run backbone on x and extract multilevel projected features.

        Args:
            x:        (B, 3, 224, 224) tensor
            view_key: one of "full", "tight", "expanded"
        Returns:
            (B, N_STAGES * PROJ_DIM) = (B, 384)
        """
        self._stage_feats.clear()
        self._register_hooks()
        _ = self.backbone(x)
        self._remove_hooks()

        pooled = []
        for stage_name in STAGE_CHANNELS:
            sk = self.stage_keys[stage_name]
            combined_key = f"{view_key}__{sk}"
            feat = self._stage_feats[stage_name]          # (B, H, W, C)
            feat = feat.mean(dim=(1, 2))                  # (B, C)  — spatial GAP
            feat = self.stage_norms[combined_key](feat)   # (B, C)
            feat = self.stage_projs[combined_key](feat)   # (B, 128)
            pooled.append(feat)

        return torch.cat(pooled, dim=-1)   # (B, 384)

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        full: torch.Tensor,
        tight: torch.Tensor,
        expanded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            full:     (B, 3, 224, 224) — full image
            tight:    (B, 3, 224, 224) — tight ROI (nodule + ~10% padding)
            expanded: (B, 3, 224, 224) — expanded ROI (2x enlarged box)
        Returns:
            logits: (B, 1)
        """
        B = full.size(0)

        # ── Single batched backbone pass for efficiency ───────────────────────
        # Stack all three views along the batch dimension so the backbone
        # (which is identical/shared weights) runs once with 3B images.
        # Then split the hook outputs per-view.
        all_views = torch.cat([full, tight, expanded], dim=0)  # (3B, 3, 224, 224)

        self._stage_feats.clear()
        self._register_hooks()
        _ = self.backbone(all_views)
        self._remove_hooks()

        # Split hook outputs and project per-view
        view_feats = []
        for i, view_key in enumerate(self.view_keys):
            pooled = []
            for stage_name in STAGE_CHANNELS:
                sk = self.stage_keys[stage_name]
                combined_key = f"{view_key}__{sk}"
                feat = self._stage_feats[stage_name]           # (3B, H, W, C)
                feat_view = feat[i * B: (i + 1) * B]          # (B,  H, W, C)
                feat_view = feat_view.mean(dim=(1, 2))         # (B,  C)
                feat_view = self.stage_norms[combined_key](feat_view)
                feat_view = self.stage_projs[combined_key](feat_view)
                pooled.append(feat_view)
            view_feats.append(torch.cat(pooled, dim=-1))       # (B, 384) per view

        # Concatenate all view features: (B, 1152)
        rep = torch.cat(view_feats, dim=-1)

        # Gated fusion
        gate = torch.sigmoid(self.gate(rep))
        fused = gate * rep                                     # (B, 1152)

        return self.classifier(fused)                          # (B, 1)

    # ── Staged freezing (same schedule as main model) ─────────────────────────

    def freeze_epoch(self, epoch: int):
        """Mirror of main model's MultiLevelSwin.freeze_epoch."""
        if epoch >= 10:
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif epoch >= 6:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last Swin stage (layers.3)
            if hasattr(self.backbone, "layers"):
                for param in self.backbone.layers[-1].parameters():
                    param.requires_grad = True
            # Unfreeze final norm
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # ── Parameter groups for discriminative LR ───────────────────────────────

    def get_param_groups(self, lr_head: float, lr_backbone: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.stage_norms.parameters())
            + list(self.stage_projs.parameters())
            + list(self.gate.parameters())
            + list(self.classifier.parameters())
        )
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups

    # ── Architecture report ───────────────────────────────────────────────────

    def report(self) -> str:
        n_total = sum(p.numel() for p in self.parameters())
        n_backbone = sum(p.numel() for p in self.backbone.parameters())
        n_head = (
            sum(p.numel() for p in self.stage_norms.parameters())
            + sum(p.numel() for p in self.stage_projs.parameters())
            + sum(p.numel() for p in self.gate.parameters())
            + sum(p.numel() for p in self.classifier.parameters())
        )
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines = [
            "MultiScaleROIModel — architecture report",
            "  Backbone: swin_tiny_patch4_window7_224 (SHARED across all views)",
            f"  Backbone params:         {n_backbone:>12,}",
            f"  Head params:             {n_head:>12,}",
            f"    stage_norms+projs:     {sum(p.numel() for p in self.stage_norms.parameters()) + sum(p.numel() for p in self.stage_projs.parameters()):>12,}",
            f"    gate:                  {sum(p.numel() for p in self.gate.parameters()):>12,}",
            f"    classifier:            {sum(p.numel() for p in self.classifier.parameters()):>12,}",
            f"  Total params:            {n_total:>12,}",
            f"  Trainable (epoch 1):     {n_trainable:>12,}",
            f"  Views:                   full / tight ROI / expanded ROI",
            f"  Stage dims:              192 / 384 / 768 -> 128 each per view",
            f"  Fused dim:               {FUSED_DIM} (= 128 x {N_STAGES} stages x {N_VIEWS} views)",
            f"  Fusion:                  element-wise gate + Linear(1152->256)->GELU->Drop->Linear(256->1)",
        ]
        return "\n".join(lines)
