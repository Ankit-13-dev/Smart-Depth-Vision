"""
models/classifier.py
─────────────────────
4-channel (RGB + Depth) ResNet18-based classifier.

Two output heads:
  - binary_head   : 2D vs 3D  (2 classes)
  - category_head : object category (10 classes)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DepthAwareClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # Load pretrained ResNet18
        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        # ── Modify first conv: 3 channels → 4 channels (RGB + Depth) ──────────
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Copy RGB weights, init depth channel with mean of RGB weights
        with torch.no_grad():
            base.conv1.weight[:, :3] = old_conv.weight
            base.conv1.weight[:, 3]  = old_conv.weight.mean(dim=1)

        # ── Backbone (everything except final FC) ──────────────────────────────
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1)

        self.dropout  = nn.Dropout(dropout)

        # ── Two heads ──────────────────────────────────────────────────────────
        self.binary_head   = nn.Linear(512, 2)          # 2D vs 3D
        self.category_head = nn.Linear(512, num_classes) # object class

    def forward(self, x):
        """
        x : (B, 4, H, W)  — channels 0-2 = RGB, channel 3 = depth
        returns:
            bin_logits : (B, 2)
            cat_logits : (B, num_classes)
        """
        feat = self.backbone(x)          # (B, 512, 1, 1)
        feat = feat.flatten(1)           # (B, 512)
        feat = self.dropout(feat)

        bin_logits = self.binary_head(feat)
        cat_logits = self.category_head(feat)

        return bin_logits, cat_logits


def build_model(num_classes: int = 10, dropout: float = 0.3,
                device: str = "cuda") -> DepthAwareClassifier:
    model = DepthAwareClassifier(num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    print(f"  Model: DepthAwareClassifier  |  "
          f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M  |  "
          f"Device: {device}")
    return model