"""
utils/dataset.py
────────────────
PyTorch Dataset and DataLoader factory for Smart Depth Vision.

Each sample is a 4-channel tensor: [R, G, B, D]  — shape (4, 224, 224)
Labels: binary (0=2D, 1=3D) and category index (0–9).
"""

import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

ROOT     = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

IMG_SIZE = CFG["dataset"]["image_size"]
PROC_DIR = ROOT / CFG["paths"]["processed_data"]


# ── Augmentation transforms (RGB only, applied consistently) ──────────────────
def _rgb_augment():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])


def _to_tensor_normalize(rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
    """
    Combine RGB + depth into a 4-channel normalized tensor.
    RGB:   normalized with ImageNet mean/std (channels 0-2)
    Depth: normalized to [0,1], then standardized (channel 3)
    """
    # RGB: HxWx3 uint8 → 3xHxW float32, ImageNet normalize
    rgb_t   = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    mean    = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std     = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_t   = (rgb_t - mean) / std

    # Depth: HxW float32 [0,1] → 1xHxW, standardized
    d_t     = torch.from_numpy(depth).float().unsqueeze(0)
    d_t     = (d_t - 0.5) / 0.5   # map [0,1] → [-1,1]

    return torch.cat([rgb_t, d_t], dim=0)   # (4, H, W)


class DepthVisionDataset(Dataset):
    """
    Loads 4-channel (RGB + depth) samples from the processed directory.

    split: "train" | "val" | "test"
    augment: apply color jitter + flip + rotation (train only)
    """

    def __init__(self, split: str = "train", augment: bool = False):
        csv_path    = PROC_DIR / "dataset.csv"
        full_df     = pd.read_csv(csv_path)
        self.df     = full_df[full_df["split"] == split].reset_index(drop=True)
        self.split_dir = PROC_DIR / split
        self.augment   = augment and (split == "train")
        self._aug      = _rgb_augment() if self.augment else None

        if len(self.df) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}'. "
                "Did you run prepare_dataset.py?"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        sid      = row["sample_id"]

        # Load RGB
        rgb_path = self.split_dir / f"{sid}_rgb.jpg"
        img_bgr  = cv2.imread(str(rgb_path))
        if img_bgr is None:
            raise FileNotFoundError(f"RGB file missing: {rgb_path}")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

        # Load depth
        dep_path = self.split_dir / f"{sid}_depth.npy"
        depth    = np.load(str(dep_path))                # (H, W) float32 [0,1]

        # Augment (RGB only — depth is spatial, so we flip/rotate consistently)
        if self.augment:
            from PIL import Image
            pil_rgb   = Image.fromarray(rgb)
            pil_rgb   = self._aug(pil_rgb)
            rgb       = np.array(pil_rgb)

            # Mirror and rotate depth to match RGB
            if np.random.rand() > 0.5:
                rgb   = rgb[:, ::-1, :].copy()
                depth = depth[:, ::-1].copy()

            angle = np.random.uniform(-10, 10)
            M     = cv2.getRotationMatrix2D(
                        (rgb.shape[1]//2, rgb.shape[0]//2), angle, 1.0)
            rgb   = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]))
            depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))

        tensor  = _to_tensor_normalize(rgb, depth)
        bin_lbl = torch.tensor(int(row["label_2d3d"]), dtype=torch.long)
        cat_lbl = torch.tensor(int(row["class_idx"]),  dtype=torch.long)

        return tensor, bin_lbl, cat_lbl


def make_loaders(batch_size: int = 32, num_workers: int = 4):
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = DepthVisionDataset("train", augment=True)
    val_ds   = DepthVisionDataset("val",   augment=False)
    test_ds  = DepthVisionDataset("test",  augment=False)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=(num_workers > 0))

    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"  Train: {len(train_ds):>5} samples  |  "
          f"Val: {len(val_ds):>4} samples  |  "
          f"Test: {len(test_ds):>4} samples")

    return train_loader, val_loader, test_loader
