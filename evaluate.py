"""
evaluate.py
────────────
Evaluates the best saved model on the test set.

Outputs:
  - Binary (2D/3D) accuracy, precision, recall, F1
  - Per-class category accuracy
  - Confusion matrix saved to models/weights/confusion_matrix.png

Run:
    python evaluate.py
"""

import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.cuda.amp import autocast
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

ROOT     = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

DS_CFG      = CFG["dataset"]
PATHS       = CFG["paths"]
TR_CFG      = CFG["training"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = ROOT / PATHS["weights"]
CATEGORIES  = DS_CFG["coco_categories"]


from utils.dataset import make_loaders
from models.classifier import build_model


@torch.no_grad()
def run_evaluation():
    print("\n" + "═"*60)
    print("  Smart Depth Vision — Evaluation")
    print("═"*60)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = WEIGHTS_DIR / "best_model.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        print("  Run: python train.py  first.")
        return

    model = build_model(
        num_classes = DS_CFG["num_classes"],
        dropout     = 0.0,   # no dropout during eval
        device      = DEVICE
    )

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"\n  Loaded checkpoint: epoch {ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.4f}")

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_loader = make_loaders(
        batch_size  = TR_CFG["batch_size"],
        num_workers = TR_CFG["num_workers"]
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    all_bin_true, all_bin_pred = [], []
    all_cat_true, all_cat_pred = [], []

    for tensors, bin_lbl, cat_lbl in test_loader:
        tensors = tensors.to(DEVICE, non_blocking=True)

        with autocast():
            bin_out, cat_out = model(tensors)

        all_bin_true.extend(bin_lbl.numpy())
        all_bin_pred.extend(bin_out.argmax(1).cpu().numpy())
        all_cat_true.extend(cat_lbl.numpy())
        all_cat_pred.extend(cat_out.argmax(1).cpu().numpy())

    all_bin_true = np.array(all_bin_true)
    all_bin_pred = np.array(all_bin_pred)
    all_cat_true = np.array(all_cat_true)
    all_cat_pred = np.array(all_cat_pred)

    # ── Binary report ─────────────────────────────────────────────────────────
    print("\n  ── Binary Classification (2D vs 3D) ──")
    print(classification_report(
        all_bin_true, all_bin_pred,
        target_names=["2D", "3D"], digits=4))

    # ── Category report ───────────────────────────────────────────────────────
    print("\n  ── Category Classification (10 classes) ──")
    print(classification_report(
        all_cat_true, all_cat_pred,
        target_names=CATEGORIES, digits=4))

    # ── Confusion matrices ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Binary
    cm_bin = confusion_matrix(all_bin_true, all_bin_pred)
    ConfusionMatrixDisplay(cm_bin, display_labels=["2D", "3D"]).plot(
        ax=axes[0], colorbar=False)
    axes[0].set_title("Binary: 2D vs 3D", fontsize=13)

    # Category
    cm_cat = confusion_matrix(all_cat_true, all_cat_pred)
    ConfusionMatrixDisplay(cm_cat, display_labels=CATEGORIES).plot(
        ax=axes[1], colorbar=False, xticks_rotation=45)
    axes[1].set_title("Category Classification", fontsize=13)

    plt.tight_layout()
    out_path = WEIGHTS_DIR / "confusion_matrix.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    bin_acc = (all_bin_true == all_bin_pred).mean()
    cat_acc = (all_cat_true == all_cat_pred).mean()

    print("═"*60)
    print(f"  Binary accuracy   : {bin_acc:.4f}  ({bin_acc*100:.2f}%)")
    print(f"  Category accuracy : {cat_acc:.4f}  ({cat_acc*100:.2f}%)")
    print(f"  Confusion matrix  : {out_path}")
    print("═"*60)
    print("\n  Next step: python inference/pipeline.py")
if __name__ == "__main__":
    run_evaluation()
