"""
data/verify_dataset.py
───────────────────────
Quick sanity check: loads 16 random samples and saves a visual grid
to data/processed/verify_grid.jpg so you can visually inspect
the RGB crops and depth maps before training.

Run:
    python data/verify_dataset.py
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

PROC_DIR   = ROOT / CFG["paths"]["processed_data"]
CSV_PATH   = PROC_DIR / "dataset.csv"
N_SHOW     = 16   # number of samples to display


def load_sample(split_dir: Path, sid: str):
    rgb_path = split_dir / f"{sid}_rgb.jpg"
    dep_path = split_dir / f"{sid}_depth.npy"
    img_bgr  = cv2.imread(str(rgb_path))
    rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    depth    = np.load(str(dep_path))
    return rgb, depth


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} not found. Run prepare_dataset.py first.")
        return

    df    = pd.read_csv(CSV_PATH)
    # Pick N_SHOW random samples from train set
    sub   = df[df["split"] == "train"].sample(n=min(N_SHOW, len(df)),
                                               random_state=0)

    cols  = 4          # 4 samples per row
    rows  = N_SHOW // cols * 2   # ×2 for rgb + depth
    fig   = plt.figure(figsize=(cols * 3, rows * 1.6))
    gs    = gridspec.GridSpec(rows, cols, figure=fig,
                              hspace=0.05, wspace=0.05)

    print(f"Visualising {len(sub)} samples …")
    for i, (_, row) in enumerate(sub.iterrows()):
        split_dir = PROC_DIR / row["split"]
        rgb, depth = load_sample(split_dir, row["sample_id"])

        label_str = "3D" if row["label_2d3d"] == 1 else "2D"
        title_str = f"{row['class_name']} [{label_str}]"

        r_row = (i // cols) * 2
        col   = i % cols

        # RGB
        ax_rgb   = fig.add_subplot(gs[r_row, col])
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(title_str, fontsize=7, pad=2)
        ax_rgb.axis("off")

        # Depth
        ax_dep   = fig.add_subplot(gs[r_row + 1, col])
        ax_dep.imshow(depth, cmap="plasma", vmin=0, vmax=1)
        ax_dep.set_title(f"σ={depth.std():.3f}", fontsize=6, pad=2)
        ax_dep.axis("off")

    out_path = PROC_DIR / "verify_grid.jpg"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Verification grid saved to: {out_path}")

    # Also print quick stats
    print("\nDataset stats:")
    print(df.groupby(["split", "label_2d3d"]).size().to_string())
    print(f"\nDepth variance  —  2D mean: {df.query('label_2d3d==0').depth_var.mean():.4f}"
          f"  3D mean: {df.query('label_2d3d==1').depth_var.mean():.4f}")
    print("\nIf 2D depth_var << 3D depth_var, your synthetic depth is working correctly.")


if __name__ == "__main__":
    main()
