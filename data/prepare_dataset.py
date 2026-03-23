"""
prepare_dataset.py  (v3 — MiDaS depth for BOTH 2D and 3D samples)
──────────────────────────────────────────────────────────────────
Key fix: NYU samples now use MiDaS-estimated depth (same as inference)
instead of raw sensor depth — this makes training/inference consistent.

3D samples → NYU RGB + MiDaS depth + YOLOv8 object crops
2D samples → COCO RGB + synthetic flat depth

Run:
    python data/prepare_dataset.py
"""

import random
import argparse
import warnings
import numpy as np
import cv2
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

PATHS           = CFG["paths"]
DS_CFG          = CFG["dataset"]
IMG_SIZE        = DS_CFG["image_size"]
MIN_CROP        = DS_CFG["min_crop_size"]
MAX_PER_CLASS   = DS_CFG["max_samples_per_class"]
DEPTH_NOISE_STD = DS_CFG["depth_2d_noise"]
CATEGORIES      = DS_CFG["coco_categories"]

NYU_DIR    = Path(r"C:\Users\aatri\OneDrive\Desktop\coding\capstone project\extracted")
NYU_RGB    = NYU_DIR / "images"

COCO_IMG_DIR  = ROOT / PATHS["coco_images"]
COCO_ANN_FILE = ROOT / PATHS["coco_ann"]
PROC_DIR      = ROOT / PATHS["processed_data"]

CLASS_TO_IDX  = {c: i for i, c in enumerate(CATEGORIES)}
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_TO_OURS = {
    "person":     "person",
    "cat":        "cat",
    "dog":        "dog",
    "chair":      "chair",
    "bottle":     "bottle",
    "cup":        "cup",
    "laptop":     "laptop",
    "book":       "book",
    "cell phone": "cell phone",
    "backpack":   "backpack",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def resize_pair(rgb, depth, size):
    h, w    = rgb.shape[:2]
    scale   = size / max(h, w)
    nh, nw  = int(h * scale), int(w * scale)
    rgb_r   = cv2.resize(rgb,   (nw, nh), interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = size - nh
    pad_w = size - nw
    rgb_out   = cv2.copyMakeBorder(rgb_r,   0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    depth_out = cv2.copyMakeBorder(depth_r, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0.0)
    return rgb_out, depth_out


def save_sample(out_dir, sample_id, rgb, depth):
    cv2.imwrite(str(out_dir / f"{sample_id}_rgb.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    np.save(str(out_dir / f"{sample_id}_depth.npy"), depth.astype(np.float32))


def make_synthetic_flat_depth(h, w, noise_std=0.01):
    rng    = np.random.default_rng()
    base_d = rng.uniform(0.3, 0.7)
    tilt_x = rng.uniform(-0.05, 0.05)
    tilt_y = rng.uniform(-0.05, 0.05)
    xs     = np.linspace(0, 1, w)
    ys     = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(xs, ys)
    depth  = base_d + tilt_x * xx + tilt_y * yy
    depth += rng.normal(0, noise_std, depth.shape)
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


# ── MiDaS loader ─────────────────────────────────────────────────────────────
def load_midas():
    print("  Loading MiDaS DPT-Large (same as inference)...")
    midas     = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
    midas     = midas.to(DEVICE).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform  = transforms.dpt_transform
    print(f"  MiDaS ready on {DEVICE}")
    return midas, transform


@torch.no_grad()
def get_midas_depth(rgb_frame, midas, transform):
    """Run MiDaS on RGB frame, return normalized depth (H,W) float32 [0,1]."""
    input_batch = transform(rgb_frame).to(DEVICE)
    prediction  = midas(input_batch)
    prediction  = F.interpolate(
        prediction.unsqueeze(1),
        size=rgb_frame.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()
    depth = prediction.cpu().numpy()
    mn, mx = depth.min(), depth.max()
    if mx - mn > 1e-8:
        depth = (depth - mn) / (mx - mn)
    return depth.astype(np.float32)


# ── 3D Samples: NYU RGB + MiDaS depth + YOLOv8 crops ─────────────────────────
def extract_nyu_samples(out_dir):
    if not NYU_RGB.exists():
        print(f"  [ERROR] NYU images not found at: {NYU_RGB}")
        return []

    print("\n─── Extracting 3D samples from NYU (YOLOv8 + MiDaS) ───")

    from ultralytics import YOLO
    yolo_model = YOLO("yolov8s.pt")

    midas, transform = load_midas()

    rgb_files = sorted(NYU_RGB.glob("*.png"))
    print(f"  Found {len(rgb_files)} NYU frames.")
    print(f"  Note: MiDaS runs on every frame — takes ~15 min total.")

    samples = []
    counts  = defaultdict(int)

    for rgb_path in tqdm(rgb_files, desc="NYU frames"):
        frame_id = rgb_path.stem

        img_bgr = cv2.imread(str(rgb_path))
        if img_bgr is None:
            continue
        rgb_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── MiDaS depth (SAME as inference pipeline) ──────────────────────
        depth_norm = get_midas_depth(rgb_frame, midas, transform)

        # ── YOLOv8 detection ───────────────────────────────────────────────
        results = yolo_model(img_bgr, verbose=False, conf=0.35)[0]

        for box in results.boxes:
            yolo_cls = yolo_model.names[int(box.cls)]
            cls_name = YOLO_TO_OURS.get(yolo_cls)
            if cls_name is None:
                continue
            if counts[cls_name] >= MAX_PER_CLASS:
                continue

            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            ih, iw = rgb_frame.shape[:2]
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(iw, x1); y1 = min(ih, y1)

            if (x1-x0) < MIN_CROP or (y1-y0) < MIN_CROP:
                continue

            rgb_crop   = rgb_frame[y0:y1, x0:x1]
            depth_crop = depth_norm[y0:y1, x0:x1]

            if rgb_crop.size == 0 or depth_crop.size == 0:
                continue

            rgb_r, depth_r = resize_pair(rgb_crop, depth_crop, IMG_SIZE)

            sample_id = f"nyu_{frame_id}_{cls_name.replace(' ','_')}_{counts[cls_name]:04d}"
            save_sample(out_dir, sample_id, rgb_r, depth_r)

            samples.append({
                "sample_id":  sample_id,
                "class_name": cls_name,
                "class_idx":  CLASS_TO_IDX[cls_name],
                "label_2d3d": 1,
                "depth_var":  float(depth_crop.std()),
            })
            counts[cls_name] += 1

    print(f"\n  NYU 3D samples: {len(samples)}")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:15s}: {cnt}")
    return samples


# ── 2D Samples: COCO crops + synthetic flat MiDaS-style depth ────────────────
def extract_coco_samples(out_dir):
    if not COCO_IMG_DIR.exists():
        print(f"  [ERROR] COCO images not found at {COCO_IMG_DIR}")
        return []

    print("\n─── Extracting 2D samples from COCO val2017 ───")

    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("  [ERROR] Run: pip install pycocotools")
        return []

    coco    = COCO(str(COCO_ANN_FILE))
    samples = []
    counts  = defaultdict(int)

    for cat_name in tqdm(CATEGORIES, desc="COCO categories"):
        cat_ids = coco.getCatIds(catNms=[cat_name])
        if not cat_ids:
            continue

        anns = coco.loadAnns(coco.getAnnIds(catIds=cat_ids))
        random.shuffle(anns)

        for ann in anns:
            if counts[cat_name] >= MAX_PER_CLASS:
                break

            x, y, w, h = [int(v) for v in ann["bbox"]]
            if w < MIN_CROP or h < MIN_CROP:
                continue

            img_info = coco.loadImgs(ann["image_id"])[0]
            img_path = COCO_IMG_DIR / img_info["file_name"]
            if not img_path.exists():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ih, iw = img_rgb.shape[:2]
            x0 = max(0, x);    y0 = max(0, y)
            x1 = min(iw, x+w); y1 = min(ih, y+h)
            if (x1-x0) < MIN_CROP or (y1-y0) < MIN_CROP:
                continue

            rgb_crop   = img_rgb[y0:y1, x0:x1]
            ch, cw     = rgb_crop.shape[:2]
            depth_crop = make_synthetic_flat_depth(ch, cw, DEPTH_NOISE_STD)
            rgb_r, depth_r = resize_pair(rgb_crop, depth_crop, IMG_SIZE)

            sample_id = f"coco_{ann['image_id']:012d}_{ann['id']:09d}"
            save_sample(out_dir, sample_id, rgb_r, depth_r)

            samples.append({
                "sample_id":  sample_id,
                "class_name": cat_name,
                "class_idx":  CLASS_TO_IDX[cat_name],
                "label_2d3d": 0,
                "depth_var":  float(depth_crop.std()),
            })
            counts[cat_name] += 1

    print(f"\n  COCO 2D samples: {len(samples)}")
    for cls, cnt in sorted(counts.items()):
        print(f"    {cls:15s}: {cnt}")
    return samples


# ── Split & Save ──────────────────────────────────────────────────────────────
def build_splits(all_samples):
    df = pd.DataFrame(all_samples)
    df["strat_key"] = df["class_name"] + "_" + df["label_2d3d"].astype(str)

    key_counts = df["strat_key"].value_counts()
    valid_keys = key_counts[key_counts >= 3].index
    df = df[df["strat_key"].isin(valid_keys)].reset_index(drop=True)

    tr = DS_CFG["train_split"]
    va = DS_CFG["val_split"]
    te = DS_CFG["test_split"]

    train_df, temp_df = train_test_split(
        df, test_size=(1-tr), stratify=df["strat_key"], random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=(te/(va+te)),
        stratify=temp_df["strat_key"], random_state=42)

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"
    return {"train": train_df, "val": val_df, "test": test_df}


def move_to_splits(splits, src_dir):
    for split_name, df in splits.items():
        dest = PROC_DIR / split_name
        dest.mkdir(parents=True, exist_ok=True)
        for row in tqdm(df.itertuples(), desc=f"  Moving {split_name}", total=len(df)):
            for ext in ("_rgb.jpg", "_depth.npy"):
                src = src_dir / f"{row.sample_id}{ext}"
                dst = dest    / f"{row.sample_id}{ext}"
                if src.exists() and not dst.exists():
                    src.rename(dst)


def print_summary(csv_path):
    df = pd.read_csv(csv_path)
    print("\n" + "═"*56)
    print("  DATASET SUMMARY")
    print("═"*56)
    print(f"  Total : {len(df)}")
    print(f"  2D    : {(df.label_2d3d==0).sum()}")
    print(f"  3D    : {(df.label_2d3d==1).sum()}")
    print()
    for split in ["train", "val", "test"]:
        s = df[df.split == split]
        print(f"  {split:6s} → {len(s):5d}  "
              f"(2D:{(s.label_2d3d==0).sum():4d}  3D:{(s.label_2d3d==1).sum():4d})")
    print("═"*56)
    print(f"\n  CSV: {csv_path}")
    print("  Next: python train.py\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nyu",  action="store_true")
    parser.add_argument("--skip-coco", action="store_true")
    args = parser.parse_args()

    stage_dir = PROC_DIR / "_staging"
    stage_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    if not args.skip_nyu:
        all_samples += extract_nyu_samples(stage_dir)
    if not args.skip_coco:
        all_samples += extract_coco_samples(stage_dir)

    if not all_samples:
        print("[ERROR] No samples extracted.")
        return

    print(f"\n  Total: {len(all_samples)}")
    splits = build_splits(all_samples)
    move_to_splits(splits, stage_dir)

    full_df  = pd.concat(splits.values(), ignore_index=True)
    csv_path = PROC_DIR / "dataset.csv"
    full_df.to_csv(csv_path, index=False)

    try:
        stage_dir.rmdir()
    except OSError:
        pass

    print_summary(csv_path)


if __name__ == "__main__":
    main()
