"""
inference/pipeline.py
──────────────────────
Real-time Smart Depth Vision inference pipeline.

What it does every frame:
  1. Capture frame from webcam (or video file)
  2. Run YOLOv8s  → bounding boxes + class names
  3. Run MiDaS DPT-Large → depth map
  4. For each detected object:
       - Crop RGB + depth using bounding box
       - Run DepthAwareClassifier → 2D or 3D + category
  5. Draw annotated output on screen

Controls:
  Q     → Quit
  S     → Save current frame as screenshot
  SPACE → Pause / Resume

Run:
    python inference/pipeline.py                  # webcam
    python inference/pipeline.py --source video.mp4   # video file
    python inference/pipeline.py --source image.jpg   # single image
"""

import sys
import argparse
import time
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORIES = CFG["dataset"]["coco_categories"]
WEIGHTS    = ROOT / CFG["paths"]["weights"] / "best_model.pt"

sys.path.insert(0, str(ROOT))
from models.classifier import build_model


# ── Colours for each class ────────────────────────────────────────────────────
CLASS_COLORS = {
    "person":     (255, 100,  50),
    "cat":        (255, 200,   0),
    "dog":        (  0, 200, 255),
    "chair":      (180,   0, 255),
    "bottle":     (  0, 255, 100),
    "cup":        (255,   0, 150),
    "laptop":     ( 50, 150, 255),
    "book":       (255, 165,   0),
    "cell phone": (  0, 255, 200),
    "backpack":   (200, 255,   0),
}
DEFAULT_COLOR = (200, 200, 200)

# 2D = red tint overlay, 3D = green tint overlay
DIM_COLOR = {0: (0, 0, 255), 1: (0, 255, 0)}   # 0=2D red, 1=3D green
DIM_LABEL = {0: "2D", 1: "3D"}


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    print("  Loading YOLOv8s...")
    from ultralytics import YOLO
    yolo = YOLO(CFG["models"]["yolo"]["variant"])

    print("  Loading MiDaS DPT-Large...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
    midas = midas.to(DEVICE).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    print("  Loading DepthAwareClassifier...")
    classifier = build_model(
        num_classes=CFG["dataset"]["num_classes"],
        dropout=0.0,
        device=DEVICE
    )
    ckpt = torch.load(str(WEIGHTS), map_location=DEVICE)
    classifier.load_state_dict(ckpt["model"])
    classifier.eval()
    print(f"  Classifier loaded from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})\n")

    return yolo, midas, transform, classifier


# ── MiDaS depth for full frame ────────────────────────────────────────────────
@torch.no_grad()
def get_depth_map(frame_rgb: np.ndarray, midas, transform) -> np.ndarray:
    """Returns depth map (H, W) float32 normalized to [0, 1]."""
    input_batch = transform(frame_rgb).to(DEVICE)
    prediction  = midas(input_batch)
    prediction  = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()
    depth = prediction.cpu().numpy()
    mn, mx = depth.min(), depth.max()
    if mx - mn > 1e-8:
        depth = (depth - mn) / (mx - mn)
    return depth.astype(np.float32)


# ── Classify a single crop ────────────────────────────────────────────────────
@torch.no_grad()
def classify_crop(rgb_crop: np.ndarray, depth_crop: np.ndarray,
                  classifier, img_size: int = 224):
    """Returns (dim_label, dim_prob, cat_label, cat_prob)."""

    # ── Depth variance heuristic (strong signal) ──────────────────────────
    # If depth variance is clearly high → 3D, clearly low → 2D
    depth_std = float(depth_crop.std())
    if depth_std > 0.08:        # clearly 3D
        dim_override = 1
    elif depth_std < 0.025:     # clearly 2D / flat
        dim_override = 0
    else:
        dim_override = None     # let classifier decide

    # Resize
    h, w = rgb_crop.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    rgb_r   = cv2.resize(rgb_crop,   (nw, nh))
    depth_r = cv2.resize(depth_crop, (nw, nh))

    # Pad to square
    ph = img_size - nh
    pw = img_size - nw
    rgb_r   = cv2.copyMakeBorder(rgb_r,   0, ph, 0, pw, cv2.BORDER_CONSTANT, 0)
    depth_r = cv2.copyMakeBorder(depth_r, 0, ph, 0, pw, cv2.BORDER_CONSTANT, 0.0)

    # To tensor (4 channels)
    rgb_t = torch.from_numpy(rgb_r.transpose(2, 0, 1)).float() / 255.0
    mean  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_t = (rgb_t - mean) / std
    dep_t = torch.from_numpy(depth_r).float().unsqueeze(0)
    dep_t = (dep_t - 0.5) / 0.5
    x     = torch.cat([rgb_t, dep_t], dim=0).unsqueeze(0).to(DEVICE)

    bin_out, cat_out = classifier(x)
    bin_prob = torch.softmax(bin_out, dim=1)[0]
    cat_prob = torch.softmax(cat_out, dim=1)[0]

    # Apply depth variance override if signal is strong
    if dim_override is not None:
        dim_idx  = dim_override
        dim_conf = depth_std * 5.0   # scale std to confidence-like value
        dim_conf = min(dim_conf, 0.99)
    else:
        dim_idx  = bin_prob.argmax().item()
        dim_conf = bin_prob[dim_idx].item()

    cat_idx  = cat_prob.argmax().item()
    cat_conf = cat_prob[cat_idx].item()

    return dim_idx, dim_conf, CATEGORIES[cat_idx], cat_conf


# ── Draw annotation on frame ──────────────────────────────────────────────────
def draw_box(frame, x0, y0, x1, y1,
             dim_idx, dim_conf, cat_name, cat_conf, yolo_class):
    color     = CLASS_COLORS.get(cat_name, DEFAULT_COLOR)
    dim_color = DIM_COLOR[dim_idx]
    dim_lbl   = DIM_LABEL[dim_idx]

    # Box
    thickness = 2
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)

    # Coloured corner markers for 2D/3D
    corner_len = 12
    cv2.line(frame, (x0, y0), (x0+corner_len, y0), dim_color, 3)
    cv2.line(frame, (x0, y0), (x0, y0+corner_len), dim_color, 3)
    cv2.line(frame, (x1, y1), (x1-corner_len, y1), dim_color, 3)
    cv2.line(frame, (x1, y1), (x1, y1-corner_len), dim_color, 3)

    # Label background
    label     = f"{dim_lbl} {dim_conf*100:.0f}% | {cat_name} {cat_conf*100:.0f}%"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale= 0.5
    thickness_txt = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness_txt)

    lx = x0
    ly = y0 - 6
    if ly - th - 4 < 0:
        ly = y1 + th + 6

    cv2.rectangle(frame, (lx, ly-th-4), (lx+tw+4, ly+baseline),
                  color, cv2.FILLED)
    cv2.putText(frame, label, (lx+2, ly-2),
                font, font_scale, (0, 0, 0), thickness_txt, cv2.LINE_AA)

    return frame


# ── Draw depth minimap ────────────────────────────────────────────────────────
def draw_depth_minimap(frame, depth_map, size=200):
    """Draw a small depth map in top-right corner."""
    h, w   = frame.shape[:2]
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
    mini  = cv2.resize(depth_color, (size, size))

    # Border
    cv2.rectangle(mini, (0, 0), (size-1, size-1), (200, 200, 200), 1)
    cv2.putText(mini, "Depth", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    frame[10:10+size, w-size-10:w-10] = mini
    return frame


# ── Draw HUD ──────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, n_2d, n_3d, paused):
    h, w = frame.shape[:2]

    # Small semi-transparent bar at very bottom of frame
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-bar_h), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame,
                f"Smart Depth Vision  |  FPS: {fps:.1f}  |  2D: {n_2d}  3D: {n_3d}  |  Q=Quit  S=Save  SPACE=Pause",
                (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "PAUSED", (w//2-40, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return frame


# ── Main inference loop ───────────────────────────────────────────────────────
def run(source=0, save_output=False):
    print("=" * 56)
    print("  Smart Depth Vision — Live Inference")
    print("=" * 56)

    yolo, midas, transform, classifier = load_models()

    # Open source
    if isinstance(source, str) and Path(source).suffix.lower() in \
            {".jpg", ".jpeg", ".png", ".bmp"}:
        # Single image mode
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Cannot read image: {source}")
            return
        process_single_image(frame, yolo, midas, transform, classifier)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # Video writer (optional)
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = str(ROOT / "inference" / "output.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, 20.0,
                                 (int(cap.get(3)), int(cap.get(4))))

    yolo_conf = CFG["models"]["yolo"]["conf"]
    img_size  = CFG["dataset"]["image_size"]

    fps_smooth = 0.0
    paused     = False
    frame_idx  = 0

    print("  Press Q to quit, S to save frame, SPACE to pause\n")

    while True:
        if not paused:
            ret, frame_bgr = cap.read()
            if not ret:
                print("  Stream ended.")
                break

            t0 = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── YOLOv8 detection ──────────────────────────────────────────────
            results  = yolo(frame_bgr, verbose=False, conf=yolo_conf)[0]

            # ── MiDaS depth ───────────────────────────────────────────────────
            depth_map = get_depth_map(frame_rgb, midas, transform)

            # ── Classify each box ─────────────────────────────────────────────
            n_2d = n_3d = 0
            out_frame = frame_bgr.copy()

            for box in results.boxes:
                yolo_cls = yolo.names[int(box.cls)]
                x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                ih, iw = frame_bgr.shape[:2]
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(iw, x1); y1 = min(ih, y1)

                if (x1-x0) < 20 or (y1-y0) < 20:
                    continue

                rgb_crop   = frame_rgb[y0:y1, x0:x1]
                depth_crop = depth_map[y0:y1, x0:x1]

                if rgb_crop.size == 0 or depth_crop.size == 0:
                    continue

                dim_idx, dim_conf, cat_name, cat_conf = classify_crop(
                    rgb_crop, depth_crop, classifier, img_size)

                if dim_idx == 0:
                    n_2d += 1
                else:
                    n_3d += 1

                draw_box(out_frame, x0, y0, x1, y1,
                         dim_idx, dim_conf, yolo_cls, cat_conf, yolo_cls)

            # ── Depth minimap ─────────────────────────────────────────────────
            draw_depth_minimap(out_frame, depth_map)

            # ── FPS ───────────────────────────────────────────────────────────
            elapsed   = time.time() - t0
            fps       = 1.0 / max(elapsed, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps

            draw_hud(out_frame, fps_smooth, n_2d, n_3d, paused)

            if writer:
                writer.write(out_frame)

            frame_idx += 1

        cv2.imshow("Smart Depth Vision", out_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            save_path = str(ROOT / "inference" / f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(save_path, out_frame)
            print(f"  Saved: {save_path}")
        elif key == ord(" "):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("  Done.")


def process_single_image(frame_bgr, yolo, midas, transform, classifier):
    """Process and display a single image."""
    yolo_conf = 0.55
    img_size  = CFG["dataset"]["image_size"]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results   = yolo(frame_bgr, verbose=False, conf=yolo_conf)[0]
    depth_map = get_depth_map(frame_rgb, midas, transform)

    out_frame = frame_bgr.copy()
    n_2d = n_3d = 0

    for box in results.boxes:
        yolo_cls = yolo.names[int(box.cls)]
        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
        ih, iw = frame_bgr.shape[:2]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(iw, x1); y1 = min(ih, y1)
        if (x1-x0) < 20 or (y1-y0) < 20:
            continue

        rgb_crop   = frame_rgb[y0:y1, x0:x1]
        depth_crop = depth_map[y0:y1, x0:x1]
        if rgb_crop.size == 0 or depth_crop.size == 0:
            continue

        dim_idx, dim_conf, cat_name, cat_conf = classify_crop(
            rgb_crop, depth_crop, classifier, img_size)
        if dim_idx == 0:
            n_2d += 1
        else:
            n_3d += 1
        draw_box(out_frame, x0, y0, x1, y1,
                 dim_idx, dim_conf, yolo_cls, cat_conf, yolo_cls)

    draw_depth_minimap(out_frame, depth_map)
    draw_hud(out_frame, 0, n_2d, n_3d, False)

    out_path = str(ROOT / "inference" / "result.jpg")
    cv2.imwrite(out_path, out_frame)
    print(f"  Result saved: {out_path}")

    # Resize window to fit screen nicely (max 900px wide)
    h, w   = out_frame.shape[:2]
    max_w  = 900
    if w > max_w:
        scale     = max_w / w
        new_w     = int(w * scale)
        new_h     = int(h * scale)
        out_frame = cv2.resize(out_frame, (new_w, new_h))

    cv2.imshow("Smart Depth Vision — Result", out_frame)
    print("  Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="0=webcam, or path to video/image file")
    parser.add_argument("--save",   action="store_true",
                        help="Save output video to inference/output.mp4")
    args = parser.parse_args()

    source = args.source
    if source != 0:
        try:
            source = int(source)
        except ValueError:
            pass   # keep as string (file path)

    run(source=source, save_output=args.save)