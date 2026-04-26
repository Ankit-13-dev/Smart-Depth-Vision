"""
inference/pipeline.py
──────────────────────
Real-time Smart Depth Vision inference pipeline.

What it does every frame:
  1. Capture frame from webcam (or video file)
  2. Run YOLOv8  → bounding boxes + class names
  3. Run MiDaS DPT-Large → depth map
  4. For each detected object:
       - Crop RGB + depth using bounding box
       - Run DepthAwareClassifier → 2D or 3D + category
       - Run CLIP → living vs non-living  (NEW in v4)
  5. Draw annotated output on screen

Controls:
  Q     → Quit
  S     → Save current frame as screenshot
  SPACE → Pause / Resume

Run:
    python inference/pipeline.py                      # webcam
    python inference/pipeline.py --source video.mp4   # video file
    python inference/pipeline.py --source image.jpg   # single image

BUG FIX HISTORY:
  v1 — original bug:
      A photo on a phone screen was labelled 3D.
      Cause: depth_crop.std() captured phone frame + hand edges,
      inflating variance above 0.07 → hard 3D override.

  v2 — over-correction:
      Introduced centre_depth_std() (inner 60% of crop only).
      This reduced variance too much for real 3D objects far from camera,
      pushing their std below 0.03 → hard 2D override on real people.

  v3 — correct fix:
      • Keep is_inside_screen(): fully solves the phone-screen case because
        YOLO already labels the phone as "cell phone".  Any detection that
        is ≥50% inside a screen box is forced 2D BEFORE depth math runs.
      • Revert to plain depth_crop.std() — no centre-crop trick needed
        since the phone case is handled by the overlap check above.
      • Lower the hard-2D threshold from 0.03 → 0.012 so that far-away
        real objects (std ≈ 0.02–0.05) fall into the classifier's middle
        zone instead of being forced 2D.

  v4 — CLIP living/non-living layer (THIS FILE):
      • Adds a CLIP-based zero-shot classifier on top of the existing
        2D/3D pipeline.  No existing thresholds or variables changed.
      • Final rules:
            - 2D output       → non-living  (image of a person ≠ alive)
            - 3D + non-living class (chair, bottle, ...)  → non-living
            - 3D + living class (person, dog, ...) + CLIP confirms → living
            - 3D + living class but CLIP rejects → non-living

  v4.1 — small bugfixes:
      • Unified the kwarg names passed to LivingNonLivingClassifier.classify():
            rgb_crop=   (numpy RGB array)
            yolo_class= (YOLO label string)
            dim_idx=    (0=2D, 1=3D from DepthAwareClassifier)
        Both webcam loop and single-image path use identical call signatures.
      • classify() return dict now uses key 'clip_conf' consistently,
        matching pipeline's life_info["clip_conf"] access.
      • Fixed `cv2.imread(...) or cv2.imread(...)` pattern, which raises
        "ValueError: ambiguous truth value of array" when the first read
        succeeds.  Replaced with an explicit None-check.

  Depth heuristic thresholds (unchanged from v3):
      std > 0.07  → hard 3D  (clearly volumetric, close real object)
      std < 0.012 → hard 2D  (extremely flat: printed photo, mirror, etc.)
      else        → classifier decides  ← far-away real objects land here
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
from models.classifier      import build_model
from models.clip_classifier import LivingNonLivingClassifier   # ← NEW


# ── Labels that represent flat screen surfaces ────────────────────────────────
SCREEN_LABELS = {"cell phone", "tv", "laptop", "monitor", "tablet"}

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

DIM_COLOR = {0: (0, 0, 255), 1: (0, 255, 0)}
DIM_LABEL = {0: "2D", 1: "3D"}

# ── NEW: Living / Non-living visuals ──────────────────────────────────────────
LIFE_COLOR = {"living": (0, 255, 120), "non-living": (60, 60, 200)}
LIFE_LABEL = {"living": "LIVING",      "non-living": "NON-LIVING"}


# ── Helper: robust depth-map reader ───────────────────────────────────────────
def _read_depth_image(path: str) -> np.ndarray:
    """
    Read a depth-map image and return a float32 array normalised to [0, 1].
    Tries IMREAD_ANYDEPTH first (preserves 16-bit), then falls back to
    IMREAD_GRAYSCALE.  Raises FileNotFoundError if neither succeeds.

    NOTE: uses explicit None-check instead of `or` to avoid the
    "ambiguous truth value of array" ValueError when the first read succeeds.
    """
    dr = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if dr is None:
        dr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if dr is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    mn, mx = float(dr.min()), float(dr.max())
    return ((dr - mn) / (mx - mn + 1e-8)).astype(np.float32)


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    print("=" * 56)
    print("  Smart Depth Vision — Loading Models  (v4.1 + CLIP)")
    print("=" * 56)

    print("  Loading YOLOv8...")
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
          f"(val_loss={ckpt['val_loss']:.4f})")

    # ── NEW: CLIP living/non-living classifier ────────────────────────────────
    life_clf = LivingNonLivingClassifier(device=DEVICE)
    print()

    return yolo, midas, transform, classifier, life_clf


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


# ── Screen-overlap check ──────────────────────────────────────────────────────
def is_inside_screen(box, screen_boxes, overlap_threshold: float = 0.50) -> bool:
    """
    Returns True if `box` overlaps any screen-device box by >= overlap_threshold
    of `box`'s own area.

    This is the primary fix for the phone-photo false-positive:
    YOLO already identifies the phone as 'cell phone', so any person/object
    sitting mostly inside that box is a flat image on a screen → 2D.
    """
    px0, py0, px1, py1 = box
    person_area = max((px1 - px0) * (py1 - py0), 1)

    for sx0, sy0, sx1, sy1 in screen_boxes:
        ix0 = max(px0, sx0)
        iy0 = max(py0, sy0)
        ix1 = min(px1, sx1)
        iy1 = min(py1, sy1)
        if ix1 > ix0 and iy1 > iy0:
            if (ix1 - ix0) * (iy1 - iy0) / person_area >= overlap_threshold:
                return True
    return False


# ── Parse YOLO boxes + collect screen device boxes ────────────────────────────
def collect_boxes(results, yolo, frame_shape):
    """
    Returns all_boxes (list of dicts) and screen_boxes (list of tuples).
    Screen boxes are collected in one pass BEFORE the classify loop so
    is_inside_screen() can check every detection in the same frame.
    """
    ih, iw = frame_shape[:2]
    all_boxes, screen_boxes = [], []

    for box in results.boxes:
        label = yolo.names[int(box.cls)]
        conf  = float(box.conf)
        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(iw, x1); y1 = min(ih, y1)

        if (x1 - x0) < 20 or (y1 - y0) < 20:
            continue

        all_boxes.append(dict(label=label, conf=conf,
                              x0=x0, y0=y0, x1=x1, y1=y1))
        if label in SCREEN_LABELS:
            screen_boxes.append((x0, y0, x1, y1))

    return all_boxes, screen_boxes


# ── Classify a single crop ────────────────────────────────────────────────────
@torch.no_grad()
def classify_crop(rgb_crop: np.ndarray, depth_crop: np.ndarray,
                  classifier, img_size: int = 224,
                  force_2d: bool = False,
                  frame_shape: tuple = None):
    """
    Returns (dim_idx, dim_conf, cat_name, cat_conf).

    force_2d: True when is_inside_screen() confirmed this is on a screen.
              Skips depth analysis, returns 2D at 99% confidence.
    """

    # ── Fast path: confirmed screen object ───────────────────────────────
    if force_2d:
        h, w  = rgb_crop.shape[:2]
        scale = img_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        rgb_r  = cv2.resize(rgb_crop, (nw, nh))
        ph, pw = img_size - nh, img_size - nw
        rgb_r  = cv2.copyMakeBorder(rgb_r, 0, ph, 0, pw, cv2.BORDER_CONSTANT, 0)

        rgb_t = torch.from_numpy(rgb_r.transpose(2, 0, 1)).float() / 255.0
        mean  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_n = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb_t = (rgb_t - mean) / std_n
        dep_p = torch.zeros(1, img_size, img_size)
        x = torch.cat([rgb_t, dep_p], dim=0).unsqueeze(0).to(DEVICE)

        _, cat_out = classifier(x)
        cat_prob   = torch.softmax(cat_out, dim=1)[0]
        cat_idx    = cat_prob.argmax().item()
        return 0, 0.99, CATEGORIES[cat_idx], cat_prob[cat_idx].item()

    # ── Depth variance heuristic ──────────────────────────────────────────
    #
    # Use the full depth_crop.std() — no centre-crop trick.
    # The phone-screen case is handled entirely by is_inside_screen() above.
    #
    # Thresholds (v4.2 — adjusted):
    #   std > 0.07  → hard 3D  (close volumetric object)
    #   std < 0.005 → hard 2D  (truly flat: mirror, printed poster on wall)
    #
    #   The previous threshold was 0.012.  That was still too high: when a
    #   person is close to the camera and fills most of the frame, MiDaS
    #   produces a fairly uniform depth map for the whole face (everything
    #   at roughly the same distance) → std ≈ 0.006–0.011 → wrongly forced
    #   to 2D.  Dropping to 0.005 means only truly flat surfaces (glass,
    #   posters flush against a wall) are hard-overridden; everything else
    #   falls into the classifier's decision zone.
    #
    #   Additionally: if the crop covers more than 25% of the frame area,
    #   skip the hard-2D override entirely — a detection that large is
    #   almost certainly a real close-up subject, not a flat poster.
    #
    #   else → classifier decides
    depth_std = float(depth_crop.std())

    # If the crop covers >25% of the frame it's almost certainly a real
    # close-up subject — skip the hard-2D override to avoid mis-labelling
    # faces that fill the frame as 2D (their depth std is naturally low).
    crop_h, crop_w = depth_crop.shape[:2]
    if frame_shape is not None:
        frame_area   = frame_shape[0] * frame_shape[1]
        large_crop   = (crop_h * crop_w) / max(frame_area, 1) > 0.25
    else:
        large_crop = False

    if depth_std > 0.07:
        dim_override = 1                    # clearly 3D
    elif depth_std < 0.005 and not large_crop:
        dim_override = 0                    # truly flat surface
    else:
        dim_override = None                 # let the model decide

    # ── Build 4-channel tensor ─────────────────────────────────────────────
    h, w  = rgb_crop.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    rgb_r   = cv2.resize(rgb_crop,   (nw, nh))
    depth_r = cv2.resize(depth_crop, (nw, nh))

    ph, pw  = img_size - nh, img_size - nw
    rgb_r   = cv2.copyMakeBorder(rgb_r,   0, ph, 0, pw, cv2.BORDER_CONSTANT, 0)
    depth_r = cv2.copyMakeBorder(depth_r, 0, ph, 0, pw, cv2.BORDER_CONSTANT, 0.0)

    rgb_t = torch.from_numpy(rgb_r.transpose(2, 0, 1)).float() / 255.0
    mean  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_n = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_t = (rgb_t - mean) / std_n
    dep_t = torch.from_numpy(depth_r).float().unsqueeze(0)
    dep_t = (dep_t - 0.5) / 0.5
    x     = torch.cat([rgb_t, dep_t], dim=0).unsqueeze(0).to(DEVICE)

    bin_out, cat_out = classifier(x)
    bin_prob = torch.softmax(bin_out, dim=1)[0]
    cat_prob = torch.softmax(cat_out, dim=1)[0]

    if dim_override is not None:
        dim_idx  = dim_override
        dim_conf = min(depth_std * 5.0, 0.99)
    else:
        dim_idx  = bin_prob.argmax().item()
        dim_conf = bin_prob[dim_idx].item()

    cat_idx = cat_prob.argmax().item()
    return dim_idx, dim_conf, CATEGORIES[cat_idx], cat_prob[cat_idx].item()


# ── Draw annotation on frame ──────────────────────────────────────────────────
def draw_box(frame, x0, y0, x1, y1,
             dim_idx, dim_conf, yolo_class, cat_conf, _unused,
             life_label=None, life_conf=None):
    color     = CLASS_COLORS.get(yolo_class, DEFAULT_COLOR)
    dim_color = DIM_COLOR[dim_idx]
    dim_lbl   = DIM_LABEL[dim_idx]

    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

    cl = 12
    cv2.line(frame, (x0, y0), (x0+cl, y0), dim_color, 3)
    cv2.line(frame, (x0, y0), (x0, y0+cl), dim_color, 3)
    cv2.line(frame, (x1, y1), (x1-cl, y1), dim_color, 3)
    cv2.line(frame, (x1, y1), (x1, y1-cl), dim_color, 3)

    label = f"{dim_lbl} {dim_conf*100:.0f}% | {yolo_class} {cat_conf*100:.0f}%"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = 0.5
    (tw, th), bl = cv2.getTextSize(label, font, fs, 1)
    lx = x0
    ly = y0 - 6 if y0 - th - 10 >= 0 else y1 + th + 6
    cv2.rectangle(frame, (lx, ly-th-4), (lx+tw+4, ly+bl), color, cv2.FILLED)
    cv2.putText(frame, label, (lx+2, ly-2), font, fs, (0, 0, 0), 1, cv2.LINE_AA)

    # ── NEW: living / non-living tag below the main label ────────────────
    if life_label is not None:
        life_txt   = f"{LIFE_LABEL[life_label]} {life_conf*100:.0f}%"
        life_color = LIFE_COLOR[life_label]
        (tw2, th2), bl2 = cv2.getTextSize(life_txt, font, fs, 1)
        ly2 = ly + th2 + 6
        if ly2 + bl2 > frame.shape[0]:
            ly2 = ly - th - 12
        cv2.rectangle(frame, (lx, ly2-th2-4),
                      (lx+tw2+4, ly2+bl2), life_color, cv2.FILLED)
        cv2.putText(frame, life_txt, (lx+2, ly2-2),
                    font, fs, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ── Depth window ──────────────────────────────────────────────────────────────
def draw_depth_separate(depth_map, frame_shape):
    h, w  = frame_shape[:2]
    dvis  = (depth_map * 255).astype(np.uint8)
    dcol  = cv2.applyColorMap(dvis, cv2.COLORMAP_PLASMA)
    dres  = cv2.resize(dcol, (w // 2, h // 2))
    cv2.putText(dres, "MiDaS Depth Map",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(dres, "Warm=Near  Cool=Far",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
    cv2.imshow("Smart Depth Vision — Depth Map", dres)


# ── HUD bar ───────────────────────────────────────────────────────────────────
def draw_hud(frame, fps, n_2d, n_3d, paused, n_living=0, n_nonliving=0):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, h-28), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame,
                f"Smart Depth Vision  |  FPS: {fps:.1f}  |  2D: {n_2d}  3D: {n_3d}"
                f"  |  Living: {n_living}  Non-living: {n_nonliving}"
                f"  |  Q=Quit  S=Save  SPACE=Pause",
                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 255, 200), 1, cv2.LINE_AA)
    if paused:
        cv2.putText(frame, "PAUSED", (w//2-40, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return frame


# ── Main inference loop ───────────────────────────────────────────────────────
def run(source=0, save_output=False, custom_depth=None):
    print("=" * 56)
    print("  Smart Depth Vision — Live Inference  (v4.1 + CLIP)")
    print("=" * 56)

    yolo, midas, transform, classifier, life_clf = load_models()

    if isinstance(source, str) and \
            Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Cannot read image: {source}")
            return
        process_single_image(frame, yolo, midas, transform, classifier,
                             life_clf, custom_depth)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    writer = None
    if save_output:
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = str(ROOT / "inference" / "output.mp4")
        writer   = cv2.VideoWriter(out_path, fourcc, 20.0,
                                   (int(cap.get(3)), int(cap.get(4))))

    yolo_conf  = CFG["models"]["yolo"]["conf"]
    img_size   = CFG["dataset"]["image_size"]
    fps_smooth = 0.0
    paused     = False
    frame_idx  = 0
    out_frame  = None

    print("  Press Q to quit, S to save frame, SPACE to pause\n")

    while True:
        if not paused:
            ret, frame_bgr = cap.read()
            if not ret:
                print("  Stream ended.")
                break

            t0        = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results   = yolo(frame_bgr, verbose=False, conf=yolo_conf)[0]

            if custom_depth is not None:
                depth_map = _read_depth_image(custom_depth)
            else:
                depth_map = get_depth_map(frame_rgb, midas, transform)

            all_boxes, screen_boxes = collect_boxes(results, yolo, frame_bgr.shape)

            n_2d = n_3d = 0
            n_living = n_nonliving = 0
            out_frame = frame_bgr.copy()

            for b in all_boxes:
                x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
                rgb_crop   = frame_rgb[y0:y1, x0:x1]
                depth_crop = depth_map[y0:y1, x0:x1]
                if rgb_crop.size == 0 or depth_crop.size == 0:
                    continue

                # Screen devices (phone, tv, laptop ...) must NOT be checked
                # against screen_boxes — they are IN screen_boxes themselves,
                # so overlap with their own box would always be 100% and every
                # phone/laptop would be wrongly forced to 2D.
                # Only non-screen objects need the overlap check (e.g. a photo
                # of a person displayed on a phone screen).
                if b["label"] in SCREEN_LABELS:
                    inside_screen = False
                else:
                    inside_screen = is_inside_screen(
                        (x0, y0, x1, y1), screen_boxes, overlap_threshold=0.50)

                dim_idx, dim_conf, cat_name, cat_conf = classify_crop(
                    rgb_crop, depth_crop, classifier, img_size,
                    force_2d=inside_screen,
                    frame_shape=frame_bgr.shape)

                # ── living / non-living via CLIP ─────────────────────────
                # Compute what fraction of the full frame this crop covers.
                # classify() uses this to decide whether CLIP is trustworthy:
                # large crops (real close-up subjects) bypass CLIP entirely
                # because CLIP cannot distinguish a live video feed from a
                # photograph — both are still frames to CLIP.
                _fh, _fw = frame_bgr.shape[:2]
                _ch, _cw = rgb_crop.shape[:2]
                crop_area_frac = (_ch * _cw) / max(_fh * _fw, 1)

                life_info = life_clf.classify(
                    rgb_crop=rgb_crop,
                    yolo_class=b["label"],
                    dim_idx=dim_idx,
                    crop_area_frac=crop_area_frac,
                )

                if life_info["label"] == "living":
                    n_living += 1
                else:
                    n_nonliving += 1

                n_2d += dim_idx == 0
                n_3d += dim_idx == 1

                draw_box(out_frame, x0, y0, x1, y1,
                         dim_idx, dim_conf, b["label"], cat_conf, b["label"],
                         life_label=life_info["label"],
                         life_conf=life_info["clip_conf"])   # ← key matches classifier

            draw_depth_separate(depth_map, out_frame.shape)

            elapsed    = time.time() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(elapsed, 1e-6)
            draw_hud(out_frame, fps_smooth, n_2d, n_3d, paused,
                     n_living=n_living, n_nonliving=n_nonliving)

            if writer:
                writer.write(out_frame)
            frame_idx += 1

        if out_frame is not None:
            cv2.imshow("Smart Depth Vision", out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s") and out_frame is not None:
            sp = str(ROOT / "inference" / f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(sp, out_frame)
            print(f"  Saved: {sp}")
        elif key == ord(" "):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("  Done.")


def process_single_image(frame_bgr, yolo, midas, transform, classifier,
                          life_clf, custom_depth=None):
    yolo_conf = CFG["models"]["yolo"]["conf"]
    img_size  = CFG["dataset"]["image_size"]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results   = yolo(frame_bgr, verbose=False, conf=yolo_conf)[0]

    if custom_depth is not None:
        print(f"  Using custom depth map: {custom_depth}")
        depth_map = _read_depth_image(custom_depth)
    else:
        depth_map = get_depth_map(frame_rgb, midas, transform)

    all_boxes, screen_boxes = collect_boxes(results, yolo, frame_bgr.shape)
    out_frame = frame_bgr.copy()
    n_2d = n_3d = 0
    n_living = n_nonliving = 0

    for b in all_boxes:
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        rgb_crop   = frame_rgb[y0:y1, x0:x1]
        depth_crop = depth_map[y0:y1, x0:x1]
        if rgb_crop.size == 0 or depth_crop.size == 0:
            continue

        # Same guard as webcam loop: screen devices must not be checked
        # against their own entry in screen_boxes.
        if b["label"] in SCREEN_LABELS:
            inside_screen = False
        else:
            inside_screen = is_inside_screen(
                (x0, y0, x1, y1), screen_boxes, overlap_threshold=0.50)

        dim_idx, dim_conf, cat_name, cat_conf = classify_crop(
            rgb_crop, depth_crop, classifier, img_size,
            force_2d=inside_screen,
            frame_shape=frame_bgr.shape)

        # ── living / non-living via CLIP ────────────────────────────────────
        _fh, _fw = frame_bgr.shape[:2]
        _ch, _cw = rgb_crop.shape[:2]
        crop_area_frac = (_ch * _cw) / max(_fh * _fw, 1)

        life_info = life_clf.classify(
            rgb_crop=rgb_crop,
            yolo_class=b["label"],
            dim_idx=dim_idx,
            crop_area_frac=crop_area_frac,
        )

        if life_info["label"] == "living":
            n_living += 1
        else:
            n_nonliving += 1

        n_2d += dim_idx == 0
        n_3d += dim_idx == 1

        draw_box(out_frame, x0, y0, x1, y1,
                 dim_idx, dim_conf, b["label"], cat_conf, b["label"],
                 life_label=life_info["label"],
                 life_conf=life_info["clip_conf"])             # ← key matches classifier

    draw_depth_separate(depth_map, out_frame.shape)
    draw_hud(out_frame, 0, n_2d, n_3d, False,
             n_living=n_living, n_nonliving=n_nonliving)

    out_path = str(ROOT / "inference" / "result.jpg")
    cv2.imwrite(out_path, out_frame)
    print(f"  Result saved: {out_path}")

    h, w = out_frame.shape[:2]
    if w > 900:
        s = 900 / w
        out_frame = cv2.resize(out_frame, (int(w*s), int(h*s)))

    cv2.imshow("Smart Depth Vision — Result", out_frame)
    print("  Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="0=webcam, or path to video/image file")
    parser.add_argument("--depth", default=None,
                        help="Path to depth map PNG (optional)")
    parser.add_argument("--save", action="store_true",
                        help="Save output video to inference/output.mp4")
    args = parser.parse_args()

    source = args.source
    if source != 0:
        try:
            source = int(source)
        except ValueError:
            pass

    run(source=source, save_output=args.save, custom_depth=args.depth)