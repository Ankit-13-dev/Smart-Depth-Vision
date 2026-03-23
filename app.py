"""
app.py
───────
Flask backend for Smart Depth Vision webpage.
Browser sends webcam frames → model predicts → JSON response back.

Run:
    python app.py

Then open: http://localhost:5000
"""

import cv2
import torch
import numpy as np
import base64
import yaml
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT     = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORIES = CFG["dataset"]["coco_categories"]
WEIGHTS    = ROOT / CFG["paths"]["weights"] / "best_model.pt"
IMG_SIZE   = CFG["dataset"]["image_size"]

sys.path.insert(0, str(ROOT))
from models.classifier import build_model

app = Flask(__name__, static_folder=str(ROOT), static_url_path='')
CORS(app)

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading models...")

from ultralytics import YOLO
yolo_model = YOLO(CFG["models"]["yolo"]["variant"])
print("  ✓ YOLOv8 loaded")

import torch.nn.functional as F
midas     = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
midas     = midas.to(DEVICE).eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = transforms.dpt_transform
print("  ✓ MiDaS loaded")

classifier = build_model(
    num_classes=CFG["dataset"]["num_classes"],
    dropout=0.0,
    device=DEVICE
)
ckpt = torch.load(str(WEIGHTS), map_location=DEVICE)
classifier.load_state_dict(ckpt["model"])
classifier.eval()
print(f"  ✓ Classifier loaded (epoch {ckpt['epoch']})")
print(f"\n  Running on: {DEVICE}")
print(f"  Open browser: http://localhost:5000\n")


# ── Depth helpers ─────────────────────────────────────────────────────────────
@torch.no_grad()
def get_depth(rgb_frame):
    inp  = midas_transform(rgb_frame).to(DEVICE)
    pred = midas(inp)
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=rgb_frame.shape[:2],
        mode="bicubic", align_corners=False
    ).squeeze()
    d = pred.cpu().numpy()
    mn, mx = d.min(), d.max()
    if mx - mn > 1e-8:
        d = (d - mn) / (mx - mn)
    return d.astype(np.float32)


@torch.no_grad()
def classify_crop(rgb_crop, depth_crop):
    h, w   = rgb_crop.shape[:2]
    scale  = IMG_SIZE / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    rgb_r  = cv2.resize(rgb_crop,   (nw, nh))
    dep_r  = cv2.resize(depth_crop, (nw, nh))
    ph, pw = IMG_SIZE-nh, IMG_SIZE-nw
    rgb_r  = cv2.copyMakeBorder(rgb_r, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    dep_r  = cv2.copyMakeBorder(dep_r, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0.0)

    rgb_t  = torch.from_numpy(rgb_r.transpose(2,0,1)).float()/255.0
    mean   = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std    = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    rgb_t  = (rgb_t - mean) / std
    dep_t  = torch.from_numpy(dep_r).float().unsqueeze(0)
    dep_t  = (dep_t - 0.5) / 0.5
    x      = torch.cat([rgb_t, dep_t], dim=0).unsqueeze(0).to(DEVICE)

    bin_out, cat_out = classifier(x)
    bin_prob = torch.softmax(bin_out, dim=1)[0]
    cat_prob = torch.softmax(cat_out, dim=1)[0]

    # Depth variance override
    depth_std = float(depth_crop.std())
    if depth_std > 0.08:
        dim_idx, dim_conf = 1, min(0.99, depth_std * 5)
    elif depth_std < 0.025:
        dim_idx, dim_conf = 0, min(0.99, (0.025 - depth_std) * 40)
    else:
        dim_idx  = bin_prob.argmax().item()
        dim_conf = bin_prob[dim_idx].item()

    cat_idx  = cat_prob.argmax().item()
    cat_conf = cat_prob[cat_idx].item()

    return {
        "dim":      "3D" if dim_idx == 1 else "2D",
        "dim_conf": round(dim_conf * 100, 1),
        "cat":      CATEGORIES[cat_idx],
        "cat_conf": round(cat_conf * 100, 1),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
             return send_from_directory(r"C:\Users\aatri\OneDrive\Desktop\coding\webpages", 'smart_depth_vision.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data      = request.json
        img_b64   = data['frame'].split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        img_np    = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return jsonify({"error": "Could not decode frame"}), 400

        # Resize for speed (max 640px wide)
        h, w = frame_bgr.shape[:2]
        if w > 640:
            scale     = 640 / w
            frame_bgr = cv2.resize(frame_bgr, (640, int(h*scale)))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # YOLOv8
        results  = yolo_model(frame_bgr, verbose=False,
                               conf=CFG["models"]["yolo"]["conf"])[0]

        # MiDaS depth
        depth_map = get_depth(frame_rgb)

        # Depth map as base64 for webpage display
        depth_vis   = (depth_map * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
        depth_small = cv2.resize(depth_color, (200, 150))
        _, depth_enc = cv2.imencode('.jpg', depth_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        depth_b64 = 'data:image/jpeg;base64,' + base64.b64encode(depth_enc).decode()

        detections = []
        ih, iw     = frame_bgr.shape[:2]

        for box in results.boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(iw, x1); y1 = min(ih, y1)

            if (x1-x0) < 20 or (y1-y0) < 20:
                continue

            rgb_crop   = frame_rgb[y0:y1, x0:x1]
            depth_crop = depth_map[y0:y1, x0:x1]

            if rgb_crop.size == 0 or depth_crop.size == 0:
                continue

            result    = classify_crop(rgb_crop, depth_crop)
            yolo_name = yolo_model.names[int(box.cls)]

            detections.append({
                "x0":       round(x0 / iw, 4),
                "y0":       round(y0 / ih, 4),
                "x1":       round(x1 / iw, 4),
                "y1":       round(y1 / ih, 4),
                "dim":      result["dim"],
                "dim_conf": result["dim_conf"],
                "cat":      yolo_name,
                "cat_conf": round(float(box.conf) * 100, 1),
            })

        return jsonify({
            "detections": detections,
            "depth_img":  depth_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
