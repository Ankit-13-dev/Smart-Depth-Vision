"""
app/spoof_detector.py
─────────────────────────────────────────────────────────
Core anti-spoofing pipeline using your three models:

Pipeline for a single frame:
  1. YOLOv8   → detect persons + bounding boxes + confidence
  2. MiDaS    → compute depth map → depth std over face region
  3. CLIP     → image-text similarity: "real face" vs "spoof face"
  4. Fusion   → weighted vote → final is_real verdict

Scoring logic:
  • depth_score : based on depth std of the detected face crop
      std > 0.08  → 3D real   (high score)
      std < 0.025 → 2D spoof  (low score)
      else        → interpolate
  • clip_score  : cosine similarity with "real face" text prompt
      vs "spoof / printed / screen" prompts
  • yolo_score  : confidence that a "person" was detected
  • combined    : 0.45 * depth + 0.35 * clip + 0.20 * yolo
  • is_real     : combined >= 0.50  AND  yolo detected a person
"""

import base64
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SpoofDetector:
    """Stateless detector — pass the models dict from models_loader.get_models()."""

    # Fusion weights (must sum to 1.0)
    W_DEPTH = 0.45
    W_CLIP  = 0.35
    W_YOLO  = 0.20

    # Decision threshold
    REAL_THRESHOLD = 0.50

    def __init__(self, models: dict):
        self.m = models
        self.device = models["device"]
        self.cfg = models["cfg"]

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """
        Full pipeline for one BGR frame.

        Returns:
          {
            "yolo_detections": [...],
            "depth_img": "data:image/jpeg;base64,...",   ← colorized depth for display
            "spoof_result": {
                "is_real": bool,
                "depth_verdict": "3D"|"2D",
                "depth_std": float,
                "depth_score": float,
                "clip_verdict": "real"|"spoof",
                "clip_score": float,
                "yolo_person_conf": float,
                "yolo_score": float,
                "combined_score": float,
            }
          }
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        # ── Step 1: YOLOv8 detection ──────────────────────────────────────────
        yolo_conf = self.cfg["models"]["yolo"]["conf"]
        results = self.m["yolo"](frame_bgr, verbose=False, conf=yolo_conf)[0]

        yolo_detections = []
        best_person_conf = 0.0
        best_person_box  = None   # (x0,y0,x1,y1) of the largest/most-confident person

        for box in results.boxes:
            cls_idx   = int(box.cls[0])
            cls_name  = self.m["yolo"].names[cls_idx]
            conf      = float(box.conf[0])
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)

            yolo_detections.append({
                "label": cls_name,
                "confidence": round(conf * 100, 1),
                "x0": round(x0 / w, 4),
                "y0": round(y0 / h, 4),
                "x1": round(x1 / w, 4),
                "y1": round(y1 / h, 4),
            })

            if cls_name == "person" and conf > best_person_conf:
                best_person_conf = conf
                best_person_box  = (x0, y0, x1, y1)

        # ── Step 2: MiDaS depth map ────────────────────────────────────────────
        depth_map = self._compute_depth(frame_rgb)

        # Build colorized depth image for frontend
        depth_img_b64 = self._depth_to_b64(depth_map)

        # Depth analysis on the face/person region
        if best_person_box:
            x0, y0, x1, y1 = best_person_box
            # Focus on the upper third of the person box (face region)
            face_y1 = y0 + (y1 - y0) // 3
            depth_crop = depth_map[y0:face_y1, x0:x1]
        else:
            # No person detected — analyze the full frame center
            cy, cx = h // 2, w // 2
            depth_crop = depth_map[
                max(0, cy - 80):min(h, cy + 80),
                max(0, cx - 80):min(w, cx + 80)
            ]

        depth_std   = float(depth_crop.std()) if depth_crop.size > 0 else 0.0
        depth_score = self._depth_std_to_score(depth_std)
        depth_verdict = "3D" if depth_score >= 0.5 else "2D"

        # ── Step 3: CLIP similarity ────────────────────────────────────────────
        clip_score, clip_verdict = self._clip_score(frame_rgb, best_person_box)

        # ── Step 4: Fusion ─────────────────────────────────────────────────────
        yolo_score = float(best_person_conf)   # 0-1
        combined = (
            self.W_DEPTH * depth_score +
            self.W_CLIP  * clip_score  +
            self.W_YOLO  * yolo_score
        )

        # Must have detected a person AND combined score above threshold
        is_real = (best_person_conf > 0.3) and (combined >= self.REAL_THRESHOLD)

        return {
            "yolo_detections": yolo_detections,
            "depth_img": depth_img_b64,
            "spoof_result": {
                "is_real":          is_real,
                "depth_verdict":    depth_verdict,
                "depth_std":        round(depth_std, 4),
                "depth_score":      round(depth_score, 3),
                "clip_verdict":     clip_verdict,
                "clip_score":       round(clip_score, 3),
                "yolo_person_conf": round(best_person_conf, 3),
                "yolo_score":       round(yolo_score, 3),
                "combined_score":   round(combined, 3),
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_depth(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Run MiDaS and return normalized [0,1] float32 depth map."""
        inp  = self.m["midas_transform"](frame_rgb).to(self.device)
        pred = self.m["midas"](inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        d = pred.cpu().numpy().astype(np.float32)
        mn, mx = d.min(), d.max()
        if mx - mn > 1e-8:
            d = (d - mn) / (mx - mn)
        return d

    @staticmethod
    def _depth_std_to_score(std: float) -> float:
        """
        Map depth standard deviation to a [0,1] 'realness' score.
        High std  →  lots of depth variation  →  3D real face → score near 1
        Low  std  →  flat depth              →  2D spoof      → score near 0
        """
        HIGH_STD = 0.08   # clearly 3D
        LOW_STD  = 0.020  # clearly 2D
        if std >= HIGH_STD:
            return min(1.0, 0.75 + (std - HIGH_STD) * 3.0)
        if std <= LOW_STD:
            return max(0.0, std / LOW_STD * 0.25)
        # linear interpolation in the ambiguous zone
        t = (std - LOW_STD) / (HIGH_STD - LOW_STD)
        return 0.25 + t * 0.50   # maps to [0.25, 0.75]

    @torch.no_grad()
    def _clip_score(
        self,
        frame_rgb: np.ndarray,
        person_box: tuple | None,
    ) -> tuple[float, str]:
        """
        Run CLIP on the face/person crop.
        Returns (score_0_to_1, "real"|"spoof")
        score = cosine-similarity with 'real' prompt cluster.
        """
        # Crop to person if available
        if person_box:
            x0, y0, x1, y1 = person_box
            h_box = y1 - y0
            # Focus on upper half (face region)
            crop = frame_rgb[y0:y0 + h_box // 2, x0:x1]
            if crop.size == 0:
                crop = frame_rgb
        else:
            crop = frame_rgb

        pil_img = Image.fromarray(crop)
        img_tensor = self.m["clip_preprocess"](pil_img).unsqueeze(0).to(self.device)
        img_feat = self.m["clip_model"].encode_image(img_tensor).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        real_sim  = float((img_feat @ self.m["real_text_feat"].T).squeeze())
        spoof_sim = float((img_feat @ self.m["spoof_text_feat"].T).squeeze())

        # Softmax to get probability
        logits = torch.tensor([real_sim, spoof_sim]) * 100.0  # temperature scale
        probs  = torch.softmax(logits, dim=0)
        real_prob = float(probs[0])

        verdict = "real" if real_prob >= 0.50 else "spoof"
        return real_prob, verdict

    @staticmethod
    def _depth_to_b64(depth_map: np.ndarray) -> str:
        """Convert float32 depth map to colorized base64 JPEG for frontend display."""
        vis = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
        small = cv2.resize(colored, (240, 180))
        _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return "data:image/jpeg;base64," + base64.b64encode(enc).decode()