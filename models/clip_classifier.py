"""
CLIP-based Living/Non-Living Classifier
Distinguishes real living objects from 2D images (photos, posters, screens)

Fix history:
  v1 — 9 non-living vs 4 living prompts, scores summed after softmax.
       Non-living won by sheer count. Real people → NON-LIVING.

  v2 — Balanced to 4 vs 4 prompts, switched to .mean() scoring.
       Labels became correct but confidence stayed low (15–25%) because
       softmax was still spread over 8 prompts (~12.5% base each).

  v3 — Binary aggregate embeddings: average all living prompts into one
       vector, all non-living into one, then softmax over just 2 vectors.
       Confidence numbers fixed. But real people (3D, large crop) were
       STILL labelled NON-LIVING because of a deeper architectural flaw.

  v4 — THIS FILE: correct architectural fix.

  ROOT CAUSE (v1–v3):
      A webcam frame of a real person is visually identical to a
      photograph of a person — it IS a still frame.  CLIP was trained
      on (image, caption) pairs from the internet where every image is
      already a photo, so it correctly identifies every face crop as
      "a photograph of a person" and scores it non-living.  CLIP cannot
      distinguish "live subject being filmed" from "photo of a subject"
      from pixel content alone.

  CORRECT ARCHITECTURE:
      The depth classifier and YOLO are the right signals for large,
      close-up subjects.  CLIP adds noise, not signal, in that regime.
      CLIP is only useful for SMALL detections (potential photo-in-scene:
      a picture on a shelf, a poster, a tiny face on a phone screen whose
      screen-overlap check was not triggered).

      Decision table:
      ┌─────────────┬────────────────┬──────────────────────┬────────────┐
      │  dim_idx    │  COCO class    │  crop_area_frac      │  Decision  │
      ├─────────────┼────────────────┼──────────────────────┼────────────┤
      │  0  (2D)    │  any           │  any                 │ NON-LIVING │
      │  1  (3D)    │  non-living    │  any                 │ NON-LIVING │
      │  1  (3D)    │  living        │  > LARGE_CROP_THRESH │ LIVING     │
      │             │                │  (bypass CLIP)       │            │
      │  1  (3D)    │  living        │  <= LARGE_CROP_THRESH│ CLIP vote  │
      │             │                │  CLIP conf >= thresh │ NON-LIVING │
      │             │                │  CLIP conf <  thresh │ LIVING     │
      └─────────────┴────────────────┴──────────────────────┴────────────┘

  LARGE_CROP_THRESH = 0.15  (crop covers >15% of frame area)
      At that size a real person in front of the camera is the overwhelm-
      ing statistical explanation; a photo-in-scene is very unlikely to
      fill 15%+ of the frame.

  CLIP_OVERRIDE_THRESHOLD = 0.80  (for small crops only)
      Raised from 0.70 so CLIP must be strongly confident before it can
      override the 3D + living verdict even for small detections.
"""

import torch
import clip
from PIL import Image
import numpy as np


# Crop must cover > this fraction of the frame to bypass CLIP entirely.
LARGE_CROP_THRESH = 0.15

# CLIP must reach this confidence (non-living side) to override a
# 3D + living-class verdict for SMALL crops.
CLIP_OVERRIDE_THRESHOLD = 0.80


class LivingNonLivingClassifier:
    def __init__(self, device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLIP] Loading model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # Prompts focused tightly on the real-vs-2D distinction.
        # Count does not affect scores (we average each group into one
        # embedding before comparing), so add as many as useful.
        living_prompts = [
            "a real person physically present in the scene",
            "a living human being captured on camera in real life",
            "a real live animal in three dimensions",
            "a live human or animal in the physical world",
            "a person standing or sitting in front of a camera",
        ]
        non_living_prompts = [
            "a photo of a person displayed on a phone or monitor screen",
            "a picture or image of a person, not a real person",
            "a digital or printed photograph of a human being",
            "a video or image of a person shown on a screen",
            "an inanimate object, furniture, or electronic device",
        ]

        self.coco_living_classes = {
            "person", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "bird",
        }

        # Build ONE aggregate embedding per category.
        # Softmax over 2 vectors → true 0–100% probability, no dilution.
        with torch.no_grad():
            def _encode_mean(prompts):
                tokens   = clip.tokenize(prompts).to(self.device)
                features = self.model.encode_text(tokens)   # (N, D)
                mean_vec = features.mean(dim=0)             # (D,)
                return mean_vec / mean_vec.norm()

            living_vec     = _encode_mean(living_prompts)
            non_living_vec = _encode_mean(non_living_prompts)
            # index 0 = living, index 1 = non-living
            self.text_features = torch.stack([living_vec, non_living_vec])

        print("[CLIP] Ready.")

    # ── Raw CLIP binary score ──────────────────────────────────────────────
    def classify_crop(self, rgb_crop: np.ndarray):
        """
        Binary CLIP classification against the two aggregate vectors.

        Args:
            rgb_crop: numpy RGB array (H, W, 3) — pipeline already converts
                      BGR→RGB before passing crops here.

        Returns:
            (label, confidence)   label ∈ {'living', 'non-living'}
                                  confidence ∈ [0.0, 1.0]
        """
        if rgb_crop is None or rgb_crop.size == 0:
            return "non-living", 0.0

        pil_image   = Image.fromarray(rgb_crop)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features  = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (100.0 * image_features @ self.text_features.T)  # (1, 2)
            probs  = logits.softmax(dim=-1)[0].cpu().numpy()          # (2,)

        living_prob, non_living_prob = float(probs[0]), float(probs[1])

        if living_prob >= non_living_prob:
            return "living", living_prob
        else:
            return "non-living", non_living_prob

    # ── Combined decision ──────────────────────────────────────────────────
    def classify(self, rgb_crop: np.ndarray, yolo_class: str,
                 dim_idx: int, crop_area_frac: float = 0.0) -> dict:
        """
        Final living / non-living decision.

        Args:
            rgb_crop       : cropped RGB numpy array (H x W x 3)
            yolo_class     : YOLO/COCO label, e.g. "person", "chair"
            dim_idx        : 0 = 2D, 1 = 3D  (from DepthAwareClassifier)
            crop_area_frac : (crop_w * crop_h) / (frame_w * frame_h)
                             Used to decide whether to invoke CLIP at all.

        Returns:
            dict with keys: label, clip_conf, clip_label, reason
        """
        coco_says_living = yolo_class.lower() in self.coco_living_classes

        # ── Rule 1: 2D detection → always non-living ──────────────────────
        if dim_idx == 0:
            clip_label, clip_conf = self.classify_crop(rgb_crop)
            return {
                "label":      "non-living",
                "clip_conf":  clip_conf,
                "clip_label": clip_label,
                "reason":     (f"2D detection — '{yolo_class}' is a flat "
                               f"image, not a living being"),
            }

        # ── Rule 2: 3D + non-living COCO class → always non-living ────────
        if not coco_says_living:
            clip_label, clip_conf = self.classify_crop(rgb_crop)
            return {
                "label":      "non-living",
                "clip_conf":  clip_conf,
                "clip_label": clip_label,
                "reason":     f"'{yolo_class}' is an inherently non-living COCO class",
            }

        # ── Rule 3: 3D + living COCO class ────────────────────────────────
        #
        # LARGE crop (> LARGE_CROP_THRESH of frame):
        #   Bypass CLIP entirely. A detection this large is statistically
        #   a real close-up subject, not a photo-in-scene.
        #   CLIP cannot distinguish a live video frame from a photograph,
        #   so invoking it here only introduces noise.
        #
        # SMALL crop (<= LARGE_CROP_THRESH):
        #   Could be a photo on a shelf, poster, or small screen image
        #   whose overlap with screen_boxes was not triggered.
        #   Consult CLIP, but require high confidence to override.

        if crop_area_frac > LARGE_CROP_THRESH:
            # Trust depth + YOLO; run CLIP only to populate clip_conf for HUD
            clip_label, clip_conf = self.classify_crop(rgb_crop)
            return {
                "label":      "living",
                "clip_conf":  clip_conf,
                "clip_label": clip_label,
                "reason":     (
                    f"Large 3D '{yolo_class}' ({crop_area_frac*100:.0f}% of frame) "
                    f"— depth+YOLO trusted, CLIP bypassed "
                    f"(CLIP said {clip_label} {clip_conf*100:.0f}%)"),
            }

        # Small 3D living-class crop — ask CLIP
        clip_label, clip_conf = self.classify_crop(rgb_crop)

        if clip_label == "living":
            final_label = "living"
            reason = (f"Small 3D '{yolo_class}' confirmed living by CLIP "
                      f"({clip_conf*100:.0f}%)")

        elif clip_conf >= CLIP_OVERRIDE_THRESHOLD:
            # CLIP very confident this is a photo-in-scene
            final_label = "non-living"
            reason = (f"CLIP strongly indicates 2D representation of "
                      f"'{yolo_class}' (conf {clip_conf*100:.0f}% >= "
                      f"threshold {CLIP_OVERRIDE_THRESHOLD*100:.0f}%)")
        else:
            # CLIP uncertain — trust depth + YOLO
            final_label = "living"
            reason = (f"Small 3D '{yolo_class}': CLIP leaned non-living "
                      f"({clip_conf*100:.0f}%) but below override threshold "
                      f"({CLIP_OVERRIDE_THRESHOLD*100:.0f}%) — treating as living")

        return {
            "label":      final_label,
            "clip_conf":  clip_conf,
            "clip_label": clip_label,
            "reason":     reason,
        }