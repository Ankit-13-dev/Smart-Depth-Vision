"""
app/models_loader.py
─────────────────────────────────────────────────────────
Singleton loader for all heavy ML models.
Called once at startup — subsequent calls return the cached dict.

Models loaded:
  • YOLOv8s     — object detection (ultralytics)
  • MiDaS DPT_Hybrid — monocular depth estimation (torch.hub)
  • CLIP ViT-B/32    — image-text similarity for spoof classification
"""

import sys
from pathlib import Path

import torch
import yaml

_models: dict | None = None   # module-level cache


def get_models() -> dict:
    """Return the loaded model bundle, loading once on first call."""
    global _models
    if _models is None:
        _models = _load_all()
    return _models


def _load_all() -> dict:
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[models_loader] device = {device}")

    # ── YOLOv8 ────────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    yolo_variant = cfg["models"]["yolo"]["variant"]
    yolo = YOLO(yolo_variant)
    print(f"  ✓ YOLOv8 ({yolo_variant}) loaded")

    # ── MiDaS ─────────────────────────────────────────────────────────────────
    # Using DPT_Hybrid — good quality / speed tradeoff (vs DPT_Large)
    midas = torch.hub.load(
        "intel-isl/MiDaS", "DPT_Hybrid",
        pretrained=True, trust_repo=True
    )
    midas = midas.to(device).eval()

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", trust_repo=True
    )
    midas_transform = midas_transforms.dpt_transform
    print("  ✓ MiDaS DPT_Hybrid loaded")

    # ── CLIP ──────────────────────────────────────────────────────────────────
    try:
        import clip as openai_clip
    except ImportError:
        raise ImportError(
            "openai-clip not found. Install it with: pip install git+https://github.com/openai/CLIP.git"
        )

    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print("  ✓ CLIP ViT-B/32 loaded")

    # ── Pre-encode CLIP text prompts (done once, reused every frame) ──────────
    # These text embeddings represent the two classes: REAL vs SPOOF
    REAL_PROMPTS = [
        "a real human face looking at the camera",
        "a live person's face",
        "a genuine living human face",
    ]
    SPOOF_PROMPTS = [
        "a printed photograph of a face",
        "a face displayed on a phone or monitor screen",
        "a face mask or mannequin",
        "a flat 2D image of a face",
    ]

    import clip as openai_clip  # re-import for tokenize
    with torch.no_grad():
        real_tokens  = openai_clip.tokenize(REAL_PROMPTS).to(device)
        spoof_tokens = openai_clip.tokenize(SPOOF_PROMPTS).to(device)
        real_text_features  = clip_model.encode_text(real_tokens).float()
        spoof_text_features = clip_model.encode_text(spoof_tokens).float()
        # Average the prompts for each class
        real_text_features  = real_text_features.mean(dim=0, keepdim=True)
        spoof_text_features = spoof_text_features.mean(dim=0, keepdim=True)
        # Normalize
        real_text_features  = real_text_features  / real_text_features.norm(dim=-1, keepdim=True)
        spoof_text_features = spoof_text_features / spoof_text_features.norm(dim=-1, keepdim=True)

    print("  ✓ CLIP text embeddings pre-computed")
    print(f"\n[models_loader] All models ready on [{device}]\n")

    return {
        "yolo":               yolo,
        "midas":              midas,
        "midas_transform":    midas_transform,
        "clip_model":         clip_model,
        "clip_preprocess":    clip_preprocess,
        "real_text_feat":     real_text_features,
        "spoof_text_feat":    spoof_text_features,
        "device":             device,
        "cfg":                cfg,
    }