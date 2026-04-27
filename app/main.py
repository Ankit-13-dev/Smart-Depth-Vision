"""
app/main.py
─────────────────────────────────────────────────────────
Smart Depth Vision — Spoof Detection API  (replaces old app.py)

Endpoints:
  POST /api/register         — new user + face enrollment
  POST /api/login            — password auth → JWT
  POST /api/enroll-face      — add face images to existing user
  POST /api/verify           — spoof check + face match (protected)
  POST /api/analyze-frame    — live webcam spoof analysis (no auth needed)
  GET  /api/logs             — fetch logs for logged-in user
  GET  /api/me               — current user info
  GET  /                     — serve the frontend SPA

Run locally:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import base64, json, os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (
    Depends, FastAPI, File, Form, HTTPException,
    UploadFile, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token, get_current_user,
    get_password_hash, verify_password,
)
from app.database import SpoofLog, User, get_db, init_db
from app.face_utils import FaceUtils
from app.models_loader import get_models
from app.spoof_detector import SpoofDetector

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Depth Vision — Spoof Detection",
    description="Face anti-spoofing using YOLOv8 + MiDaS + CLIP + DeepFace",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize DB and preload all heavy models once."""
    init_db()
    print("✓ Database initialized")
    get_models()   # loads YOLOv8 + MiDaS + CLIP into memory
    print("✓ All models loaded and ready")

# ── Serve frontend ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

# ── Pydantic schemas ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class FrameRequest(BaseModel):
    """Base64-encoded frame from webcam."""
    frame: str   # data:image/jpeg;base64,...

class VerifyRequest(BaseModel):
    frame: str   # base64 webcam frame
    username: Optional[str] = None   # if None, face-match against all users

# ── Helpers ────────────────────────────────────────────────────────────────────

def decode_b64_image(b64_str: str) -> np.ndarray:
    """Decode a base64 data-URL or raw base64 string to BGR numpy array."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image from base64 string")
    return frame


def _resize_for_inference(frame: np.ndarray, max_w: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
    return frame

# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/register", summary="Register a new user (no face yet)")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=get_password_hash(req.password),
        face_embeddings=json.dumps([]),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.username})
    return {
        "message": "Registered successfully. Now enroll your face.",
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "face_enrolled": False,
    }


@app.post("/api/login", summary="Login with username + password")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    embeddings = json.loads(user.face_embeddings or "[]")
    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "face_enrolled": len(embeddings) > 0,
    }


@app.get("/api/me", summary="Get current user info")
async def me(current_user: User = Depends(get_current_user)):
    embeddings = json.loads(current_user.face_embeddings or "[]")
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "face_enrolled": len(embeddings) > 0,
        "face_count": len(embeddings),
        "registered_at": current_user.registered_at.isoformat(),
    }

# ── Face enrollment endpoint ───────────────────────────────────────────────────

@app.post("/api/enroll-face", summary="Upload 3-4 face images to enroll")
async def enroll_face(
    images: List[UploadFile] = File(..., description="3-4 face images from different angles"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Accepts 3-4 images. Extracts face embeddings using DeepFace and stores them.
    The more angles the better — frontal, left profile, right profile, slight tilt.
    """
    if len(images) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least 2 face images.")
    if len(images) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 images allowed.")

    face_utils = FaceUtils()
    new_embeddings = []
    errors = []

    for i, img_file in enumerate(images):
        raw = await img_file.read()
        np_arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            errors.append(f"Image {i+1}: Could not decode")
            continue
        try:
            embedding = face_utils.extract_embedding(frame)
            new_embeddings.append(embedding.tolist())
        except Exception as e:
            errors.append(f"Image {i+1}: {str(e)}")

    if len(new_embeddings) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Could not detect a clear face in enough images. Errors: {errors}. "
                   "Please use well-lit, front-facing photos."
        )

    # Merge with existing embeddings (allow incremental enrollment)
    existing = json.loads(current_user.face_embeddings or "[]")
    all_embeddings = existing + new_embeddings

    user_in_db = db.query(User).filter(User.id == current_user.id).first()
    user_in_db.face_embeddings = json.dumps(all_embeddings)
    db.commit()

    return {
        "message": f"Face enrolled successfully! {len(new_embeddings)} embedding(s) stored.",
        "total_embeddings": len(all_embeddings),
        "errors": errors,
    }


@app.post("/api/enroll-face-b64", summary="Enroll face via base64 images (webcam capture)")
async def enroll_face_b64(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Same as enroll-face but accepts {"images": ["data:image/jpeg;base64,...", ...]}
    Used by the frontend webcam capture flow.
    """
    images_b64: List[str] = payload.get("images", [])
    if len(images_b64) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 images.")

    face_utils = FaceUtils()
    new_embeddings = []
    errors = []

    for i, b64 in enumerate(images_b64):
        try:
            frame = decode_b64_image(b64)
            embedding = face_utils.extract_embedding(frame)
            new_embeddings.append(embedding.tolist())
        except Exception as e:
            errors.append(f"Image {i+1}: {str(e)}")

    if len(new_embeddings) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Face detection failed on most images. {errors}"
        )

    existing = json.loads(current_user.face_embeddings or "[]")
    all_embeddings = existing + new_embeddings

    user_in_db = db.query(User).filter(User.id == current_user.id).first()
    user_in_db.face_embeddings = json.dumps(all_embeddings)
    db.commit()

    return {
        "message": f"Face enrolled! {len(new_embeddings)} new embedding(s) added.",
        "total_embeddings": len(all_embeddings),
        "errors": errors,
    }

# ── Core analysis endpoints ────────────────────────────────────────────────────

@app.post("/api/analyze-frame", summary="Analyze a single frame for spoof signals (no auth)")
async def analyze_frame(req: FrameRequest):
    """
    Real-time analysis endpoint. Called by the frontend every ~500ms from webcam.
    Returns YOLO detections, MiDaS depth map, CLIP verdict, and combined spoof score.
    """
    try:
        frame = decode_b64_image(req.frame)
        frame = _resize_for_inference(frame)
        models = get_models()
        detector = SpoofDetector(models)
        result = detector.analyze(frame)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/api/verify", summary="Full verification: spoof check + face identity match")
async def verify(
    req: VerifyRequest,
    db: Session = Depends(get_db),
):
    """
    The main verification endpoint used for access control.
    1. Runs spoof detection pipeline (YOLOv8 + MiDaS + CLIP)
    2. If not a spoof, attempts face match against stored user embeddings
    3. Returns: is_real, face_matched, matched_user, confidence, detailed scores
    """
    try:
        frame = decode_b64_image(req.frame)
        frame = _resize_for_inference(frame)
        models = get_models()
        detector = SpoofDetector(models)

        # Step 1 — spoof analysis
        analysis = detector.analyze(frame)
        spoof_result = analysis.get("spoof_result", {})
        is_real = spoof_result.get("is_real", False)

        # Step 2 — face matching (only if not flagged as spoof)
        face_matched = False
        matched_user = None
        face_distance = None

        if is_real:
            face_utils = FaceUtils()
            try:
                query_embedding = face_utils.extract_embedding(frame)

                # Determine which users to match against
                if req.username:
                    users = db.query(User).filter(User.username == req.username).all()
                else:
                    users = db.query(User).filter(User.is_active == True).all()

                best_match = None
                best_distance = float("inf")

                for user in users:
                    stored = json.loads(user.face_embeddings or "[]")
                    if not stored:
                        continue
                    distance = face_utils.match_embedding(
                        query_embedding,
                        [np.array(e) for e in stored]
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_match = user

                THRESHOLD = 0.40  # cosine distance threshold (lower = stricter)
                if best_match and best_distance < THRESHOLD:
                    face_matched = True
                    matched_user = best_match.username
                    face_distance = round(float(best_distance), 4)

            except Exception as fe:
                print(f"[WARN] Face matching error: {fe}")

        # Step 3 — log the attempt
        log = SpoofLog(
            user_id=None,
            depth_verdict=spoof_result.get("depth_verdict"),
            clip_verdict=spoof_result.get("clip_verdict"),
            combined_score=spoof_result.get("combined_score"),
            is_spoof=not is_real,
            face_matched=face_matched,
        )
        if matched_user:
            user_obj = db.query(User).filter(User.username == matched_user).first()
            if user_obj:
                log.user_id = user_obj.id
        db.add(log)
        db.commit()

        return {
            "is_real": is_real,
            "face_matched": face_matched,
            "matched_user": matched_user,
            "face_distance": face_distance,
            "access_granted": is_real and face_matched,
            "spoof_details": spoof_result,
            "detections": analysis.get("yolo_detections", []),
            "depth_img": analysis.get("depth_img"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")


@app.get("/api/logs", summary="Get spoof attempt logs for current user")
async def get_logs(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logs = (
        db.query(SpoofLog)
        .filter(SpoofLog.user_id == current_user.id)
        .order_by(SpoofLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": l.id,
            "timestamp": l.timestamp.isoformat(),
            "depth_verdict": l.depth_verdict,
            "clip_verdict": l.clip_verdict,
            "combined_score": l.combined_score,
            "is_spoof": l.is_spoof,
            "face_matched": l.face_matched,
        }
        for l in logs
    ]