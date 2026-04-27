"""
app/database.py
─────────────────────────────────────────────────────────
SQLAlchemy ORM models + SQLite session management.
Database file is stored at: ./spoof_detection.db
(relative to wherever you run uvicorn from — project root)
"""

import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    Integer, String, Text, create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# ── Database location ──────────────────────────────────────────────────────────
# Store the DB file one level up from app/ (i.e. in the project root)
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "spoof_detection.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ─────────────────────────────────────────────────────────────────────

class User(Base):
    """Registered user with face embeddings for identity verification."""
    __tablename__ = "users"

    id                = Column(Integer, primary_key=True, index=True)
    username          = Column(String(64), unique=True, index=True, nullable=False)
    email             = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password   = Column(String(256), nullable=False)

    # JSON-serialized list of face embedding vectors (list of list[float])
    # Each enrollment session adds 3-4 embeddings; multiple sessions accumulate.
    face_embeddings   = Column(Text, default="[]", nullable=False)

    registered_at     = Column(DateTime, default=datetime.datetime.utcnow)
    last_login        = Column(DateTime, nullable=True)
    is_active         = Column(Boolean, default=True)

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"


class SpoofLog(Base):
    """Audit log for every verification attempt."""
    __tablename__ = "spoof_logs"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, nullable=True)   # NULL = unknown / anonymous attempt
    timestamp       = Column(DateTime, default=datetime.datetime.utcnow)

    # Per-model verdicts
    depth_verdict   = Column(String(8), nullable=True)   # "2D" | "3D"
    depth_std       = Column(Float, nullable=True)
    clip_verdict    = Column(String(16), nullable=True)  # "real" | "spoof"
    clip_confidence = Column(Float, nullable=True)
    yolo_person_conf= Column(Float, nullable=True)

    # Final scores
    combined_score  = Column(Float, nullable=True)   # 0-1, higher = more real
    is_spoof        = Column(Boolean, nullable=True)
    face_matched    = Column(Boolean, nullable=True)

    def __repr__(self):
        return (
            f"<SpoofLog id={self.id} user_id={self.user_id} "
            f"is_spoof={self.is_spoof} ts={self.timestamp}>"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables (idempotent — safe to call on every startup)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it on exit."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()