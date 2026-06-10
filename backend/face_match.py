"""Local face embedding + similarity helper for SPL stage-2 search.

Uses InsightFace `buffalo_l` (CPU). Embeddings are 512-d float32. Cosine
similarity is the score everyone compares against. ~150 ms per embed
on the Hetzner CPX32 once the model is warm.

Disk cache lives at /var/www/spl-tool/backend/face_cache/<sha256>.npy
keyed by the SHA256 of the avatar URL (URLs themselves are stable
enough — when scrapers refresh an avatar they typically write to a new
URL).

Built 2026-05-29 for the stage-2 "This is the person, now search again"
flow. No external API calls; cost per embed is local CPU only.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from typing import Optional

import numpy as np

_FACE_APP = None
_FACE_APP_LOCK = threading.Lock()

CACHE_DIR = os.environ.get(
    "FACE_CACHE_DIR", "/var/www/spl-tool/backend/face_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)


def _app():
    """Lazy-load the InsightFace app on first use (~5s cold start)."""
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP
    with _FACE_APP_LOCK:
        if _FACE_APP is not None:
            return _FACE_APP
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
    return _FACE_APP


def _cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.npy")


def _decode(image_bytes: bytes):
    import cv2
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def embed_url(url: str, image_bytes: Optional[bytes] = None) -> Optional[np.ndarray]:
    """Return a 512-d L2-normalised embedding for the largest face in the
    image at `url`, or None if no face is detected / bytes unavailable.

    Pass `image_bytes` to avoid a network refetch when the caller has
    already pulled the avatar (typical).
    """
    if not url:
        return None
    cp = _cache_path(url)
    if os.path.exists(cp):
        try:
            return np.load(cp)
        except Exception:
            pass

    if image_bytes is None:
        try:
            import httpx
            r = httpx.get(url, timeout=8, follow_redirects=True)
            if r.status_code != 200 or not r.content:
                return None
            image_bytes = r.content
        except Exception:
            return None

    img = _decode(image_bytes)
    if img is None:
        return None

    faces = _app().get(img)
    if not faces:
        return None
    # Take the highest-confidence face if more than one is detected.
    faces.sort(key=lambda f: f.det_score, reverse=True)
    emb = faces[0].embedding.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    try:
        np.save(cp, emb)
    except Exception:
        pass
    return emb


def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is None.

    Inputs are expected to be already L2-normalised (embed_url normalises),
    so this is just a dot product.
    """
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def mean_sim_to_anchors(candidate_emb, anchor_embs) -> float:
    """Mean cosine similarity from one candidate embedding to a list of
    anchor embeddings (the confirmed faces).
    """
    if candidate_emb is None or not anchor_embs:
        return 0.0
    sims = [cosine_sim(candidate_emb, a) for a in anchor_embs if a is not None]
    if not sims:
        return 0.0
    return float(np.mean(sims))


def warm_up() -> dict:
    """Force model load + a tiny embed pass. Call at app startup so the
    first real request doesn't pay the 5-second cold-start tax.
    """
    t0 = time.time()
    app = _app()
    # Dummy 64x64 grey image — won't have a face, but exercises the model
    # graph + warms the ONNX runtime caches.
    dummy = np.full((64, 64, 3), 128, dtype=np.uint8)
    app.get(dummy)
    return {"warm_ms": int((time.time() - t0) * 1000)}
