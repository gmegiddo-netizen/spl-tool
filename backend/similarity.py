"""Per-avatar similarity primitives for SPL carousel triage.

Two free local signals computed once per avatar at ingest time, packed
small, shipped inline in the profile stream event. Frontend uses them
to rerank the carousel queue on every vote (confirm OR reject) with
zero network round-trip.

  pHash:        Pillow + imagehash, 64-bit perceptual hash → 8 bytes
                base64 (~12 chars). Hamming distance on the frontend
                detects "same picture even after re-encode / scale /
                light crop."

  Face embed:   Reuses face_match.embed_url (InsightFace buffalo_l,
                already disk-cached at /var/www/spl-tool/backend/
                face_cache). 512 float32 → 2048 bytes raw → ~2.7 KB
                base64. Cosine on the frontend detects "different
                photo, same person."

Built 2026-05-30 for the carousel-triage bidirectional rerank
(boost-on-confirm + penalty-on-reject) without paying any API cost.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_phash_b64(image_bytes: bytes) -> str:
    """64-bit perceptual hash → 8-byte little-endian → base64. Returns
    "" on any failure (caller should treat empty as "no pHash signal").
    """
    if not image_bytes:
        return ""
    try:
        from PIL import Image
        import imagehash
        img = Image.open(io.BytesIO(image_bytes))
        h = imagehash.phash(img)         # ImageHash, 8x8 by default
        # h.hash is a 8x8 ndarray of bool; pack into 8 bytes (uint64-LE).
        n = 0
        for bit in h.hash.flatten():
            n = (n << 1) | (1 if bool(bit) else 0)
        return base64.b64encode(n.to_bytes(8, "big")).decode("ascii")
    except Exception as e:
        logger.debug(f"phash failed: {e}")
        return ""


def compute_face_emb_b64(url: str, image_bytes: bytes) -> str:
    """512-d float32 L2-normalised face embedding → base64 (~2.7 KB).
    Returns "" if no face is detected or any failure. Disk cache in
    face_match.embed_url means a re-fetched avatar is essentially free.
    """
    if not image_bytes:
        return ""
    try:
        import face_match  # local module
        emb = face_match.embed_url(url or "phash_only", image_bytes)
        if emb is None:
            return ""
        # emb is float32, 512-d, L2-normalised.
        return base64.b64encode(emb.tobytes()).decode("ascii")
    except Exception as e:
        logger.debug(f"face_emb failed: {e}")
        return ""


def attach_similarity_payload(prof: dict, image_bytes: bytes) -> None:
    """Compute pHash + face_emb from `image_bytes` and attach to `prof`
    in place under keys `phash_b64` and `face_emb_b64`. Either may be
    "" if computation fails (no face detected, bad bytes, etc.); the
    frontend handles missing signals gracefully.

    SYNCHRONOUS — caller MUST run this via asyncio.to_thread to keep
    the event loop free (face embed is ~150 ms CPU per face).
    """
    if not image_bytes:
        prof["phash_b64"] = ""
        prof["face_emb_b64"] = ""
        return
    prof["phash_b64"] = compute_phash_b64(image_bytes)
    prof["face_emb_b64"] = compute_face_emb_b64(
        prof.get("image_url") or "", image_bytes
    )
