"""BrightData Google Lens reverse-image client for SPL stage-2 search.

One call ≈ $0.0015–0.003 per BD's SERP pricing. Returns the URLs where
Google Lens believes the input image also appears, optionally filtered
to social-platform domains.

Disk cache at /var/www/spl-tool/backend/lens_cache/<sha256>.json keyed
by the input image URL — avatars are stable enough that re-querying
within a session is wasteful.

Built 2026-05-29.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Iterable
from urllib.parse import urlparse

import httpx

CACHE_DIR = os.environ.get(
    "LENS_CACHE_DIR", "/var/www/spl-tool/backend/lens_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)

BD_KEY = os.environ.get("BRIGHTDATA_API_KEY", "")
# BD SERP endpoint accepting Google Lens queries with brd_lens=exact for
# tight matches. Per BD docs, the lens parameter restricts to "exact"
# (same image variants), "visual" (similar), "homework", "products".
# We want exact for cross-platform handle discovery.
BD_LENS_URL = "https://api.brightdata.com/request"
# Verified 2026-05-29 that zone `serp_api1` accepts Google Lens URLs and
# returns parsed JSON when brd_json=1 is in the query string. Override
# via env var if you stand up a dedicated lens zone later.
BD_LENS_ZONE = os.environ.get("BRIGHTDATA_SERP_ZONE", "serp_api1")

# Social platforms we care about for stage-2 discovery.
_PLAT_HOST = {
    "X":         ("twitter.com", "x.com"),
    "Facebook":  ("facebook.com", "fb.com"),
    "Instagram": ("instagram.com",),
    "LinkedIn":  ("linkedin.com",),
    "TikTok":    ("tiktok.com",),
}
_ALL_HOSTS = tuple(h for hs in _PLAT_HOST.values() for h in hs)


def _cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


def _classify_platform(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    for plat, hosts in _PLAT_HOST.items():
        if any(host == h or host.endswith("." + h) for h in hosts):
            return plat
    return ""


def search_by_image(image_url: str, *, max_results: int = 30) -> list[dict]:
    """Send `image_url` to BrightData Google Lens (exact-match mode) and
    return a list of {url, platform, title, score} dicts for hits on the
    five social platforms we care about. Empty list on any error.
    """
    if not image_url or not BD_KEY:
        return []
    cp = _cache_path(image_url)
    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # BrightData /request: POST { zone, url, format }. The `url` is a
    # Google Lens search URL with brd_lens=exact_matches and brd_json=1.
    # Image URL must be URL-encoded inside the Lens query string.
    from urllib.parse import quote
    google_lens = (
        f"https://lens.google.com/uploadbyurl"
        f"?url={quote(image_url, safe='')}&hl=en&brd_json=1&brd_lens=exact_matches"
    )
    payload = {
        "zone": BD_LENS_ZONE,
        "url": google_lens,
        "format": "json",
    }
    try:
        r = httpx.post(
            BD_LENS_URL,
            headers={
                "Authorization": f"Bearer {BD_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if r.status_code != 200:
            return []
        outer = r.json()
        # BD wraps the upstream response: outer = {status_code, headers,
        # body (string)}. Parse the inner body if present.
        body = outer.get("body") if isinstance(outer, dict) else None
        if isinstance(body, str):
            try:
                data = json.loads(body)
            except Exception:
                data = {"_raw_body": body}
        elif isinstance(outer, dict):
            data = outer
        else:
            data = {}
    except Exception:
        return []

    out: list[dict] = []
    seen: set[str] = set()
    # Try several response shapes — BD's Lens parser has been moving:
    # - data["exact_matches"] = [...]
    # - data["visual_matches"] = [...]
    # - data["matches"] = [...]
    # - data["organic"] = [...]      (less likely for Lens, but try)
    # - last-ditch regex over the raw body string for social URLs.
    candidates: list[dict] = []
    for key in ("exact_matches", "visual_matches", "matches", "results", "organic"):
        v = data.get(key)
        if isinstance(v, list):
            candidates.extend(v)
    for hit in candidates[:max_results]:
        if not isinstance(hit, dict):
            continue
        u = hit.get("url") or hit.get("link") or hit.get("source") or ""
        if not u or u in seen:
            continue
        plat = _classify_platform(u)
        if not plat:
            continue
        seen.add(u)
        out.append({
            "url": u,
            "platform": plat,
            "title": hit.get("title") or "",
            "lens_score": float(hit.get("score") or hit.get("confidence") or 0.0),
        })

    # Fallback: scan raw response for social URLs even if no structured
    # match list was returned. Catches BD parser drift; better than 0
    # results when the data is in there but the shape changed.
    if not out:
        import re as _re
        haystack = json.dumps(data) if isinstance(data, dict) else str(data)
        url_re = _re.compile(
            r"https?://(?:www\.)?(?:twitter\.com|x\.com|facebook\.com|fb\.com|"
            r"instagram\.com|linkedin\.com/in|tiktok\.com/@)/?[A-Za-z0-9_.\-]+"
        )
        for m in url_re.findall(haystack)[:max_results]:
            if m in seen:
                continue
            plat = _classify_platform(m)
            if not plat:
                continue
            seen.add(m)
            out.append({
                "url": m,
                "platform": plat,
                "title": "",
                "lens_score": 0.5,  # uncertain — flagged for human review
            })

    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
    except Exception:
        pass
    return out


def discover_new_profiles(
    confirmed_avatars: Iterable[str],
    already_have: Iterable[str],
) -> list[dict]:
    """For each confirmed avatar URL, run BD Lens, collect platform hits,
    drop any whose URL matches one we already have. Return a deduped
    list of new candidate {url, platform, ...} dicts.
    """
    have = {(u or "").rstrip("/").lower() for u in already_have}
    seen: set[str] = set()
    out: list[dict] = []
    for av in confirmed_avatars:
        if not av:
            continue
        for hit in search_by_image(av):
            key = hit["url"].rstrip("/").lower()
            if key in have or key in seen:
                continue
            seen.add(key)
            out.append(hit)
    return out
