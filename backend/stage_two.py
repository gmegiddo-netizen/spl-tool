"""SPL stage-2 search — metadata-driven re-query.

Rewritten 2026-05-29: dropped the BD Google Lens + InsightFace face-match
pipeline (it was slow and BD's Lens parser wasn't returning usable
match URLs). The new design uses the ground-truth metadata we now have
from the confirmed profile(s) to do BETTER stage-1-style discovery:

  1. Extract: display name, job title/company, location, confirmed handle
  2. Handle extrapolation: try the confirmed handle on platforms we don't
     have yet (most public figures reuse handles across networks).
  3. Richer Serper.dev SERP queries: include company/position/city so
     the search disambiguates between same-name people.
  4. Lightweight per-candidate verification: query the SPL avatar
     endpoints to confirm the profile exists + has a real picture.
  5. Merge with confirmed (pre-checked) + remaining (unchanged).

Total wall ≈ 2–4s for typical N=1–2 confirmed inputs. Cost ≈ $0.003 per
stage-2 call (Serper.dev: ~$0.001/query × ~3 platforms).

face_match.py and reverse_image.py are NOT imported here anymore but
left in place in case we want to re-enable them later. The /api/search-
again endpoint signature is unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Iterable, Optional
from urllib.parse import urlparse, quote

import httpx

# username_sweep was wired in 2026-05-29 to catch low-key handles on
# the long tail of platforms (Reddit, GitHub, Threads, etc.). Disabled
# 2026-05-29 same day: the scan product only supports the big-5 (X /
# FB / IG / LI / TT), so returning hits on YouTube/Pinterest/Steam was
# noise the customer couldn't act on. The module stays on disk for
# future use if we add support for more platforms downstream.
# import username_sweep  # noqa: F401

log = logging.getLogger(__name__)

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SPL_BASE = "http://127.0.0.1:8801"

# Platform → URL builder for handle extrapolation.
_PLAT_URL = {
    "X":         lambda h: f"https://x.com/{h.lstrip('@')}",
    "Facebook":  lambda h: f"https://facebook.com/{h.lstrip('@')}",
    "Instagram": lambda h: f"https://instagram.com/{h.lstrip('@')}",
    "LinkedIn":  lambda h: f"https://linkedin.com/in/{h.lstrip('@')}",
    "TikTok":    lambda h: f"https://tiktok.com/@{h.lstrip('@')}",
}
_PLAT_HOST = {
    "X":         ("twitter.com", "x.com"),
    "Facebook":  ("facebook.com",),
    "Instagram": ("instagram.com",),
    "LinkedIn":  ("linkedin.com",),
    "TikTok":    ("tiktok.com",),
}
_PLAT_USERNAME_RE = {
    "X":         re.compile(r"(?:x|twitter)\.com/([A-Za-z0-9_]+)", re.I),
    "Facebook":  re.compile(r"facebook\.com/([A-Za-z0-9_.]+)", re.I),
    "Instagram": re.compile(r"instagram\.com/([A-Za-z0-9_.]+)", re.I),
    "LinkedIn":  re.compile(r"linkedin\.com/in/([A-Za-z0-9_\-]+)", re.I),
    "TikTok":    re.compile(r"tiktok\.com/@([A-Za-z0-9_.]+)", re.I),
}
ALL_PLATFORMS = list(_PLAT_URL.keys())


def _classify(url: str) -> tuple[str, str]:
    """Return (platform, username) for a URL or ("", "") if unrecognised."""
    if not url:
        return "", ""
    for plat, pat in _PLAT_USERNAME_RE.items():
        m = pat.search(url)
        if m:
            return plat, m.group(1)
    return "", ""


def _extract_metadata(confirmed: list[dict]) -> dict[str, Any]:
    """Pull the most-useful identity hints out of the confirmed
    profiles. LinkedIn wins for name + position + company because BD
    enriches it; X / FB fall back to their own display_name/bio.
    """
    md = {
        "display_names": [],
        "company": "",
        "position": "",
        "city": "",
        "bio_fragments": [],
        "handles": set(),
    }
    # Prefer LinkedIn for the canonical name + position + company.
    for c in confirmed:
        if c.get("platform") == "LinkedIn":
            if c.get("display_name"): md["display_names"].insert(0, c["display_name"])
            if c.get("company"):      md["company"]  = c["company"]
            if c.get("position"):     md["position"] = c["position"]
            if c.get("city"):         md["city"]     = c["city"]
        else:
            if c.get("display_name"): md["display_names"].append(c["display_name"])
        if c.get("username"):
            md["handles"].add(c["username"].lstrip("@"))
        bio = (c.get("bio") or "").strip()
        if bio and len(bio) <= 200:
            md["bio_fragments"].append(bio)
    md["handles"] = list(md["handles"])
    # Pick a canonical display name (longest non-empty; usually the
    # fullest version, e.g. "Scott Galloway" beats "Scott G").
    md["display_names"] = sorted(set(md["display_names"]), key=len, reverse=True)
    md["display_name"] = md["display_names"][0] if md["display_names"] else ""
    return md


def _build_serp_queries(md: dict, platform: str) -> list[str]:
    """One or two enriched SERP queries per missing platform. Each
    query tries to disambiguate via company/position/city.
    """
    name = md.get("display_name") or ""
    if not name:
        return []
    host = _PLAT_HOST[platform][0]
    qs = []
    # Query A: name + best contextual hint + site filter
    hint = md.get("company") or md.get("position") or md.get("city") or ""
    base = f'"{name}"'
    if hint:
        base += f' {hint}'
    qs.append(f'{base} site:{host}')
    # Query B (only if we have BOTH name and a hint): redundant but
    # different enough to catch profiles the first misses (different
    # snippet rankings).
    if hint and (md.get("city") and md.get("city") != hint):
        qs.append(f'"{name}" "{md["city"]}" site:{host}')
    return qs


async def _empty_list():
    """Return [] when one of the three task families has no entries —
    keeps the three-way asyncio.gather signature consistent.
    """
    return []


async def _serper(client: httpx.AsyncClient, query: str) -> list[dict]:
    """Google SERP via Serper.dev with BrightData fallback when Serper
    is unavailable (out of credits / down). Matches the resilience the
    stage-1 google_search has — without it, stage-2 silently returns
    empty when Serper is down.
    """
    if not query:
        return []
    # Try Serper first
    if SERPER_API_KEY:
        try:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 8},
                timeout=4,
            )
            if r.status_code == 200:
                return (r.json() or {}).get("organic", []) or []
        except Exception:
            pass
    # BD fallback (zone serp_api1). Same pattern as profile_finder.google_search.
    bd_key = os.getenv("BRIGHTDATA_API_KEY", "")
    if not bd_key:
        return []
    bd_zone = os.getenv("BRIGHTDATA_SERP_ZONE", "serp_api1")
    from urllib.parse import quote as _qq
    try:
        google_url = f"https://www.google.com/search?q={_qq(query)}&brd_json=1&num=10"
        r = await client.post(
            "https://api.brightdata.com/request",
            headers={
                "Authorization": f"Bearer {bd_key}",
                "Content-Type": "application/json",
            },
            json={"zone": bd_zone, "url": google_url, "format": "json"},
            timeout=15.0,
        )
        if r.status_code != 200:
            return []
        outer = r.json()
        body = outer.get("body") if isinstance(outer, dict) else None
        if isinstance(body, str):
            try:
                import json as _j
                data = _j.loads(body)
            except Exception:
                return []
        elif isinstance(outer, dict):
            data = outer
        else:
            return []
        organic = data.get("organic") or data.get("results") or []
        # Map BD fields to the Serper shape stage-2 expects.
        return [
            {
                "title": r.get("title", ""),
                "link":  r.get("link") or r.get("url", ""),
                "snippet": r.get("description") or r.get("snippet", ""),
            }
            for r in organic[:8]
            if (r.get("link") or r.get("url"))
        ]
    except Exception:
        return []


# Platform path segments that look like usernames but aren't (reels,
# explore pages, login etc.). Anything that lands here is a non-profile
# URL we shouldn't suggest as a candidate.
_PLATFORM_PATH_BLOCKLIST = {
    "reel", "reels", "tv", "p", "explore", "live", "video", "videos",
    "photos", "posts", "watch", "search", "story", "stories", "about",
    "help", "login", "home", "feed", "settings", "messages", "tag",
    "tags", "hashtag", "discover", "trending", "directory", "pages",
    "groups", "events", "marketplace", "share", "embed",
    "i", "intent",                       # twitter/x intent URLs
    "company", "school", "showcase",     # linkedin non-/in/ paths
    "people", "public", "profile", "user", "users",  # generic profile-list paths
}


def _username_resembles_subject(username: str, display_name: str) -> bool:
    """Drop SERP hits where the username has zero connection to the
    subject's name. E.g. for "Scott Galloway", `profgalloway` passes
    (contains "galloway") but `dianaelainem` fails. Catches the very
    common SERP failure mode where the snippet *mentions* the subject
    but the actual account belongs to someone else.
    """
    if not display_name or not username:
        return True
    u = username.lower().replace(".", "").replace("_", "").replace("-", "")
    tokens = [t.lower() for t in re.split(r"\s+", display_name) if len(t) >= 3]
    if not tokens:
        return True
    # At least one name token must appear (or be a prefix-match) in the
    # username. 4 chars is the floor for prefix matches to avoid
    # "scot" matching every "scott*" account incidentally.
    for t in tokens:
        if t in u:
            return True
        if len(t) >= 4 and (u.startswith(t[:4]) or t.startswith(u[:4])):
            return True
    return False


def _candidate_url(url: str, platform: str) -> str:
    """Normalise a Google result URL to a canonical platform-handle URL,
    or empty string if it doesn't look like a profile URL.
    """
    p, u = _classify(url)
    if not p or not u or p != platform:
        return ""
    if u.lower() in _PLATFORM_PATH_BLOCKLIST:
        return ""
    # Reject FB numeric-only IDs (e.g. /100055619064163) — those are
    # opaque page IDs that don't disambiguate.
    if p == "Facebook" and u.isdigit():
        return ""
    return _PLAT_URL[p](u)


def _result_matches_subject(hit: dict, display_name: str) -> bool:
    """True if the SERP result's title or snippet contains at least one
    name token from the confirmed display_name. Stops random
    Galloway-MENTIONS (a TikToker who talked ABOUT Galloway) from
    showing up as "his" account.
    """
    if not display_name:
        return True   # no name to check against — let the URL filter do the work
    tokens = [t for t in re.split(r"\s+", display_name) if len(t) >= 3]
    if not tokens:
        return True
    haystack = ((hit.get("title") or "") + " " + (hit.get("snippet") or "")).lower()
    # Need TWO name tokens to match (catches "Scott Galloway" but not
    # "Scott Smith" or "Sarah Galloway").
    matches = sum(1 for t in tokens if t.lower() in haystack)
    return matches >= min(2, len(tokens))


async def _check_handle_exists(
    client: httpx.AsyncClient, platform: str, handle: str
) -> Optional[dict]:
    """Cheap existence probe: try the SPL avatar endpoint for the
    platform. If it returns image bytes, the handle exists. Returns a
    candidate dict or None.
    """
    handle = (handle or "").lstrip("@")
    if not handle:
        return None
    url = _PLAT_URL[platform](handle)
    # Use the platform-direct avatar endpoint if available, else fall
    # back to unavatar.io.
    direct = {
        "X":         f"{SPL_BASE}/api/avatar/x?u={quote(url)}",
        "Instagram": f"{SPL_BASE}/api/avatar/instagram?u={quote(url)}",
        "TikTok":    f"{SPL_BASE}/api/avatar/tiktok?u={quote(url)}",
    }.get(platform)
    img_url = direct or f"https://unavatar.io/{platform.lower()}/{handle}"
    try:
        r = await client.get(img_url, timeout=6, follow_redirects=True)
        # Accept any image-y response that's > 1 KB (skips the
        # "no profile pic" 1×1 placeholder some endpoints return).
        ct = (r.headers.get("content-type") or "").lower()
        if r.status_code == 200 and ct.startswith("image") and len(r.content) > 1024:
            return {
                "platform": platform,
                "url": url,
                "username": handle,
                "display_name": handle,
                "image_url": img_url,
                "bio": "",
                "followers": "",
                "score": 75,                   # "Likely" — handle-match but unverified
                "reasoning": f"Same handle as confirmed account.",
                "_stage2_new": True,
            }
    except Exception:
        return None
    return None


async def _serp_candidates(
    client: httpx.AsyncClient,
    md: dict,
    platform: str,
    seen_urls: set,
) -> list[dict]:
    """Run enriched SERP queries for `platform`, map results to
    candidate profile URLs, skip ones we already have.
    """
    queries = _build_serp_queries(md, platform)
    if not queries:
        return []
    # Fire all queries for this platform in parallel.
    results = await asyncio.gather(*(_serper(client, q) for q in queries))
    name = md.get("display_name") or ""
    out: list[dict] = []
    for hits in results:
        for h in hits[:5]:
            link = h.get("link") or ""
            canon = _candidate_url(link, platform)
            if not canon or canon.rstrip("/").lower() in seen_urls:
                continue
            if not _result_matches_subject(h, name):
                continue
            _, uname = _classify(canon)
            if not _username_resembles_subject(uname, name):
                continue
            seen_urls.add(canon.rstrip("/").lower())
            out.append({
                "platform": platform,
                "url": canon,
                "username": uname,
                "display_name": (h.get("title") or uname).split(" | ")[0].strip(),
                "image_url": "",                # filled below if cheap
                "bio": h.get("snippet", "") or "",
                "followers": "",
                "score": 70,                   # "Possible" — SERP-derived
                "reasoning": "Found via metadata-enriched search.",
                "_stage2_new": True,
            })
            break  # one hit per query is enough
    return out


async def _fill_avatar(client: httpx.AsyncClient, card: dict) -> None:
    """Best-effort: pull an avatar URL through the SPL platform-direct
    endpoints so the chip renders with a face instead of initials.
    Non-fatal — if avatar fetch fails the chip still shows initials.
    """
    if card.get("image_url"):
        return
    plat = card.get("platform", "")
    handle = card.get("username", "")
    if not plat or not handle:
        return
    direct = {
        "X":         f"{SPL_BASE}/api/avatar/x?u={quote(_PLAT_URL['X'](handle))}",
        "Instagram": f"{SPL_BASE}/api/avatar/instagram?u={quote(_PLAT_URL['Instagram'](handle))}",
        "TikTok":    f"{SPL_BASE}/api/avatar/tiktok?u={quote(_PLAT_URL['TikTok'](handle))}",
    }.get(plat)
    if direct:
        card["image_url"] = direct


async def run(
    confirmed: list[dict[str, Any]],
    unknown:   list[dict[str, Any]],
    rejected:  list[dict[str, Any]],
) -> dict[str, Any]:
    """Stage-2 entry — metadata-driven re-query with 3-bucket triage.

    confirmed: anchors + pre-checked in the output (today's behaviour).
    unknown:   carry through to the output unchanged, NOT dedup-blocked.
    rejected:  dropped from output AND seeded into seen_urls so the new
               SERP/handle pass cannot re-suggest them.

    Returns: {profiles, diagnostics}.
    """
    diag = {
        "handle_probes": 0,
        "serp_queries": 0,
        "new_candidates": 0,
        "missing_platforms": [],
    }

    md = _extract_metadata(confirmed)
    have_platforms = {c.get("platform", "") for c in confirmed}
    # Dedup-block: confirmed + rejected + unknown URLs all get added to
    # seen_urls so the new SERP/handle pass won't re-suggest them. The
    # difference between buckets shows up in the OUTPUT: confirmed are
    # pre-checked, unknown carry through unselected, rejected are dropped.
    seen_urls = {
        (c.get("url") or "").rstrip("/").lower()
        for c in (confirmed + unknown + rejected)
        if c.get("url")
    }
    missing = [p for p in ALL_PLATFORMS if p not in have_platforms]
    diag["missing_platforms"] = missing
    diag["unknown_carried"] = len(unknown)
    diag["rejected_blocked"] = len(rejected)

    async with httpx.AsyncClient() as client:
        # 1. Handle extrapolation — try each confirmed handle on each
        #    missing big-5 platform in parallel.
        handle_tasks = []
        for h in md["handles"]:
            for plat in missing:
                handle_tasks.append(_check_handle_exists(client, plat, h))
        diag["handle_probes"] = len(handle_tasks)

        # 2. SERP re-query per missing big-5 platform in parallel.
        serp_tasks = [_serp_candidates(client, md, plat, seen_urls) for plat in missing]
        diag["serp_queries"] = sum(len(_build_serp_queries(md, p)) for p in missing)

        # Fire both families in parallel.
        handle_results, serp_results = await asyncio.gather(
            asyncio.gather(*handle_tasks) if handle_tasks else _empty_list(),
            asyncio.gather(*serp_tasks)   if serp_tasks   else _empty_list(),
        )
        handle_hits = [h for h in handle_results if h]
        serp_hits: list[dict] = []
        for lst in serp_results:
            serp_hits.extend(lst)

        # Drop handle hits whose URLs are also in seen (a SERP hit may
        # have landed first), and let handle hits trump SERP hits
        # because they're a stronger signal.
        handle_url_set = {h["url"].rstrip("/").lower() for h in handle_hits}
        serp_hits = [s for s in serp_hits
                     if s["url"].rstrip("/").lower() not in handle_url_set]

        new_candidates = handle_hits + serp_hits
        diag["new_candidates"] = len(new_candidates)

        # 3. Best-effort avatar fill for SERP-derived cards (handle
        #    hits already have one).
        await asyncio.gather(*(_fill_avatar(client, c) for c in serp_hits))

    # 4. Pre-check confirmed cards and bump their score.
    for c in confirmed:
        c["_carry_confirmed"] = True
        c["score"] = max(int(c.get("score", 0)), 95)

    # 5. Merge: confirmed first (pre-checked), then new candidates,
    #    then unknown carryovers (rendered unselected, customer can
    #    triage them again next round). Rejected are NOT in the output.
    for u in unknown:
        # Strip any prior carry/select state so they render unchecked.
        u.pop("_carry_confirmed", None)
    merged = (
        list(confirmed)
        + sorted(new_candidates, key=lambda x: -int(x.get("score", 0)))
        + sorted(unknown,        key=lambda x: -int(x.get("score", 0)))
    )

    return {"profiles": merged, "diagnostics": diag}
