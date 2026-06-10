import asyncio
import html as _html
import json
import os
import re
import urllib.parse
from urllib.parse import quote_plus
import httpx
import anthropic
from dotenv import load_dotenv

load_dotenv()

BRIGHTDATA_API_KEY = os.getenv("BRIGHTDATA_API_KEY", "")
SERP_ZONE = os.getenv("SERP_ZONE", "serp_api1")
# ITER27 2026-05-28: Serper.dev replaces BD SERP. Free tier 2,500 queries/mo, no card.
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "53c3cffc5a8c4204592066609e448915bcd644bf")
# v7.64 2026-06-01: operator dropped Serper (out of credits / $50-capped). BD SERP
# is now the PRIMARY path so we don't waste a round-trip on Serper's instant 400
# "Not enough credits" before falling back. Set SERP_PROVIDER=serper to restore
# Serper-first (e.g. after a top-up). "brightdata" (default) = BD primary, Serper
# only as a safety fallback if BD itself returns nothing.
SERP_PROVIDER = os.getenv("SERP_PROVIDER", "brightdata").strip().lower()
# ITER28 2026-05-28: ScrapeCreators for TT-native search (bypasses Google indexing gap).
# Closes Ben Gvir's TT (Hebrew handle Google misses). ~3 credits ($0.006) per call.
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY", "GWj19GDRZdWLcuWQCx185pehj582")
# BrightData Scraping Browser (browser_api) zone — renders the public LinkedIn
# page as a real browser so we can recover the real media.licdn.com avatar when
# the dataset API only returns the static.licdn.com ghost (default_avatar=True).
SCRAPING_BROWSER_ZONE = os.getenv("SCRAPING_BROWSER_ZONE", "scraping_browser1")
LINKEDIN_DATASET_ID = "gd_l1viktl72bvl7bjuj0"
TIKTOK_DATASET_ID = "gd_l1villgoiiidt09ci"
# Instagram Profiles dataset (same one the main app's scraper_instagram uses).
# Provides the real cdninstagram avatar (profile_image_link); the logged-out
# og:image only returns the IG logo, so BD is the reliable avatar source.
INSTAGRAM_DATASET_ID = "gd_l1vikfch901nx3by4"
def _log_spl_bd_usage(event: str):
    """Log SPL BrightData usage to accountability DB so CSC doesn't flag as ghost charges."""
    try:
        import sqlite3, uuid
        from datetime import datetime, timezone
        db_path = "/var/www/accountability/accountability.db"
        conn = sqlite3.connect(db_path)
        bid = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO balance_log (id, service, balance, credits_remaining, event, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (bid, "brightdata", None, None, f"spl_{event}", now)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass



PLATFORMS = {
    "X": "x.com",
    "Facebook": "facebook.com",
    "Instagram": "instagram.com",
    "TikTok": "tiktok.com",
    "LinkedIn": "linkedin.com",
}


# Regex to extract profile URLs from search results
PROFILE_URL_PATTERNS = {
    "X": re.compile(r'https?://(?:x|twitter)\.com/([a-zA-Z0-9_]+)(?:\?|$|/)'),
    "Facebook": re.compile(r'https?://(?:www\.)?facebook\.com/([a-zA-Z0-9_.]+)(?:\?|$|/)'),
    "Instagram": re.compile(r'https?://(?:www\.)?instagram\.com/([a-zA-Z0-9_.]+)(?:\?|$|/)'),
    "TikTok": re.compile(r'https?://(?:www\.)?tiktok\.com/@([a-zA-Z0-9_.]+)(?:\?|$|/)'),
    "LinkedIn": re.compile(r'https?://(?:\w+\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)(?:\?|$|/)'),
}

CANONICAL_URL = {
    "X": "https://x.com/{username}",
    "Facebook": "https://www.facebook.com/{username}",
    "Instagram": "https://www.instagram.com/{username}/",
    "TikTok": "https://www.tiktok.com/@{username}",
    "LinkedIn": "https://www.linkedin.com/in/{username}/",
}


async def google_search(query: str, client: httpx.AsyncClient, raw_sink: list | None = None) -> list[dict]:
    """Search Google via Serper.dev and return organic results.
    ITER27 2026-05-28: switched from BrightData SERP. Serper is ~3-5× faster (sub-second
    typical vs BD's 3-5s) and ~5× cheaper. Same {title, link, description} shape — only
    Serper calls the snippet field 'snippet' instead of BD's 'description'.
    2026-05-29: falls back to BrightData SERP when Serper is out of credits
    (HTTP 400 "Not enough credits") so coverage doesn't drop to zero. BD is
    ~3× per-query but only fires on Serper failure — conditional cost.

    v7.56 (serp_haiku safety net): when `raw_sink` is provided, ALL organic
    results (title/link/snippet — the full set, including ones search_platform
    later filters out) are appended to it for the single Haiku extraction pass,
    and Part A knowledge-graph / peopleAlsoSearch social URLs are appended as
    {"_kg_url": ...} markers. This is free — the data is already in the response.
    The normal return value (top-5 organic) is unchanged.
    """
    # v7.64 2026-06-01: BD-primary by default (SERP_PROVIDER). When BD is
    # primary we skip the now-dead Serper call entirely (it returns an
    # instant 400 "Not enough credits" — a wasted round-trip on the
    # critical path). Serper stays available as a fallback only if BD
    # yields nothing, and SERP_PROVIDER=serper restores the old order.
    if SERP_PROVIDER == "serper":
        res = await _serper_search(query, client, raw_sink)
        if res:
            return res
        return await _bd_serp_search(query, client, raw_sink)
    # brightdata (default): BD first, Serper only as last-ditch fallback.
    res = await _bd_serp_search(query, client, raw_sink)
    if res:
        return res
    return await _serper_search(query, client, raw_sink)


async def _serper_search(query: str, client: httpx.AsyncClient, raw_sink: list | None = None) -> list[dict]:
    """Serper.dev organic search. Returns [] on any failure (incl. the
    out-of-credits 400) so the caller can fall through to BD."""
    try:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": 10},
            timeout=4,
        )

        if resp.status_code == 200:
            data = resp.json()
            organic = data.get("organic", [])
            if raw_sink is not None:
                try:
                    for r in organic:
                        raw_sink.append({
                            "title": r.get("title", ""),
                            "link": r.get("link", ""),
                            "snippet": r.get("snippet", ""),
                        })
                    _harvest_kg_social(data, raw_sink)
                except Exception:
                    pass
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "description": r.get("snippet", ""),
                    "image": r.get("imageUrl", ""),
                }
                for r in organic[:5]
            ]
        import logging
        logging.getLogger("spl").warning(
            f"Serper {resp.status_code}: {resp.text[:200]}"
        )
    except Exception:
        pass
    return []


async def _bd_serp_search(query: str, client: httpx.AsyncClient, raw_sink: list | None = None) -> list[dict]:
    """BrightData dedicated SERP API (zone serp_api1, brd_json=1 → structured
    JSON; this is the FAST SERP endpoint, not a Web-Unlocker render). Returns
    [] on failure."""
    try:
        bd_key = os.getenv("BRIGHTDATA_API_KEY", "")
        if not bd_key:
            return []
        bd_zone = os.getenv("BRIGHTDATA_SERP_ZONE", "serp_api1")
        from urllib.parse import quote
        google_url = f"https://www.google.com/search?q={quote(query)}&brd_json=1&num=10"
        resp = await client.post(
            "https://api.brightdata.com/request",
            headers={
                "Authorization": f"Bearer {bd_key}",
                "Content-Type": "application/json",
            },
            json={"zone": bd_zone, "url": google_url, "format": "json"},
            # 2026-05-30 v3: 3s (was 4s). Aggressively trims the wall
            # at the cost of losing slow-platform SERP hits. Handle-guess
            # covers FB/IG/TT/X via SC; LinkedIn relies on this SERP for
            # discovery so 3s is the minimum that doesn't kill LI.
            # v7.64: BD is now PRIMARY (Serper dropped). serp_api1 + brd_json=1
            # is BD's structured SERP API — sub-second to ~2-3s typical, far
            # faster than a Web-Unlocker page render. Timeout held at 3.0s.
            timeout=3.0,
        )
        if resp.status_code != 200:
            return []
        outer = resp.json()
        body = outer.get("body") if isinstance(outer, dict) else None
        if isinstance(body, str):
            try:
                data = __import__("json").loads(body)
            except Exception:
                return []
        elif isinstance(outer, dict):
            data = outer
        else:
            return []
        organic = data.get("organic") or data.get("results") or []
        if raw_sink is not None:
            try:
                for r in organic:
                    if r.get("link") or r.get("url"):
                        raw_sink.append({
                            "title": r.get("title", ""),
                            "link": r.get("link") or r.get("url", ""),
                            "snippet": r.get("description") or r.get("snippet", ""),
                        })
                # BD brd_json may carry a knowledge panel / social block.
                _harvest_kg_social(data, raw_sink)
            except Exception:
                pass
        return [
            {
                "title": r.get("title", ""),
                "link": r.get("link") or r.get("url", ""),
                "description": r.get("description") or r.get("snippet", ""),
                "image": r.get("imageUrl") or r.get("image") or r.get("thumbnail") or "",
            }
            for r in organic[:5]
            if (r.get("link") or r.get("url"))
        ]
    except Exception:
        return []


# ───────────────────── v7.56: SERP Haiku safety net ─────────────────────
# A free Part A (knowledge-graph / rich-field social-link harvest) + a single
# cheap Part B (~$0.001) Haiku extraction over RAW organic results, recovering
# the subject's real profiles that the strict regex discovery throws away.

_SOCIAL_HOST_PLATFORM = [
    ("x.com", "X"), ("twitter.com", "X"),
    ("facebook.com", "Facebook"), ("fb.com", "Facebook"),
    ("instagram.com", "Instagram"),
    ("tiktok.com", "TikTok"),
    ("linkedin.com", "LinkedIn"),
]


def _platform_for_url(url: str) -> str | None:
    low = (url or "").lower()
    for host, plat in _SOCIAL_HOST_PLATFORM:
        if host in low:
            return plat
    return None


def _harvest_kg_social(data: dict, raw_sink: list) -> None:
    """Part A (free): pull social profile URLs out of the SERP provider's
    knowledge-graph / rich fields (Serper: knowledgeGraph.attributes/website +
    peopleAlsoSearch; BD brd_json: knowledge panel / social) and append them to
    the raw sink as {"_kg_url": url} markers so they enter the candidate pool.
    No-ops gracefully if the provider returns nothing usable."""
    urls: list[str] = []

    def _eat(v):
        if isinstance(v, str) and v.startswith("http") and _platform_for_url(v):
            urls.append(v)

    try:
        kg = data.get("knowledgeGraph") or data.get("knowledge_graph") or {}
        if isinstance(kg, dict):
            _eat(kg.get("website"))
            _eat(kg.get("url"))
            attrs = kg.get("attributes") or {}
            if isinstance(attrs, dict):
                for v in attrs.values():
                    _eat(v)
            # Serper sometimes returns a "profiles"/"socialProfiles" list.
            for key in ("profiles", "socialProfiles", "social"):
                lst = kg.get(key)
                if isinstance(lst, list):
                    for it in lst:
                        if isinstance(it, dict):
                            _eat(it.get("link") or it.get("url"))
                        else:
                            _eat(it)
        # Generic top-level "social" block some BD panels expose.
        soc = data.get("social")
        if isinstance(soc, list):
            for it in soc:
                if isinstance(it, dict):
                    _eat(it.get("link") or it.get("url"))
                else:
                    _eat(it)
    except Exception:
        pass

    for u in urls:
        raw_sink.append({"_kg_url": u})


async def scrapecreators_tiktok_search(query: str, client: httpx.AsyncClient, limit: int = 5) -> list[dict]:
    """ITER28: TikTok-native user search via ScrapeCreators.
    Bypasses Google — finds TT handles SERP misses (especially non-English profiles).
    Cost: ~3 credits ($0.006) per search.
    Returns same shape as google_search() so it can be merged into search_platform().
    """
    try:
        resp = await client.get(
            "https://api.scrapecreators.com/v1/tiktok/search/users",
            params={"query": query, "limit": limit},
            headers={"x-api-key": SCRAPECREATORS_API_KEY},
            timeout=4,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for u in (data.get("user_list", []) or [])[:limit]:
            ui = u.get("user_info", {})
            uid = ui.get("unique_id")
            if not uid:
                continue
            results.append({
                "title": ui.get("nickname", ""),
                "link": f"https://www.tiktok.com/@{uid}",
                "description": (ui.get("signature") or "")[:200],
            })
        return results
    except Exception:
        return []


SKIP_USERNAMES = {
    "search", "explore", "login", "signup", "help", "about", "hashtag",
    "p", "reel", "stories", "reels", "discover", "tag", "music", "live",
    "watch", "share", "settings", "notifications", "messages", "i",
    # Facebook generic paths that Google SERP returns alongside real profiles
    "media", "photo.php", "photo", "photos", "video", "videos",
    "groups", "events", "marketplace", "gaming", "ads", "business",
    "developers", "privacy", "policies", "pages", "friends", "people",
    "bookmarks", "saved", "fundraisers", "offers", "jobs", "public",
    "dialog", "plugins", "flx", "permalink.php", "story.php", "sharer",
}

# URL fragments that indicate a post/article, not a profile page
POST_INDICATORS = [
    "/status/", "/posts/", "/post/", "/p/", "/reel/", "/photo/",
    "/video/", "/activity-", "/pulse/", "/articles/", "/article/",
    "/events/", "/groups/", "/pages/", "/tag/", "/discover/", "/music/",
]


def _is_profile_url(link: str) -> bool:
    """Return True if the URL looks like a profile page, not a post/article."""
    lower = link.lower()
    return not any(indicator in lower for indicator in POST_INDICATORS)


PLATFORM_SEARCH_HINTS = {
    "X": ["twitter profile", "x.com"],
    "Facebook": ["facebook profile", "facebook.com"],
    "Instagram": ["instagram profile", "instagram.com"],
    "TikTok": ["tiktok profile", "tiktok.com"],
    "LinkedIn": ["linkedin profile", "linkedin.com/in"],
}


async def search_platform(name: str, description: str, platform: str, site_domain: str, client: httpx.AsyncClient, raw_sink: list | None = None) -> list[dict]:
    """Search Google for a person's profile on a specific platform.
    Returns a list of candidate profiles (may be empty).

    v7.56: when `raw_sink` is passed, the RAW organic results (incl. the ones
    filtered out below) from this platform's queries are accumulated into it so
    the caller can run the single serp_haiku safety-net pass over the full set.
    Existing behavior (regex filtering / returned candidates) is unchanged."""
    hints = PLATFORM_SEARCH_HINTS.get(platform, [site_domain])

    # Two SERP queries via Serper.dev, in parallel for best coverage.
    tasks = [
        google_search(f'{name} {hints[0]}', client, raw_sink=raw_sink),
        google_search(f'{name} {description} {hints[-1]}', client, raw_sink=raw_sink),
    ]
    # ITER28: TikTok also fires SC native search to bypass Google indexing gap
    if platform == "TikTok":
        tasks.append(scrapecreators_tiktok_search(name, client))
    results = await asyncio.gather(*tasks)
    all_results = sum(results, [])

    if not all_results:
        return []

    # Extract ALL matching profile URLs, deduplicated by username
    pattern = PROFILE_URL_PATTERNS.get(platform)
    if not pattern:
        return []

    seen_usernames = set()
    candidates = []

    for result in all_results:
        link = result.get("link", "")

        # Skip posts/articles — we want profile pages
        if not _is_profile_url(link):
            continue

        match = pattern.search(link)
        if not match:
            continue

        username = match.group(1)
        username_lower = username.lower()

        # Skip generic pages and duplicates
        if username_lower in SKIP_USERNAMES or username_lower in seen_usernames:
            continue
        seen_usernames.add(username_lower)

        serp_image = result.get("image", "")
        # 2026-06-01 v7.39 (LI avatar fix): for LinkedIn only, ride the free
        # SERP thumbnail when it's a plausible real CDN image. Route through
        # the main app's same-origin /api/img-proxy so referer-gating works.
        # Otherwise leave image_url "" so the avatar endpoint fallback (incl.
        # the BD Scraping-Browser render) still applies.
        li_image_url = ""
        if platform == "LinkedIn" and serp_image.startswith("http") and (
            "licdn" in serp_image
            or "googleusercontent" in serp_image
            or "gstatic" in serp_image
        ):
            li_image_url = "/api/img-proxy?u=" + urllib.parse.quote(serp_image, safe="")

        candidates.append({
            "platform": platform,
            "url": CANONICAL_URL[platform].format(username=username),
            "username": username,
            "source": "google_search",
            "search_title": result.get("title", ""),
            "search_snippet": result.get("description", ""),
            "serp_image": serp_image,
            "bio": result.get("description", ""),
            "display_name": result.get("title", ""),
            "image_url": li_image_url,
        })

    # Limit to top 3 candidates per platform to control scoring/enrichment costs
    return candidates[:3]


# Allowed extraction platforms for the Haiku safety net.
_SN_PLATFORMS = {"X", "Facebook", "Instagram", "TikTok", "LinkedIn"}
# Common alias spellings Haiku may emit.
_SN_PLATFORM_ALIAS = {
    "twitter": "X", "x": "X", "x (twitter)": "X", "x/twitter": "X",
    "facebook": "Facebook", "fb": "Facebook",
    "instagram": "Instagram", "ig": "Instagram",
    "tiktok": "TikTok", "tik tok": "TikTok",
    "linkedin": "LinkedIn",
}


def _sn_name_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if len(t) >= 2}


def _sn_handle_tokens(handle: str) -> set[str]:
    # Split a handle on non-alphanumerics AND on case boundaries, lowercase.
    h = handle or ""
    parts = re.split(r"[^A-Za-z0-9]+", h)
    parts += re.findall(r"[A-Z]?[a-z]+|[0-9]+", h)
    return {p.lower() for p in parts if p}


def serp_haiku_safety_net(
    raw_results: list[dict],
    name: str,
    description: str,
    existing_keys: set[str],
    normalize_url=None,
) -> list[dict]:
    """v7.56 SERP Haiku safety net (Part B, ~$0.001/search).

    ONE Haiku call over the RAW organic results (title/link/snippet — the full
    set, including ones search_platform's regex dropped) + the Part-A
    knowledge-graph URLs ({"_kg_url": ...} markers) collected in `raw_results`.
    Extracts ONLY social profiles that clearly belong to THIS person, name-gates
    the output, and returns candidate dicts in the SAME shape as
    search_platform() output (source="serp_haiku") for the EXISTING per-platform
    Haiku verify + FP-guard to judge. They do NOT bypass verification.

    `existing_keys`: set of "platform|handle" (lowercased) already in the pool —
    used to dedup so we only ADD genuinely new candidates.
    `normalize_url`: optional callable (main.normalize_profile_url) used to
    canonicalize messy extracted URLs.

    Returns the list of NEW candidate dicts. On timeout / parse error / no input
    it returns [] and never raises (never blocks or crashes discovery).
    """
    import logging
    log = logging.getLogger("spl")
    if not raw_results:
        return []

    # Split KG markers (Part A) from organic rows.
    kg_urls = [r["_kg_url"] for r in raw_results if isinstance(r, dict) and r.get("_kg_url")]
    organic = [r for r in raw_results if isinstance(r, dict) and not r.get("_kg_url")]

    # Build a compact, de-duplicated digest for the prompt (bound the tokens).
    lines = []
    seen_lines = set()
    for r in organic:
        link = (r.get("link") or "").strip()
        title = (r.get("title") or "").strip()
        snip = (r.get("snippet") or "").strip()
        key = (link, title[:60])
        if key in seen_lines:
            continue
        seen_lines.add(key)
        lines.append(f"- {title} | {link} | {snip}"[:320])
        if len(lines) >= 40:
            break
    digest = "\n".join(lines)

    extracted: list[dict] = []

    # ── Part A → candidates (free): KG social links go straight in (still verified)
    for u in kg_urls:
        plat = _platform_for_url(u)
        if plat:
            extracted.append({"platform": plat, "handle_or_url": u, "_src": "kg"})

    # ── Part B: single Haiku call over the raw digest ─────────────────
    if digest:
        prompt = (
            f"Google results for **{name}** — *{description}*.\n"
            f"From these titles/snippets/links, list ONLY social media profiles "
            f"(X/Twitter, Facebook, Instagram, LinkedIn, TikTok) that clearly "
            f"belong to THIS specific person. Exclude look-alikes, fan/parody/"
            f"news accounts, and anyone who isn't clearly this person. If unsure, omit.\n\n"
            f"Results:\n{digest}\n\n"
            f'Return STRICT JSON only: [{{"platform":"...","handle_or_url":"..."}}]'
        )
        try:
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            # Grab the first JSON array in the text.
            mb = re.search(r"\[.*\]", raw, re.DOTALL)
            items = json.loads(mb.group(0)) if mb else json.loads(raw)
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and it.get("handle_or_url"):
                        extracted.append({
                            "platform": it.get("platform", ""),
                            "handle_or_url": str(it.get("handle_or_url")),
                            "_src": "haiku",
                        })
        except Exception as e:
            log.info(f"[serp_haiku] extraction skipped ({type(e).__name__}: {str(e)[:80]})")

    if not extracted:
        log.info("[serp_haiku] extracted 0 raw items (KG + Haiku)")
        return []
    log.info(
        f"[serp_haiku] extracted {len(extracted)} raw items: "
        + ", ".join(f"{e.get('platform','?')}:{e.get('handle_or_url','?')}({e.get('_src')})" for e in extracted)
    )

    name_tok = _sn_name_tokens(name)
    new_candidates: list[dict] = []
    new_keys: set[str] = set()

    for it in extracted:
        # Normalize platform label.
        plat_raw = (it.get("platform") or "").strip()
        plat = plat_raw if plat_raw in _SN_PLATFORMS else _SN_PLATFORM_ALIAS.get(plat_raw.lower())
        hou = (it.get("handle_or_url") or "").strip()
        if not hou:
            continue

        # Derive a canonical URL + username.
        username = ""
        url = ""
        if hou.startswith("http") or "/" in hou or "." in hou:
            # Looks like a URL → normalize.
            canon = ""
            if normalize_url:
                try:
                    canon = normalize_url(hou) or ""
                except Exception:
                    canon = ""
            target = canon or hou
            if not plat:
                plat = _platform_for_url(target)
            if plat:
                pat = PROFILE_URL_PATTERNS.get(plat)
                if pat:
                    m = pat.search(target)
                    if m:
                        username = m.group(1)
                if username:
                    url = CANONICAL_URL[plat].format(username=username)
                else:
                    url = canon or target
        else:
            # Bare handle.
            username = hou.lstrip("@")
            if plat and re.match(r"^[A-Za-z0-9_.\-]+$", username):
                url = CANONICAL_URL[plat].format(username=username)

        if not plat or plat not in _SN_PLATFORMS or not url:
            continue
        if not username:
            continue
        ulow = username.lower()
        if ulow in SKIP_USERNAMES:
            continue

        # ── Name-gate (cheap sanity filter): drop handles whose normalized
        # form shares NO token with the subject name, UNLESS a raw snippet
        # strongly ties this exact handle/url to them.
        h_tok = _sn_handle_tokens(username)
        shares = bool(name_tok & h_tok)
        if not shares:
            tied = False
            ul = url.lower()
            for r in organic:
                blob = ((r.get("title") or "") + " " + (r.get("snippet") or "")).lower()
                link = (r.get("link") or "").lower()
                if (ulow in link or ulow in blob) and (name_tok and len(name_tok & set(re.findall(r"[a-z0-9]+", blob))) >= max(1, len(name_tok) - 1)):
                    tied = True
                    break
            if not tied:
                log.info(f"[serp_haiku] name-gate dropped {plat}|@{username}")
                continue

        key = f"{plat.lower()}|{ulow}"
        if key in existing_keys or key in new_keys:
            continue
        new_keys.add(key)
        new_candidates.append({
            "platform": plat,
            "url": url,
            "username": username,
            "source": "serp_haiku",
            "search_title": "",
            "search_snippet": "",
            "serp_image": "",
            "bio": "",
            "display_name": "",
            "image_url": "",
        })

    return new_candidates


def _is_real_li_avatar(url: str) -> bool:
    """Real LinkedIn headshots live on media.licdn.com; the static.licdn.com
    aero-v1 asset is the grey ghost / default avatar."""
    if not url:
        return False
    low = url.lower()
    return "media.licdn.com" in low and "static.licdn.com" not in low


# Cache the Scraping Browser CDP endpoint for the process lifetime (avoids an
# extra BD API round-trip per call). Value: full wss:// URL with credentials.
_SB_CDP_URL: str | None = None


async def _get_scraping_browser_cdp() -> str | None:
    """Build the BrightData Scraping Browser CDP websocket URL from the zone
    password + account customer id (fetched once, cached)."""
    global _SB_CDP_URL
    if _SB_CDP_URL:
        return _SB_CDP_URL
    import logging
    log = logging.getLogger("spl")
    try:
        async with httpx.AsyncClient() as client:
            pw_resp = await client.get(
                f"https://api.brightdata.com/zone/passwords?zone={SCRAPING_BROWSER_ZONE}",
                headers={"Authorization": f"Bearer {BRIGHTDATA_API_KEY}"}, timeout=15,
            )
            if pw_resp.status_code != 200:
                log.warning(f"Scraping Browser password fetch failed: {pw_resp.status_code}")
                return None
            pw = (pw_resp.json().get("passwords") or [None])[0]
            st_resp = await client.get(
                "https://api.brightdata.com/status",
                headers={"Authorization": f"Bearer {BRIGHTDATA_API_KEY}"}, timeout=15,
            )
            cust = st_resp.json().get("customer") if st_resp.status_code == 200 else None
        if not pw or not cust:
            log.warning("Scraping Browser: missing password or customer id")
            return None
        _SB_CDP_URL = (
            f"wss://brd-customer-{cust}-zone-{SCRAPING_BROWSER_ZONE}:{pw}@brd.superproxy.io:9222"
        )
        return _SB_CDP_URL
    except Exception as e:
        log.error(f"Scraping Browser CDP setup failed: {e}")
        return None


async def scraping_browser_linkedin_avatar(username: str) -> str:
    """Render the public LinkedIn profile page via the BrightData Scraping
    Browser and extract the real media.licdn.com profile-displayphoto URL.

    Used only as a fallback when the dataset API returns the ghost/default
    avatar — the dataset and ScrapeCreators both surface only the
    static.licdn.com placeholder for some profiles (e.g. profgalloway), but the
    rendered public page exposes the genuine headshot. (LI avatar fix)"""
    import logging
    log = logging.getLogger("spl")
    cdp = await _get_scraping_browser_cdp()
    if not cdp:
        return ""
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(cdp, timeout=60000)
            try:
                page = await browser.new_page()
                await page.goto(
                    f"https://www.linkedin.com/in/{username}/",
                    wait_until="domcontentloaded", timeout=60000,
                )
                try:
                    await page.wait_for_load_state("networkidle", timeout=20000)
                except Exception:
                    pass
                await asyncio.sleep(2)
                content = ""
                for _ in range(5):
                    try:
                        content = await page.content()
                        break
                    except Exception:
                        await asyncio.sleep(2)
                # v7.59: also grab og:image meta -- on a LinkedIn profile
                # page this is the SUBJECT's own representative image (when
                # the page exposes a real one), the single most reliable
                # source and immune to sidebar/"People also viewed" noise.
                og_image = ""
                try:
                    og_image = await page.locator(
                        'meta[property="og:image"]'
                    ).first.get_attribute("content") or ""
                except Exception:
                    og_image = ""
            finally:
                await browser.close()
        _log_spl_bd_usage("linkedin_scraping_browser")
        # v7.59 LinkedIn wrong-photo fix.
        #
        # A rendered public LinkedIn profile contains MANY
        # media.licdn.com profile-displayphoto URLs: not just the
        # subject's own, but everyone in the "People also viewed" /
        # browsemap aside, connections, recommended profiles, etc. The
        # old code grabbed the FIRST profile-displayphoto in DOM order,
        # which on many profiles (e.g. uriblau) is a sidebar person's
        # photo -> the wrong face was attached to the subject's chip and
        # poisoned the face-rerank anchor. We now scope extraction to the
        # SUBJECT's own photo only, and return "" (initials) rather than
        # risk a wrong person. Better blank than wrong.
        dp_re = re.compile(
            r'https://media\.licdn\.com/dms/image/[^\s"\'\\]*'
            r'profile-displayphoto[^\s"\'\\]*'
        )

        # 1) og:image meta -- the subject's representative image. Accept
        #    only a real media.licdn.com profile-displayphoto (never the
        #    static.licdn.com ghost / default avatar).
        og = _html.unescape((og_image or "").strip())
        if _is_real_li_avatar(og) and "profile-displayphoto" in og:
            log.info(
                f"Scraping Browser avatar for {username}: og:image (subject)"
            )
            return og

        # 2) Top-card-scoped fallback. The subject's photo, when present in
        #    the rendered DOM, appears in the profile top card -- which
        #    comes BEFORE the "People also viewed" / browsemap aside and
        #    its sidebar avatars. So only accept a profile-displayphoto URL
        #    that occurs before the first such related-profiles boundary.
        #    Everything at or after that boundary is a related person and
        #    is explicitly excluded -- never a page-wide first-match.
        boundary = len(content)
        for marker in ("browsemap", "PEOPLE_ALSO_VIEWED",
                       "people-also-viewed", "pv-browsemap",
                       "<aside"):
            i = content.find(marker)
            if i != -1:
                boundary = min(boundary, i)
        top_card = content[:boundary]
        m = dp_re.search(top_card)
        if m:
            url = _html.unescape(m.group(0))
            log.info(
                f"Scraping Browser avatar for {username}: top-card photo"
            )
            return url

        # 3) Neither the subject's og:image nor a top-card photo was found
        #    (the page exposed only related-profile photos / ghosts).
        #    Return blank so the frontend shows initials -- never a
        #    sidebar person's face.
        log.warning(
            f"Scraping Browser: no SUBJECT photo for {username} "
            f"(og:image ghost/absent, no top-card displayphoto); "
            f"returning blank to avoid wrong-person photo"
        )
    except Exception as e:
        log.error(f"Scraping Browser avatar fetch failed for {username}: {e}")
    return ""


async def brightdata_linkedin(username: str) -> dict | None:
    """Fetch detailed LinkedIn profile via BrightData datasets API.

    The dataset API frequently returns the static.licdn.com ghost avatar
    (default_avatar=True) even for profiles that have a real public headshot.
    When that happens we fall back to the Scraping Browser, which renders the
    public page and recovers the genuine media.licdn.com avatar. (LI avatar fix)"""
    import logging
    log = logging.getLogger("spl")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={LINKEDIN_DATASET_ID}&format=json",
                headers={
                    "Authorization": f"Bearer {BRIGHTDATA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=[{"url": f"https://www.linkedin.com/in/{username}/"}],
                timeout=15,
            )
            data = resp.json()
            snapshot_id = data.get("snapshot_id")
            _log_spl_bd_usage("linkedin_enrichment")
            if not snapshot_id:
                log.warning(f"LinkedIn enrichment: no snapshot_id for {username}")
                return None

            # Poll (max ~30 seconds wall-clock).
            # 2026-05-31 v7.32: was range(6) × sleep(1) ≈ 6s effective —
            # too short, BD typically needs 8-10s for LinkedIn, leading to
            # spurious "couldn't find profile" errors on perfectly valid
            # public profiles (e.g. gur-megiddo-1929a379 → ~9.6s).
            for attempt in range(15):
                await asyncio.sleep(2)
                poll_resp = await client.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json",
                    headers={"Authorization": f"Bearer {BRIGHTDATA_API_KEY}"},
                    timeout=15,
                )
                if poll_resp.status_code == 200:
                    poll_data = poll_resp.json()
                    if isinstance(poll_data, list) and poll_data and poll_data[0].get("name"):
                        p = poll_data[0]
                        bd_avatar = p.get("avatar", "")
                        log.info(f"LinkedIn enrichment OK for {username}: name={p.get('name')[:30]}, avatar={'yes' if bd_avatar else 'no'}, default_avatar={p.get('default_avatar')}, position={p.get('position', '')[:30]}")
                        from profile_scraper import format_followers
                        # ITER2 2026-05-28: SB ghost-avatar fallback DISABLED for latency.
                        # Cost when active: +30-60s per ghost profile (Galloway, Shapiro).
                        # When disabled, profiles with default_avatar=True keep the BD ghost URL.
                        image_url = bd_avatar
                        # if not _is_real_li_avatar(image_url) or p.get("default_avatar"):
                        #     real = await scraping_browser_linkedin_avatar(username)
                        #     if real:
                        #         image_url = real
                        return {
                            "bio": p.get("about", ""),
                            "display_name": p.get("name", ""),
                            "image_url": image_url,
                            "position": p.get("position", ""),
                            "city": p.get("city", ""),
                            "company": p.get("current_company_name", ""),
                            "followers": format_followers(p.get("followers")),
                        }
                    elif isinstance(poll_data, dict) and poll_data.get("status") == "running":
                        continue
                    else:
                        log.warning(f"LinkedIn enrichment unexpected response for {username}: {str(poll_data)[:200]}")
                        break
            log.warning(f"LinkedIn enrichment timeout for {username} after 30s")
    except Exception as e:
        log.error(f"LinkedIn enrichment exception for {username}: {e}")
    return None




async def brightdata_tiktok(username: str) -> dict | None:
    """Fetch TikTok profile via BrightData datasets API."""
    import logging
    log = logging.getLogger("spl")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={TIKTOK_DATASET_ID}&format=json",
                headers={
                    "Authorization": f"Bearer {BRIGHTDATA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=[{"url": f"https://www.tiktok.com/@{username}"}],
                timeout=15,
            )
            data = resp.json()
            snapshot_id = data.get("snapshot_id")
            if not snapshot_id:
                _log_spl_bd_usage("tiktok_enrichment")
                log.warning(f"TikTok enrichment: no snapshot_id for {username}")
                return None

            for attempt in range(6):
                await asyncio.sleep(1)
                poll_resp = await client.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json",
                    headers={"Authorization": f"Bearer {BRIGHTDATA_API_KEY}"},
                    timeout=15,
                )
                if poll_resp.status_code == 200:
                    poll_data = poll_resp.json()
                    if isinstance(poll_data, list) and poll_data and poll_data[0].get("nickname"):
                        p = poll_data[0]
                        from profile_scraper import format_followers
                        log.info(f"TikTok enrichment OK for {username}: name={p.get('nickname', '')[:30]}, followers={p.get('followers', '')}")
                        return {
                            "bio": p.get("biography", ""),
                            "display_name": p.get("nickname", ""),
                            "image_url": p.get("profile_pic_url_hd", "") or p.get("profile_pic_url", ""),
                            "followers": format_followers(p.get("followers")),
                            "is_verified": p.get("is_verified", False),
                            "videos_count": p.get("videos_count", 0),
                        }
                    elif isinstance(poll_data, dict) and poll_data.get("status") == "running":
                        continue
                    else:
                        log.warning(f"TikTok enrichment unexpected response for {username}: {str(poll_data)[:200]}")
                        break
            log.warning(f"TikTok enrichment timeout for {username} after 30s")
    except Exception as e:
        log.error(f"TikTok enrichment exception for {username}: {e}")
    return None

async def brightdata_instagram(username: str) -> dict | None:
    """Fetch Instagram profile (incl. real avatar) via BrightData datasets API."""
    import logging
    log = logging.getLogger("spl")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={INSTAGRAM_DATASET_ID}&format=json",
                headers={
                    "Authorization": f"Bearer {BRIGHTDATA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=[{"url": f"https://www.instagram.com/{username}/"}],
                timeout=5,  # ITER22 2026-05-28: was 15s; trigger should be fast
            )
            data = resp.json()
            snapshot_id = data.get("snapshot_id")
            _log_spl_bd_usage("instagram_enrichment")
            if not snapshot_id:
                log.warning(f"Instagram enrichment: no snapshot_id for {username}")
                return None

            # ITER25 2026-05-28: IG poll cap range(4) — balance between BD success rate and wall-clock
            for attempt in range(4):
                await asyncio.sleep(1)
                poll_resp = await client.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json",
                    headers={"Authorization": f"Bearer {BRIGHTDATA_API_KEY}"},
                    timeout=15,
                )
                if poll_resp.status_code == 200:
                    poll_data = poll_resp.json()
                    if isinstance(poll_data, list) and poll_data and (
                        poll_data[0].get("full_name") or poll_data[0].get("profile_name") or poll_data[0].get("account")
                    ):
                        p = poll_data[0]
                        from profile_scraper import format_followers
                        avatar = p.get("profile_image_link") or p.get("profile_pic_url_hd") or p.get("profile_pic_url", "")
                        log.info(f"Instagram enrichment OK for {username}: name={(p.get('full_name') or p.get('profile_name') or '')[:30]}, avatar={'yes' if avatar else 'no'}")
                        return {
                            "bio": p.get("biography", ""),
                            "display_name": p.get("full_name", "") or p.get("profile_name", ""),
                            "image_url": avatar,
                            "followers": format_followers(p.get("followers")),
                            "is_verified": p.get("is_verified", False),
                            "is_private": bool(p.get("is_private")),
                        }
                    elif isinstance(poll_data, dict) and poll_data.get("status") == "running":
                        continue
                    else:
                        log.warning(f"Instagram enrichment unexpected response for {username}: {str(poll_data)[:200]}")
                        break
            log.warning(f"Instagram enrichment timeout for {username} after 30s")
    except Exception as e:
        log.error(f"Instagram enrichment exception for {username}: {e}")
    return None


async def find_profiles(usernames: list[str], name: str = "", description: str = "", known_links: dict = None, parallel_after_serp=None) -> list[dict]:
    """
    New pipeline:
    1. If user provided known links, use those directly
    2. Google search for each platform
    3. Enrich LinkedIn with BrightData if found
    4. Return all found profiles
    """
    known_links = known_links or {}
    profiles = []

    async with httpx.AsyncClient() as client:
        # Phase 1: Search Google for each platform concurrently
        search_tasks = {}
        for platform, domain in PLATFORMS.items():
            if platform in known_links and known_links[platform]:
                # User provided a direct link — use it
                url = known_links[platform]
                pattern = PROFILE_URL_PATTERNS.get(platform)
                username = ""
                if pattern:
                    match = pattern.search(url)
                    if match:
                        username = match.group(1)
                profiles.append({
                    "platform": platform,
                    "url": url,
                    "username": username,
                    "source": "user_provided",
                    "bio": "",
                    "display_name": "",
                    "image_url": "",
                    "verified": True,
                })
            else:
                search_tasks[platform] = search_platform(name, description, platform, domain, client)

        if search_tasks:
            results = await asyncio.gather(*search_tasks.values())
            for platform, candidates in zip(search_tasks.keys(), results):
                # search_platform now returns a list of candidates
                profiles.extend(candidates)

    # Phases 2/3/4 + optional parallel-after-serp work, all in parallel via asyncio.gather.
    # ITER8 2026-05-28: parallel_after_serp callback (typically verify_profiles) runs
    # CONCURRENTLY with BD enrichment so the LLM verification overlaps the BD poll wait.
    first_li = next((p for p in profiles if p["platform"] == "LinkedIn"  and p.get("username")), None)
    first_tt = next((p for p in profiles if p["platform"] == "TikTok"    and p.get("username")), None)
    first_ig = next((p for p in profiles if p["platform"] == "Instagram" and p.get("username")), None)

    async def _none(): return None
    async def _enrich():
        # ITER25 2026-05-28: IG-only BD enrichment for real IG avatars.
        # LI/TT covered by unavatar.io URLs in main.py (fast, no BD wait).
        ig_task = brightdata_instagram(first_ig["username"]) if first_ig else _none()
        ig_res, = await asyncio.gather(ig_task)
        if first_ig and ig_res:
            for k, v in ig_res.items():
                if v: first_ig[k] = v
            first_ig["source"] = "brightdata"

    # ITER20 2026-05-28: IG BD enrichment re-enabled (in parallel with verify).
    # X/FB/TT/LI avatars come from fallback URLs (unavatar.io, graph.facebook.com).
    if parallel_after_serp is not None:
        await asyncio.gather(_enrich(), parallel_after_serp(profiles))
    else:
        await _enrich()

    return profiles
