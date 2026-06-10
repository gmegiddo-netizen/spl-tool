import asyncio
import os
import re
import sqlite3
from playwright.async_api import async_playwright

# Path to the accountability database (shared X accounts)
ACCOUNTABILITY_DB = "/var/www/accountability/accountability.db"

# Simple in-memory cache for profile pictures (key: platform/username -> dict)
_AVATAR_CACHE: dict = {}


def _get_x_auth_token() -> str:
    """Read an active X auth_token from the accountability database."""
    if not os.path.exists(ACCOUNTABILITY_DB):
        return ""
    try:
        conn = sqlite3.connect(ACCOUNTABILITY_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT auth_token FROM x_accounts "
            "WHERE status IN ('active', 'backup') AND auth_token IS NOT NULL "
            "ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        conn.close()
        return row["auth_token"] if row else ""
    except Exception:
        return ""


def format_followers(num) -> str:
    """Format a follower count: 40164791 -> 40.2M, 525200 -> 525K, etc."""
    if num is None or num == "":
        return ""
    s = str(num).replace(",", "").strip()
    # If already formatted (contains K/M/B), keep it
    if any(c in s.upper() for c in "KMB"):
        return s
    try:
        n = float(s)
        if n >= 1_000_000_000:
            return f"{n/1_000_000_000:.1f}B"
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}K".replace(".0K", "K")
        return str(int(n))
    except (ValueError, TypeError):
        return ""


FOLLOWER_PATTERNS = [
    re.compile(r'([\d,\.]+[KMB]?)\s*(?:followers|likes|fans|subscribers)', re.IGNORECASE),
]


def extract_followers(text: str) -> str:
    """Extract follower/likes count from a text snippet."""
    if not text:
        return ""
    for pattern in FOLLOWER_PATTERNS:
        m = pattern.search(text)
        if m:
            return format_followers(m.group(1))
    return ""


def clean_display_name(raw: str, username: str = "") -> str:
    """Strip platform suffixes like '(@handle) / Posts / X' from display names."""
    if not raw:
        return username or ""
    name = raw
    # Remove "(@username) / Posts / X" style endings
    name = re.sub(r'\s*\(@[^\)]+\).*$', '', name)
    # Remove "/ Posts / X", "/ Twitter", etc
    name = re.sub(r'\s*[/\|·]\s*(Posts|Twitter|X|Facebook|LinkedIn|Instagram|TikTok).*$', '', name, flags=re.IGNORECASE)
    # Remove trailing emoji-only words (like 🟠)
    name = name.strip()
    return name or (username or raw)


# ==================== AVATAR FETCHING (NO 3RD PARTY) ====================

async def fetch_x_authenticated(profiles: list[dict]) -> None:
    """Fetch X avatars + followers using authenticated session."""
    import logging
    log = logging.getLogger("spl")
    x_profiles = [p for p in profiles if p["platform"] == "X" and not p.get("image_url") and p.get("username")]
    if not x_profiles:
        return

    auth_token = _get_x_auth_token()
    if not auth_token:
        log.warning("X auth token unavailable")
        return

    log.info(f"X authenticated fetch: {len(x_profiles)} profiles")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            await context.add_cookies([{
                "name": "auth_token",
                "value": auth_token,
                "domain": ".x.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
            }])
            tasks = [_fetch_x_profile(context, prof) for prof in x_profiles]
            await asyncio.gather(*tasks)
            await browser.close()
        for prof in x_profiles:
            log.info(f"  X {prof['username']}: img={'YES' if prof.get('image_url') else 'NO'} fol={prof.get('followers','-')}")
    except Exception as e:
        log.error(f"X auth fetch failed: {e}")


async def _fetch_x_profile(context, profile: dict) -> None:
    """Load X profile page, extract avatar and follower count."""
    username = profile['username']
    try:
        page = await context.new_page()
        await page.goto(f"https://x.com/{username}", wait_until="domcontentloaded", timeout=15000)
        try:
            await page.wait_for_selector(f'[data-testid="UserAvatar-Container-{username}"]', timeout=8000)
        except Exception:
            pass

        # Avatar — must target THIS profile's avatar specifically
        try:
            el = await page.query_selector(f'[data-testid="UserAvatar-Container-{username}"] img')
            if el:
                src = await el.get_attribute("src")
                if src and "profile_images" in src:
                    profile["image_url"] = src
        except Exception:
            pass

        # Followers — multiple selectors, case-insensitive
        if not profile.get("followers"):
            try:
                # Find all anchor tags that link to followers/verified_followers
                follower_links = await page.query_selector_all(f'a[href*="/{username}/followers"], a[href*="/{username}/verified_followers"]')
                for link in follower_links:
                    text = await link.inner_text()
                    m = re.search(r'([\d,\.]+[KMB]?)', text)
                    if m:
                        profile["followers"] = format_followers(m.group(1))
                        break
            except Exception:
                pass

        await page.close()
    except Exception:
        pass


async def fetch_facebook_avatars(profiles: list[dict]) -> None:
    """Fetch Facebook avatars + follower count via Playwright (facebookexternalhit UA)."""
    import logging
    log = logging.getLogger("spl")
    fb_profiles = [p for p in profiles if p["platform"] == "Facebook" and not p.get("image_url") and p.get("username")]
    if not fb_profiles:
        return

    log.info(f"Facebook fetch: {len(fb_profiles)} profiles")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                # A real browser UA returns the actual fbcdn.net profile photo in
                # og:image. The old facebookexternalhit bot UA returned a
                # lookaside.fbsbx.com/crawler URL that serves HTML (not an image),
                # so the <img> never rendered. (avatar fix 2026-05-27)
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
            tasks = [_fetch_fb_profile(context, prof) for prof in fb_profiles]
            await asyncio.gather(*tasks)
            await browser.close()
        for prof in fb_profiles:
            log.info(f"  FB {prof['username']}: img={'YES' if prof.get('image_url') else 'NO'} fol={prof.get('followers','-')}")
    except Exception as e:
        log.error(f"FB fetch failed: {e}")


def _is_real_avatar_url(url: str, platform: str) -> bool:
    """Reject known placeholder / non-image URLs so the form falls back to the
    initials placeholder instead of showing a broken image. (avatar fix)"""
    if not url:
        return False
    low = url.lower()
    # Facebook crawler endpoint returns HTML, not an image
    if "lookaside.fbsbx.com" in low or "lookaside.facebook.com" in low:
        return False
    # LinkedIn ghost / generic static placeholder (BrightData returns this when
    # it can't capture the real avatar) — static.licdn.com/aero-v1/... is the
    # grey silhouette SVG, never a real headshot (which lives on media.licdn.com)
    if "static.licdn.com" in low:
        return False
    # Instagram logged-out og:image is the IG logo bundled asset, not an avatar
    if "/rsrc.php/" in low or low.endswith("instagram_logo.png"):
        return False
    return True


async def _fetch_fb_profile(context, profile: dict) -> None:
    """Load a Facebook profile page and extract og:image + followers."""
    try:
        page = await context.new_page()
        await page.goto(f"https://www.facebook.com/{profile['username']}", wait_until="domcontentloaded", timeout=12000)

        og = await page.query_selector('meta[property="og:image"]')
        if og:
            img = await og.get_attribute("content")
            if img and _is_real_avatar_url(img, "Facebook"):
                profile["image_url"] = img

        if not profile.get("followers"):
            meta_desc = await page.query_selector('meta[name="description"]')
            if meta_desc:
                desc = await meta_desc.get_attribute("content") or ""
                m = re.search(r'([\d,\.]+[KMB]?)\s*(?:likes|followers)', desc, re.IGNORECASE)
                if m:
                    profile["followers"] = format_followers(m.group(1))
        await page.close()
    except Exception:
        pass


# ==================== MAIN ENTRY ====================

async def scrape_profiles(profiles: list[dict]) -> list[dict]:
    """Enrich profiles with avatars, followers, and clean names. NO 3rd party deps."""
    for profile in profiles:
        profile.setdefault("bio", "")
        profile.setdefault("display_name", "")
        profile.setdefault("image_url", "")
        profile.setdefault("followers", "")

        # Drop placeholder / non-image avatar URLs (e.g. LinkedIn ghost SVG from
        # BrightData, FB lookaside HTML) so they never reach the form. (avatar fix)
        if profile.get("image_url") and not _is_real_avatar_url(profile["image_url"], profile.get("platform", "")):
            profile["image_url"] = ""

        # Clean display name
        if profile.get("display_name"):
            profile["display_name"] = clean_display_name(profile["display_name"], profile.get("username", ""))

        # Format any pre-existing follower numbers (e.g., from BrightData LinkedIn)
        if profile.get("followers"):
            profile["followers"] = format_followers(profile["followers"])

        # Extract followers from bio if still missing (Google snippet)
        if not profile.get("followers") and profile.get("bio"):
            followers = extract_followers(profile["bio"])
            if followers:
                profile["followers"] = followers

        # Apply cache
        if profile.get("username"):
            cache_key = f"{profile['platform']}/{profile['username']}"
            if cache_key in _AVATAR_CACHE:
                cached = _AVATAR_CACHE[cache_key]
                if not profile.get("image_url"):
                    profile["image_url"] = cached.get("image_url", "")
                if not profile.get("followers"):
                    profile["followers"] = cached.get("followers", "")

    # Run X authenticated and Facebook fetches in parallel
    await asyncio.gather(
        fetch_x_authenticated(profiles),
        fetch_facebook_avatars(profiles),
    )

    # Cache successful fetches
    for profile in profiles:
        if profile.get("username") and (profile.get("image_url") or profile.get("followers")):
            cache_key = f"{profile['platform']}/{profile['username']}"
            _AVATAR_CACHE[cache_key] = {
                "image_url": profile.get("image_url", ""),
                "followers": profile.get("followers", ""),
            }

    return profiles
