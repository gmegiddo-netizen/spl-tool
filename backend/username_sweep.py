"""Cross-platform username sweep — given a confirmed handle, probe
~40 platforms in parallel and return the ones where the handle exists.

Built 2026-05-29 for SPL stage-2 "low-key profile" discovery. The big-5
platforms (X / FB / IG / LI / TT) are deliberately NOT in this list —
they're already covered by stage-1 SERP and the handle-extrapolation
in stage_two.py. This module catches the long tail of platforms
ordinary people reuse handles on (Reddit, GitHub, Twitch, SoundCloud,
etc.) that don't show up in Google SERP.

Each platform has a (name, url_template, success_status_codes) tuple.
We do a HEAD request (falling back to GET on 405) and treat
`status_code in success_status_codes` as "handle exists." Platforms
where 200 doesn't reliably mean "exists" (SPA frontends that render
"not found" inside HTML) are excluded from this list — adding them
without a content-regex check produces too many false positives.

No external API calls. The HTTP request cost is server bandwidth +
target site's tolerance. Tight 4s timeout per platform keeps total
wall ~4–6s in the parallel sweep.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Iterable, Optional

import httpx

log = logging.getLogger(__name__)

# (display_name, url_template, success_status_codes)
# Curated for status-code reliability — anywhere 200 means "user found,"
# 404 means "doesn't exist." Platforms with always-200 SPA frontends are
# excluded to keep false-positive rate low.
PLATFORMS: list[tuple[str, str, set[int]]] = [
    # Developer / professional
    ("GitHub",        "https://github.com/{h}",                        {200}),
    ("GitLab",        "https://gitlab.com/{h}",                        {200}),
    ("Dev.to",        "https://dev.to/{h}",                            {200}),
    ("Medium",        "https://medium.com/@{h}",                       {200}),
    ("Substack",      "https://{h}.substack.com/",                     {200}),
    ("AngelList",     "https://wellfound.com/u/{h}",                   {200}),
    ("ProductHunt",   "https://www.producthunt.com/@{h}",              {200}),
    ("StackOverflow", "https://stackoverflow.com/users/{h}",           {200}),
    ("Behance",       "https://www.behance.net/{h}",                   {200}),
    ("Dribbble",      "https://dribbble.com/{h}",                      {200}),

    # Reading / writing
    ("Goodreads",     "https://www.goodreads.com/{h}",                 {200}),
    ("Wattpad",       "https://www.wattpad.com/user/{h}",              {200}),

    # Media / music / video
    ("YouTube",       "https://www.youtube.com/@{h}",                  {200}),
    ("Vimeo",         "https://vimeo.com/{h}",                         {200}),
    ("SoundCloud",    "https://soundcloud.com/{h}",                    {200}),
    ("Last.fm",       "https://www.last.fm/user/{h}",                  {200}),
    ("Bandcamp",      "https://{h}.bandcamp.com/",                     {200}),
    ("Mixcloud",      "https://www.mixcloud.com/{h}/",                 {200}),

    # Social / community
    ("Reddit",        "https://www.reddit.com/user/{h}/about.json",    {200}),
    ("Threads",       "https://www.threads.net/@{h}",                  {200}),
    ("BlueSky",       "https://bsky.app/profile/{h}.bsky.social",      {200}),
    ("Mastodon",      "https://mastodon.social/@{h}",                  {200}),
    ("Tumblr",        "https://{h}.tumblr.com/",                       {200}),
    ("Pinterest",     "https://www.pinterest.com/{h}/",                {200}),
    ("Quora",         "https://www.quora.com/profile/{h}",             {200}),
    ("Telegram",      "https://t.me/{h}",                              {200}),

    # Live-streaming / gaming
    ("Twitch",        "https://www.twitch.tv/{h}",                     {200}),
    ("Kick",          "https://kick.com/{h}",                          {200}),
    ("Steam",         "https://steamcommunity.com/id/{h}",             {200}),
    ("Chess.com",     "https://www.chess.com/member/{h}",              {200}),
    ("Roblox",        "https://www.roblox.com/users/profile?username={h}", {200, 302}),

    # Fitness / health
    ("Strava",        "https://www.strava.com/athletes/{h}",           {200}),
    ("MyFitnessPal",  "https://www.myfitnesspal.com/profile/{h}",      {200}),

    # Photography
    ("Flickr",        "https://www.flickr.com/people/{h}",             {200}),
    ("500px",         "https://500px.com/p/{h}",                       {200}),

    # Misc
    ("Patreon",       "https://www.patreon.com/{h}",                   {200}),
    ("Kofi",          "https://ko-fi.com/{h}",                         {200}),
    ("Buymeacoffee",  "https://www.buymeacoffee.com/{h}",              {200}),
    ("Linktree",      "https://linktr.ee/{h}",                         {200}),
    ("Cashapp",       "https://cash.app/${h}",                         {200}),
    ("Venmo",         "https://account.venmo.com/u/{h}",               {200}),

    # Dating-adjacent profiles that are public
    ("About.me",      "https://about.me/{h}",                          {200}),
]

# Handles known to be invalid before we waste a probe.
_HANDLE_RE = re.compile(r"^[A-Za-z0-9_.\-]{2,40}$")

# Conservative request timeouts — slow sites should not anchor the
# total wall time. Per-platform 4s; whole sweep waits for at most
# whichever is slower.
_PER_REQUEST_TIMEOUT = 4.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)


async def _probe(
    client: httpx.AsyncClient,
    name: str,
    tmpl: str,
    ok_codes: set[int],
    handle: str,
) -> Optional[dict]:
    url = tmpl.format(h=handle)
    try:
        # HEAD first (cheap); fall back to GET on 405 Method Not Allowed
        # or on any error.
        r = await client.head(url, timeout=_PER_REQUEST_TIMEOUT, follow_redirects=True)
        if r.status_code == 405:
            r = await client.get(url, timeout=_PER_REQUEST_TIMEOUT, follow_redirects=True)
    except Exception:
        return None

    if r.status_code in ok_codes:
        return {
            "platform_other": name,                   # NOT in the big-5 set
            "platform": name,                         # for chip rendering
            "url": url,
            "username": handle,
            "display_name": handle,
            "image_url": "",
            "bio": "",
            "followers": "",
            "score": 72,                              # "Possible" / handle-match
            "reasoning": f"Same handle exists on {name}.",
            "_stage2_new": True,
            "_sweep": True,
        }
    return None


async def sweep_username(
    handle: str,
    exclude_platforms: Iterable[str] = (),
) -> list[dict]:
    """Probe every platform in PLATFORMS in parallel; return the ones
    where the handle exists. Skips platforms in `exclude_platforms`
    (typically the big-5 already covered by stage-1).
    """
    handle = (handle or "").lstrip("@").strip()
    if not handle or not _HANDLE_RE.match(handle):
        return []
    skip = {p.lower() for p in exclude_platforms}
    transport = httpx.AsyncHTTPTransport(retries=0)
    async with httpx.AsyncClient(
        transport=transport,
        headers={"User-Agent": _USER_AGENT, "Accept-Language": "en"},
        follow_redirects=True,
        timeout=httpx.Timeout(_PER_REQUEST_TIMEOUT),
    ) as client:
        tasks = [
            _probe(client, name, tmpl, codes, handle)
            for name, tmpl, codes in PLATFORMS
            if name.lower() not in skip
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, dict):
            out.append(r)
    return out
