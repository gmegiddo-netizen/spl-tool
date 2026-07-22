import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from username_gen import generate_usernames
from profile_finder import find_profiles
from profile_scraper import scrape_profiles
from verifier import verify_profiles, verify_profiles_single_platform
from similarity import attach_similarity_payload  # 2026-05-30 carousel rerank
import namefix_helper  # 2026-06-03 name-spelling/cap fix (Candice->Candace)
import stage_two  # noqa: F401 — registers the /api/search-again handler below

load_dotenv()

# v7.61 all-followers: follower counts now render on EVERY network, not
# just TikTok. A bounded, cached per-network fetch (_fetch_followers)
# fills X / LinkedIn / Facebook (and any IG/TT miss) for the ≤2 chips
# actually emitted per platform, and for manual-adds via verify_url.
app = FastAPI(title="SPL - Social Profile Lookup")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
CONFIDENCE_THRESHOLD = 50  # ITER26 2026-05-28: was 60; let marginal-but-legit profiles through


# 2026-05-30 v7: Facebook returns a generic SEO meta description when
# the profile page isn't publicly readable — "X is on Facebook. Join
# Facebook to connect with X and others you may know. Facebook gives
# people the power to share and makes the world more open and
# connected." We surface that as the "bio", which is misleading: it's
# not the user's bio at all. Detect + strip the boilerplate; preserve
# any useful tail like "Lives in <city>".
# 2026-05-31 v7.18: extended to every social network. Each platform
# returns a generic SEO meta description in its open-graph tags when
# the SERP snippet is the only bio source — we shouldn't show that as
# the user's bio. Detect + strip; preserve useful tails like "Lives in
# <city>" (FB) and "<N> followers · <M> following" (IG/TT).
_SEO_PATTERNS = (
    # Facebook
    "is on facebook",
    "join facebook to connect",
    "facebook gives people the power",
    "others you may know",
    # Instagram
    "see photos and videos from",
    "on instagram, and discover other accounts",
    "discover other accounts you'll love",
    "discover other accounts youll love",
    "photos and videos from friends on instagram",
    "'s profile picture",
    "profile picture. ",
    " posts · @",
    " posts. @",
    # TikTok
    "watch the latest video from",
    "check out their videos, sound off in the comments",
    "latest videos from",
    # X / Twitter
    "the latest posts from",
    "the latest tweets from",
    # LinkedIn
    "the world's largest professional community",
    "view the profiles of professionals named",
    "view profile on linkedin",
)

# 2026-05-31 v7.35: news-snippet leak detection.
# When SC/BD have no bio for an account, profile_finder falls back to
# the Google SERP snippet (profile_finder.py ~line 285: bio =
# result["description"]). For chips that are journalists or news-style
# accounts, the SERP snippet is often the latest news ARTICLE / POST
# about (or by) them, not a profile bio. Example seen 2026-05-31:
#   "אפריל פורסמה ב\"פטריוטים\" הקלטה של אליעד שרגא ..."
# (an April news-event report, not a bio at all).
#
# Detection is two-signal-minimum and length-gated (>= 120 chars) so
# legit short bios that share a single keyword (e.g. someone whose bio
# mentions "April") stay intact. See _looks_like_news_snippet below.
_HE_MONTHS = (
    "ינואר", "פברואר", "מרץ", "מארס", "אפריל", "מאי", "יוני",
    "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
)
_HE_REPORT_VERBS = (
    "פורסם", "פורסמה", "פורסמו",
    "דווח", "דווחה", "דווחו",
    "נחשף", "נחשפה", "נחשפו",
    "הוקלט", "הוקלטה", "הוקלטו",
    "תיעד", "תיעדה",
    "הודיע", "הודיעה",
    "אמר ל", "אמרה ל",
)
_EN_REPORT_PHRASES = (
    "said in a statement",
    "told reporters",
    "in an interview with",
    "was quoted as saying",
    "according to a report",
    "according to the report",
    "earlier this week",
    "earlier this month",
    "earlier this year",
)
_EN_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_EN_REPORT_VERBS_PAST = (
    " announced ", " revealed ", " disclosed ", " confirmed ",
    " published ", " unveiled ", " was published ", " was reported ",
)


def _has_quoted_publication(text: str) -> bool:
    import re as _re
    # Hebrew: ב"..." or ב'...' (quoted publication after the בּ preposition)
    if _re.search(r"ב[\"״][^\"״]{2,}[\"״]", text):
        return True
    if _re.search(r"ב['׳][^'׳]{2,}['׳]", text):
        return True
    return False


def _looks_like_news_snippet(bio: str) -> bool:
    """Conservative: require length >= 120 AND >= 2 independent signals.

    The length floor protects short legit bios. The two-signal floor
    protects long legit bios that happen to share a single keyword
    (e.g. a journalist bio that mentions a month, or one that quotes
    a book title)."""
    if not bio or len(bio) < 120:
        return False

    text = bio.strip()
    low = text.lower()
    signals = 0

    # S1: Hebrew month at the very start
    for m in _HE_MONTHS:
        if text.startswith(m) or text.startswith(" " + m):
            signals += 1
            break

    # S2: Hebrew reporting verb anywhere
    if any(v in text for v in _HE_REPORT_VERBS):
        signals += 1

    # S3: quoted publication name (Hebrew-style ב"...")
    if _has_quoted_publication(text):
        signals += 1

    # S4: English reporting phrase
    if any(p in low for p in _EN_REPORT_PHRASES):
        signals += 1

    # S5: English month at start + a past-tense reporting verb
    starts_with_month = any(
        low.startswith(m + " ") or low.startswith("in " + m + " ")
        for m in _EN_MONTHS
    )
    has_past_verb = any(v in (" " + low + " ") for v in _EN_REPORT_VERBS_PAST)
    if starts_with_month and has_past_verb:
        signals += 1

    # S6: "described as" / "תיאר כ"
    if "תיאר כ" in text or " described as " in low:
        signals += 1

    return signals >= 2


import re as _re  # v7.47: module-level for the bio-artifact regexes below


# ============================================================================
# v7.47 NEW: always-on UI / scraping-artifact stripper.
# Runs on EVERY bio (before the SEO/news gates) so it also cleans the
# common SERP-leak case where a profile bio is suffixed with UI chrome
# (Follow button text in many languages, Read-more expanders, app icon
# private-use glyphs, leaked metric counts/tab labels). Conservative:
# only strips when a UI token is STANDALONE / a trailing or leading
# fragment surrounded by separators -- never inside a real sentence.
# ============================================================================

# Private-use-area + zero-width/bidi/control glyphs (app icon chars like the
# reported leading glyph). Keep \n \t \r -- they are legit bio separators.
_PUA_RE = _re.compile(
    u"["
    u"-"            # BMP Private Use Area (app icon glyphs)
    u"\U000F0000-\U000FFFFD"    # Supplementary PUA-A
    u"\U00100000-\U0010FFFD"    # Supplementary PUA-B
    u"​‌"            # ZWSP, ZWNJ  (NOT ‍ ZWJ -- emoji joiner)
    u"‎‏"            # LRM / RLM
    u"‪-‮"            # bidi embeddings / overrides
    u"⁠-⁤"            # word-joiner / invisible operators
    u"﻿"                   # BOM / zero-width no-break space
    u"]"
    u"|[\x00-\x08\x0b\x0c\x0e-\x1f]"   # C0 controls except \t \n \r
)

# Boundary policy (CONSERVATIVE):
#   * A UI token is stripped only when it is a real "UI fragment", i.e.
#     - bounded by a HARD separator (bullet-type ·•‣⁃ , pipe |, or a newline)
#       on the side facing the rest of the bio, OR
#     - it sits at the very START or very END of the bio (trailing/leading
#       fragment), where intervening plain spaces are allowed.
#   * Plain spaces and hyphens do NOT, by themselves, make a word "standalone"
#     mid-sentence -- otherwise "Follow your dreams" / "I write about X" /
#     "50 million followers" would be wrongly gutted.
_HARD_SEP = u"·•‣⁃\\|\n\r"            # chars that strongly delimit UI chrome
_HSEP = u"[" + _HARD_SEP + u"]"        # single hard separator
# A " . " (dot flanked by spaces) and " - " (spaced hyphen) are also commonly
# how a SERP renders the bullet separator between bio and UI chrome
# (the reported case: "<glyph> . follow . Read more"). Treat a spaced dot/
# spaced-hyphen as a hard separator too -- but NOT a dot glued to a word
# (sentence-ending period like "Periodista." must NOT be a mid-sep).
_SOFT_SEP = u"(?:[ \\t]+[.\\-][ \\t]+)"
# left edge: string start, OR a hard separator, OR a spaced-dot separator.
_LEFT = u"(?:^[ \\t]*|" + _HSEP + u"[ \\t]*|" + _SOFT_SEP + u")"
# right edge (lookahead): string end, OR a hard sep, OR a spaced-dot separator.
_RIGHT = u"(?=[ \\t]*$|[ \\t]*" + _HSEP + u"|" + _SOFT_SEP + u")"
# Trailing edge: end-of-string reached after optional trailing spaces/punct.
_TRAIL = u"(?=[ \\t]*[.,;:]?[ \\t]*$)"

# Expander / generic UI verbs.
_EXPANDERS = [
    r"read\s+more", r"see\s+more", r"show\s+more", r"read\s+less",
    r"see\s+less", r"show\s+less", r"see\s+translation", r"view\s+translation",
    r"see\s+original", r"view\s+profile", r"show\s+this\s+thread",
]
# Dotted/ellipsis expanders ("...more", "… more", "...See more") -- these are
# unambiguous UI chrome, so strip them even mid-space at the trailing edge.
_DOTMORE_RE = _re.compile(
    u"[ \\t]*(?:\\.{2,}|…)[ \\t]*(?:see\\s+|show\\s+|read\\s+)?more[ \\t]*"
    + _RIGHT,
    _re.IGNORECASE | _re.UNICODE,
)
# Follow-button words across languages.
_FOLLOW_WORDS = [
    r"follow", r"following", r"followed", r"message",
    u"מעקב",                 # HE Follow (noun)
    u"עוקב",                  # HE Following
    u"עקוב",                  # HE Follow (imp)
    u"שלח\\s+הודעה",  # HE "send message"
    r"seguir", r"siguiendo",                       # ES
    u"suivre", u"s'abonner", u"abonné",     # FR
    r"folgen", r"abonnieren",                      # DE
    r"segui",                                       # IT
    u"متابعة", u"متابعون",  # AR
    u"Подписаться",  # RU
]
# Bare tab labels.
_TAB_LABELS = [r"posts", r"reels", r"followers", r"following", r"about", r"videos", r"reposts", r"highlights"]

# v7.60 biofix2: LinkedIn profile section headers / tab labels. These leak
# from the LinkedIn `about` enrichment + SERP snippets ("Personal details",
# "Contact info", "Activity", "Experience"...). Same conservative
# separator/edge-bounded matching as the other UI fragments -- so a real
# sentence like "I have experience in journalism" is NOT gutted.
_LI_LABELS = [
    r"personal\s+details", r"contact\s+info", r"contact\s+information",
    r"highlights", r"activity", r"experience", r"education", r"featured",
    r"skills", r"recommendations", r"interests", r"see\s+all", r"show\s+all",
    r"see\s+more", r"show\s+more", r"connect",
]

# v7.60 biofix2: the subset of LI labels that are SAFE to strip purely on
# TRAILING position (token reaches the end of the bio after optional
# punctuation), in addition to the separator-bounded matching above. These
# are multi-word headers / unambiguous CTAs that essentially never end a
# real human bio: "...journalist. Personal details" / "...Acme. Connect".
# The ambiguous single common words (experience, education, activity,
# interests, skills, featured, highlights, recommendations) are DELIBERATELY
# excluded here -- they only strip when separator-bounded, so a legit bio
# ending in "...years of experience" is preserved.
_LI_LABELS_TRAILING = [
    r"personal\s+details", r"contact\s+info", r"contact\s+information",
    r"see\s+all", r"show\s+all", r"see\s+more", r"show\s+more", r"connect",
]


def _ui_fragment_re(words):
    """Match any of `words` only as a hard/soft-separator-bounded UI fragment
    -- never a plain space-bounded word mid-sentence."""
    body = "|".join(words)
    return _re.compile(
        _LEFT + u"(?:" + body + u")" + _RIGHT,
        _re.IGNORECASE | _re.UNICODE,
    )


def _ui_trailing_re(words, require_wb=True):
    """Match any of `words` as a TRAILING fragment: the token reaches the end
    of the string (after optional trailing punctuation/spaces). A preceding
    word boundary is required so we only nibble a genuine trailing token, not
    part of a larger word. This is what removes 'Marketing strategist Show
    more', 'Periodista. Seguir', 'compositor See translation' -- while
    'Follow your dreams ...' is safe because Follow is NOT at the end."""
    body = "|".join(words)
    pre = u"(?:(?<=^)|(?<=\\s)|(?<=[.,;:]))" if require_wb else u""
    return _re.compile(
        pre + u"(?:" + body + u")" + _TRAIL,
        _re.IGNORECASE | _re.UNICODE,
    )

_EXPANDER_RE = _ui_fragment_re(_EXPANDERS)
_FOLLOW_RE = _ui_fragment_re(_FOLLOW_WORDS)
_TAB_RE = _ui_fragment_re(_TAB_LABELS)
_LI_LABEL_RE = _ui_fragment_re(_LI_LABELS)  # v7.60 biofix2
_LI_LABEL_TRAIL_RE = _ui_trailing_re(_LI_LABELS_TRAILING)  # v7.60 biofix2
# v7.60 biofix2: strip an LI label that occupies a WHOLE sentence/segment
# (flanked by ./!/?, a hard separator, or a string edge). Catches the
# period-glued leaks 'journalist. Personal details. Contact info' and
# 'News anchor. Highlights. See all'. Conservative: the label must be the
# entire segment, so mid-sentence uses like 'experience in X' are safe.
_LI_SENTENCE_RE = _re.compile(
    u"(?:^|(?<=[.!?\u00B7\u2022\u2023\u2043|\\n\\r]))"
    u"[ \\t]*(?:" + u"|".join(_LI_LABELS) + u")[ \\t]*"
    u"[ \\t]*[.!?]?(?=[ \\t]*(?:$|[.!?\u00B7\u2022\u2023\u2043|\\n\\r]))",
    _re.IGNORECASE | _re.UNICODE,
)
# v7.60 biofix2: FOLLOW/MESSAGE words as a WHOLE sentence/segment too. A
# real human bio sentence is never just "Follow." / "Message." -- those
# are unambiguous CTA chrome even when period-glued (e.g. ". Follow."),
# which the separator-bounded _FOLLOW_RE deliberately leaves alone. Same
# conservative whole-segment match as _LI_SENTENCE_RE.
_FOLLOW_SENTENCE_RE = _re.compile(
    u"(?:^|(?<=[.!?\u00B7\u2022\u2023\u2043|\\n\\r]))"
    u"[ \\t]*(?:" + u"|".join(_FOLLOW_WORDS) + u")[ \\t]*"
    u"[ \\t]*[.!?]?(?=[ \\t]*(?:$|[.!?\u00B7\u2022\u2023\u2043|\\n\\r]))",
    _re.IGNORECASE | _re.UNICODE,
)
# Trailing variants. Expander phrases are unambiguous UI chrome -> always strip
# trailing. Follow words can be strippable trailing tokens too (Seguir, etc.).
# Tab labels are NOT stripped purely on trailing position (too homonymous:
# "...he posts" would be too risky) -- they need a hard/soft separator.
_EXPANDER_TRAIL_RE = _ui_trailing_re(_EXPANDERS)
_FOLLOW_TRAIL_RE = _ui_trailing_re(_FOLLOW_WORDS)

# Leaked metric counts: "1.2M followers", "340 following", "57 posts" ...
# Only at a UI boundary (leading/trailing/hard-sep) -- so "50 million followers"
# inside a sentence is preserved.
_COUNT_RE = _re.compile(
    _LEFT + u"\\d[\\d.,]*\\s*(?:[KkMmBb]\\b)?\\s*"
    u"(?:followers?|following|posts?|likes?|friends|subscribers?|"
    u"talking\\s+about\\s+this)"
    + _RIGHT,
    _re.IGNORECASE | _re.UNICODE,
)


def _strip_ui_artifacts(bio):
    """Conservative, always-on UI/scrape-artifact stripper. Returns cleaned bio."""
    if not bio:
        return bio
    s = bio
    # 1) Drop private-use / zero-width / control glyphs entirely.
    s = _PUA_RE.sub(" ", s)
    glyphs_removed = (s != bio)
    # 2) Iteratively strip UI-fragment tokens. Replace with a hard sep marker so
    #    the NEXT pass still sees a boundary for chained fragments
    #    ("· Follow · Read more"). Loop until stable.
    for _ in range(6):
        before = s
        s = _DOTMORE_RE.sub(u"\n", s)
        s = _COUNT_RE.sub(u"\n", s)
        s = _EXPANDER_RE.sub(u"\n", s)
        s = _FOLLOW_RE.sub(u"\n", s)
        s = _TAB_RE.sub(u"\n", s)
        s = _LI_LABEL_RE.sub(u"\n", s)  # v7.60 biofix2: LinkedIn section headers
        s = _LI_LABEL_TRAIL_RE.sub(u"", s)  # v7.60 biofix2: trailing LI headers/CTAs
        s = _LI_SENTENCE_RE.sub(u"\n", s)  # v7.60 biofix2: whole-sentence LI labels
        s = _FOLLOW_SENTENCE_RE.sub(u"\n", s)  # v7.60 biofix2: whole-sentence Follow/Message
        # trailing single-fragment cleanup (runs after sep-bounded strips so
        # chained "x · Follow · Read more" is reduced to a trailing "Read more")
        s = _EXPANDER_TRAIL_RE.sub(u"", s)
        s = _FOLLOW_TRAIL_RE.sub(u"", s)
        if s == before:
            break
    fragments_removed = (s != (bio if not glyphs_removed else _PUA_RE.sub(" ", bio)))

    # CONSERVATIVE GUARD: if we removed NOTHING (no glyphs, no UI fragments),
    # return the bio byte-for-byte unchanged. We must not "normalize" a clean
    # user bio (e.g. rewrite their chosen "|" separators to "·", or squeeze
    # their intentional double spaces) -- only tidy when we actually cut chrome.
    if not glyphs_removed and not fragments_removed:
        # ...but never return an all-whitespace string as a "bio".
        return bio if bio.strip() else ""

    # 3) Tidy up ONLY the residue from what we removed. Preserve internal
    #    newlines (legit multi-line bios); collapse separator runs and dangling
    #    punctuation a removed fragment left behind (e.g. "Periodista. Seguir"
    #    -> "Periodista.").
    # collapse runs of bullet separators (possibly created above) to one " · "
    s = _re.sub(u"[ \\t]*(?:[·•‣⁃\\|][ \\t]*)+[ \\t]*", u" · ", s)
    # squeeze horizontal whitespace only (keep newlines)
    s = _re.sub(u"[ \\t]+", u" ", s)
    # collapse 3+ newlines to 2, strip spaces hugging newlines
    s = _re.sub(u"[ \\t]*\n[ \\t]*", u"\n", s)
    s = _re.sub(u"\n{3,}", u"\n\n", s)
    # v7.66 GOAL 5: a removed whole-segment label (e.g. "Personal details",
    # "Contact info", "See all") left behind by _LI_SENTENCE_RE is replaced
    # with "\n", which can orphan the label's trailing sentence-terminator
    # onto its own line — the leak "journalist. Personal details. Contact
    # info." cleaned to "journalist.\n.". Drop any line that is ONLY
    # punctuation/separators (no letters/digits/emoji), then re-tidy newlines.
    # Runs only inside this "something WAS removed" tidy block, so a clean
    # user bio is never touched.
    s = _re.sub(
        u"(?m)^[\\s.!?·•‣⁃\\|/–—,;:_~`\"'“”‘’]+$",
        u"", s,
    )
    s = _re.sub(u"\n{2,}", u"\n", s)
    # strip leading/trailing separator / newline / whitespace noise
    # (v7.66: add . ! ? so a dangling terminator left by a removed trailing
    # label — "Marketing strategist. See more" → "Marketing strategist." —
    # is tidied too; only inside the removed-something block, so clean bios
    # keep their own punctuation untouched).
    s = _re.sub(u"^[\\s.!?·•‣⁃\\|/–—,;:]+", u"", s)
    s = _re.sub(u"[\\s.!?·•‣⁃\\|/–—,;:]+$", u"", s)
    return s.strip()


# v7.60 biofix2: bio finalizer. Runs on the OUTPUT of every clean path.
#   (a) strip matched wrapping quotes (straight + smart) from both ends,
#   (b) if what remains is empty / only quote chars / only whitespace /
#       only punctuation+separators, return "" so the frontend renders no
#       bio row (kills the literal '""' leak, e.g. @alufbenn: "").
# A legit quoted bio like '"Building the future"' keeps its TEXT.
_WRAP_QUOTES = "\"'\u201C\u201D\u2018\u2019\u00AB\u00BB\u2039\u203A"
# A string is "empty-ish" when, after removing quote chars, whitespace,
# and punctuation/separators, nothing is left. We strip those classes and
# check if ANYTHING remains -- so letters, digits, AND emoji/symbols count
# as real content (an emoji-only bio is kept), while a "" / " . " / "\u201C\u201D"
# string becomes empty.
_BIO_STRIP_NONCONTENT_RE = _re.compile(
    "[\\s\"'\u201C\u201D\u2018\u2019\u00AB\u00BB\u2039\u203A"
    "\u00B7\u2022\u2023\u2043\\|/.,;:!?\\-\u2013\u2014_~`@#%^&*()\\[\\]{}<>+=]+",
    _re.UNICODE,
)


def _finalize_bio(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # Strip ALL matched wrapping quote pairs from the ends (handles
    # nested / repeated quoting like \'\u201C"x"\u201D\'). Only strip when BOTH
    # ends carry a quote char, so we never lop a single stray quote off
    # one side of real text.
    for _ in range(4):
        s = s.strip()
        if len(s) >= 2 and s[0] in _WRAP_QUOTES and s[-1] in _WRAP_QUOTES:
            s = s[1:-1]
            continue
        break
    s = s.strip()
    # Empty / quote-only / whitespace-only / punctuation-only -> no bio.
    if not _BIO_STRIP_NONCONTENT_RE.sub("", s).strip():
        return ""
    return s


def _clean_social_bio(bio: str) -> str:
    """Public entry: clean via the inner worker, then finalize
    (strip wrapping quotes + drop empty/quote-only). v7.60 biofix2."""
    return _finalize_bio(_clean_social_bio_inner(bio))


def _clean_social_bio_inner(bio: str) -> str:
    if not bio:
        return ""
    # v7.47: always-on UI/scraping-artifact strip FIRST (handles SERP UI-chrome
    # leaks that do not trip the SEO/news gates, e.g. "<glyph> . follow . Read
    # more" with Follow-button text in many languages, app icon PUA glyphs,
    # leaked counts/tab labels). Conservative: untouched bios pass through
    # byte-for-byte unchanged. See _strip_ui_artifacts above.
    bio = _strip_ui_artifacts(bio)
    if not bio:
        return ""
    low = bio.lower()
    has_seo = any(pat in low for pat in _SEO_PATTERNS)
    looks_news = _looks_like_news_snippet(bio)

    if not has_seo and not looks_news:
        return bio  # Not SEO boilerplate AND not a news snippet — pass through.

    # News-snippet path (no SEO overlap): nothing salvageable, return empty.
    if looks_news and not has_seo:
        return ""

    import re as _re
    cleaned = bio.strip()
    cleaned = _re.sub(r"\b(read more|see more|ראה עוד)\b\.?$", "", cleaned,
                      flags=_re.IGNORECASE).strip()
    # IG SERP often ends with "Follow . Message" / "Follow .. Message"
    cleaned = _re.sub(r"(follow\s*[.\s]+\s*message)\s*\.?$", "", cleaned,
                      flags=_re.IGNORECASE).strip()
    tails = []
    m_lives = _re.search(r"\b(lives in [^.]+?\.?)\s*$", cleaned, flags=_re.IGNORECASE)
    if m_lives:
        tails.append(m_lives.group(1).strip().rstrip("."))
    m_follow = _re.search(r"(\d[\d,.]*\s*followers?\s*[·•|]+\s*\d[\d,.]*\s*following)",
                          bio, flags=_re.IGNORECASE)
    if m_follow and not tails:
        tails.insert(0, m_follow.group(1).strip())
    return " · ".join(tails) if tails else ""


# Back-compat alias.
def _clean_fb_bio(bio: str) -> str:
    return _clean_social_bio(bio)


def transliterate_name(name: str) -> str:
    """Use Claude to transliterate non-Latin names."""
    if all(c.isascii() or c.isspace() for c in name):
        return name
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": f"Transliterate this name to its most common English/Latin spelling. Return ONLY the transliterated name, nothing else: {name}"}],
    )
    return resp.content[0].text.strip()


class SearchRequest(BaseModel):
    name: str
    description: str
    link_x: Optional[str] = ""
    link_facebook: Optional[str] = ""
    link_instagram: Optional[str] = ""
    link_tiktok: Optional[str] = ""
    link_linkedin: Optional[str] = ""
    # 2026-05-30 (Phase 2 carousel triage): URLs the customer has
    # explicitly rejected in a prior attempt. Backend filters these
    # out so refined searches never re-suggest them.
    rejected_urls: Optional[list[str]] = []
    # 2026-05-31 v7.6: patience hint. "normal" (default) uses the
    # speed-tuned timeouts of round 1. "long" doubles them throughout
    # the pipeline so the recal-round second search doesn't drop late
    # candidates (handle-guess probes, avatar fetches, IG enrichment).
    # The carousel reveal timer in the frontend is also longer for
    # round 2, so users don't see chips appearing under a stale "no
    # more" state.
    patience: Optional[str] = "normal"


class ProfileResult(BaseModel):
    platform: str
    url: str
    username: str
    display_name: str
    bio: str
    image_url: str
    image_data: str = ""           # 2026-05-30: data:image/jpeg;base64 inline
    # 2026-05-30: per-avatar similarity payload, computed once at ingest
    # time (similarity.attach_similarity_payload) and shipped inline so
    # the carousel can rerank locally on every vote (confirm OR reject)
    # without a network round-trip. Either may be "" if computation
    # failed (no face detected, bad bytes); frontend handles gracefully.
    phash_b64: str = ""             # 8-byte perceptual hash, base64
    face_emb_b64: str = ""          # 512 float32 InsightFace, base64
    is_private: bool = False        # 2026-05-31 v7.14: free-detected at discovery
    score: int
    reasoning: str
    position: str = ""
    company: str = ""
    city: str = ""
    followers: str = ""


class SearchResponse(BaseModel):
    profiles: list[ProfileResult]
    candidates_searched: int
    total_found: int
    elapsed_seconds: float
    # 2026-06-03: authoritative subject name derived from the located profile
    # display names (corrects operator typos like 'Candice'->'Candace') and
    # Title-Cased. Falls back to the Title-Cased typed query when no confident
    # profile match exists. Free/local string logic, no API.
    authoritative_name: str = ""


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# ITER29 2026-05-28: minimal /api/img-proxy for local form testing.
# The production form routes referer-gated CDN URLs (cdninstagram, fbcdn, licdn,
# fbsbx) through this endpoint so the browser can render them. Locally, without
# this, those avatars 404. We fetch with platform-appropriate Referer + UA and
# stream the bytes back.
from fastapi.responses import Response
import urllib.parse as _urlparse

_IMG_PROXY_REFERERS = {
    "Instagram": "https://www.instagram.com/",
    "Facebook": "https://www.facebook.com/",
    "LinkedIn": "https://www.linkedin.com/",
    "TikTok": "https://www.tiktok.com/",
    "X": "https://x.com/",
}
_IMG_PROXY_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_IMG_HOST_ALLOWLIST = (
    # CDN hosts (referer-gated — proxy adds Referer header)
    "cdninstagram.com", "fbcdn.net", "licdn.com", "fbsbx.com", "twimg.com",
    # Avatar redirect services (added 2026-05-28 to defeat browser ad-blockers
    # that block these as third-party trackers — Brave Shield, uBlock, FF
    # Strict mode all blacklist unavatar.io and graph.facebook.com by default).
    "unavatar.io", "graph.facebook.com", "ui-avatars.com",
)


# ITER34 2026-05-28: platform-direct avatar resolvers. unavatar.io is rate-limiting
# (HTTP 429 on X / TT / LI after heavy testing today). These bypass it by going
# straight to the platform: X via x.com profile-page scrape, TT via the
# ScrapeCreators profile API. Both responses cached in-memory for 1h to avoid
# pounding upstream on every form load.
import time as _time
_avatar_cache: dict = {}  # key -> (content_bytes, content_type, expires_at_ts)
_AVATAR_CACHE_TTL = 3600


# v7.66 MONEY-LEAK FIX: defensive handle normalizer. A few upstream paths
# (user-provided known links, some serp/handle-guess merges) can hand a chip a
# `username` that is actually a FULL profile URL (e.g.
# "https://instagram.com/or-rilov" or a url-encoded form). When that reaches an
# avatar/follower endpoint as `?u=<url>` / `handle=<url>`, ScrapeCreators is
# billed for a malformed-but-tolerated call (~$0.002–0.003 each) that's
# redundant with the correct bare-handle probe. This collapses any URL-ish
# value down to the bare handle so we only ever spend on the canonical handle.
# Idempotent + cheap; bare handles pass through unchanged.
import re as _re_hn
from urllib.parse import unquote as _unquote_hn
_HANDLE_FROM_URL_RE = _re_hn.compile(
    r"(?:x|twitter|facebook|fb|instagram|tiktok|linkedin)\.com/"
    r"(?:in/|@)?([A-Za-z0-9_.\-]+)",
    _re_hn.IGNORECASE,
)


def _normalize_handle_param(u: str) -> str:
    if not u:
        return u
    s = _unquote_hn(u).strip()
    # Only act when it actually looks like a URL — leave bare handles alone.
    if "://" in s or "/" in s or "." in s and ".com" in s.lower():
        m = _HANDLE_FROM_URL_RE.search(s)
        if m:
            return m.group(1).lstrip("@")
    return u.lstrip("@")


# 2026-06-08 #2: ONE pooled keep-alive httpx client reused across every SC/CDN
# fetch, instead of a fresh client (= fresh TLS handshake) per call. _pooled() is
# a drop-in for `async with _httpx.AsyncClient(...) as cl:` — it hands out the
# shared client and never closes it, so no code reshape / dedent is needed.
import httpx as _httpx_pool
_SHARED_HTTP = None
def _get_http():
    global _SHARED_HTTP
    if _SHARED_HTTP is None or _SHARED_HTTP.is_closed:
        _SHARED_HTTP = _httpx_pool.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            limits=_httpx_pool.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _SHARED_HTTP
class _PooledClientCM:
    async def __aenter__(self):
        return _get_http()
    async def __aexit__(self, *exc):
        return False
def _pooled():
    return _PooledClientCM()


async def _fetch_bytes(url: str, headers: dict | None = None, timeout: float = 8.0):
    import httpx as _httpx
    try:
        async with _pooled() as cl:
            r = await cl.get(url, headers=headers or {}, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content, r.headers.get("content-type", "image/jpeg")
    except Exception:
        return None
    return None


# ─── v7.61 all-followers: per-network follower-count fetch ──────────────
# TikTok + Instagram already arrive with a follower count baked into the
# handle-guess / BD-IG data we fetch anyway. X, LinkedIn and Facebook do
# NOT — we never asked their profile endpoints for it. This bounded,
# cached helper fills that gap for the 1-2 chips actually EMITTED per
# platform (never for every handle-guess), so a real follower count shows
# on every network.
#
# Cached by "platform|handle" (1h TTL) like the avatar cache, so repeat
# handles cost nothing. Returns a RAW integer string (e.g. "240263644");
# the frontend's formatCount() turns it into "240.3M" — matching exactly
# how TikTok's already rendered. Returns "" when the platform/profile
# type carries no public count (e.g. FB personal profiles).
_followers_cache: dict = {}  # "platform|handle" -> (followers_str, expires_at_ts)
_FOLLOWERS_CACHE_TTL = 3600


async def _fetch_followers(platform: str, handle: str) -> str:
    """Fetch a follower count for one EMITTED chip. Bounded (one call,
    short timeout), cached by platform|handle. ~$0.003/uncached handle.
    Returns a raw integer string, or "" when none is available."""
    if not platform or not handle:
        return ""
    h = _normalize_handle_param(handle).strip()   # v7.66: strip URL-as-handle leak
    if not h:
        return ""
    ck = f"{platform.lower()}|{h.lower()}"
    hit = _followers_cache.get(ck)
    if hit and hit[1] > _time.time():
        return hit[0]

    import httpx as _httpx
    from profile_finder import SCRAPECREATORS_API_KEY
    out = ""
    try:
        if platform == "X":
            # SC twitter profile → legacy.followers_count (free of the
            # x.com page-scrape's unreliability). One cheap call.
            async with _pooled() as cl:
                r = await cl.get(
                    "https://api.scrapecreators.com/v1/twitter/profile",
                    params={"handle": h},
                    headers={"x-api-key": SCRAPECREATORS_API_KEY},
                    timeout=10.0,
                )
            if r.status_code == 200:
                d = r.json() or {}
                fc = (((d.get("legacy") or {}).get("followers_count"))
                      or d.get("followers_count"))
                if fc:
                    out = str(fc)
        elif platform == "Facebook":
            # SC FB profile → followerCount (pages). FB PERSONAL profiles
            # carry no public count → stays "".
            async with _pooled() as cl:
                r = await cl.get(
                    "https://api.scrapecreators.com/v1/facebook/profile",
                    params={"url": f"https://www.facebook.com/{h}"},
                    headers={"x-api-key": SCRAPECREATORS_API_KEY},
                    timeout=12.0,
                )
            if r.status_code == 200:
                d = r.json() or {}
                fc = d.get("followerCount") or d.get("likeCount")
                if fc:
                    out = str(fc)
        elif platform == "LinkedIn":
            # SC LI rarely exposes a public count (returns "private or not
            # publicly available"); BD LinkedIn carries `followers` (already
            # run through format_followers, e.g. "1.8M"). Normalise to a raw
            # int string when BD hands back a plain number; otherwise keep
            # the BD-formatted string (frontend strips non-digits anyway).
            from profile_finder import brightdata_linkedin
            enriched = await brightdata_linkedin(h)
            if enriched:
                fc = enriched.get("followers") or ""
                if fc:
                    out = str(fc)
        elif platform == "Instagram":
            # Belt-and-suspenders: SC IG profile → edge_followed_by.count.
            # (Normally already filled by handle-guess / BD-IG.)
            async with _pooled() as cl:
                r = await cl.get(
                    "https://api.scrapecreators.com/v1/instagram/profile",
                    params={"handle": h},
                    headers={"x-api-key": SCRAPECREATORS_API_KEY},
                    timeout=10.0,
                )
            if r.status_code == 200:
                d = r.json() or {}
                user = ((d.get("data") or {}).get("user") or {})
                fb_d = (user.get("edge_followed_by") or {})
                fc = (fb_d.get("count") if isinstance(fb_d, dict) else None) \
                    or user.get("follower_count")
                if fc:
                    out = str(fc)
        elif platform == "TikTok":
            # v7.64: SERP-discovered TikTok chips reach the emit loop with NO
            # follower count (only handle-guess/BD-TT chips arrive pre-filled),
            # so the operator saw blank counts on SERP TT chips. Fill them from
            # the same SC tiktok/profile endpoint the avatar/enrich path uses.
            async with _pooled() as cl:
                r = await cl.get(
                    "https://api.scrapecreators.com/v1/tiktok/profile",
                    params={"handle": h},
                    headers={"x-api-key": SCRAPECREATORS_API_KEY},
                    timeout=10.0,
                )
            if r.status_code == 200:
                d = r.json() or {}
                user = (d.get("user") or {})
                stats = (d.get("statsV2") or d.get("stats") or {})
                fc = stats.get("followerCount") or user.get("followerCount")
                if fc:
                    out = str(fc)
    except Exception as e:
        logger.warning(f"_fetch_followers {platform}|{h} failed: {e}")
        out = ""

    _followers_cache[ck] = (out, _time.time() + _FOLLOWERS_CACHE_TTL)
    return out


@app.get("/api/avatar/x")
async def avatar_x(u: str = ""):
    """X avatar: scrape x.com/{handle}, regex twimg URL, fetch + cache + return."""
    if not u:
        return Response(status_code=400)
    u = _normalize_handle_param(u)   # v7.66: collapse URL-as-handle → bare handle
    cache_key = f"x:{u.lower()}"
    cached = _avatar_cache.get(cache_key)
    if cached and cached[2] > _time.time():
        return Response(content=cached[0], media_type=cached[1],
                        headers={"Cache-Control": "public, max-age=3600"})

    import re as _re
    page = await _fetch_bytes(
        f"https://x.com/{u}",
        headers={"User-Agent": _IMG_PROXY_UA},
        timeout=5.0,
    )
    if not page:
        return Response(status_code=502)
    body, _ = page
    m = _re.search(rb'pbs\.twimg\.com/profile_images/[^"]+', body)
    if not m:
        return Response(status_code=404)
    avatar_url = "https://" + m.group(0).decode("utf-8", errors="ignore").split('"')[0].split("'")[0]
    # Upgrade _normal (48x48) → _400x400. Same path & query; only the size token differs.
    avatar_url = _re.sub(r"_(normal|bigger|mini)(?=\.[a-z]+(\?|$))", "_400x400", avatar_url)
    fetched = await _fetch_bytes(
        avatar_url,
        headers={"User-Agent": _IMG_PROXY_UA, "Referer": "https://x.com/"},
        timeout=5.0,
    )
    if not fetched:
        return Response(status_code=502)
    content, ct = fetched
    _avatar_cache[cache_key] = (content, ct, _time.time() + _AVATAR_CACHE_TTL)
    return Response(content=content, media_type=ct,
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/avatar/instagram")
async def avatar_instagram(u: str = ""):
    """Instagram avatar via ScrapeCreators profile API (profile_pic_url_hd → cdn jpeg).
    Replaces unreliable BD-IG enrichment + ui-avatars initials fallback. ~$0.002 per
    uncached handle, sub-2s. Cdninstagram URLs are referer-gated; we add the right
    Referer when fetching upstream so the browser doesn't need to."""
    if not u:
        return Response(status_code=400)
    u = _normalize_handle_param(u)   # v7.66: collapse URL-as-handle → bare handle
    cache_key = f"ig:{u.lower()}"
    cached = _avatar_cache.get(cache_key)
    if cached and cached[2] > _time.time():
        return Response(content=cached[0], media_type=cached[1],
                        headers={"Cache-Control": "public, max-age=3600"})

    import httpx as _httpx
    from profile_finder import SCRAPECREATORS_API_KEY
    try:
        async with _pooled() as cl:
            r = await cl.get(
                "https://api.scrapecreators.com/v1/instagram/profile",
                params={"handle": u},
                headers={"x-api-key": SCRAPECREATORS_API_KEY},
                timeout=6.0,
            )
        if r.status_code != 200:
            return Response(status_code=502)
        data = r.json()
        user = ((data.get("data") or {}).get("user") or {})
        avatar_url = user.get("profile_pic_url_hd") or user.get("profile_pic_url")
        if not avatar_url:
            return Response(status_code=404)
    except Exception:
        return Response(status_code=502)

    fetched = await _fetch_bytes(
        avatar_url,
        headers={"User-Agent": _IMG_PROXY_UA, "Referer": "https://www.instagram.com/"},
        timeout=6.0,
    )
    if not fetched:
        return Response(status_code=502)
    content, ct = fetched
    _avatar_cache[cache_key] = (content, ct, _time.time() + _AVATAR_CACHE_TTL)
    return Response(content=content, media_type=ct,
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/avatar/tiktok")
async def avatar_tiktok(u: str = ""):
    """TikTok avatar via ScrapeCreators profile API.

    2026-05-29 ghost-rejection: TT returns a default silhouette for
    accounts that haven't uploaded a picture. Those come back as
    ~1–3 KB JPEGs. We try avatarLarger (1080×1080) first because the
    higher-res variant is rarely served as a default. If the response
    is still < 5 KB we treat it as no-avatar and 404 so the renderer
    falls back to initials.
    """
    if not u:
        return Response(status_code=400)
    u = _normalize_handle_param(u)   # v7.66: collapse URL-as-handle → bare handle
    cache_key = f"tt:{u.lower()}"
    cached = _avatar_cache.get(cache_key)
    if cached and cached[2] > _time.time():
        return Response(content=cached[0], media_type=cached[1],
                        headers={"Cache-Control": "public, max-age=3600"})

    import httpx as _httpx
    from profile_finder import SCRAPECREATORS_API_KEY
    try:
        async with _pooled() as cl:
            r = await cl.get(
                "https://api.scrapecreators.com/v1/tiktok/profile",
                params={"handle": u},
                headers={"x-api-key": SCRAPECREATORS_API_KEY},
                timeout=5.0,
            )
        if r.status_code != 200:
            return Response(status_code=502)
        data = r.json()
        user = (data.get("user") or {})
        # Prefer the highest-res variant first — defaults usually come
        # at the medium size and are not regenerated for larger sizes.
        candidates = [user.get("avatarLarger"), user.get("avatarMedium"), user.get("avatarThumb")]
        candidates = [c for c in candidates if c]
        if not candidates:
            return Response(status_code=404)
    except Exception:
        return Response(status_code=502)

    # Try each URL in order, accept the first one that returns >= 5KB
    # of image bytes. < 5KB is the TikTok default-silhouette giveaway.
    best = None
    for avatar_url in candidates:
        fetched = await _fetch_bytes(
            avatar_url,
            headers={"User-Agent": _IMG_PROXY_UA, "Referer": "https://www.tiktok.com/"},
            timeout=5.0,
        )
        if fetched and len(fetched[0]) >= 2500:   # audit-F7: was 5000 (rejected real small avatars)
            best = fetched
            break
        if fetched and (best is None or len(fetched[0]) > len(best[0])):
            best = fetched

    if not best:
        return Response(status_code=502)
    # If all variants returned a ghost-sized image, this is a default-
    # silhouette account — respond 404 so the chip renders as initials.
    if len(best[0]) < 2500:   # audit-F7: was 5000
        return Response(status_code=404)
    content, ct = best
    _avatar_cache[cache_key] = (content, ct, _time.time() + _AVATAR_CACHE_TTL)
    return Response(content=content, media_type=ct,
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/avatar/facebook")
async def avatar_facebook(u: str = ""):
    """Facebook avatar via ScrapeCreators profile API (profilePicLarge →
    fbcdn jpeg). Replaces the broken graph.facebook.com fallback for
    personal profiles. ~$0.002 per uncached handle, ~1–2s. fbcdn URLs
    are referer-gated; we add the right Referer when fetching upstream
    so the browser doesn't need to.
    """
    if not u:
        return Response(status_code=400)
    u = _normalize_handle_param(u)   # v7.66: collapse URL-as-handle → bare handle (also strips @)
    cache_key = f"fb:{u.lower()}"
    cached = _avatar_cache.get(cache_key)
    if cached and cached[2] > _time.time():
        return Response(content=cached[0], media_type=cached[1],
                        headers={"Cache-Control": "public, max-age=3600"})

    import httpx as _httpx
    from profile_finder import SCRAPECREATORS_API_KEY
    profile_url = f"https://www.facebook.com/{u}"
    try:
        async with _pooled() as cl:
            r = await cl.get(
                "https://api.scrapecreators.com/v1/facebook/profile",
                params={"url": profile_url},
                headers={"x-api-key": SCRAPECREATORS_API_KEY},
                timeout=8.0,
            )
        if r.status_code != 200:
            return Response(status_code=502)
        data = r.json()
        if not data.get("success"):
            return Response(status_code=404)
        avatar_url = (
            data.get("profilePicLarge")
            or data.get("profilePicMedium")
            or data.get("profilePicSmall")
        )
        if not avatar_url:
            return Response(status_code=404)
    except Exception:
        return Response(status_code=502)

    fetched = await _fetch_bytes(
        avatar_url,
        headers={"User-Agent": _IMG_PROXY_UA, "Referer": "https://www.facebook.com/"},
        timeout=6.0,
    )
    if not fetched:
        return Response(status_code=502)
    content, ct = fetched
    _avatar_cache[cache_key] = (content, ct, _time.time() + _AVATAR_CACHE_TTL)
    return Response(content=content, media_type=ct,
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/avatar/linkedin")
async def avatar_linkedin(u: str = ""):
    """LinkedIn avatar via ScrapeCreators profile API (image → licdn jpeg).
    Replaces the unreliable unavatar.io/linkedin fallback. ~$0.002 per
    uncached handle, ~2–4s. Some profiles are private — those 404.
    """
    if not u:
        return Response(status_code=400)
    u = _normalize_handle_param(u)   # v7.66: collapse URL-as-handle → bare handle (also strips @)
    cache_key = f"li:{u.lower()}"
    cached = _avatar_cache.get(cache_key)
    if cached and cached[2] > _time.time():
        return Response(content=cached[0], media_type=cached[1],
                        headers={"Cache-Control": "public, max-age=3600"})

    import httpx as _httpx
    from profile_finder import SCRAPECREATORS_API_KEY

    async def _bd_scraping_browser_fallback():
        # 2026-06-01 v7.39: when SC yields the ghost / no real avatar, fall
        # back to the BrightData Scraping Browser render which exposes the
        # genuine media.licdn.com headshot (e.g. profgalloway). Slow (~5-10s
        # browser render) but the frontend loads avatars lazily so it won't
        # block the carousel. Wrapped so a failure returns 404 (initials),
        # never 500. Cost is logged via _log_spl_bd_usage inside the helper.
        try:
            from profile_finder import scraping_browser_linkedin_avatar
            bd_url = await scraping_browser_linkedin_avatar(u)
            if not bd_url:
                return None
            bd_fetched = await _fetch_bytes(
                bd_url,
                headers={"User-Agent": _IMG_PROXY_UA,
                         "Referer": "https://www.linkedin.com/"},
                timeout=12.0,
            )
            if not bd_fetched or len(bd_fetched[0]) < 2000:
                return None
            bd_content, bd_ct = bd_fetched
            _avatar_cache[cache_key] = (bd_content, bd_ct,
                                        _time.time() + _AVATAR_CACHE_TTL)
            return Response(content=bd_content, media_type=bd_ct,
                            headers={"Cache-Control": "public, max-age=3600"})
        except Exception:
            return None

    profile_url = f"https://www.linkedin.com/in/{u}"
    avatar_url = ""
    try:
        async with _pooled() as cl:
            r = await cl.get(
                "https://api.scrapecreators.com/v1/linkedin/profile",
                params={"url": profile_url},
                headers={"x-api-key": SCRAPECREATORS_API_KEY},
                timeout=8.0,
            )
        if r.status_code == 200:
            data = r.json()
            if data.get("success"):
                cand = data.get("image") or data.get("profile_picture") or ""
                # Reject the well-known LinkedIn ghost-SVG (small static asset
                # on static.licdn.com that means "no avatar"). It serves but
                # it's the empty placeholder.
                if cand and "static.licdn.com/aero" not in cand:
                    avatar_url = cand
    except Exception:
        avatar_url = ""

    if avatar_url:
        fetched = await _fetch_bytes(
            avatar_url,
            headers={"User-Agent": _IMG_PROXY_UA, "Referer": "https://www.linkedin.com/"},
            timeout=6.0,
        )
        if fetched and len(fetched[0]) >= 2000:
            content, ct = fetched
            _avatar_cache[cache_key] = (content, ct, _time.time() + _AVATAR_CACHE_TTL)
            return Response(content=content, media_type=ct,
                            headers={"Cache-Control": "public, max-age=3600"})
        # < 2KB is another LinkedIn-ghost variant — fall through to BD.

    # SC produced no real avatar → BrightData Scraping-Browser fallback.
    bd_resp = await _bd_scraping_browser_fallback()
    if bd_resp is not None:
        return bd_resp
    return Response(status_code=404)


@app.get("/api/img-proxy")
async def img_proxy(p: str = "", u: str = ""):
    """Stream a referer-gated CDN avatar. Allowlist-only on URL host."""
    import httpx as _httpx
    if not u:
        return Response(status_code=400)
    try:
        host = _urlparse.urlparse(u).hostname or ""
        if not any(h in host for h in _IMG_HOST_ALLOWLIST):
            return Response(status_code=403)
        headers = {
            "User-Agent": _IMG_PROXY_UA,
            "Referer": _IMG_PROXY_REFERERS.get(p, "https://www.google.com/"),
        }
        async with _pooled() as cl:
            r = await cl.get(u, headers=headers, timeout=8)
        if r.status_code != 200:
            return Response(status_code=r.status_code)
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "image/jpeg"),
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception:
        return Response(status_code=502)


logger = logging.getLogger("spl")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─── 2026-05-29: stage-1 handle-guess helper ──────────────────────────
# Generates First.Last / FirstLast / FirstLast / Last.First etc. variants
# from the search name and probes SC FB + IG profile endpoints. Any
# variant that resolves to a real profile with a matching display name
# is returned as a candidate dict in the same shape as find_profiles
# output, ready to be merged into the candidate pool. Skips platforms
# already represented by ≥2 candidates in `existing` (already a good
# match — don't waste SC credit).
import re as _ihg_re

_HANDLE_ASCII_RE = _ihg_re.compile(r"[^a-z0-9]")


def _name_variants(name: str, include_longshots: bool = True) -> list[str]:
    parts = [p for p in (name or "").lower().split() if p]
    if len(parts) < 2:
        # audit-F14 2026-07-06: single-token names previously got no guesses. Emit a
        # tiny bare-handle set (free existence probes; paid enrich only on a real hit).
        if len(parts) == 1:
            _solo = _HANDLE_ASCII_RE.sub("", parts[0])
            return [_solo, f"real{_solo}", f"{_solo}official"] if len(_solo) >= 3 else []
        return []
    first = _HANDLE_ASCII_RE.sub("", parts[0])
    last  = _HANDLE_ASCII_RE.sub("", parts[-1])
    if not first or not last:
        return []
    # 2026-05-30 v5: widened from 4 to ~12 variants. Adds common
    # numeric-suffix patterns (firstlast91, firstlast07), short-first
    # (just first letter + last), and last-first reorder. Each variant
    # is one SC call (~$0.002) per platform, so cost goes from ~16 to
    # ~48 calls per scan (~$0.10 extra). Worth it: the Or-Rilov case
    # showed we miss real accounts whose handles include digits.
    # 2026-05-31 v7.9: trimmed from 14 to 8 high-hit-rate variants to
    # cut SC probe cost. Dropped: lastfirst (lastfirst already covers
    # the reorder), short-first-with-dot, firstlast1/07/thefirstlast/
    # realfirstlast — all low-hit longshots. Kept the 8 below which
    # account for &gt;90% of real handles we've observed in scans.
    variants = [
        f"{first}.{last}",
        f"{first}{last}",
        f"{first}_{last}",
        f"{last}.{first}",
        f"{last}{first}",
        f"{first[0]}{last}",          # short-first: "orilov"
    ]
    # 2026-06-01 v7.39: the two numeric-suffix longshots below are low-hit
    # and only worth probing on the expand pass (rich/thin subjects), not
    # on every initial search. Gated behind include_longshots.
    if include_longshots:
        variants += [
            f"{first}{last}91",
            f"{first}.{last}.7",          # the @or.rilov.7 pattern
        ]
    # Dedup, keep order. Length 3-40, no double-dots/underscores.
    seen = set()
    out = []
    for v in variants:
        v = v.strip(".").strip("_")
        if v not in seen and 3 <= len(v) <= 40:
            seen.add(v)
            out.append(v)
    return out


def _name_tokens_match(profile_name: str, search_name: str) -> bool:
    """Strict check: ALL ≥3-char tokens from the search name must appear
    (case-insensitive) in the profile's display name. Catches "Carlos
    Mendoza" matching only profiles where both names appear — keeps out
    random "Carlos" or "Mendoza" only accounts.
    """
    if not profile_name or not search_name:
        return False
    pn = profile_name.lower()
    tokens = [t for t in search_name.lower().split() if len(t) >= 3]
    if not tokens:
        return False
    for t in tokens:
        if t not in pn:
            return False
    return True


def _handle_from_url(url: str) -> str:
    """Best-effort trailing handle/slug from a profile URL (known-URL seed)."""
    if not url:
        return ""
    u = url.split("?")[0].split("#")[0].rstrip("/")
    return u.rsplit("/", 1)[-1].lstrip("@").strip()


async def _guess_handle_candidates(name: str, existing: list[dict], include_longshots: bool = False, seed_handles=None) -> list[dict]:
    """Stage-1 enrichment — handle-guess.

    2026-05-31 rewrite: split into three lanes to cut paid SC traffic ~70%.

      Lane A — TikTok: free direct HTTPS GET to tiktok.com/@handle. 200 = exists.
      Lane B — IG + FB: free direct HTTPS GET with body-pattern check.
      Lane C — X + LinkedIn: keep paid SC probes (Cloudflare/login wall).

    Only variants that pass the free existence check go on to paid SC
    enrichment. Ambiguous (timeout/error/unknown response) → fall back
    to SC for that variant so we don't silently drop real profiles.
    """
    from profile_finder import SCRAPECREATORS_API_KEY
    import asyncio
    import random as _hg_random
    import httpx as _httpx

    # v7.40: free lanes (FB/IG/TT) probe at no cost before any paid call, so
    # give them the full 8 variants incl. digit longshots (recovers real
    # digit-handle candidates like @or.rilov.7). The paid X lane stays on the
    # 6 core variants since every X probe is a paid SC call.
    core_variants = _name_variants(name, include_longshots=False)   # 6 core
    full_variants = _name_variants(name, include_longshots=True)    # 8 (adds digit longshots)
    variants = full_variants   # keep `variants` defined for any later refs / guard below
    if not variants:
        return []

    BROWSER_UA = _IMG_PROXY_UA   # reuse module-level browser UA
    PROBE_TIMEOUT = 6.0          # direct GETs
    SC_TIMEOUT = 8.0             # paid SC enrichment

    # ── Variant pools per platform ───────────────────────────────────
    fb_variants = list(full_variants)
    ig_variants = list(full_variants)
    tt_variants = list({v.replace(".", "") for v in full_variants} | set(full_variants))
    x_variants  = list({v.replace(".", "").replace("_", "") for v in core_variants}
                       | {v.replace(".", "_") for v in core_variants})
    x_variants  = [v for v in x_variants if 4 <= len(v) <= 15]

    # 2026-06-07 KNOWN-URL SEED: when the user pasted a Known Profile URL, probe
    # its EXACT handle (+ dotted/undotted forms) on every OTHER network, ahead
    # of name-derived guesses — highest-precision signal we have. Folds into the
    # existing free-check → SC fan-out below (no extra round): FB/IG/TT seeds
    # still pay only if their free existence check passes; X seeds are paid.
    _seed_base = []
    for _h in (seed_handles or []):
        _h = (_h or "").strip().lstrip("@").lower()
        if _h:
            _seed_base.append(_h)
            _seed_base.append(_h.replace(".", "").replace("_", ""))   # stripped form
    _seed_base = [s for s in dict.fromkeys(_seed_base) if s]
    if _seed_base:
        fb_variants = list(dict.fromkeys(_seed_base + fb_variants))
        ig_variants = list(dict.fromkeys(_seed_base + ig_variants))
        tt_variants = list(dict.fromkeys(_seed_base + tt_variants))
        _x_seed = []
        for _h in (seed_handles or []):
            _h = (_h or "").strip().lstrip("@").lower()
            if _h:
                _x_seed.append(_h.replace(".", "").replace("_", ""))
                _x_seed.append(_h.replace(".", "_"))
        _x_seed = [v for v in _x_seed if 4 <= len(v) <= 15]
        x_variants = list(dict.fromkeys(_x_seed + x_variants))
        logger.info(f"HG_SEED known-url handles={_seed_base!r}")

    # ── Shared SC enrichment (returns candidate dict or None) ───────
    async def _sc_enrich(platform: str, variant: str):
        try:
            if platform == "Facebook":
                api_url = "https://api.scrapecreators.com/v1/facebook/profile"
                params = {"url": f"https://www.facebook.com/{variant}"}
            elif platform == "Instagram":
                api_url = "https://api.scrapecreators.com/v1/instagram/profile"
                params = {"handle": variant}
            elif platform == "TikTok":
                api_url = "https://api.scrapecreators.com/v1/tiktok/profile"
                params = {"handle": variant}
            elif platform == "X":
                api_url = "http://127.0.0.1:8801/api/avatar/x"
                params = {"u": variant}
            else:
                return None
            async with _pooled() as cl:
                r = await cl.get(api_url, params=params,
                                 headers={"x-api-key": SCRAPECREATORS_API_KEY},
                                 timeout=SC_TIMEOUT)
            if r.status_code != 200:
                return None

            bio = ""
            followers = ""
            if platform == "X":
                if not r.content or len(r.content) < 2000:
                    return None
                display = variant
            else:
                data = r.json()
                if not data.get("success", True):
                    return None
                is_priv = False  # v7.14: read free is_private signal from SC per-platform
                if platform == "Facebook":
                    display = data.get("name", "")
                    bio = (data.get("pageIntro") or data.get("intro") or "").strip()
                    bio = _clean_fb_bio(bio)
                    is_priv = False  # v7.40: don't trust SC privacy signal for Facebook
                elif platform == "Instagram":
                    user = ((data.get("data") or {}).get("user") or {})
                    display = user.get("full_name", "") or user.get("username", "")
                    bio = (user.get("biography") or "").strip()
                    fb_d = (user.get("edge_followed_by") or {})
                    fc = fb_d.get("count") if isinstance(fb_d, dict) else None
                    if fc: followers = str(fc)
                    is_priv = bool(user.get("is_private"))
                elif platform == "TikTok":
                    user = (data.get("user") or {})
                    display = user.get("nickname", "") or user.get("uniqueId", "")
                    bio = (user.get("signature") or "").strip()
                    stats = (data.get("statsV2") or data.get("stats") or {})
                    fc = stats.get("followerCount")
                    if fc: followers = str(fc)
                    is_priv = bool(user.get("privateAccount") or user.get("secret"))
                if not display:
                    return None
                if not _name_tokens_match(display, name):
                    return None

            profile_url = {
                "Facebook":  f"https://www.facebook.com/{variant}",
                "Instagram": f"https://www.instagram.com/{variant}",
                "TikTok":    f"https://www.tiktok.com/@{variant}",
                "X":         f"https://x.com/{variant}",
            }[platform]
            return {
                "platform": platform,
                "url": profile_url.rstrip("/"),
                "username": variant,
                "display_name": display,
                "bio": bio,
                "followers": followers,
                "image_url": {
                    "Facebook":  f"/api/spl/api/avatar/facebook?u={variant}",
                    "Instagram": f"/api/spl/api/avatar/instagram?u={variant}",
                    "TikTok":    f"/api/spl/api/avatar/tiktok?u={variant}",
                    "X":         f"/api/spl/api/avatar/x?u={variant}",
                }[platform],
                "score": 72,
                "reasoning": f"Found via {platform} handle pattern guess.",
                "source": "handle_guess",
                "is_private": is_priv,
            }
        except Exception:
            return None

    # ── Lane A — TikTok existence (FREE) ─────────────────────────────
    # TikTok's SPA HTML embeds a JSON __UNIVERSAL_DATA__ blob with the
    # real status. Hit: "statusCode":0 and "uniqueId":"<handle>" present.
    # Miss: "statusCode":10221 (user not found). Substring search on the
    # error-component strings is unreliable — those phrases ship in
    # every page's JS bundle.
    tt_sem = asyncio.Semaphore(5)   # audit-F13
    async def _tt_exists(variant: str):
        async with tt_sem:
            await asyncio.sleep(_hg_random.uniform(0.0, 0.4))   # audit-F13: less jitter → more guesses ready by early-drain
            url = f"https://www.tiktok.com/@{variant}"
            try:
                async with _httpx.AsyncClient(follow_redirects=True,
                                              timeout=PROBE_TIMEOUT) as cl:
                    r = await cl.get(url, headers={
                        "User-Agent": BROWSER_UA,
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
                    })
                body = r.text or ""
                if r.status_code == 404:
                    return ("miss", variant)
                if r.status_code != 200:
                    return ("ambig", variant)
                # Bot-gated stub (~1.1KB shell) — no signal, fall back to SC
                if len(body) < 5000:
                    return ("ambig", variant)
                if '"statusCode":10221' in body:
                    return ("miss", variant)
                if '"statusCode":0' in body and f'"uniqueId":"{variant}"' in body:
                    return ("hit", variant)
                # Other status codes (auth-walled, region-blocked) → ambig
                return ("ambig", variant)
            except Exception:
                return ("ambig", variant)

    # ── Lane B — IG + FB existence (FREE body pattern) ──────────────
    # NOTE on yield: IG serves a generic login-wall HTML for every
    # handle from datacenter IPs (no per-handle signal in the body).
    # FB returns HTTP 400 for every handle from non-logged-in
    # datacenter UAs. Both lanes therefore mostly emit "ambig" and
    # fall back to SC. The probes are still cheap and occasionally
    # catch a clean 404, so we keep them — the design preserves
    # correctness (no false drops) and is ready to start saving once
    # FB/IG behaviour shifts.
    name_tokens_l = [t for t in (name or "").lower().split() if len(t) >= 3]
    _COMMON_HEADERS = {
        "User-Agent": BROWSER_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    }

    ig_sem = asyncio.Semaphore(5)   # audit-F13
    async def _ig_exists(variant: str):
        async with ig_sem:
            await asyncio.sleep(_hg_random.uniform(0.0, 0.4))   # audit-F13: less jitter → more guesses ready by early-drain
            url = f"https://www.instagram.com/{variant}/"
            try:
                async with _httpx.AsyncClient(follow_redirects=True,
                                              timeout=PROBE_TIMEOUT) as cl:
                    r = await cl.get(url, headers=_COMMON_HEADERS)
                body = r.text or ""
                low = body.lower()
                if r.status_code == 404:
                    return ("miss", variant)
                # Real-profile signals (rare from datacenter IPs, but free)
                if "profilepage_" in low:
                    return ("hit", variant)
                if 'property="og:title"' in low and variant.lower() in low:
                    return ("hit", variant)
                # Definitive miss strings
                if ("sorry, this page isn't available" in low
                        or "page not found" in low):
                    return ("miss", variant)
                return ("ambig", variant)
            except Exception:
                return ("ambig", variant)

    fb_sem = asyncio.Semaphore(5)   # audit-F13
    async def _fb_exists(variant: str):
        async with fb_sem:
            await asyncio.sleep(_hg_random.uniform(0.0, 0.4))   # audit-F13: less jitter → more guesses ready by early-drain
            url = f"https://www.facebook.com/{variant}"
            try:
                async with _httpx.AsyncClient(follow_redirects=True,
                                              timeout=PROBE_TIMEOUT) as cl:
                    r = await cl.get(url, headers=_COMMON_HEADERS)
                body = r.text or ""
                low = body.lower()
                if r.status_code == 404:
                    return ("miss", variant)
                if r.status_code in (400, 403):
                    # FB blocks raw GETs from datacenter UA — no signal
                    return ("ambig", variant)
                if ("this content isn't available right now" in low
                        or "page not found" in low
                        or "this page isn't available" in low):
                    return ("miss", variant)
                if any(tok in low for tok in name_tokens_l):
                    return ("hit", variant)
                return ("ambig", variant)
            except Exception:
                return ("ambig", variant)

    # ── Run Lane A + Lane B existence checks in parallel ────────────
    tt_check_t = asyncio.gather(*(_tt_exists(v) for v in tt_variants))
    ig_check_t = asyncio.gather(*(_ig_exists(v) for v in ig_variants))
    fb_check_t = asyncio.gather(*(_fb_exists(v) for v in fb_variants))
    tt_checks, ig_checks, fb_checks = await asyncio.gather(
        tt_check_t, ig_check_t, fb_check_t
    )

    def _survivors(checks):
        # hit or ambig → run paid SC enrichment. miss → drop.
        return [v for status, v in checks if status in ("hit", "ambig")]

    tt_keep = _survivors(tt_checks)
    ig_keep = _survivors(ig_checks)
    fb_keep = _survivors(fb_checks)

    # ── Lane C — X + LinkedIn always go through paid SC ──────────────
    # (LinkedIn was not previously probed by this fn; preserve prior
    # behaviour and only handle X here.)
    sc_calls: list[tuple[str, str]] = []
    for v in fb_keep:
        sc_calls.append(("Facebook", v))
    for v in ig_keep:
        sc_calls.append(("Instagram", v))
    for v in tt_keep:
        sc_calls.append(("TikTok", v))
    for v in x_variants:
        sc_calls.append(("X", v))

    # Per-platform telemetry
    def _hits(checks):
        return sum(1 for s, _ in checks if s == "hit")
    def _saved(checks):
        return sum(1 for s, _ in checks if s == "miss")
    logger.info(
        f"HG_TIER1 TikTok exist_checks={len(tt_checks)} hits={_hits(tt_checks)} "
        f"sc_calls={len(tt_keep)} sc_saved={_saved(tt_checks)}"
    )
    logger.info(
        f"HG_TIER1 Instagram exist_checks={len(ig_checks)} hits={_hits(ig_checks)} "
        f"sc_calls={len(ig_keep)} sc_saved={_saved(ig_checks)}"
    )
    logger.info(
        f"HG_TIER1 Facebook exist_checks={len(fb_checks)} hits={_hits(fb_checks)} "
        f"sc_calls={len(fb_keep)} sc_saved={_saved(fb_checks)}"
    )
    logger.info(
        f"HG_TIER1 X exist_checks=0 hits=0 sc_calls={len(x_variants)} sc_saved=0"
    )

    if not sc_calls:
        return []
    results = await asyncio.gather(*(_sc_enrich(p, v) for p, v in sc_calls))
    return [r for r in results if r]


@app.post("/api/search", response_model=SearchResponse)
async def search_profiles(req: SearchRequest):
    start = time.time()
    logger.info(f"SEARCH request: name={req.name!r} desc={req.description!r}")

    t = {}
    t0 = time.time()
    # Transliterate name if needed
    latin_name = transliterate_name(req.name)
    t["transliterate"] = round(time.time() - t0, 2)
    logger.info(f"  transliterated: {latin_name!r}")

    # Build known links from user input
    known_links = {}
    if req.link_x:
        known_links["X"] = req.link_x
    if req.link_facebook:
        known_links["Facebook"] = req.link_facebook
    if req.link_instagram:
        known_links["Instagram"] = req.link_instagram
    if req.link_tiktok:
        known_links["TikTok"] = req.link_tiktok
    if req.link_linkedin:
        known_links["LinkedIn"] = req.link_linkedin

    # Generate usernames for fallback
    usernames = generate_usernames(latin_name)[:10]

    # ITER8 2026-05-28: run verify_profiles IN PARALLEL with BD enrichment phases.
    # find_profiles takes parallel_after_serp=callback that runs alongside enrichment.
    name_for_verify = f"{req.name} ({latin_name})" if latin_name != req.name else req.name
    verify_holder = {"result": None, "duration": 0.0}

    async def _parallel_verify(serp_profiles):
        # Snapshot the SERP-stage profiles (no BD enrichment data yet),
        # run sync verify in a thread so it doesn't block the event loop.
        if not serp_profiles:
            return
        snapshot = [dict(p) for p in serp_profiles]
        t_v = time.time()
        verify_holder["result"] = await asyncio.to_thread(
            verify_profiles, snapshot, name_for_verify, req.description
        )
        verify_holder["duration"] = round(time.time() - t_v, 2)
        logger.info(f"  PHASE verify_profiles (parallel) {verify_holder['duration']}s")

    # 2026-05-29: launch handle-guess EAGERLY so it runs in parallel
    # with find_profiles + verify. Saves ~3–5s wall vs sequential.
    t_guess = time.time()
    _seed_handles = [h for h in (_handle_from_url(u) for u in known_links.values()) if h]
    guess_task = asyncio.create_task(_guess_handle_candidates(req.name, [], seed_handles=_seed_handles))

    t0 = time.time()
    found = await find_profiles(
        usernames=usernames,
        name=f"{req.name} {latin_name}" if latin_name != req.name else req.name,
        description=req.description,
        known_links=known_links,
        parallel_after_serp=_parallel_verify,
    )
    t["find_profiles"] = round(time.time() - t0, 2)
    t["verify_profiles"] = verify_holder["duration"]
    logger.info(f"  PHASE find_profiles {t['find_profiles']}s, {len(found)} candidates")
    for p in found:
        logger.info(f"    {p.get('platform')}: {p.get('url')} (src={p.get('source')})")

    # Await the parallel handle-guess and merge its hits.
    guessed = await guess_task
    t["handle_guess"] = round(time.time() - t_guess, 2)
    if guessed:
        have = {(p.get("url") or "").rstrip("/").lower() for p in found}
        added = [g for g in guessed if (g.get("url") or "").rstrip("/").lower() not in have]
        found.extend(added)
        logger.info(f"  PHASE handle_guess {t['handle_guess']}s (parallel) — added {len(added)} candidates")

    # ITER24 2026-05-28: avatar fill — direct URLs for X/FB/TT/LI + real IG fetch via HTTP scrape
    from urllib.parse import quote as _qq
    import re as _re
    import httpx as _httpx
    BROWSER_UA = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    async def _fetch_ig_avatar(p, client):
        """Fetch IG public page, extract real avatar URL from scontent.cdninstagram.com pattern."""
        u = (p.get("username") or "").lstrip("@")
        if not u:
            return
        try:
            r = await client.get(
                f"https://www.instagram.com/{u}/",
                headers={"User-Agent": BROWSER_UA},
                timeout=3, follow_redirects=True,
            )
            if r.status_code != 200:
                return
            m = _re.search(r'(scontent[^"]*cdninstagram[^"]*?profile_pic[^"]*)', r.text)
            if not m:
                # Fallback pattern (without "profile_pic" tag)
                m = _re.search(r'(scontent[^"]*cdninstagram[^"]*\.jpg[^"]*)', r.text)
            if m:
                url = m.group(1).replace("&amp;", "&")
                p["image_url"] = url
        except Exception:
            pass

    if found:
        t0 = time.time()
        # Pre-fill non-IG URLs (zero latency — browser fetches at render time)
        for p in found:
            p.setdefault("bio", "")
            p.setdefault("display_name", p.get("username", ""))
            p.setdefault("image_url", "")
            p.setdefault("followers", "")
            if not p.get("image_url"):
                platform = p.get("platform", "")
                u = (p.get("username") or "").lstrip("@")
                if not u:
                    continue
                if platform == "X":
                    p["image_url"] = f"/api/spl/api/avatar/x?u={u}"  # ITER34: bypass unavatar 429
                elif platform == "Facebook":
                    p["image_url"] = f"/api/spl/api/avatar/facebook?u={u}"  # 2026-05-29: SC FB endpoint
                elif platform == "TikTok":
                    p["image_url"] = f"/api/spl/api/avatar/tiktok?u={u}"  # ITER34: bypass unavatar 429
                elif platform == "LinkedIn":
                    p["image_url"] = f"/api/spl/api/avatar/linkedin?u={u}"  # 2026-05-29: SC LI endpoint
                # IG handled below via HTTP scrape

        # For IG: parallel HTTP fetch of public page, regex-extract cdninstagram URL
        ig_profiles = [p for p in found if p.get("platform") == "Instagram" and not p.get("image_url")]
        if ig_profiles:
            async with _pooled() as ig_client:
                await asyncio.gather(*[_fetch_ig_avatar(p, ig_client) for p in ig_profiles])

        # ITER34: IG without BD enrichment now routes through SC IG profile
        # endpoint instead of ui-avatars initials. Same 1-SC-credit cost as TT.
        for p in found:
            if p.get("platform") == "Instagram" and not p.get("image_url"):
                u = (p.get("username") or "").lstrip("@")
                if u:
                    p["image_url"] = f"/api/spl/api/avatar/instagram?u={u}"
        t["scrape_profiles"] = round(time.time() - t0, 2)
        logger.info(f"  PHASE scrape_profiles {t['scrape_profiles']}s (avatar fill incl. IG HTTP)")

    # Merge verify scores back into enriched profiles by URL
    total_found = len(found)
    if found and verify_holder["result"]:
        score_map = {v.get("url"): v for v in verify_holder["result"] if v.get("url")}
        for p in found:
            v = score_map.get(p.get("url"))
            if v:
                p["score"] = v.get("score", 0)
                p["reasoning"] = v.get("reasoning", "")
            else:
                p.setdefault("score", 0)
                p.setdefault("reasoning", "")
        # 2026-05-31 v7.8 wider-net: CONFIDENCE_THRESHOLD drop removed.
        # Was: survivors = [p for p in found if score >= 50], with a
        # per-platform top-1 fallback. Now: ALL Haiku-scored candidates
        # emit regardless of score — the user judges in the carousel.
        # Constant kept defined for any external reference.

    # Sanitize reasoning - strip backend implementation details
    import re as _re
    BACKEND_TERMS = _re.compile(r'\b(brightdata|bright data|google search|serp|haiku|claude|anchor profile|api|metadata|source[: ]+brightdata|fp guard|false[ -]?positive|username|display[ _-]?name|headline|name absent|guard)\b', _re.IGNORECASE)
    for p in found:
        reasoning = p.get("reasoning", "")
        if reasoning:
            # Remove sentences mentioning backend terms
            sentences = _re.split(r'(?<=[.!?])\s+', reasoning)
            clean = [s for s in sentences if not BACKEND_TERMS.search(s)]
            p["reasoning"] = " ".join(clean) if clean else reasoning

    # Per-platform cap: keep more candidates so the downstream avatar
    # filter has alternatives to fall back on when the top hit has no
    # real picture. 2026-05-29: was 1-or-2; now 1 (dominant) / 3 (close
    # call) so we surface ~5 results per query on regular-person searches
    # where many candidates exist but only some have usable avatars.
    by_platform: dict[str, list[dict]] = {}
    for p in found:
        by_platform.setdefault(p["platform"], []).append(p)

    filtered = []
    for platform, profs in by_platform.items():
        profs.sort(key=lambda x: x.get("score", 0), reverse=True)
        top = profs[0]
        if len(profs) == 1:
            filtered.append(top)
        else:
            second = profs[1]
            # Dominant: top > 85 AND second < 70 → keep just top
            if top.get("score", 0) > 85 and second.get("score", 0) < 70:
                filtered.append(top)
            else:
                # 2026-05-31: top 6 (was 4). Loosened per-platform cap to
                # let through more legit candidates the operator can
                # triage. Tier-1 free existence checks now cover most of
                # the cost concern that originally drove the cap down.
                filtered.extend(profs[:6])
    found = sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)

    # 2026-05-30: avatar fetch + inline. Goes beyond the prior validate-
    # only flow: actually downloads the bytes and stashes them on the
    # profile as `_image_bytes` so they're returned inline (base64) in
    # the response. Customer's browser doesn't need a second roundtrip
    # per chip — pictures appear with the chip, no async load lag.
    # Drops profiles whose avatar didn't materialise (incl. LI ghosts)
    # to keep coverage at ~99%.
    t_av = time.time()
    async def _avatar_fetch(p):
        """Return (ok, bytes_or_None, content_type)."""
        iu = p.get("image_url", "") or ""
        if not iu:
            return False, None, ""
        if iu.startswith("/api/spl/"):
            local_url = "http://127.0.0.1:8801" + iu[len("/api/spl"):]
        elif iu.startswith("/"):
            local_url = "http://127.0.0.1:8801" + iu
        else:
            local_url = iu
        try:
            async with _pooled() as cl:
                # 2026-05-30: 5s timeout (was 8s) — slowest single avatar
                # used to anchor the whole parallel wall. Faster fail =
                # faster overall response. Avatars that genuinely need
                # more time get caught by their second-warm-cache hit.
                r = await cl.get(local_url, timeout=5.0, follow_redirects=True)
            if r.status_code != 200 or not r.content:
                return False, None, ""
            ct = (r.headers.get("content-type") or "").lower()
            if not ct.startswith("image"):
                return False, None, ""
            if len(r.content) < 2000:
                return False, None, ""
            return True, r.content, ct
        except Exception:
            return False, None, ""

    if found:
        fetched = await asyncio.gather(*[_avatar_fetch(p) for p in found])
        before = len(found)
        kept = []
        import base64 as _b64
        for p, (ok, data, ct) in zip(found, fetched):
            if not ok:
                # Drop unconditionally (incl. LI). The new 99%-target
                # goal trumps the per-platform special-cases; LI w/o pic
                # is now treated the same as any other no-pic candidate.
                continue
            ct = ct or "image/jpeg"
            # Inline as data: URI on a NEW key (the existing image_url
            # stays so face-rerank etc. still know the source).
            p["image_data"] = "data:" + ct + ";base64," + _b64.b64encode(data).decode("ascii")
            kept.append(p)
        # 2026-05-30: compute pHash + face embedding for every kept
        # avatar in parallel via thread pool. Reused face_match disk
        # cache means repeats are essentially free.
        if kept:
            avatar_bytes_map = {p.get("url"): data
                                for p, (ok, data, ct) in zip(found, fetched) if ok}
            await asyncio.gather(*[
                asyncio.to_thread(attach_similarity_payload, p,
                                  avatar_bytes_map.get(p.get("url"), b""))
                for p in kept
            ])
        found = kept
        t_avatar = round(time.time() - t_av, 2)
        t["avatar_filter"] = t_avatar
        logger.info(f"  PHASE avatar_inline {t_avatar}s — kept {len(found)}/{before}")

    total = round(time.time() - start, 2)
    other = round(total - sum(t.values()), 2)
    logger.info(f"  PHASE_BREAKDOWN total={total}s transliterate={t.get('transliterate',0)} find={t.get('find_profiles',0)} scrape={t.get('scrape_profiles',0)} verify={t.get('verify_profiles',0)} avatar_filter={t.get('avatar_filter',0)} other={other}")
    logger.info(f"  after scoring + dedup + avatar filter: {len(found)} profiles, elapsed={total}s")

    elapsed = round(time.time() - start, 2)

    return SearchResponse(
        profiles=[
            ProfileResult(
                platform=p.get("platform", ""),
                url=p.get("url", ""),
                username=p.get("username", ""),
                display_name=p.get("display_name", ""),
                bio=p.get("bio", ""),
                image_url=p.get("image_url", ""),
                image_data=p.get("image_data", ""),
                phash_b64=p.get("phash_b64", ""),
                face_emb_b64=p.get("face_emb_b64", ""),
                is_private=bool(p.get("is_private", False)),
                score=p.get("score", 0),
                reasoning=p.get("reasoning", ""),
                position=p.get("position", ""),
                company=p.get("company", ""),
                city=p.get("city", ""),
                followers=p.get("followers", ""),
            )
            for p in found
        ],
        candidates_searched=len(usernames),
        total_found=total_found,
        elapsed_seconds=elapsed,
        authoritative_name=namefix_helper.authoritative_display_name(
            req.name, found),
    )


# ITER30 2026-05-28: streaming variant of /api/search.
# Emits profiles via NDJSON as soon as Haiku verify completes, then continues
# emitting one at a time at ~400ms intervals (the "dice game" UX). BD IG
# enrichment runs in parallel; when it completes (typically after verify), an
# avatar_update event upgrades the IG avatar from ui-avatars init to the real
# cdninstagram URL.
from fastapi.responses import StreamingResponse as _StreamingResponse
from profile_finder import (
    PLATFORMS,
    PROFILE_URL_PATTERNS,
    CANONICAL_URL,
    search_platform,
    brightdata_instagram,
    serp_haiku_safety_net,  # v7.56 SERP Haiku safety net
)


# v7.67 enrich-on-emit: avatar inline + follower fetch now start the MOMENT
# each chip is emitted (concurrent tasks overlapping discovery) and patch in
# via avatar_update/followers_update as they finish — instead of blocking each
# platform's emit on a whole-batch avatar gather and deferring followers to a
# single post-loop batch. See the _enrich_one / _enrich_tasks drain below.
@app.post("/api/search-stream")
async def search_profiles_stream(req: SearchRequest):
    async def gen():
        import json as _json
        import re as _re
        import httpx as _httpx
        from urllib.parse import quote as _qq

        start = time.time()

        def _emit(obj):
            return (_json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

        # Transliteration (skipped for ASCII names — usually instant)
        # audit-F8 2026-07-06: off the event loop — the sync Anthropic call (~0.3-1s
        # for non-Latin names) otherwise blocks EVERY concurrent request on the worker.
        latin_name = await asyncio.to_thread(transliterate_name, req.name)
        name_for_verify = (
            f"{req.name} ({latin_name})" if latin_name != req.name else req.name
        )

        known_links = {}
        if req.link_x: known_links["X"] = req.link_x
        if req.link_facebook: known_links["Facebook"] = req.link_facebook
        if req.link_instagram: known_links["Instagram"] = req.link_instagram
        if req.link_tiktok: known_links["TikTok"] = req.link_tiktok
        if req.link_linkedin: known_links["LinkedIn"] = req.link_linkedin

        # 2026-05-30 Phase 2: block-list of URLs the customer rejected
        # in prior attempts of this session (carousel-triage refined
        # search). Normalised lowercase + trailing-slash stripped for
        # matching during emit.
        _rejected_block = {
            (u or "").rstrip("/").lower()
            for u in (req.rejected_urls or []) if u
        }

        # 2026-05-30 v4: single source of truth for per-profile dedup.
        # Three emit sites (early-hg drain, per-platform verify drain,
        # late-hg drain) used to each carry their own seen-sets; under
        # certain ordering the same profile slipped through twice.
        # Key = "platform|@handle" (handle preferred — survives URL
        # variants like x.com vs twitter.com, trailing slashes, etc.);
        # falls back to normalised URL when handle is empty.
        _emitted_keys: set[str] = set()

        # 2026-05-30 v6: REVERTED the pHash dedup. The user's original
        # complaint ("two cards for the exact same profile on the same
        # network") was about handle duplication, NOT image duplication.
        # Same image on DIFFERENT handles = distinct accounts that the
        # carousel rerank turns into a feature: confirm one → the
        # others rescore to positions 1, 2 (their face_sim = 1.0).
        # That IS the success criterion. So we keep handle dedup only.

        def _profile_key(p) -> str:
            plat = (p.get("platform") or "").lower()
            handle = (p.get("username") or "").lstrip("@").lower()
            if handle:
                return f"{plat}|@{handle}"
            url = (p.get("url") or "").strip().lower()
            url = url.split("?")[0].rstrip("/")
            return f"{plat}|url:{url}"

        # 2026-05-31 v7.3: near-handle index for cross-handle dedup.
        # Per-platform map: normalized handle (alphanumeric-only,
        # lowercase) → list of (display_name_lower, phash_b64) tuples
        # of profiles already emitted under that normalized form.
        _norm_handle_seen: dict[str, dict[str, list[tuple[str, str]]]] = {}

        # 2026-06-04 EXACT-DUP COLLAPSE: same real profile re-discovered
        # under different guessed handles (FB redirects konstantin.kisin,
        # kkisin, konstantinkisin -> ONE page; each variant becomes its own
        # candidate with a distinct (platform,handle) key, so neither
        # _emitted_keys nor the phash-gated NEAR_HANDLE guard collapses
        # them and the picker shows two identical cards). Signature =
        # (platform, normalized display name, normalized follower count):
        # the operator-visible same name + same followers + same network
        # tuple. Applies to EVERY platform (X/FB/IG/LI/TT) — per-platform
        # parity — and still lets genuinely distinct same-network pages
        # through (they differ on name OR follower count).
        _emitted_sig: set[tuple[str, str, str]] = set()

        def _norm_followers(f) -> str:
            # Reduce any follower representation (230K, 230,000,
            # 230000, ) to a canonical digit string for comparison.
            s = str(f or "").strip().lower().replace(",", "")
            if not s:
                return ""
            mult = 1
            if s.endswith("k"):
                mult, s = 1_000, s[:-1]
            elif s.endswith("m"):
                mult, s = 1_000_000, s[:-1]
            try:
                return str(int(round(float(s) * mult)))
            except (ValueError, TypeError):
                return s

        def _norm_handle(h: str) -> str:
            return "".join(c for c in (h or "").lower() if c.isalnum())

        def _should_emit(p) -> bool:
            # 2026-05-30 v7.2: drop Facebook chips whose handle is
            # purely numeric — that's FB's internal user ID, an
            # alternate URL form for the same account that also has a
            # vanity URL. Always-redundant; never useful to the user.
            plat = (p.get("platform") or "").lower()
            handle = (p.get("username") or "").lstrip("@")
            if plat == "facebook" and handle.isdigit():
                logger.info(f"  FB_NUMERIC_ID dropped facebook|@{handle}")
                return False

            k = _profile_key(p)
            if k in _emitted_keys:
                return False

            # 2026-05-31 v7.3: near-handle dedup. Two handles that
            # collapse to the same alphanumeric string (e.g.
            # "itzik.pinchas" vs "itzikpinchas") AND carry the same
            # display name AND the same image are the same account
            # under variant URL forms (FB ignores dots in vanity paths).
            # Drop the duplicate. Requires ALL THREE matches so genuine
            # alts with similar handles but different identity stay.
            norm = _norm_handle(handle)
            if norm:
                display_low = (p.get("display_name") or "").strip().lower()
                ph = p.get("phash_b64") or ""
                prior_list = _norm_handle_seen.get(plat, {}).get(norm, [])
                for prior_display, prior_ph in prior_list:
                    same_name = display_low and prior_display and display_low == prior_display
                    same_image = ph and prior_ph and ph == prior_ph
                    if same_name and same_image:
                        logger.info(f"  NEAR_HANDLE_DUP dropped {plat}|@{handle} (matches prior norm={norm})")
                        return False
                _norm_handle_seen.setdefault(plat, {}).setdefault(norm, []).append(
                    (display_low, ph)
                )

            # 2026-06-04 EXACT-DUP COLLAPSE (all platforms): if a profile
            # with this exact (platform, display name, follower count)
            # signature already emitted, this is the SAME account under a
            # variant handle (e.g. FB vanity aliases) — drop it. Requires a
            # non-empty display name AND a non-empty follower count so we
            # never collapse two count-less or name-less chips by accident.
            _disp_sig = (p.get("display_name") or "").strip().lower()
            _foll_sig = _norm_followers(p.get("followers"))
            if _disp_sig and _foll_sig:
                _sig = (plat, _disp_sig, _foll_sig)
                if _sig in _emitted_sig:
                    logger.info(
                        f"  EXACT_DUP_PROFILE dropped {plat}|@{handle} "
                        f"(same name+followers as prior: {_disp_sig!r}/{_foll_sig})"
                    )
                    return False
                _emitted_sig.add(_sig)

            _emitted_keys.add(k)
            # 2026-05-30 v7 + v7.18: strip generic SEO meta descriptions
            # from every platform's bio (FB / IG / TT / X / LinkedIn).
            # SERP snippets often contain only this boilerplate.
            p["bio"] = _clean_social_bio(p.get("bio") or "")
            return True

        # 2026-05-31 v7.6: patience-driven timeout scale. 'long' = 2× —
        # used by the recal-round second search where catching late
        # candidates matters more than wall time.
        TS = 2.0 if (req.patience or "normal").lower() == "long" else 1.0
        logger.info(f"  patience={req.patience} ts_scale={TS}")

        yield _emit({"type": "phase", "name": "started", "ts": 0.0})

        # 2026-05-30: fire handle-guess + inline its avatars from t=0
        # in parallel with SERP. By verify_done time, the bytes are
        # already inlined → emits are instant. Especially important for
        # queries SERP misses entirely (Sarah Chen) — handle-guess hits
        # become the only chips.
        async def _hg_then_inline():
            import base64 as _b64
            import httpx as _hg_httpx
            async def _inline(p):
                # 2026-05-31 v7.8 wider-net: never DROP a handle-guess chip on
                # avatar failure. If the fetch succeeds, attach image_data +
                # phash + face_emb (existing behavior). If it fails for any
                # reason, leave image_data empty and return the chip anyway —
                # the frontend renders an initials placeholder.
                p.setdefault("image_data", "")
                iu = p.get("image_url") or ""
                if not iu:
                    return p
                if iu.startswith("/api/spl/"):
                    url = "http://127.0.0.1:8801" + iu[len("/api/spl"):]
                elif iu.startswith("/"):
                    url = "http://127.0.0.1:8801" + iu
                else:
                    url = iu
                try:
                    async with _hg_pooled() as cl:
                        r = await cl.get(url, timeout=5.0 * TS, follow_redirects=True)
                    if r.status_code != 200 or not r.content:
                        return p
                    ct = (r.headers.get("content-type") or "image/jpeg").lower()
                    if not ct.startswith("image") or len(r.content) < 2000:
                        return p
                    p["image_data"] = "data:" + ct + ";base64," + _b64.b64encode(r.content).decode("ascii")
                    # 2026-05-30: compute pHash + face embedding from
                    # the same bytes (free, no refetch). Runs in this
                    # coro's thread offload via to_thread so the event
                    # loop stays responsive during the ~150ms embed.
                    await asyncio.to_thread(attach_similarity_payload, p, r.content)
                    return p
                except Exception:
                    return p
            _seed_handles = [h for h in (_handle_from_url(u) for u in known_links.values()) if h]
            # audit-F4 2026-07-06: feed the TRANSLITERATED name — _name_variants strips
            # non-ASCII to empty, so a raw Hebrew/Arabic name produced ZERO handle
            # guesses (the SERP-miss safety-net lane was dead for exactly those people).
            _hg_name = latin_name if (latin_name and latin_name != req.name) else req.name
            guessed = await _guess_handle_candidates(_hg_name, [], seed_handles=_seed_handles)
            inlined = await asyncio.gather(*[_inline(g) for g in guessed])
            return [g for g in inlined if g is not None]

        stream_hg_task = asyncio.create_task(_hg_then_inline())

        async with _pooled() as client:
            # Phase 1: SERP fan-out
            search_tasks = {}
            profiles = []
            # v7.56: accumulate the RAW organic results (incl. the ones the
            # regex filter drops) + Part-A knowledge-graph social links across
            # every platform's queries, for the single serp_haiku safety net.
            serp_raw_sink: list = []
            for platform, domain in PLATFORMS.items():
                if platform in known_links and known_links[platform]:
                    url = known_links[platform]
                    pattern = PROFILE_URL_PATTERNS.get(platform)
                    username = ""
                    if pattern:
                        m = pattern.search(url)
                        if m: username = m.group(1)
                    profiles.append({
                        "platform": platform, "url": url, "username": username,
                        "source": "user_provided", "bio": "", "display_name": "",
                        "image_url": "", "verified": True,
                    })
                else:
                    search_tasks[platform] = search_platform(
                        f"{req.name} {latin_name}" if latin_name != req.name else req.name,
                        req.description, platform, domain, client,
                        raw_sink=serp_raw_sink,
                    )

            if search_tasks:
                results = await asyncio.gather(*search_tasks.values())
                for platform, candidates in zip(search_tasks.keys(), results):
                    profiles.extend(candidates)

            # audit-F5 2026-07-06: the SERP Haiku safety net (recovers regex-dropped
            # profiles) used to BLOCK up to 3s before serp_done was emitted, delaying
            # first-chip. Kick it off concurrently, emit serp_done now, and await+merge
            # its results just before the verify fan-out below — every recovered
            # candidate still flows through the SAME per-platform verify + FP-guard
            # (nothing bypasses scoring), but the 3s now overlaps the handle-guess
            # early drain instead of sitting on the critical path.
            _sn_existing_keys = set()
            for _p in profiles:
                _u = (_p.get("username") or "").lstrip("@").lower()
                if _u:
                    _sn_existing_keys.add(f"{(_p.get('platform') or '').lower()}|{_u}")
            _sn_name = (
                f"{req.name} {latin_name}" if latin_name != req.name else req.name
            )
            _safety_net_task = asyncio.create_task(asyncio.wait_for(
                asyncio.to_thread(
                    serp_haiku_safety_net,
                    serp_raw_sink, _sn_name, req.description,
                    _sn_existing_keys, _normalize_profile_url,
                ),
                timeout=3.0 * TS,
            ))

            yield _emit({"type": "phase", "name": "serp_done",
                         "ts": round(time.time() - start, 2),
                         "candidate_count": len(profiles)})

            # 2026-05-30: as soon as SERP is done, drain handle-guess if
            # it's also ready (it usually is — SC probes start at t=0 and
            # complete in 3-7s, before BD SERP). Emitting here drops the
            # first-chip time on low-SERP queries from ~12s to ~7-9s.
            if stream_hg_task.done():
                try:
                    early_hg = stream_hg_task.result() or []
                    seen = {(p.get("url") or "").rstrip("/").lower() for p in profiles}
                    for g in early_hg:
                        k = (g.get("url") or "").rstrip("/").lower()
                        if k in seen:
                            continue
                        if k in _rejected_block:
                            continue            # Phase 2 block-list
                        if not _should_emit(g):
                            continue            # 2026-05-30 v4: global dedup
                        # Mark as already-emitted so the post-verify drain skips it.
                        g["_early_emitted"] = True
                        seen.add(k)
                        # No stagger on the first early hit; small stagger after.
                        if 'first_emit_done_early' in dir():
                            await asyncio.sleep(0.08)
                        yield _emit({"type": "profile", "data": dict(g),
                                     "ts": round(time.time() - start, 2)})
                    # Add the early-emitted ones to a tracker the post-verify
                    # drain can dedupe against.
                    emitted_hg_urls = {k for k in seen}
                except Exception as _eh_err:
                    logger.warning(f"early hg drain failed: {_eh_err}")
                    emitted_hg_urls = set()
            else:
                emitted_hg_urls = set()

            # audit-F5 2026-07-06: await + merge the safety-net recoveries here, so all
            # candidates (regex + recovered) are present before the verify fan-out and
            # the first_ig scan below. This await overlaps the early-hg drain above.
            try:
                _sn_new = await _safety_net_task
                if _sn_new:
                    _sn_added: dict[str, int] = {}
                    for _c in _sn_new:
                        _sn_added[_c["platform"]] = _sn_added.get(_c["platform"], 0) + 1
                    profiles.extend(_sn_new)
                    logger.info(f"  [serp_haiku] added {len(_sn_new)} (per-platform {_sn_added})")
                else:
                    logger.info("  [serp_haiku] added 0 (no new profiles)")
            except asyncio.TimeoutError:
                logger.info("  [serp_haiku] timed out — added 0")
            except Exception as _sn_err:
                logger.info(f"  [serp_haiku] error — added 0: {_sn_err}")

            # ITER31 2026-05-28: per-platform parallel verify.
            # Old flow: single Haiku call over ALL candidates → ~5s before any profile
            # could be scored, so first emit landed at ~serp_done + 5s = ~8s wall.
            # New flow: split candidates by platform, fire one small Haiku call per
            # platform in parallel via asyncio.gather(asyncio.to_thread(...)). Each
            # call sees 1–3 candidates and returns in ~1s, so the first platform's
            # profiles emit at ~serp_done + 1s. Cost: 5× Haiku calls (~$0.025 total
            # vs $0.005 — still negligible).
            first_ig = next(
                (p for p in profiles if p["platform"] == "Instagram" and p.get("username")),
                None,
            )

            async def _enrich_ig():
                if first_ig:
                    ig_res = await brightdata_instagram(first_ig["username"])
                    if ig_res:
                        for k, v in ig_res.items():
                            if v: first_ig[k] = v
                        first_ig["source"] = "brightdata"

            # 2026-05-31: only spend BD credit (~$0.05) when SERP actually
            # found an Instagram candidate. Previously we created the task
            # unconditionally — it'd no-op inside _enrich_ig but the await
            # path is now cleanly guarded too.
            if first_ig:
                ig_task = asyncio.create_task(_enrich_ig())
            else:
                ig_task = None

            # Group SERP candidates by platform (deep-copy so per-platform mutations
            # don't bleed across tasks)
            by_plat_in: dict[str, list[dict]] = {}
            for p in profiles:
                by_plat_in.setdefault(p["platform"], []).append(dict(p))

            BACKEND_TERMS = _re.compile(
                r'\b(brightdata|bright data|google search|serp|haiku|claude|'
                r'anchor profile|api|metadata|source[: ]+brightdata|fp guard|'
                r'false[ -]?positive|username|display[ _-]?name|headline|'
                r'name absent|guard)\b', _re.IGNORECASE,
            )

            def _post_process_platform(platform: str, scored: list[dict]) -> list[dict]:
                """Reasoning sanitize + hybrid top-1/top-2 limit, scoped to
                a single platform's candidates.
                2026-05-31 v7.8 wider-net: CONFIDENCE_THRESHOLD drop +
                low-confidence fallback removed — every Haiku-scored chip
                survives. Hybrid top-1/top-2 cap retained: still want to
                limit to the strongest 1-2 per platform so we don't drown
                the carousel in low-quality matches from a single source."""
                if not scored:
                    return []
                survivors = list(scored)

                # Sanitize reasoning
                for p in survivors:
                    r = p.get("reasoning", "")
                    if r:
                        sents = _re.split(r'(?<=[.!?])\s+', r)
                        clean = [s for s in sents if not BACKEND_TERMS.search(s)]
                        p["reasoning"] = " ".join(clean) if clean else r

                # Hybrid limit: 1 if dominant, 2 if close
                survivors.sort(key=lambda x: x.get("score", 0), reverse=True)
                if len(survivors) <= 1:
                    out = survivors
                else:
                    top, second = survivors[0], survivors[1]
                    if top.get("score", 0) > 85 and second.get("score", 0) < 70:
                        out = [top]
                    elif len(survivors) >= 3 and survivors[2].get("score", 0) >= 70:
                        # audit-F10 2026-07-06: surface a strong 3rd (personal+brand,
                        # main+finsta) — already scored, so no extra API calls.
                        out = survivors[:3]
                    else:
                        out = [top, second]

                # Avatar fast-fill
                for p in out:
                    p.setdefault("bio", "")
                    p.setdefault("display_name", p.get("username", ""))
                    p.setdefault("image_url", "")
                    p.setdefault("followers", "")
                    if not p.get("image_url"):
                        u = (p.get("username") or "").lstrip("@")
                        if not u:
                            continue
                        if platform == "X":
                            p["image_url"] = f"/api/spl/api/avatar/x?u={u}"  # ITER34: bypass unavatar 429
                        elif platform == "Facebook":
                            p["image_url"] = f"/api/spl/api/avatar/facebook?u={u}"  # 2026-05-29: SC FB
                        elif platform == "TikTok":
                            p["image_url"] = f"/api/spl/api/avatar/tiktok?u={u}"  # ITER34: bypass unavatar 429
                        elif platform == "LinkedIn":
                            p["image_url"] = f"/api/spl/api/avatar/linkedin?u={u}"  # 2026-05-29: SC LI
                        elif platform == "Instagram":
                            # ITER34: SC IG profile endpoint (replaces ui-avatars init).
                            # BD-IG enrichment running in parallel may still upgrade
                            # the first IG candidate via avatar_update; this endpoint
                            # serves the rest with real cdninstagram photos.
                            p["image_url"] = f"/api/spl/api/avatar/instagram?u={u}"
                return out

            async def _verify_one_platform(platform: str, plat_profiles: list[dict]):
                """Run Haiku on one platform's candidates in a thread; return
                (platform, post_processed_results, raw_count) tuple.
                2026-05-30: 2.5s hard timeout. If Haiku is slow, fall back
                to SERP-derived scoring (skip verify entirely for this
                platform) — keeps wall under control.
                """
                t_v = time.time()
                try:
                    scored = await asyncio.wait_for(
                        asyncio.to_thread(
                            verify_profiles_single_platform,
                            plat_profiles, name_for_verify, req.description,
                        ),
                        timeout=2.5 * TS,   # v7.6: 5.0s on patience='long'
                    )
                except asyncio.TimeoutError:
                    # Fallback: assign a baseline score by SERP rank.
                    # 2026-05-31 v7.7: do NOT stamp "(verify timeout)"
                    # into reasoning — that's operator-facing language
                    # leaking to the customer card. Leave reasoning as
                    # whatever the SERP step already produced (often
                    # empty), which the frontend renders as no checklist.
                    scored = []
                    for i, p in enumerate(plat_profiles):
                        c = dict(p)
                        c["score"] = max(40, 70 - i * 5)
                        scored.append(c)
                    logger.info(f"  [stream] verify {platform} TIMEOUT — used SERP-rank fallback")
                duration = round(time.time() - t_v, 2)
                logger.info(f"  [stream] verify {platform} {duration}s ({len(plat_profiles)} cands)")
                return platform, _post_process_platform(platform, scored), duration

            # Fan out: one verify task per platform that has candidates
            verify_tasks = [
                asyncio.create_task(_verify_one_platform(plat, profs))
                for plat, profs in by_plat_in.items()
                if profs
            ]
            # 2026-06-08 #1: warm the avatar cache DURING verify. Handles are
            # known from SERP now; firing the top-2 candidates' avatar fetches per
            # platform in parallel with the (1-5s) Haiku verify means the cache is
            # hot by the time a chip emits, so the picture appears WITH the card
            # instead of ~0.5-2s later. Fire-and-forget; never blocks the stream.
            _AV_RESOLVERS = {"X": avatar_x, "Facebook": avatar_facebook,
                             "Instagram": avatar_instagram, "TikTok": avatar_tiktok,
                             "LinkedIn": avatar_linkedin}
            async def _prewarm_avatar(_plat, _handle):
                try:
                    _fn = _AV_RESOLVERS.get(_plat)
                    if _fn and _handle:
                        await _fn(u=_handle)   # populates _avatar_cache; result ignored
                except Exception:
                    pass
            for _pf_plat, _pf_profs in by_plat_in.items():
                for _pf in (_pf_profs or [])[:2]:
                    _pf_h = (_pf.get("username") or "").lstrip("@")
                    if _pf_h:
                        asyncio.create_task(_prewarm_avatar(_pf_plat, _pf_h))

            ig_already_emitted = False
            emitted_final: list[dict] = []
            verify_durations: dict[str, float] = {}

            # 2026-05-30: stagger trimmed from 0.75s → 0.1s. With 10+
            # chips per query (handle-guess added candidates), the old
            # cadence added 7-8s of artificial wait. Frontend animation
            # still feels staggered at 100ms but total wall closes much
            # faster. If the "dice game" cadence is needed back, the
            # frontend can throttle the SSE consumption instead.
            EMIT_STAGGER_S = 0.1
            first_emit_done = False
            # v7.64 perf: chips emitted without a follower count get queued
            # here; we fetch them all in PARALLEL after the emit loops and
            # patch each in via a followers_update SSE event (same async
            # pattern the avatar uses). Non-blocking, so the first card paints
            # immediately instead of waiting on a serialized 10-12s-timeout
            # SC/BD profile call per chip (the v7.61 regression).
            # v7.67 enrich-on-emit: the MOMENT a chip is emitted we kick off its
            # enrichment (avatar inline → image_data+phash+face, AND follower
            # fetch when missing) as a CONCURRENT task — overlapping the rest of
            # discovery instead of (a) blocking each platform's emit on a
            # whole-batch avatar gather [old line 2344] or (b) deferring every
            # follower fetch to a single post-loop batch [old _followers_pending
            # drain]. After the main emit loop these tasks are drained with
            # asyncio.as_completed and each result is yielded as an avatar_update
            # / followers_update SSE patch the instant it finishes. Because the
            # fetches already ran during discovery, the patches go out almost
            # immediately at loop-end. Bounded (≤2 emitted chips/platform via
            # _should_emit), cached, no new paid calls, no dropped candidates.
            _enrich_tasks: list = []        # list[asyncio.Task] -> _EnrichResult
            # v7.67 queue: each enrichment task pushes its finished patch dict
            # onto this queue. The generator drains it (non-blocking) at every
            # natural await point — the per-chip EMIT_STAGGER sleeps AND while it
            # waits on handle-guess — so a patch goes out the INSTANT its fetch
            # finishes, INTERLEAVED with ongoing discovery, instead of being held
            # until a single post-loop drain. A final as_completed sweep flushes
            # anything still in flight before the stream closes (no lost patch).
            _patch_q: "asyncio.Queue" = asyncio.Queue()
            import base64 as _b64
            import httpx as _stream_httpx

            def _drain_patch_events():
                """Yield SSE patch events for every enrichment result currently
                ready on the queue. Non-blocking — returns immediately when the
                queue is empty. Safe to call from anywhere in the generator
                body (a generator may only yield from its own body, so we drain
                here rather than from the producer)."""
                _out = []
                while True:
                    try:
                        _res = _patch_q.get_nowait()
                    except Exception:
                        break
                    if not _res:
                        continue
                    _av = _res.get("avatar")
                    if _av and _av.get("image_data"):
                        _ave = {"type": "avatar_update",
                                "url": _res.get("url"),
                                "phash_b64": _av.get("phash_b64", ""),
                                "face_emb_b64": _av.get("face_emb_b64", ""),
                                "ts": round(time.time() - start, 2)}
                        if _av.get("image_data"):
                            _ave["image_data"] = _av["image_data"]
                        _out.append(_ave)
                    _fc = _res.get("followers")
                    if _fc:
                        _out.append({"type": "followers_update",
                                     "url": _res.get("url"),
                                     "followers": _fc, "bio": "",
                                     "ts": round(time.time() - start, 2)})
                return _out

            async def _inline_avatar(prof):
                """Fetch the avatar bytes, set image_data, and compute the
                pHash + face embedding (carousel rerank). Mutates prof in
                place. v7.8: never drops a chip on failure (empty image_data
                → frontend initials placeholder)."""
                prof.setdefault("image_data", "")
                iu = prof.get("image_url") or ""
                if not iu:
                    return
                if iu.startswith("/api/spl/"):
                    url = "http://127.0.0.1:8801" + iu[len("/api/spl"):]
                elif iu.startswith("/"):
                    url = "http://127.0.0.1:8801" + iu
                else:
                    url = iu
                try:
                    async with _pooled() as cl:
                        r = await cl.get(url, timeout=5.0 * TS, follow_redirects=True)
                    if r.status_code != 200 or not r.content:
                        return
                    ct = (r.headers.get("content-type") or "image/jpeg").lower()
                    if not ct.startswith("image") or len(r.content) < 2000:
                        return
                    prof["image_data"] = (
                        "data:" + ct + ";base64," +
                        _b64.b64encode(r.content).decode("ascii")
                    )
                    # compute pHash + face embedding from the same bytes for the
                    # carousel rerank. Offload to a thread so the ~150ms embed
                    # doesn't block the event loop.
                    await asyncio.to_thread(attach_similarity_payload, prof, r.content)
                except Exception:
                    return

            async def _enrich_one(prof, need_followers: bool):
                """Concurrent per-chip enrichment kicked off at emit time.
                Runs the avatar inline and (when missing) the follower fetch in
                PARALLEL, then returns a dict of the SSE patches to yield. Never
                raises — a hiccup on one chip can't abort the stream."""
                _patches = {"url": prof.get("url")}
                try:
                    async def _av():
                        before = (prof.get("image_data") or "",
                                  prof.get("phash_b64") or "",
                                  prof.get("face_emb_b64") or "")
                        await _inline_avatar(prof)
                        after = (prof.get("image_data") or "",
                                 prof.get("phash_b64") or "",
                                 prof.get("face_emb_b64") or "")
                        if after != before and (after[0] or after[1] or after[2]):
                            # 2026-06-08 #3: the browser already paints the picture
                            # from image_url (our same-origin /api/avatar proxy), so
                            # ship only the tiny ranking payload (phash/face) and skip
                            # the ~30KB base64 image_data UNLESS the chip has no
                            # image_url to render from (then it's the only source).
                            _avp = {"phash_b64": after[1], "face_emb_b64": after[2]}
                            # audit-F3 2026-07-06: always ship the already-fetched
                            # base64 (not only when image_url is absent). The browser's
                            # own second fetch of /api/avatar can 404/502 independently
                            # (rate-limit, ghost gate, cross-worker cache miss); painting
                            # the known-good bytes we already hold kills most blank cards.
                            if after[0]:
                                _avp["image_data"] = after[0]
                            _patches["avatar"] = _avp

                    async def _fl():
                        if not need_followers:
                            return
                        fc = await _fetch_followers(prof.get("platform", ""),
                                                    prof.get("username", ""))
                        if fc:
                            prof["followers"] = fc
                            _patches["followers"] = fc

                    await asyncio.gather(_av(), _fl(), return_exceptions=True)
                except Exception:
                    pass
                # v7.67 queue: push the finished patch so the generator can flush
                # it at its next await point (interleaved with discovery). Only
                # push when there's actually something to patch.
                try:
                    if _patches.get("avatar") or _patches.get("followers"):
                        _patch_q.put_nowait(_patches)
                except Exception:
                    pass
                return _patches
            # v7.55: followers/bio backfill from handle-guess.
            # Handle-guess _sc_enrich already fetched follower counts for
            # IG (edge_followed_by) and TikTok (followerCount). When a SERP
            # chip and a guessed chip resolve to the SAME profile URL, dedup
            # keeps the (followerless) SERP chip and the post-verify drain
            # discards the guessed one - so its follower count was lost.
            # Build a url -> {followers, bio} map from whatever handle-guess
            # data is ready and backfill SERP chips at emit time. No new
            # paid calls: this only copies data already fetched.
            def _hg_followers_map():
                m = {}
                try:
                    if stream_hg_task.done():
                        for g in (stream_hg_task.result() or []):
                            k = (g.get("url") or "").rstrip("/").lower()
                            if k:
                                m[k] = {"followers": g.get("followers") or "",
                                        "bio": g.get("bio") or ""}
                except Exception:
                    pass
                return m

            for fut in asyncio.as_completed(verify_tasks):
                platform, plat_out, dur = await fut
                verify_durations[platform] = dur
                # If IG is in this batch and BD enrichment already completed, prefer
                # the BD avatar over the ui-avatars stub.
                if platform == "Instagram" and first_ig and first_ig.get("source") == "brightdata":
                    for p in plat_out:
                        if p.get("url") == first_ig.get("url"):
                            # v7.55: merge avatar AND followers/bio from BD
                            # enrichment. Previously only the avatar was copied,
                            # so the fetched follower count + bio were dropped and
                            # the chip rendered with no follower row.
                            if first_ig.get("image_url"):
                                p["image_url"] = first_ig["image_url"]
                            if first_ig.get("followers") and not p.get("followers"):
                                p["followers"] = first_ig["followers"]
                            if first_ig.get("bio") and not p.get("bio"):
                                p["bio"] = first_ig["bio"]
                # v7.67 enrich-on-emit: the whole-batch avatar gather that used
                # to run HERE (blocking each platform's emit on every avatar
                # fetch, creating the between-card gaps) is gone. Each chip's
                # avatar is now fetched as a concurrent task started the moment
                # that chip is emitted (see below), so emits no longer wait on
                # network I/O for pictures.
                _hg_fmap = _hg_followers_map()
                for p in plat_out:
                    # v7.55: inherit followers/bio from a same-URL handle-guess
                    # chip (data already fetched) so dedup doesn't drop them.
                    _hgk = (p.get("url") or "").rstrip("/").lower()
                    _hgd = _hg_fmap.get(_hgk)
                    if _hgd:
                        if _hgd.get("followers") and not p.get("followers"):
                            p["followers"] = _hgd["followers"]
                        if _hgd.get("bio") and not p.get("bio"):
                            p["bio"] = _hgd["bio"]
                    # Phase 2: skip URLs the customer rejected on a
                    # previous attempt of this carousel-triage session.
                    if (p.get("url") or "").rstrip("/").lower() in _rejected_block:
                        continue
                    if not _should_emit(p):
                        continue                # 2026-05-30 v4: global dedup
                    if first_emit_done:
                        await asyncio.sleep(EMIT_STAGGER_S)
                    first_emit_done = True
                    if p.get("platform") == "Instagram":
                        ig_already_emitted = True
                    emitted_final.append(p)
                    yield _emit({"type": "profile", "data": dict(p),
                                 "ts": round(time.time() - start, 2)})
                    # v7.67 enrich-on-emit: kick off this chip's avatar (+
                    # follower fetch when missing) as a CONCURRENT task the
                    # instant it's emitted, so the work overlaps the rest of
                    # discovery. The patch is yielded later as soon as the task
                    # finishes (drained right after the loop). No blocking here.
                    _enrich_tasks.append(asyncio.create_task(
                        _enrich_one(p, need_followers=not p.get("followers"))))
                    # v7.67 queue: flush any enrichment patches that have already
                    # completed (e.g. a fast cached avatar from an earlier chip),
                    # interleaved with the ongoing emit loop.
                    for _pe in _drain_patch_events():
                        yield _emit(_pe)

            yield _emit({"type": "phase", "name": "verify_done",
                         "ts": round(time.time() - start, 2),
                         "per_platform_ms": {k: int(v * 1000) for k, v in verify_durations.items()}})
            for _pe in _drain_patch_events():
                yield _emit(_pe)

            # Handle-guess + inline both ran in background since t=0;
            # some may have already been early-emitted right after SERP.
            # Drain any remaining hits here. Dedup against both the SERP
            # emissions AND any early-emitted hg hits.
            try:
                # v7.67 queue: do NOT block flat on handle-guess. While it runs,
                # poll it in short slices and flush any enrichment patches that
                # finished in the meantime — so avatar/follower patches for the
                # already-emitted cards go out DURING the hg wait (the window
                # where they were previously stuck), not all at once after it.
                while not stream_hg_task.done():
                    _hg_done, _ = await asyncio.wait({stream_hg_task}, timeout=0.2)
                    for _pe in _drain_patch_events():
                        yield _emit(_pe)
                    if _hg_done:
                        break
                for _pe in _drain_patch_events():
                    yield _emit(_pe)
                guessed_inlined = await stream_hg_task
                seen_urls = {(p.get("url") or "").rstrip("/").lower() for p in emitted_final}
                seen_urls |= emitted_hg_urls
                # v7.55: index already-emitted chips by URL so a colliding
                # guessed chip can backfill followers/bio onto the survivor.
                _emitted_by_url = {}
                for _ep in emitted_final:
                    _ek = (_ep.get("url") or "").rstrip("/").lower()
                    if _ek:
                        _emitted_by_url[_ek] = _ep
                for g in (guessed_inlined or []):
                    key = (g.get("url") or "").rstrip("/").lower()
                    if key in seen_urls or g.get("_early_emitted"):
                        # v7.55: dedup dropped this guessed chip, but it may
                        # carry follower/bio data the surviving chip lacks
                        # (handle-guess finished after the SERP chip emitted).
                        # Patch the survivor in place via followers_update.
                        _surv = _emitted_by_url.get(key)
                        if _surv is not None:
                            _newf = g.get("followers") or ""
                            _newb = g.get("bio") or ""
                            _upd = {}
                            if _newf and not _surv.get("followers"):
                                _surv["followers"] = _newf
                                _upd["followers"] = _newf
                            if _newb and not _surv.get("bio"):
                                _surv["bio"] = _newb
                                _upd["bio"] = _newb
                            if _upd:
                                yield _emit({"type": "followers_update",
                                             "url": _surv.get("url"),
                                             "followers": _upd.get("followers", ""),
                                             "bio": _upd.get("bio", ""),
                                             "ts": round(time.time() - start, 2)})
                        continue
                    if key in _rejected_block:
                        continue                # Phase 2 block-list
                    if not _should_emit(g):
                        continue                # 2026-05-30 v4: global dedup
                    await asyncio.sleep(EMIT_STAGGER_S)
                    emitted_final.append(g)
                    yield _emit({"type": "profile", "data": dict(g),
                                 "ts": round(time.time() - start, 2)})
                    # v7.67 enrich-on-emit: same concurrent avatar (+ follower)
                    # enrichment for a drained handle-guess chip, started the
                    # instant it's emitted.
                    _enrich_tasks.append(asyncio.create_task(
                        _enrich_one(g, need_followers=not g.get("followers"))))
                    for _pe in _drain_patch_events():
                        yield _emit(_pe)
            except Exception as _hg_err:
                logger.warning(f"stream handle_guess failed: {_hg_err}")
            for _pe in _drain_patch_events():
                yield _emit(_pe)

            # v7.67 queue: most enrichment patches already flushed via the queue
            # at the await points above (during the emit loop + the hg wait).
            # This final sweep awaits any still-in-flight task with as_completed
            # and, after each one finishes (its result is now on the queue),
            # drains the queue immediately — so the LAST patches go out the
            # instant their fetch lands, and no task is left orphaned before the
            # stream closes. The queue is the single emit path, so there's no
            # double-emit. Wrapped so a hiccup can never abort the stream.
            if _enrich_tasks:
                try:
                    _drain_deadline = time.time() + 6.0 * TS   # audit-F9: bound the enrich tail
                    for _et in asyncio.as_completed(_enrich_tasks):
                        if time.time() > _drain_deadline:
                            break   # stragglers keep running; their patches ride the *_update channel
                        try:
                            await asyncio.wait_for(_et, timeout=max(0.2, _drain_deadline - time.time()))
                        except Exception:
                            pass
                        for _pe in _drain_patch_events():
                            yield _emit(_pe)
                except Exception as _en_err:
                    logger.warning(f"enrich-on-emit drain failed: {_en_err}")
                    for _et in _enrich_tasks:
                        if not _et.done():
                            try:
                                await _et
                            except Exception:
                                pass
                # Final flush — anything that landed on the queue after the last
                # as_completed iteration.
                for _pe in _drain_patch_events():
                    yield _emit(_pe)

            # IG enrichment may finish after the IG profile has already been emitted
            # with a ui-avatars stub. If BD lands a real cdninstagram URL, push an
            # avatar_update so the frontend swaps in place.
            ig_pre_url = first_ig.get("image_url") if first_ig else None
            try:
                # ITER33 2026-05-28: was 8.0 — give late BD-IG enrichment 3 more
                # seconds to land before we give up and finalize the stream.
                # v7.6: scaled by patience hint — round-2 patience='long' → 22s.
                # 2026-05-31: skip the await entirely when no IG SERP hit
                # (ig_task is None — see CHANGE 1 above).
                if ig_task is not None:
                    await asyncio.wait_for(ig_task, timeout=4.0 * TS)   # audit-F9: bounded tail (was 11s); upgrade rides avatar_update
            except asyncio.TimeoutError:
                pass
            if ig_already_emitted and first_ig and first_ig.get("source") == "brightdata":
                new_url = first_ig.get("image_url")
                # Only emit if the in-flight URL changed AND it's a real BD avatar
                # (the IG fast-fill above may have already used it before emit)
                emitted_ig_url = next(
                    (p.get("image_url") for p in emitted_final if p.get("url") == first_ig.get("url")),
                    None,
                )
                if new_url and new_url != emitted_ig_url:
                    yield _emit({"type": "avatar_update",
                                 "url": first_ig["url"], "image_url": new_url,
                                 "ts": round(time.time() - start, 2)})
                # v7.55: late BD-IG enrichment also carries followers + bio.
                # If the IG chip emitted before BD landed (no follower row),
                # patch it in place now.
                _ig_emitted = next(
                    (p for p in emitted_final if p.get("url") == first_ig.get("url")),
                    None,
                )
                if _ig_emitted is not None:
                    _igf = first_ig.get("followers") or ""
                    _igb = first_ig.get("bio") or ""
                    _igupd = {}
                    if _igf and not _ig_emitted.get("followers"):
                        _ig_emitted["followers"] = _igf
                        _igupd["followers"] = _igf
                    if _igb and not _ig_emitted.get("bio"):
                        _ig_emitted["bio"] = _igb
                        _igupd["bio"] = _igb
                    if _igupd:
                        yield _emit({"type": "followers_update",
                                     "url": first_ig.get("url"),
                                     "followers": _igupd.get("followers", ""),
                                     "bio": _igupd.get("bio", ""),
                                     "ts": round(time.time() - start, 2)})

            yield _emit({"type": "done", "elapsed": round(time.time() - start, 2),
                         "count": len(emitted_final),
                         "authoritative_name": namefix_helper.authoritative_display_name(
                             req.name, emitted_final)})

    return _StreamingResponse(gen(), media_type="application/x-ndjson")


class VerifyUrlRequest(BaseModel):
    url: str


# 2026-06-01 v7.43: comprehensive table-driven cross-platform URL
# normalizer. Extracts the profile identifier (handle OR numeric/vanity
# user-id) from ANY link form for each of the 5 networks and rebuilds
# the canonical profile URL. Handles:
#   - regional/mobile/locale subdomains (xx.linkedin.com, m./mbasic./
#     web./free./l.facebook.com, fb.com, fb.me, en.instagram.com,
#     mobile.twitter.com, vm./vt.tiktok.com, mwlite.linkedin.com)
#   - FB numeric id from profile_id/id/fbid query on ANY path (incl.
#     /friends/suggestions/, /messages/...) BEFORE any subpath trim
#   - FB /profile.php?id=, /people/<Name>/<digits>, /<vanity>/<subpath>
#   - LI legacy /pub/<name>/<a>/<b>/<c> reverse-join, /in/<lang>/<vanity>
#     locale trim, /in/<vanity>/<subpath> trim
#   - IG /<handle> (drops /p/ /reel/ /stories/ posts)
#   - X /<handle> (drops /status/...), /i/user/<digits> numeric id form
#   - TT /@<handle>; unresolvable short links (vm./vt., /t/<short>) → None
#   - tracking params everywhere (utm_*, mibextid, __tn__, fref, rdid,
#     ref, sk, igsh, igshid, hl, trk, lipi, miniProfile, originalSubdomain,
#     lang, s, t, ref_src ...), trailing slashes, fragments, @-prefixes,
#     missing scheme, surrounding whitespace.
# Returns a dict {platform, canonical_url, identifier} or None if the URL
# is genuinely unrecognizable / unresolvable.
def normalize_profile_url(raw):
    import re as _re
    if not raw or not isinstance(raw, str):
        return None
    u = raw.strip()
    if not u:
        return None
    # Strip a lone leading "@" (e.g. user pasted "@shtekler" — but that's a
    # bare handle with no platform, handled by the bare-handle guard below).
    if u.startswith("@"):
        u = u[1:]
    # Add scheme if missing — e.g. "facebook.com/shtekler"
    if not _re.match(r"^https?://", u, _re.IGNORECASE):
        # Bare token (no dot before first slash) → no platform context.
        if "." not in u.split("/")[0]:
            return None
        u = "https://" + u
    # Parse: scheme, host, path, query, fragment
    m = _re.match(r"^(https?)://([^/?#]+)([^?#]*)(\?[^#]*)?(#.*)?$", u, _re.IGNORECASE)
    if not m:
        return None
    host  = m.group(2).lower().strip()
    path  = m.group(3) or ""
    query = (m.group(4) or "")[1:]  # drop leading "?"; fragment dropped

    # ── Parse query into a dict (case-insensitive keys) ──
    qd = {}
    for pair in query.split("&"):
        if not pair:
            continue
        k, _, v = pair.partition("=")
        if k:
            qd[k.lower()] = v

    def _digits(s):
        m2 = _re.match(r"^(\d+)$", s or "")
        return m2.group(1) if m2 else None

    # ── Identify platform from host ──
    # Map every known host alias to a canonical platform key.
    h = host
    # Facebook family
    if (h in ("fb.com", "fb.me") or h.endswith(".fb.com") or
            h == "facebook.com" or h.endswith(".facebook.com")):
        platform = "Facebook"
    elif h == "instagram.com" or h.endswith(".instagram.com"):
        platform = "Instagram"
    elif h in ("twitter.com", "x.com") or h.endswith(".twitter.com") or h.endswith(".x.com"):
        platform = "X"
    elif h == "linkedin.com" or h.endswith(".linkedin.com"):
        platform = "LinkedIn"
    elif h == "tiktok.com" or h.endswith(".tiktok.com"):
        platform = "TikTok"
    else:
        return None

    # Strip a trailing slash for segment math; keep leading slash.
    segs = [s for s in path.split("/") if s != ""]

    # ═══════════════════════════ FACEBOOK ═══════════════════════════
    if platform == "Facebook":
        CANON = "https://www.facebook.com"
        # 1. Numeric id from query — runs FIRST, before any subpath logic.
        #    profile_id / id / fbid on ANY path (incl. /friends/suggestions/,
        #    /messages/t/, /profile.php).
        for key in ("profile_id", "id", "fbid"):
            d = _digits(qd.get(key, ""))
            if d:
                return {"platform": "Facebook",
                        "canonical_url": f"{CANON}/profile.php?id={d}",
                        "identifier": d}
        low = path.lower()
        # 2. /people/<Name>/<digits> → profile.php?id=<digits>
        pm = _re.match(r"^/people/[^/]+/(\d+)/?$", path)
        if pm:
            return {"platform": "Facebook",
                    "canonical_url": f"{CANON}/profile.php?id={pm.group(1)}",
                    "identifier": pm.group(1)}
        # 3. /profile.php with no usable id query (already handled above) →
        #    unrecognizable.
        if low.startswith("/profile.php"):
            return None
        # 4. /share/... short links — unresolvable without following a
        #    redirect. Don't fabricate.
        if low.startswith("/share") or low.startswith("/groups") or low.startswith("/messages") or low.startswith("/friends"):
            return None
        # 5. /<vanity>(/posts|/photos|/videos|/about|/friends|/reels|...) →
        #    strip trailing subpaths, keep first segment as the vanity.
        if not segs:
            return None
        vanity = segs[0]
        # Reject reserved/non-profile first segments.
        if vanity.lower() in ("profile.php", "people", "share", "groups",
                               "pages", "watch", "marketplace", "events",
                               "gaming", "story.php", "permalink.php",
                               "photo.php", "media", "pg", "home.php",
                               "login", "help", "settings", "bookmarks"):
            return None
        return {"platform": "Facebook",
                "canonical_url": f"{CANON}/{vanity}",
                "identifier": vanity}

    # ═══════════════════════════ INSTAGRAM ══════════════════════════
    if platform == "Instagram":
        CANON = "https://www.instagram.com"
        if not segs:
            return None
        first = segs[0].lower()
        # Post / reel / story / explore / tv URLs are not profiles.
        if first in ("p", "reel", "reels", "stories", "explore", "tv",
                     "accounts", "directory", "about", "developer"):
            return None
        handle = segs[0].lstrip("@")
        if not _re.match(r"^[A-Za-z0-9_.]+$", handle):
            return None
        return {"platform": "Instagram",
                "canonical_url": f"{CANON}/{handle}",
                "identifier": handle}

    # ═══════════════════════════════ X ══════════════════════════════
    if platform == "X":
        CANON = "https://x.com"
        if not segs:
            return None
        first = segs[0].lower()
        # /i/user/<digits> → numeric id form (no handle available).
        if first == "i":
            im = _re.match(r"^/i/user/(\d+)/?$", path)
            if im:
                return {"platform": "X",
                        "canonical_url": f"{CANON}/i/user/{im.group(1)}",
                        "identifier": im.group(1)}
            return None
        # Reserved non-profile first segments.
        if first in ("home", "explore", "search", "settings", "messages",
                     "notifications", "compose", "hashtag", "intent",
                     "share", "login", "signup", "tos", "privacy", "about"):
            return None
        handle = segs[0].lstrip("@")
        if not _re.match(r"^[A-Za-z0-9_]+$", handle):
            return None
        return {"platform": "X",
                "canonical_url": f"{CANON}/{handle}",
                "identifier": handle}

    # ═══════════════════════════ LINKEDIN ═══════════════════════════
    if platform == "LinkedIn":
        CANON = "https://www.linkedin.com"
        low = path.lower()
        # Legacy /pub/<name>/<a>/<b>/<c> → /in/<name>-<c><b><a>
        pub_m = _re.match(r"^/pub/([^/]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)/?$", path)
        if pub_m:
            vanity, a, b, c = pub_m.group(1), pub_m.group(2), pub_m.group(3), pub_m.group(4)
            return {"platform": "LinkedIn",
                    "canonical_url": f"{CANON}/in/{vanity}-{c}{b}{a}",
                    "identifier": f"{vanity}-{c}{b}{a}"}
        # /in/<vanity> with optional locale prefix and trailing subpaths.
        if low.startswith("/in/"):
            rest = path[4:]  # after "/in/"
            r_segs = [s for s in rest.split("/") if s != ""]
            if not r_segs:
                return None
            # Locale prefix: /in/<lang>/<vanity>
            LANGS = {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "sv",
                     "da", "no", "fi", "ru", "ja", "ko", "zh", "ar", "he",
                     "tr", "cs", "hu", "id", "th", "vi", "ms", "tl", "hi",
                     "uk", "ro", "el", "bg", "hr", "sk", "sl"}
            if len(r_segs) >= 2 and r_segs[0].lower() in LANGS:
                vanity = r_segs[1]
            else:
                vanity = r_segs[0]
            # Drop trailing subpaths (details/..., recent-activity, edit,
            # overlay, etc.) — already handled by taking r_segs[idx].
            if not _re.match(r"^[A-Za-z0-9_.\-%]+$", vanity):
                return None
            return {"platform": "LinkedIn",
                    "canonical_url": f"{CANON}/in/{vanity}",
                    "identifier": vanity}
        # /pub/dir, /company, /school, /feed etc. are not personal profiles.
        return None

    # ═══════════════════════════ TIKTOK ═════════════════════════════
    if platform == "TikTok":
        CANON = "https://www.tiktok.com"
        # Short links (vm./vt. hosts, or /t/<short>) are unresolvable
        # without following a redirect — return None rather than mangle.
        if host.startswith("vm.") or host.startswith("vt.") or path.lower().startswith("/t/"):
            return None
        # /@<handle>(/video/...|/live|...) → keep @handle.
        am = _re.match(r"^/@([A-Za-z0-9_.]+)", path)
        if am:
            handle = am.group(1)
            return {"platform": "TikTok",
                    "canonical_url": f"{CANON}/@{handle}",
                    "identifier": handle}
        return None

    return None


# Backward-compatible wrapper: existing callers (verify_url and friends)
# expect a canonical URL *string* ("" when unrecognizable). The downstream
# verify_url re-derives platform/username from this string via regex, so we
# keep returning the canonical_url here.
def _normalize_profile_url(raw: str) -> str:
    res = normalize_profile_url(raw)
    if not res:
        return ""
    return res["canonical_url"]


@app.post("/api/verify-url")
async def verify_url(req: VerifyUrlRequest):
    """Verify a profile URL exists. Customer-facing errors only —
    private accounts get a specific message so they don't paste again."""
    import re as _re
    import httpx as _httpx
    import base64 as _b64
    from profile_finder import brightdata_linkedin, SCRAPECREATORS_API_KEY

    # v7.44 (2026-06-01): manually-added chips must render EXACTLY like
    # discovered ones. Discovered chips show a real photo because the
    # /api/search-stream emit path inlines the avatar BYTES into an
    # `image_data` (base64 data: URI) field (see _inline_avatar) and
    # attaches the pHash/face-embedding similarity payload. A manual chip
    # that only carries a live `image_url` shows grey initials whenever that
    # avatar endpoint 404s. This helper replicates the stream/_enrich
    # approach: fetch the profile's own `/api/avatar/<platform>` endpoint
    # server-side, base64-inline it into `image_data`, and attach the same
    # phash_b64/face_emb_b64 payload — so a manual chip is byte-for-byte the
    # same shape as a discovered one. If the avatar can't be fetched we leave
    # image_data empty and the frontend falls back to initials (same as
    # discovery). Best-effort: never let an avatar failure block a verify.
    async def _inline_profile_assets(prof: dict) -> dict:
        prof.setdefault("image_data", "")
        prof.setdefault("phash_b64", "")
        prof.setdefault("face_emb_b64", "")
        iu = prof.get("image_url") or ""
        if not iu:
            return prof
        if iu.startswith("/api/spl/"):
            fetch_url = "http://127.0.0.1:8801" + iu[len("/api/spl"):]
        elif iu.startswith("/"):
            fetch_url = "http://127.0.0.1:8801" + iu
        else:
            fetch_url = iu
        try:
            async with _pooled() as _cl:
                rr = await _cl.get(fetch_url, timeout=6.0, follow_redirects=True)
            if rr.status_code != 200 or not rr.content or len(rr.content) < 2000:
                return prof
            ct = (rr.headers.get("content-type") or "image/jpeg").lower()
            if not ct.startswith("image"):
                return prof
            prof["image_data"] = (
                "data:" + ct + ";base64," + _b64.b64encode(rr.content).decode("ascii")
            )
            try:
                await asyncio.to_thread(attach_similarity_payload, prof, rr.content)
            except Exception:
                pass
        except Exception:
            return prof
        return prof

    url = _normalize_profile_url(req.url)
    if not url:
        return {"valid": False, "error": "Please paste a full profile link (https://...)."}

    # Detect platform + extract username from the canonical URL.
    # LinkedIn allows letters/digits/dash/underscore/dot in the vanity.
    # Facebook allows letters/digits/dash/underscore/dot AND /profile.php?id=<numeric>.
    # IG, TT, X allow letters/digits/underscore/dot (TT also leading @).
    platform = ""
    username = ""
    if _re.search(r"(?:^|//)(?:x|twitter)\.com/", url, _re.IGNORECASE):
        platform = "X"
        m = _re.search(r"(?:x|twitter)\.com/([a-zA-Z0-9_]+)", url, _re.IGNORECASE)
        if m: username = m.group(1)
    elif _re.search(r"facebook\.com/profile\.php", url, _re.IGNORECASE):
        platform = "Facebook"
        m = _re.search(r"[?&]id=(\d+)", url)
        if m: username = m.group(1)
    elif _re.search(r"facebook\.com/", url, _re.IGNORECASE):
        platform = "Facebook"
        m = _re.search(r"facebook\.com/([a-zA-Z0-9_.\-]+)", url, _re.IGNORECASE)
        if m: username = m.group(1)
    elif _re.search(r"instagram\.com/", url, _re.IGNORECASE):
        platform = "Instagram"
        m = _re.search(r"instagram\.com/([a-zA-Z0-9_.]+)", url, _re.IGNORECASE)
        if m: username = m.group(1)
    elif _re.search(r"tiktok\.com/@?", url, _re.IGNORECASE):
        platform = "TikTok"
        m = _re.search(r"tiktok\.com/@?([a-zA-Z0-9_.\-]+)", url, _re.IGNORECASE)
        if m: username = m.group(1).lstrip("@")
    elif _re.search(r"linkedin\.com/in/", url, _re.IGNORECASE):
        platform = "LinkedIn"
        m = _re.search(r"linkedin\.com/in/([a-zA-Z0-9_\-\.]+)", url, _re.IGNORECASE)
        if m: username = m.group(1)
    else:
        return {"valid": False, "error": "Use a link from X, Facebook, Instagram, TikTok, or LinkedIn."}

    if not username:
        return {"valid": False, "error": "We couldn't find a username in that link."}

    # SC-direct verification per platform.
    async with _pooled() as cl:
        api_url = None; params = None
        if platform == "Facebook":
            api_url = "https://api.scrapecreators.com/v1/facebook/profile"
            params = {"url": f"https://www.facebook.com/{username}"}
        elif platform == "Instagram":
            api_url = "https://api.scrapecreators.com/v1/instagram/profile"
            params = {"handle": username}
        elif platform == "TikTok":
            api_url = "https://api.scrapecreators.com/v1/tiktok/profile"
            params = {"handle": username}
        elif platform == "X":
            # No SC profile-by-handle for X; use our avatar endpoint as existence check.
            api_url = "http://127.0.0.1:8801/api/avatar/x"
            params = {"u": username}
        elif platform == "LinkedIn":
            # BD LinkedIn enrichment for the actual data.
            # 2026-05-31 v7.32: BD poll now waits up to 30s (was effectively
            # ~6s, which timed out before BD returned for valid profiles like
            # gur-megiddo-1929a379 — BD typically needs 8-10s).
            enriched = await brightdata_linkedin(username)
            if enriched:
                _li_prof = {
                    "platform": "LinkedIn", "url": url, "username": username,
                    "display_name": enriched.get("display_name", "") or username,
                    "bio":          enriched.get("bio", ""),
                    "image_url":    enriched.get("image_url", ""),
                    "followers":    enriched.get("followers", ""),
                    "position":     enriched.get("position", ""),
                    "company":      enriched.get("company", ""),
                    "city":         enriched.get("city", ""),
                    "is_private":   False,
                }
                return {"valid": True, "profile": await _inline_profile_assets(_li_prof)}
            # Safety net: BD didn't return. Try SC LinkedIn endpoint.
            # SC may return success:true with a "private or not publicly
            # available" message — surface that as a specific error rather
            # than a generic "not found".
            try:
                sc_r = await cl.get(
                    "https://api.scrapecreators.com/v1/linkedin/profile",
                    params={"url": f"https://www.linkedin.com/in/{username}/"},
                    headers={"x-api-key": SCRAPECREATORS_API_KEY},
                    timeout=10.0,
                )
                if sc_r.status_code == 200:
                    sc_d = sc_r.json() or {}
                    sc_msg = (sc_d.get("message") or "").lower()
                    # If SC has profile data, use it.
                    if sc_d.get("name") or sc_d.get("display_name") or sc_d.get("full_name"):
                        _li_sc_prof = {
                            "platform": "LinkedIn", "url": url, "username": username,
                            "display_name": sc_d.get("name") or sc_d.get("display_name") or sc_d.get("full_name") or username,
                            "bio":       sc_d.get("about") or sc_d.get("bio") or "",
                            "image_url": sc_d.get("image") or sc_d.get("profile_picture") or "",
                            "followers": str(sc_d.get("followers") or ""),
                            "position":  sc_d.get("position") or sc_d.get("headline") or "",
                            "company":   sc_d.get("company") or "",
                            "city":      sc_d.get("city") or sc_d.get("location") or "",
                            "is_private": False,
                        }
                        return {"valid": True, "profile": await _inline_profile_assets(_li_sc_prof)}
                    # Privacy / unavailable message.
                    if "private" in sc_msg or "not publicly available" in sc_msg:
                        return {"valid": False,
                                "error": "This LinkedIn profile isn't publicly visible — we can only scan public profiles."}
            except Exception:
                pass
            # Both BD and SC failed — give a clearer error than before.
            return {"valid": False,
                    "error": "We couldn't reach that LinkedIn profile right now. The link looks right; please try again in a moment."}

        try:
            r = await cl.get(api_url, params=params,
                             headers={"x-api-key": SCRAPECREATORS_API_KEY},
                             timeout=12.0)
        except Exception:
            return {"valid": False, "error": "Couldn't reach the profile right now. Please try again."}

    if platform == "X":
        # Avatar endpoint: 200 + non-trivial body = profile exists.
        if r.status_code != 200 or not r.content or len(r.content) < 2000:
            return {"valid": False, "error": "We couldn't find that X profile. Double-check the link."}
        # v7.61 all-followers: manual-add X chips now show a count too,
        # pulled from the SC twitter profile endpoint (bounded + cached).
        _x_followers = await _fetch_followers("X", username)
        _x_prof = {
            "platform": "X", "url": url, "username": username,
            "display_name": username, "bio": "", "image_url": f"/api/spl/api/avatar/x?u={username}",
            "followers": _x_followers, "position": "", "company": "", "city": "", "is_private": False,
        }
        return {"valid": True, "profile": await _inline_profile_assets(_x_prof)}

    if r.status_code != 200:
        return {"valid": False, "error": "We couldn't find that profile. Double-check the link."}
    data = r.json() or {}

    # ── Platform-specific parse + privacy detection ──
    display = ""; bio = ""; image_url = ""; followers = ""; is_priv = False

    if platform == "Facebook":
        # SC FB shape: { success, account_status, message, name, pageIntro, ... }
        status = (data.get("account_status") or "").lower()
        # v7.40: SC reports account_status="private" for PUBLIC personal FB profiles
        # (logged-out scraper can't see a personal timeline). Back off entirely —
        # never block a manually-added public FB URL. (mirrors LI SC privacy probe disable, v7.34)
        # if status == "private" or (data.get("message") or "").lower().startswith("profile is private"):
        #     return {"valid": False,
        #             "error": "This Facebook profile is private — private profiles can't be scanned."}
        # v7.41: SC's FB endpoint also returns 200 with an EMPTY name for real,
        # public personal profiles (e.g. /shtekler) — same unreliability as the
        # privacy signal. For a MANUALLY-PASTED url the user is asserting this
        # profile exists, so do NOT reject on a missing name — accept it and let
        # BrightData resolve it at scan time. Fall back to the URL handle as the
        # display name when SC gives us nothing.
        display = data.get("name", "") or username
        bio = _clean_social_bio((data.get("pageIntro") or data.get("intro") or "").strip())
        is_priv = False  # v7.40: don't trust SC privacy signal for Facebook
        image_url = f"/api/spl/api/avatar/facebook?u={username}"
        # v7.61 all-followers: FB PAGES carry followerCount/likeCount in the
        # same SC response we already fetched — surface it (personal profiles
        # have none → stays "").
        _fb_fc = data.get("followerCount") or data.get("likeCount")
        if _fb_fc:
            followers = str(_fb_fc)

    elif platform == "Instagram":
        user = ((data.get("data") or {}).get("user") or {})
        if not user:
            return {"valid": False, "error": "We couldn't find that Instagram profile. Double-check the link."}
        display = user.get("full_name", "") or user.get("username", "") or username
        bio = (user.get("biography") or "").strip()
        is_priv = bool(user.get("is_private"))
        fb_d = (user.get("edge_followed_by") or {})
        fc = fb_d.get("count") if isinstance(fb_d, dict) else None
        if fc: followers = str(fc)
        image_url = f"/api/spl/api/avatar/instagram?u={username}"
        if is_priv:
            return {"valid": False,
                    "error": "This Instagram profile is private — private profiles can't be scanned."}

    elif platform == "TikTok":
        user = (data.get("user") or {})
        if not user:
            return {"valid": False, "error": "We couldn't find that TikTok profile. Double-check the link."}
        display = user.get("nickname", "") or user.get("uniqueId", "") or username
        bio = (user.get("signature") or "").strip()
        is_priv = bool(user.get("privateAccount") or user.get("secret"))
        stats = (data.get("statsV2") or data.get("stats") or {})
        fc = stats.get("followerCount")
        if fc: followers = str(fc)
        image_url = f"/api/spl/api/avatar/tiktok?u={username}"
        if is_priv:
            return {"valid": False,
                    "error": "This TikTok profile is private — private profiles can't be scanned."}

    _prof = {
        "platform": platform, "url": url, "username": username,
        "display_name": display or username,
        "bio": bio, "image_url": image_url, "followers": followers,
        "position": "", "company": "", "city": "", "is_private": is_priv,
    }
    return {"valid": True, "profile": await _inline_profile_assets(_prof)}


# ─── Stage-2 "This is the person, now search again" 2026-05-29 ────────
class SearchAgainRequest(BaseModel):
    confirmed: list[dict] = []
    unknown:   list[dict] = []      # triage: "?" — carry through, no dedup-block
    rejected:  list[dict] = []      # triage: "✗" — drop + add to dedup-block
    # Back-compat with the pre-triage payload (`remaining` = stage-1 cards
    # the customer didn't select). If a client still sends `remaining` it
    # will be treated as `rejected` (the conservative default — old
    # behaviour was to drop them too).
    remaining: list[dict] = []


# ── Dynamic recalibration questions (2026-05-30 v7.1) ────────────────
# Carousel UX: when triage exhausts candidates with <2 confirmed, we
# show recalibration questions. Each question is its own carousel card
# (slide-in). To avoid asking about info the user ALREADY provided
# (e.g., they said "Madrid" → don't ask city), Haiku inspects the search
# name + description + any confirmed/rejected metadata and returns up
# to 3 fresh, context-aware questions.
class RecalQuestionsRequest(BaseModel):
    name: str
    description: str
    confirmed: list[dict] = []
    rejected: list[dict] = []
    attempt: int = 1
    asked_keys: list[str] = []   # keys asked in PRIOR recal rounds (don't repeat)


@app.post("/api/recal-questions")
async def recal_questions(req: RecalQuestionsRequest):
    """Generate up to 3 dynamic disambiguating questions, tailored to
    the gaps NOT already covered by the search query + confirmed
    metadata. Uses Haiku (~$0.0005 per call, fits the $1 budget easily).
    """
    # Build a compact context for Haiku.
    confirmed_lines = []
    for p in (req.confirmed or [])[:3]:
        bits = [f"{p.get('platform','')}:{p.get('username','')}"]
        for k in ("display_name", "bio", "city", "company", "position", "followers"):
            v = (p.get(k) or "").strip()
            if v: bits.append(f"{k}={v}")
        confirmed_lines.append(" | ".join(bits))
    rejected_lines = [f"{p.get('platform','')}:{p.get('username','')}" for p in (req.rejected or [])[:5]]

    prompt = f"""Generate up to 3 short questions to help narrow a person search.

Original search:
  name: {req.name}
  description: {req.description or '(none provided)'}

Confirmed candidates (subject's known accounts):
{chr(10).join('  - ' + l for l in confirmed_lines) if confirmed_lines else '  (none)'}

Already-rejected candidate handles:
  {', '.join(rejected_lines) if rejected_lines else '(none)'}

Previously asked question keys (skip these):
  {', '.join(req.asked_keys) if req.asked_keys else '(none)'}

Instructions:
- Generate up to 3 questions ONLY about info NOT already known from the search description or confirmed metadata.
- Phrase each question naturally, referencing context the user gave. Examples:
    description "Israeli, Connecteam" → "Which city in Israel?" (NOT "what country?")
    description "Madrid" + employer unknown → "Where do they work?" (NOT "what city?")
    description "designer" + employer unknown → "Where do they work as a designer?"
- Pick from these keys: location, employer, role, age, other.
- If the description ALREADY covers a key clearly, skip it.
- If asked_keys lists a key, skip it.
- Keep each label short (≤ 40 chars).
- Placeholder = short hint of what to type.

Return JSON ONLY, this exact shape (no commentary, no markdown fences):
{{"questions":[{{"key":"location","label":"Which city in Israel?","placeholder":"e.g. Tel Aviv"}}, ...]}}"""

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip any code-fence wrapper Haiku occasionally adds.
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip().rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        qs = data.get("questions", [])[:3]
        # Filter out any asked_keys that slipped through.
        asked = set(req.asked_keys or [])
        qs = [q for q in qs if q.get("key") not in asked]
        return {"questions": qs}
    except Exception as e:
        logger.warning(f"recal-questions failed: {e}")
        # Fallback: static set, minus already-asked keys + description hits.
        defaults = [
            {"key":"location", "label":"City or country?", "placeholder":"e.g. Toronto, Canada"},
            {"key":"employer", "label":"Where do they work?", "placeholder":"e.g. Wayfair"},
            {"key":"role",     "label":"Job title or role?", "placeholder":"e.g. UX designer"},
            {"key":"age",      "label":"Approximate age?", "placeholder":"e.g. 30s, 45-55"},
            {"key":"other",    "label":"Anything distinctive?", "placeholder":"e.g. marathon runner"},
        ]
        asked = set(req.asked_keys or [])
        return {"questions": [q for q in defaults if q["key"] not in asked][:3]}


@app.post("/api/search-again")
async def search_again(req: SearchAgainRequest):
    """Re-run discovery using confirmed accounts' metadata as the seed.

    Pipeline (see stage_two.py): metadata-enriched Serper queries +
    handle extrapolation across missing platforms. Confirmed cards
    return with `_carry_confirmed`. Unknown cards carry through
    unchanged. Rejected cards are dropped from output AND seeded into
    seen_urls so the new search can't re-suggest them.

    2026-05-30 v6: attaches pHash + face_emb to every returned profile
    so the carousel rerank can use them locally — same payload as
    /api/search-stream emits. Lets the auto-trigger-on-first-confirm
    flow surface face-matched chips at queue positions 0+1.
    """
    import stage_two
    if not req.confirmed:
        return {"profiles": [], "diagnostics": {"error": "no_confirmed"}}
    rejected = list(req.rejected) + list(req.remaining)
    result = await stage_two.run(req.confirmed, req.unknown, rejected)

    # Attach pHash + face_emb to every NEW (not _carry_confirmed) chip.
    # Fetch avatar bytes once, then run similarity in a thread pool.
    import base64 as _b64, httpx as _httpx
    new_profs = [p for p in result.get("profiles", []) if not p.get("_carry_confirmed")]
    if new_profs:
        async def _enrich(p):
            iu = p.get("image_url") or ""
            if not iu:
                p["phash_b64"] = ""; p["face_emb_b64"] = ""; return
            url = iu
            if iu.startswith("/api/spl/"):  url = "http://127.0.0.1:8801" + iu[len("/api/spl"):]
            elif iu.startswith("/"):        url = "http://127.0.0.1:8801" + iu
            try:
                async with _pooled() as cl:
                    r = await cl.get(url, timeout=6.0, follow_redirects=True)
                if r.status_code != 200 or not r.content or len(r.content) < 2000:
                    p["phash_b64"] = ""; p["face_emb_b64"] = ""; return
                ct = (r.headers.get("content-type") or "image/jpeg").lower()
                if ct.startswith("image"):
                    p["image_data"] = "data:" + ct + ";base64," + _b64.b64encode(r.content).decode("ascii")
                await asyncio.to_thread(attach_similarity_payload, p, r.content)
            except Exception:
                p["phash_b64"] = ""; p["face_emb_b64"] = ""
        await asyncio.gather(*[_enrich(p) for p in new_profs])
    return result


# Note: InsightFace warm-up removed 2026-05-29 along with the face-match
# pipeline. stage_two.py is now metadata-driven (Serper re-query +
# handle extrapolation); no model load needed at startup.


# 2026-05-31 v7.6: /api/face-rerank + FaceRerankCandidate + FaceRerankRequest
# + _verdict_from_sim REMOVED. The carousel triage already does face
# similarity locally during the vote loop using the inline pHash +
# face_emb payload shipped on every chip. This endpoint duplicated that
# work and drove a noisy post-click halo UI (green/orange/red badges)
# on the final chip view. face_match.embed_url is still used by
# similarity.py for the ingest-time embedding computation.
