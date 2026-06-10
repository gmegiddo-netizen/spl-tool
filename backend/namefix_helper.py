"""
Shared name-spelling/capitalization helpers for the 'Candice'->'Candace' fix.
Free string logic only — NO API calls, no added run-time.

Two enforcement rules:
  (b) Title-case: first letter of each name word upper-cased, hyphen/apostrophe aware.
  (a) Authoritative profile name: when scraped social profiles exist, prefer the
      real display name from those profiles over the operator's typed query —
      but only when we're confident they refer to the same person (shared token
      or close edit-distance), so legitimately-different queries aren't clobbered.
"""
import re


def smart_title_name(s: str) -> str:
    """Rule (b). Title-case a person name token-by-token.
    - Upper-cases the first letter of each whitespace-separated token.
    - Handles hyphen and apostrophe sub-parts: o'brien -> O'Brien,
      smith-jones -> Smith-Jones.
    - Leaves names that already contain internal capitals alone (deliberate
      styling like 'GurMegiddo' or all-caps acronyms 'NASA').
    """
    if not s:
        return s
    s = s.strip()
    if not s or s in ("—", "-", "?"):
        return s
    # Already has a deliberate internal capital anywhere after the first char
    # of any token -> leave as-is (don't mangle styled/acronym names).
    for tok in s.split():
        if any(c.isupper() for c in tok[1:]):
            return s

    def _cap_part(part: str) -> str:
        if not part:
            return part
        return part[:1].upper() + part[1:]

    out_tokens = []
    for tok in s.split():
        # split on hyphen and apostrophe but keep the separators
        pieces = re.split(r"([-'’])", tok)
        rebuilt = "".join(_cap_part(pc) if pc not in ("-", "'", "’") else pc
                          for pc in pieces)
        out_tokens.append(rebuilt)
    return " ".join(out_tokens)


def _tokens(s: str):
    return [t for t in re.split(r"[^0-9a-zÀ-ɏ]+", (s or "").lower()) if t]


def _lev(a: str, b: str) -> int:
    """Tiny Levenshtein (free, local)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _close(a: str, b: str) -> bool:
    """Two tokens are 'close' typos of each other (free, local)."""
    if a == b:
        return True
    if len(a) < 3 or len(b) < 3:
        return False
    d = _lev(a, b)
    # 1 edit for short words, up to 2 for longer ones
    return d <= (1 if max(len(a), len(b)) <= 6 else 2)


def authoritative_display_name(typed_name: str, profiles, *, prefer_verified: bool = True) -> str:
    """Rules (a)+(b) combined.

    typed_name : the operator's raw typed query (may contain a typo / be lower-case).
    profiles   : iterable of dicts with at least 'display_name' and optionally
                 'verified'/'is_verified'. (Scraped social profiles.)

    Returns the name to DISPLAY downstream:
      - If a confident authoritative profile name is found -> that name (title-cased
        only if it has no internal caps).
      - Otherwise -> the title-cased typed name (Rule b fallback).
    """
    typed_clean = (typed_name or "").strip()
    typed_titled = smart_title_name(typed_clean)
    if not profiles:
        return typed_titled

    # Gather candidate display names, scored.
    cands = []  # (score, display_name)
    for p in profiles:
        dn = (p.get("display_name") or "").strip()
        if not dn:
            continue
        verified = bool(p.get("verified") or p.get("is_verified"))
        ntok = len(_tokens(dn))
        score = 0
        if verified and prefer_verified:
            score += 100
        # Prefer multi-word "First Last" person names over single-token handles
        # (e.g. 'CANDACE' single token vs 'Candace Owens').
        score += min(ntok, 3) * 10
        # Prefer names that are NOT all-caps (TikTok 'CANDACE') unless nothing else.
        if dn.isupper():
            score -= 15
        cands.append((score, dn))

    if not cands:
        return typed_titled

    # Tally identical display names (case-insensitive) to find consensus.
    from collections import defaultdict
    by_norm = defaultdict(lambda: [0, 0, None])  # norm -> [count, best_score, raw]
    for score, dn in cands:
        key = dn.lower()
        slot = by_norm[key]
        slot[0] += 1
        if slot[2] is None or score > slot[1]:
            slot[1] = score
            slot[2] = dn

    # Rank by (consensus count, score).
    ranked = sorted(by_norm.values(), key=lambda v: (v[0], v[1]), reverse=True)
    best_count, best_score, best_name = ranked[0]

    # Confidence gate: only override the typed name if the authoritative name
    # plausibly refers to the same person. Confident if ANY token of the typed
    # name exactly matches OR is a close typo of any token of the best name.
    typed_toks = _tokens(typed_clean)
    best_toks = _tokens(best_name)
    confident = False
    if typed_toks and best_toks:
        for tt in typed_toks:
            for bt in best_toks:
                if tt == bt or _close(tt, bt):
                    confident = True
                    break
            if confident:
                break
    # If the operator typed nothing meaningful, trust the profiles outright.
    if not typed_toks:
        confident = True

    if confident:
        return smart_title_name(best_name)
    return typed_titled


if __name__ == "__main__":
    cases = [
        ("candice owens", [
            {"display_name": "Candace Owens", "is_verified": 1},
            {"display_name": "CANDACE", "is_verified": 1},
            {"display_name": "Candace Owens", "is_verified": 1},
        ], "Candace Owens"),
        ("brett cooper", [
            {"display_name": "Brett Cooper", "is_verified": 1},
            {"display_name": "brettcooper_", "is_verified": 1},
        ], "Brett Cooper"),
        ("o'brien", [], "O'Brien"),
        ("smith-jones", [], "Smith-Jones"),
        ("GurMegiddo", [], "GurMegiddo"),
        ("NASA", [], "NASA"),
        ("candace owens", [], "Candace Owens"),
        # Legit different query must NOT be clobbered by an unrelated profile.
        ("joe rogan", [{"display_name": "Candace Owens", "is_verified": 1}], "Joe Rogan"),
        # single token typed, profile is the authority
        ("candice", [{"display_name": "Candace Owens", "is_verified": 1}], "Candace Owens"),
        ("", [{"display_name": "Candace Owens", "is_verified": 1}], "Candace Owens"),
    ]
    ok = True
    for typed, profs, expect in cases:
        got = authoritative_display_name(typed, profs)
        status = "OK " if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"{status} typed={typed!r:20} -> {got!r:18} (expect {expect!r})")
    print("ALL PASS" if ok else "SOME FAILED")
