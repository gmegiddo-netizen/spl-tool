import json
import re
import anthropic


def _fast_score(profile: dict, name: str, description: str) -> tuple[int, str]:
    """ITER13 2026-05-28: rule-based scoring with FALSE-POSITIVE GUARDS.
    Key insight from QA: news articles ABOUT the subject have the subject's name in the
    SERP snippet/title, so a "name in bio" check returns 100 for wrong entities like
    @thedailyshow or @Channel4News. Fix: the USERNAME must contain at least one name
    part for the profile to score above 50.
    """
    name_parts = [p.lower() for p in re.findall(r"[A-Za-z]+", name) if len(p) >= 2]
    desc_words = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", description))
    desc_words -= {"the", "and", "for", "with", "from"}

    username = (profile.get("username") or "").lower()
    bio_blob = " ".join([
        str(profile.get("bio", "")),
        str(profile.get("search_snippet", "")),
        str(profile.get("search_title", "")),
        str(profile.get("display_name", "")),
        str(profile.get("position", "")),
        str(profile.get("company", "")),
    ]).lower()

    # GUARD 1: username must contain at least one name part >=3 chars.
    # Without that, this is almost certainly someone else's account that just
    # mentions the subject in a post/article.
    name_in_username = any(p in username for p in name_parts if len(p) >= 3)

    # GUARD 2: detect news-style display name like "Spoke to X about ..." or
    # "X has ..." patterns — these are article headlines, not profile names.
    dn = (profile.get("display_name") or "").lower()
    article_starts = ("spoke to ", "watch ", "meet ", "read ")
    looks_like_article = any(dn.startswith(s) for s in article_starts) or dn.endswith("...")

    score = 30
    reasons = []

    # Name parts present in bio
    name_hits = sum(1 for p in name_parts if p in bio_blob)
    if name_parts:
        name_frac = name_hits / len(name_parts)
        if name_frac == 1.0:
            score += 40; reasons.append("name match in bio")
        elif name_frac >= 0.5:
            score += 25; reasons.append("partial name in bio")
        else:
            score += 5; reasons.append("weak name signal")

    # Description keyword overlap
    desc_hits = sum(1 for w in desc_words if w in bio_blob)
    if desc_words and desc_hits:
        frac = desc_hits / max(len(desc_words), 1)
        bonus = min(30, int(frac * 50))
        score += bonus
        reasons.append(f"{desc_hits}/{len(desc_words)} desc keywords")

    # BD enrichment bonus
    if profile.get("source") == "brightdata" and profile.get("bio"):
        score += 10
        reasons.append("BD enriched")

    # User-provided URL is gold
    if profile.get("source") == "user_provided":
        score = max(score, 95)
        reasons.append("user-provided URL")

    # APPLY GUARDS: cap score if the username doesn't contain a name part
    # or display name looks like a news article headline
    if not name_in_username:
        score = min(score, 35)
        reasons.append("username lacks name match → likely wrong entity")
    if looks_like_article:
        score = min(score, 25)
        reasons.append("display name looks like article headline")

    score = max(0, min(100, score))
    return score, " · ".join(reasons) if reasons else "no signal"


def _fp_guard(profile: dict, name: str) -> tuple[int | None, str | None]:
    """ITER14 false-positive guard. Returns (cap_score, extra_reason) if the
    profile fails a quality check, else (None, None).
    Catches: (a) news articles ABOUT the subject (username unrelated to name),
             (b) article-headline display names.
    """
    name_parts = [p.lower() for p in re.findall(r"[A-Za-z]+", name) if len(p) >= 3]
    username = (profile.get("username") or "").lower()
    dn = (profile.get("display_name") or "").lower()
    article_starts = ("spoke to ", "watch ", "meet ", "read more about ", "read about ")

    # GUARD 1: username OR display_name must contain a name part. ITER29: display_name
    # added so legitimate-but-named-differently accounts (e.g. @otzma_yehudit with
    # display_name "ITAMAR BEN GVIR") pass when BD enrichment confirms the identity.
    if name_parts and not any(p in username for p in name_parts) and not any(p in dn for p in name_parts):
        # Exception: explicitly accept user-provided URLs
        if profile.get("source") == "user_provided":
            return (None, None)
        return (35, "FP guard: name absent from both username and display_name")

    # GUARD 2: display name looks like a news headline
    if any(dn.startswith(s) for s in article_starts) or (dn.endswith("...") and len(dn) > 50):
        return (25, "FP guard: display name is article headline")

    return (None, None)


def verify_profiles(profiles: list[dict], name: str, description: str) -> list[dict]:
    """ITER14 2026-05-28: Haiku verify ALWAYS runs (no rule-based bypass).
    Post-process scores with false-positive guards to catch news articles ABOUT
    the subject that the LLM might score too high.
    """
    if not profiles:
        return []
    # Run Haiku verify (slim prompt from iter 9)
    scored = _verify_with_haiku(profiles, name, description)
    # Apply false-positive guards as a post-process score cap
    for p in scored:
        cap, extra = _fp_guard(p, name)
        if cap is not None and p.get("score", 0) > cap:
            p["score"] = cap
            # v7.48: keep the down-scoring effect but do NOT leak the operator
            # FP-guard diagnostic into customer-facing `reasoning` (it was
            # surfacing as a ✓ positive). Route the diagnostic to logs +
            # an internal-only field instead.
            p["_fp_guard"] = extra
            print(f"[fp_guard] {p.get('platform','?')}/{p.get('username','?')} capped to {cap}: {extra}")
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


def verify_profiles_single_platform(profiles: list[dict], name: str, description: str) -> list[dict]:
    """ITER31 2026-05-28: per-platform variant of verify_profiles.
    Same scoring + FP-guard pipeline as verify_profiles, but expects the input
    list to all be from a single platform. Designed to be called once per
    platform in parallel from the streaming endpoint so that each platform's
    profiles can be emitted as soon as Haiku finishes that small batch — first
    profile arrives at ~SERP_done + ~1s instead of ~SERP_done + ~5s on the
    full candidate set.

    Trade-off vs single-call verify: no cross-platform disambiguation context
    (e.g. the LinkedIn anchor's verified bio can't help score an ambiguous X
    candidate). The FP guard + per-platform threshold limit the blast radius.
    """
    if not profiles:
        return []
    scored = _verify_with_haiku(profiles, name, description)
    for p in scored:
        cap, extra = _fp_guard(p, name)
        if cap is not None and p.get("score", 0) > cap:
            p["score"] = cap
            # v7.48: keep the down-scoring effect but do NOT leak the operator
            # FP-guard diagnostic into customer-facing `reasoning`. Route it to
            # logs + an internal-only field instead.
            p["_fp_guard"] = extra
            print(f"[fp_guard] {p.get('platform','?')}/{p.get('username','?')} capped to {cap}: {extra}")
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


def _verify_with_haiku(profiles: list[dict], name: str, description: str) -> list[dict]:
    """Original Haiku verify, kept as fallback for ambiguous cases."""
    if not profiles:
        return []

    # Separate anchor profiles (LinkedIn with rich data) from candidates
    anchors = [p for p in profiles if p.get("source") == "brightdata" and p.get("bio")]
    candidates = [p for p in profiles if p not in anchors]

    # Build anchor context
    anchor_text = ""
    if anchors:
        anchor_lines = []
        for a in anchors:
            parts = [f"  Platform: {a['platform']}", f"  Name: {a.get('display_name', 'N/A')}"]
            if a.get("bio"):
                parts.append(f"  Bio: {a['bio']}")
            if a.get("position"):
                parts.append(f"  Position: {a['position']}")
            if a.get("company"):
                parts.append(f"  Company: {a['company']}")
            if a.get("city"):
                parts.append(f"  Location: {a['city']}")
            if a.get("image_url"):
                parts.append(f"  Avatar URL: {a['image_url']}")
            anchor_lines.append("\n".join(parts))
        anchor_text = "\n\nVerified anchor profiles (high confidence):\n" + "\n\n".join(anchor_lines)

    # Build candidate list
    candidate_entries = []
    all_to_score = anchors + candidates
    for i, p in enumerate(all_to_score):
        parts = [f"Profile {i+1}:", f"  Platform: {p['platform']}", f"  URL: {p['url']}", f"  Username: {p.get('username', 'N/A')}"]
        if p.get("display_name"):
            parts.append(f"  Display Name: {p['display_name']}")
        if p.get("bio"):
            parts.append(f"  Bio: {p['bio']}")
        if p.get("position"):
            parts.append(f"  Position: {p['position']}")
        if p.get("image_url"):
            parts.append(f"  Has profile image: yes")
        candidate_entries.append("\n".join(parts))

    profiles_text = "\n\n".join(candidate_entries)

    # ITER9 2026-05-28: prompt slimmed (was ~40 lines / ~1500 input tokens, now ~10 lines / ~400)
    prompt = f"""Score each profile 0-100 for whether it belongs to: {name} ({description}).

Rules:
- Profession contradiction (description says "lawyer", bio says "CFO") → score 10-25
- Name matches + bio aligns with description → 90+
- Name matches + bio consistent (no contradiction) → 70-89
- Name matches + thin/empty bio → 50
- Wrong entity / not a name match → 0-9
- Trust bio over name; same name ≠ same person

Profiles:
{profiles_text}

Return ONLY a JSON array, brief reasoning (<15 words each):
[{{"profile_index":1,"score":N,"reasoning":"..."}}]"""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,  # ITER9 2026-05-28: shorter reasoning + smaller candidate set → 600 fits
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        scores = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        for p in all_to_score:
            p["score"] = 50 if p.get("source") == "brightdata" else 30
            p["reasoning"] = ""  # v7.7: don't leak engine status to customer
        return all_to_score

    for score_entry in scores:
        idx = score_entry.get("profile_index", 0) - 1
        if 0 <= idx < len(all_to_score):
            all_to_score[idx]["score"] = score_entry.get("score", 0)
            all_to_score[idx]["reasoning"] = score_entry.get("reasoning", "")

    all_to_score.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_to_score
