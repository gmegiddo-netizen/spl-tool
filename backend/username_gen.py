import re

# Common first name expansions
NAME_VARIANTS = {
    "bill": ["william", "bill"],
    "william": ["william", "bill"],
    "bob": ["robert", "bob"],
    "robert": ["robert", "bob"],
    "mike": ["michael", "mike"],
    "michael": ["michael", "mike"],
    "jim": ["james", "jim"],
    "james": ["james", "jim"],
    "joe": ["joseph", "joe"],
    "joseph": ["joseph", "joe"],
    "dick": ["richard", "dick"],
    "richard": ["richard", "dick"],
    "tom": ["thomas", "tom"],
    "thomas": ["thomas", "tom"],
    "dan": ["daniel", "dan"],
    "daniel": ["daniel", "dan"],
    "dave": ["david", "dave"],
    "david": ["david", "dave"],
    "steve": ["steven", "steve", "stephen"],
    "steven": ["steven", "steve"],
    "ed": ["edward", "ed"],
    "edward": ["edward", "ed"],
    "tony": ["anthony", "tony"],
    "anthony": ["anthony", "tony"],
    "chris": ["christopher", "chris"],
    "christopher": ["christopher", "chris"],
    "matt": ["matthew", "matt"],
    "matthew": ["matthew", "matt"],
    "nick": ["nicholas", "nick"],
    "nicholas": ["nicholas", "nick"],
    "alex": ["alexander", "alex"],
    "alexander": ["alexander", "alex"],
    "sam": ["samuel", "sam"],
    "samuel": ["samuel", "sam"],
    "ben": ["benjamin", "ben"],
    "benjamin": ["benjamin", "ben"],
    "jeff": ["jeffrey", "jeff"],
    "jeffrey": ["jeffrey", "jeff"],
    "liz": ["elizabeth", "liz"],
    "elizabeth": ["elizabeth", "liz"],
    "jen": ["jennifer", "jen"],
    "jennifer": ["jennifer", "jen"],
    "kate": ["katherine", "kate", "catherine"],
    "katherine": ["katherine", "kate"],
    "sue": ["susan", "sue"],
    "susan": ["susan", "sue"],
}


def generate_usernames(full_name: str) -> list[str]:
    """Generate candidate usernames from a full name."""
    parts = re.sub(r"[^a-zA-Z\s]", "", full_name).lower().split()
    if not parts:
        return []

    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""
    middle = parts[1] if len(parts) > 2 else ""

    # Get name variants
    first_variants = NAME_VARIANTS.get(first, [first])
    if first not in first_variants:
        first_variants.append(first)

    candidates = set()

    for f in first_variants:
        if last:
            candidates.update([
                f"{f}{last}",
                f"{f}.{last}",
                f"{f}_{last}",
                f"{f}-{last}",
                f"{f[0]}{last}",
                f"{f[0]}.{last}",
                f"{f}{last[0]}",
                f"{last}{f}",
                f"{last}.{f}",
                f"{last}_{f}",
                f"{last}{f[0]}",
                f"{f}{last}official",
            ])
            if middle:
                candidates.update([
                    f"{f}{middle[0]}{last}",
                    f"{f}.{middle[0]}.{last}",
                ])

    if not last:
        candidates.update(first_variants)

    return sorted(candidates)
