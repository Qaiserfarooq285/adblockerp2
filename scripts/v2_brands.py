#!/usr/bin/env python3
"""Brand vocabulary for the football betting-ad blocker.

The rails in this footage carry betting and non-betting sponsors at identical
size, angle and brightness, so nothing geometric separates them -- only the
brand.  This module is the single place that decides which is which.

OCR on 1080p LED boards is lossy at the frame edges ("Betano" -> "etano",
"PREDICTSTREET" -> "TREET"), so matching is stem-based rather than exact.
Stems are deliberately long enough to be unambiguous inside this corpus.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Substrings that identify a betting/prediction-market brand.  Each must be
# long enough that no non-betting sponsor in the corpus contains it.
BETTING_STEMS = (
    "betano", "etano", "betan", "3etano", "detano", "betanc",
    "kalshi", "kalsh", "alshi", "kals", "lshi", "ka1sh", "k1sh", "kal5h",
    "kal", "lsh", "klsh", "kls", "alsh", "1shi",
    "predictstreet", "predictstre", "edictstreet", "dictstreet", "ictstreet",
    "redictstr", "predictst", "predicts",
)

# Full-brand aliases from the Ontario operator register (input PDF).  These do
# not appear on FIFA rails today but cost nothing to watch for.
PDF_BRAND_STEMS = (
    "bet365", "betvictor", "tonybet", "betway", "bwin", "pinnacle",
    "unibet", "888sport", "leovegas", "betfair", "williamhill", "paddypower",
    "draftkings", "fanduel", "pointsbet", "betrivers", "caesarssports",
    "polymarket", "stake.com", "thescorebet", "bally bet", "ballybet",
)

# Sponsors that share the rails and must never be hidden.  Listed so the
# annotator can assert a frame really was surveyed rather than simply empty.
NON_BETTING = (
    "fifa", "tsn", "visa", "lays", "adidas", "marriott", "bonvoy", "globant",
    "lenovo", "doordash", "aramco", "qatar", "mercado", "libre", "bankofamerica",
    "mcdonald", "interrapidisimo", "beactive", "kraken", "michelob", "hisense",
    "coca", "budweiser", "verizon", "powerade", "unilever", "worldcup",
)

# Words that look brand-ish but are scoreboard / venue furniture.
IGNORE = (
    "worldcup", "kansas", "atlanta", "florida", "texas", "newyork", "newjersey",
    "gamein30", "argentina", "brazil", "mexico", "norway", "colegiales", "zamora",
)

_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(text: str) -> str:
    """Lowercase and strip everything that is not a letter or digit."""
    return _ALNUM.sub("", str(text).lower())


def is_betting(text: str, fuzzy: float = 0.86) -> bool:
    """True when an OCR string names a betting or prediction-market brand."""
    t = norm(text)
    if len(t) < 3:
        return False
    for stem in BETTING_STEMS:
        if stem in t:
            return True
    for stem in PDF_BRAND_STEMS:
        s = norm(stem)
        if len(s) >= 4 and s in t:
            return True
    # Fuzzy pass catches single-character OCR damage in the middle of a word.
    for stem in ("betano", "kalshi", "predictstreet"):
        if abs(len(t) - len(stem)) <= 3:
            if SequenceMatcher(None, t, stem).ratio() >= fuzzy:
                return True
    return False


def is_non_betting(text: str) -> bool:
    """True when an OCR string names a known non-betting sponsor."""
    t = norm(text)
    return any(s in t for s in NON_BETTING)


def is_ignorable(text: str) -> bool:
    t = norm(text)
    return len(t) < 3 or any(s in t for s in IGNORE)


def classify(text: str) -> str:
    """Return 'bet', 'neg' or 'other' for one OCR string."""
    if is_betting(text):
        return "bet"
    if is_non_betting(text):
        return "neg"
    return "other"
