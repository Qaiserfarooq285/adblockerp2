#!/usr/bin/env python3
"""The Brand Library: which sponsors to hide and which to keep, editable live.

v2_brands.py is the built-in vocabulary -- the brands the model was trained
on and a register of licensed operators.  Everything the operator decides on
the operator page lands here instead, in <learn_dir>/brands.json, and takes
precedence over the built-ins:

    hide   a betting brand: every board reading this name is painted over
    keep   a sponsor that must stay visible, even if a built-in list or the
           model says otherwise
    ignored  text that is not a sponsor at all (score graphics, venue signs):
           never asked about again

A brand is recognised by its *aliases*: the name as OCR reads it, so a brand
added by clicking a board carries exactly the spelling the text reader
produces for that artwork.  A brand with no readable name is recognised by
picture instead (see brand_scan.py); the operator's crops are kept as
`examples` for that and as artwork for training.

Changes take effect in the live pipeline within a frame (the engine re-reads
the verdicts of every tracked board when `version` changes) and are saved at
once, so they survive a restart and are used by the retraining job.
"""
from __future__ import annotations

import json
import re
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

from v2_brands import is_betting, is_ignorable, is_non_betting, norm

ACTIONS = ("hide", "keep")

# A suggestion shown next to an unknown sponsor, never a decision.
BETTING_HINTS = ("bet", "casino", "poker", "slots", "odds", "wager", "sportsbook", "jackpot",
                 "lotto", "lottery", "bingo", "punter", "gambl", "predict", "parlay")


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return s[:40] or "brand"


def alias_hit(t: str, a: str) -> bool:
    """Does the normalised OCR string `t` name the normalised alias `a`?

    Substring for aliases of 4+ characters (a board reads "STAKE.COM" or
    "stakesport"); a 3-character alias only as (nearly) the whole string, or
    "bet" would match "alphabet".  Long aliases also match fuzzily, to absorb
    one or two misread characters.
    """
    if len(a) < 3 or len(t) < 3:
        return False
    if a in t and (len(a) >= 4 or len(t) <= len(a) + 1):
        return True
    return (len(a) >= 5 and abs(len(t) - len(a)) <= 2
            and SequenceMatcher(None, t, a).ratio() >= 0.84)


def sponsor_like(text: str, conf: float, min_conf: float = 0.75) -> bool:
    """Worth asking the operator about: confident, word-like, not a number."""
    t = norm(text)
    letters = sum(c.isalpha() for c in t)
    return conf >= min_conf and len(t) >= 4 and letters >= 3 and letters >= 0.6 * len(t)


def betting_hint(text: str) -> bool:
    t = norm(text)
    return any(h in t for h in BETTING_HINTS)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class BrandLibrary:
    """Thread-safe, file-backed.  Readers poll `version` (an int) to notice
    changes; the file is re-read if another process (the retraining job, a
    second pipeline) changed it."""

    def __init__(self, path: Path | str | None):
        """`path=None`: an in-memory library (the built-in lists only, plus
        whatever is added in this process) -- nothing is read or written."""
        self.path = Path(path) if path else None
        self.lock = threading.RLock()
        self.brands: dict[str, dict] = {}
        self.ignored: set[str] = set()
        self.version = 0
        self._mtime = None
        self.load()

    # -- persistence ------------------------------------------------------------

    def load(self) -> None:
        with self.lock:
            if self.path is None or not self.path.exists():
                return
            try:
                d = json.loads(self.path.read_text())
            except (OSError, ValueError) as exc:
                print(f"[brands] cannot read {self.path} ({exc}); keeping the last good copy")
                return
            brands = {}
            for b in d.get("brands", []):
                if b.get("action") in ACTIONS and b.get("id"):
                    b.setdefault("aliases", [])
                    b.setdefault("examples", [])
                    b.setdefault("picture", False)
                    brands[b["id"]] = b
            self.brands = brands
            self.ignored = set(d.get("ignored", []))
            self.version = max(self.version + 1, int(d.get("version", 0)))
            self._mtime = self.path.stat().st_mtime

    def save(self) -> None:
        with self.lock:
            self.version += 1
            if self.path is None:
                return
            d = {"version": self.version, "updated": _now(),
                 "brands": sorted(self.brands.values(), key=lambda b: b["id"]),
                 "ignored": sorted(self.ignored)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(d, indent=1))
            tmp.replace(self.path)
            self._mtime = self.path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        if self.path is None:
            return False
        try:
            m = self.path.stat().st_mtime
        except OSError:
            return False
        if m != self._mtime:
            self.load()
            return True
        return False

    # -- editing ----------------------------------------------------------------

    def find(self, name_or_alias: str) -> dict | None:
        """The brand with this id, name or alias (normalised), if any."""
        k = norm(name_or_alias)
        with self.lock:
            if slug(name_or_alias) in self.brands:
                return self.brands[slug(name_or_alias)]
            for b in self.brands.values():
                if norm(b["name"]) == k or k in b["aliases"]:
                    return b
        return None

    def upsert(self, name: str, action: str, aliases=(), example: str | None = None,
               picture: bool | None = None, source: str = "operator",
               name_is_alias: bool = True) -> dict:
        """Add a brand, or update the existing one with this name/alias.

        `name_is_alias=False` for a name that is not written on the board
        (a picture-only logo the operator called "Logo 3").
        """
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        name = str(name).strip()[:60] or "Unnamed logo"
        keys = [norm(a) for a in [*([name] if name_is_alias else []), *aliases]]
        keys = [k for k in dict.fromkeys(keys) if len(k) >= 3]
        with self.lock:
            b = self.find(name) if name_is_alias else self.brands.get(slug(name))
            if b is None:
                for k in keys:
                    b = self.find(k)
                    if b is not None:
                        break
            if b is None:
                bid = slug(name)
                n = 2
                while bid in self.brands:
                    bid = f"{slug(name)}-{n}"
                    n += 1
                b = {"id": bid, "name": name, "action": action, "aliases": [], "examples": [],
                     "picture": False, "source": source, "created": _now()}
                self.brands[bid] = b
            b["action"] = action
            for k in keys:
                if k not in b["aliases"]:
                    b["aliases"].append(k)
            if example and example not in b["examples"]:
                b["examples"] = (b["examples"] + [example])[-12:]
            if picture is not None:
                b["picture"] = bool(picture)
            elif not b["aliases"]:
                b["picture"] = True         # nothing to read: recognise it by picture
            b["updated"] = _now()
            for k in keys:
                self.ignored.discard(k)
            self.save()
            return dict(b)

    def edit(self, brand_id: str, action: str | None = None, name: str | None = None,
             picture: bool | None = None, add_alias: str | None = None,
             remove_alias: str | None = None) -> dict | None:
        with self.lock:
            b = self.brands.get(brand_id)
            if b is None:
                return None
            if action is not None:
                if action not in ACTIONS:
                    raise ValueError(f"action must be one of {ACTIONS}")
                b["action"] = action
            if name:
                b["name"] = str(name).strip()[:60]
            if picture is not None:
                b["picture"] = bool(picture)
            if add_alias and len(norm(add_alias)) >= 3 and norm(add_alias) not in b["aliases"]:
                b["aliases"].append(norm(add_alias))
            if remove_alias:
                b["aliases"] = [a for a in b["aliases"] if a != norm(remove_alias)]
            b["updated"] = _now()
            self.save()
            return dict(b)

    def add_example(self, brand_id: str, rel: str, picture: bool | None = None) -> None:
        with self.lock:
            b = self.brands.get(brand_id)
            if b is None:
                return
            if rel not in b["examples"]:
                b["examples"] = (b["examples"] + [rel])[-12:]
            if picture is not None:
                b["picture"] = bool(picture)
            self.save()

    def remove(self, brand_id: str) -> bool:
        with self.lock:
            if self.brands.pop(brand_id, None) is None:
                return False
            self.save()
            return True

    def ignore(self, text: str) -> None:
        k = norm(text)
        if len(k) < 3:
            return
        with self.lock:
            self.ignored.add(k)
            self.save()

    def unignore(self, text: str) -> None:
        with self.lock:
            self.ignored.discard(norm(text))
            self.save()

    # -- reading ----------------------------------------------------------------

    def classify(self, text: str) -> tuple[str, str | None]:
        """One OCR string -> ('bet' | 'neg' | 'ignore' | 'unknown', brand id).

        The operator's decisions come first, then the built-in lists.  Built-in
        matches return the id 'builtin'.
        """
        t = norm(text)
        if len(t) < 3:
            return "ignore", None
        hide = keep = keep_near = None
        with self.lock:
            for b in self.brands.values():
                if any(alias_hit(t, a) for a in b["aliases"]):
                    if b["action"] == "hide":
                        hide = hide or b["id"]
                    else:
                        keep = keep or b["id"]
                elif b["action"] == "keep" and any(
                        len(a) >= 5 and SequenceMatcher(None, t, a).ratio() >= 0.75 for a in b["aliases"]):
                    keep_near = keep_near or b["id"]
            if hide:
                return "bet", hide
            if keep:
                return "neg", keep
            if t in self.ignored or any(len(i) >= 4 and i in t for i in self.ignored):
                return "ignore", None
        if is_betting(text):
            # The built-in stems also catch misreadings ("KALSH1"); a brand the
            # operator chose to keep must not come back through one of them.
            return ("neg", keep_near) if keep_near else ("bet", "builtin")
        if is_non_betting(text):
            return "neg", "builtin"
        if is_ignorable(text):
            return "ignore", None
        return "unknown", None

    def verdict(self, texts) -> tuple[str, str | None]:
        """All texts read on one board -> ('bet' | 'neg' | 'unknown', brand id).

        Same precedence as v2_render.is_betting_board: a betting read anywhere
        on the board wins over a neighbour's wordmark bleeding into the crop.
        """
        found = {"bet": None, "neg": None}
        for t in texts:
            c, bid = self.classify(t)
            if c in found and found[c] is None:
                found[c] = bid
        if found["bet"] is not None:
            return "bet", found["bet"]
        if found["neg"] is not None:
            return "neg", found["neg"]
        return "unknown", None

    def picture_brands(self) -> list[dict]:
        """Hide brands recognised by picture (their examples are templates)."""
        with self.lock:
            return [dict(b) for b in self.brands.values()
                    if b["action"] == "hide" and b.get("picture") and b.get("examples")]

    def snapshot(self) -> dict:
        with self.lock:
            return {"version": self.version,
                    "brands": sorted((dict(b) for b in self.brands.values()),
                                     key=lambda b: (b["action"], b["name"].lower())),
                    "ignored": sorted(self.ignored)}
