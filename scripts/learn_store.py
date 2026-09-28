#!/usr/bin/env python3
"""Where the pipeline keeps what it learns from, and the helpers that write it.

    <learn_dir>/
      brands.json          the Brand Library (brand_library.py)
      review/              sponsors the scan could not place: queue.json, crops, frames
      examples/<brand>/    logo crops per brand: picture matching + training artwork
      samples/<day>/       training frames: ORIGINAL full-resolution picture (.jpg)
                           plus its board labels (.json)
      recordings/          the original feed (MPEG-TS, stream copy), for re-harvesting
      models/              versions the retraining job produced; current.json = in use
      jobs/                retraining job state and logs

On RunPod this is /workspace/learning, on the persistent volume.  Nothing in
here is committed to git except the README.

A training sample never contains a painted frame: every image is taken before
the renderer touches it.  Labels are stored with the text that was read and
who decided (operator / OCR / model), so the retraining job re-resolves them
against the Brand Library as it is *then* -- a board seen today as an unknown
sponsor becomes a positive example once the operator marks that brand.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from collections import Counter, deque
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBDIRS = ("review/crops", "review/frames", "examples", "samples", "recordings", "models", "jobs")


def default_learn_dir() -> Path:
    """LEARN_DIR, else the RunPod volume, else <repo>/learning."""
    if os.environ.get("LEARN_DIR"):
        return Path(os.environ["LEARN_DIR"])
    ws = Path("/workspace")
    if ws.is_dir() and os.access(ws, os.W_OK):
        return ws / "learning"
    return ROOT / "learning"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float))
    tmp.replace(path)


def imwrite(path: Path, img: np.ndarray, quality: int = 90) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    tmp = path.with_suffix(".part")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)
    return True


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


class LearnDir:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        for sub in SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    @property
    def brands_file(self) -> Path:
        return self.root / "brands.json"

    def rel(self, path: Path | str) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def resolve(self, rel: str) -> Path | None:
        """A path inside the learn dir, or None (no escaping it with ..)."""
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root.resolve())
        except ValueError:
            return None
        return p


# ---------------------------------------------------------------------- review

class ReviewQueue:
    """Sponsors seen on the boards that nobody has decided about yet.

    Keyed by the normalised text; a near-identical reading (one OCR character
    off) joins the existing entry.  Each entry keeps its clearest crop for the
    operator and up to `frames_per_item` full frames, which become training
    samples once the operator decides.
    """

    def __init__(self, ld: LearnDir, max_items: int = 150, frames_per_item: int = 3,
                 frame_gap_s: float = 10.0):
        self.ld = ld
        self.path = ld.root / "review" / "queue.json"
        self.max_items = max_items
        self.frames_per_item = frames_per_item
        self.frame_gap_s = frame_gap_s
        self.lock = threading.RLock()
        self.items: dict[str, dict] = {}
        self._n = 0
        self.dirty = False
        if self.path.exists():
            try:
                for it in json.loads(self.path.read_text()):
                    self.items[it["id"]] = it
                self._n = max((int(i.split("-")[-1]) for i in self.items), default=0)
            except (OSError, ValueError, KeyError):
                self.items = {}

    def _match(self, key: str) -> str | None:
        for iid, it in self.items.items():
            k = it["key"]
            if k == key or (min(len(k), len(key)) >= 5 and
                            SequenceMatcher(None, k, key).ratio() >= 0.86):
                return iid
        return None

    def offer(self, key: str, text: str, conf: float, crop: np.ndarray | None = None,
              frame: np.ndarray | None = None, quad=None, hint: bool = False) -> None:
        now = time.time()
        with self.lock:
            iid = self._match(key)
            if iid is None:
                if len(self.items) >= self.max_items:
                    self._evict()
                self._n += 1
                iid = f"c-{self._n}"
                self.items[iid] = {"id": iid, "key": key, "text": text, "conf": 0.0, "count": 0,
                                   "first": now, "last": now, "crop": None, "frames": [],
                                   "hint": bool(hint)}
            it = self.items[iid]
            it["count"] += 1
            it["last"] = now
            if crop is not None and (it["crop"] is None or conf > it["conf"] + 0.03):
                p = self.ld.root / "review" / "crops" / f"{iid}.jpg"
                if imwrite(p, crop, 88):
                    it["crop"] = self.ld.rel(p)
                    it["conf"], it["text"] = float(conf), text
            if (frame is not None and quad is not None and len(it["frames"]) < self.frames_per_item
                    and (not it["frames"] or now - it["frames"][-1]["t"] >= self.frame_gap_s)):
                p = self.ld.root / "review" / "frames" / f"{iid}_{len(it['frames'])}.jpg"
                if imwrite(p, frame, 92):
                    it["frames"].append({"file": self.ld.rel(p), "t": now, "text": text,
                                         "quad": np.asarray(quad, np.float32).round(1).tolist(),
                                         "w": int(frame.shape[1]), "h": int(frame.shape[0])})
            self.dirty = True

    def _evict(self) -> None:
        victim = min(self.items.values(), key=lambda it: (it["count"], it["last"]))
        self._delete_files(self.items.pop(victim["id"]))

    def _delete_files(self, it: dict) -> None:
        for rel in [it.get("crop")] + [f["file"] for f in it.get("frames", [])]:
            p = self.ld.resolve(rel) if rel else None
            if p is not None and p.exists():
                p.unlink()

    def pending(self, min_count: int = 2) -> list[dict]:
        with self.lock:
            out = [dict(it) for it in self.items.values() if it["count"] >= min_count]
        return sorted(out, key=lambda it: (-it["count"], -it["last"]))

    def take(self, iid: str) -> dict | None:
        """Remove an entry to act on it; its files stay until `finish`."""
        with self.lock:
            it = self.items.pop(iid, None)
            self.dirty = True
            return it

    def finish(self, it: dict) -> None:
        self._delete_files(it)

    def drop_matching(self, classify) -> int:
        """Forget entries the library now decides (e.g. a brand typed in by
        name).  `classify(text) -> (cls, id)`.  Returns how many were dropped."""
        n = 0
        with self.lock:
            for iid in list(self.items):
                if classify(self.items[iid]["text"])[0] != "unknown":
                    self._delete_files(self.items.pop(iid))
                    n += 1
            self.dirty = self.dirty or n > 0
        return n

    def save(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            write_json(self.path, list(self.items.values()))
            self.dirty = False


# --------------------------------------------------------------------- samples

class SampleStore:
    """Training frames: the ORIGINAL picture plus the boards on it.

    boards: [{"quad": [[x, y] * 4], "label": "hide" | "keep" | "other",
              "brand": id or None, "texts": [...], "by": "operator" | "ocr" |
              "picture" | "model"}]
    """

    def __init__(self, ld: LearnDir, max_total: int = 40000):
        self.ld = ld
        self.dir = ld.root / "samples"
        self.max_total = max_total
        self.lock = threading.Lock()
        self._n = 0
        self.counts: Counter = Counter()
        for jf in self.dir.glob("*/*.json"):
            try:
                self.counts[json.loads(jf.read_text()).get("source", "?")] += 1
            except (OSError, ValueError):
                pass

    def add(self, frame: np.ndarray | None, boards: list[dict], source: str,
            meta: dict | None = None, image_path: Path | None = None) -> str | None:
        """Save one sample; `image_path` moves an already-written JPEG in
        instead of encoding `frame`.  Returns the sample's relative path."""
        with self.lock:
            self._n += 1
            n = self._n
        day = time.strftime("%Y%m%d", time.gmtime())
        stem = f"{time.strftime('%H%M%S', time.gmtime())}_{source}_{os.getpid() % 1000:03d}{n:05d}"
        img = self.dir / day / f"{stem}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        if image_path is not None:
            try:
                shutil.move(str(image_path), img)
            except OSError:
                return None
            h, w = (meta or {}).get("h"), (meta or {}).get("w")
            if not (h and w):
                im = cv2.imread(str(img))
                if im is None:
                    return None
                h, w = im.shape[:2]
        else:
            if frame is None or not imwrite(img, frame, 92):
                return None
            h, w = frame.shape[:2]
        rec = {"image": img.name, "w": int(w), "h": int(h), "source": source,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "boards": [{**b, "quad": np.asarray(b["quad"], np.float32).round(1).tolist()}
                          for b in boards],
               **({"meta": meta} if meta else {})}
        write_json(img.with_suffix(".json"), rec)
        with self.lock:
            self.counts[source] += 1
            total = sum(self.counts.values())
        if total > self.max_total:
            self.prune()
        return self.ld.rel(img.with_suffix(".json"))

    def remove_recording(self, name: str) -> int:
        """Drop the samples harvested from one recording (before re-reading it)."""
        n = 0
        for jf in self.dir.glob("*/*_harvest_*.json"):
            try:
                if (json.loads(jf.read_text()).get("meta") or {}).get("recording") != name:
                    continue
            except (OSError, ValueError):
                continue
            for p in (jf, jf.with_suffix(".jpg")):
                if p.exists():
                    p.unlink()
            n += 1
        with self.lock:
            self.counts["harvest"] -= n
        return n

    def prune(self) -> None:
        """Over the cap: drop the oldest automatic samples first; operator
        decisions are the most expensive to get again, so they go last."""
        order = {"replay": 0, "harvest": 1, "scan": 2, "review": 3, "operator": 4}
        files = []
        for jf in self.dir.glob("*/*.json"):
            src = jf.stem.split("_")[1] if "_" in jf.stem else "?"
            files.append((order.get(src, 1), jf.stem, jf))
        files.sort()
        excess = len(files) - int(0.9 * self.max_total)
        for _, _, jf in files[:max(0, excess)]:
            src = jf.stem.split("_")[1]
            for p in (jf, jf.with_suffix(".jpg")):
                if p.exists():
                    p.unlink()
            with self.lock:
                self.counts[src] -= 1

    def summary(self) -> dict:
        with self.lock:
            c = {k: v for k, v in self.counts.items() if v > 0}
        return {"total": sum(c.values()), "by_source": c}


# ------------------------------------------------------------------- snapshots

class Snapshots:
    """The last `seconds` of ORIGINAL frames, a few per second, as JPEG.

    The operator pauses the player, picks the clearest of these around that
    moment and draws on it.  Frames are copied on the caller's thread (a
    memcpy) and encoded on a worker thread.
    """

    def __init__(self, seconds: float = 90.0, per_second: float = 3.0, fps: float = 30.0):
        self.stride = max(1, int(round(fps / per_second)))
        self.items: deque = deque(maxlen=max(4, int(seconds * per_second)))
        self.q: "queue.Queue" = queue.Queue(maxsize=6)
        self.lock = threading.Lock()
        self._sid = 0
        self._n = 0
        threading.Thread(target=self._run, name="snapshots", daemon=True).start()

    def offer(self, idx: int, frame: np.ndarray, t_out: float, session: int = 0) -> None:
        self._n += 1
        if self._n % self.stride:
            return
        try:
            self.q.put_nowait((idx, frame.copy(), float(t_out), time.time(), session))
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            idx, frame, t_out, wall, session = self.q.get()
            ok, full = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            h, w = frame.shape[:2]
            tw = 320
            thumb = cv2.resize(frame, (tw, max(2, int(h * tw / w))), interpolation=cv2.INTER_AREA)
            ok2, th = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not (ok and ok2):
                continue
            with self.lock:
                self._sid += 1
                self.items.append({"id": self._sid, "idx": idx, "t": t_out, "wall": wall,
                                   "session": session, "w": w, "h": h,
                                   "jpg": full.tobytes(), "thumb": th.tobytes()})

    def get(self, sid: int) -> dict | None:
        with self.lock:
            for it in self.items:
                if it["id"] == sid:
                    return it
        return None

    def listing(self) -> list[dict]:
        with self.lock:
            return [{k: it[k] for k in ("id", "idx", "t", "wall", "session", "w", "h")}
                    for it in self.items]

    def frame(self, sid: int) -> np.ndarray | None:
        it = self.get(sid)
        if it is None:
            return None
        return cv2.imdecode(np.frombuffer(it["jpg"], np.uint8), cv2.IMREAD_COLOR)


class OutputClock:
    """Output timestamp <-> the wall clock in the HLS playlist.

    The HLS muxer stamps each segment with EXT-X-PROGRAM-DATE-TIME counted
    from when its output started, so a frame's date is start + (t - t_first).
    The player reports the date it is showing (hls.js `playingDate`); this
    turns it back into our output timestamp to find the frame on screen.
    Good to a fraction of a second, which is why the operator gets a strip of
    frames around that moment to choose from.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.wall0: float | None = None
        self.t0: float | None = None
        self.session = 0

    def start(self, t: float) -> None:
        with self.lock:
            self.wall0, self.t0 = time.time(), float(t)
            self.session += 1

    def t_at(self, wall: float) -> float | None:
        with self.lock:
            if self.wall0 is None:
                return None
            return self.t0 + (wall - self.wall0)


def prune_recordings(folder: Path, max_gb: float, pattern: str = "original-*.ts") -> int:
    """Delete the oldest recordings in `folder` beyond `max_gb`.  Returns bytes freed.

    The newest file is never touched: it is the one being written.
    """
    files = sorted(Path(folder).glob(pattern), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    limit = max_gb * 1e9
    freed = 0
    for p in files[:-1]:
        if total <= limit:
            break
        size = p.stat().st_size
        p.unlink()
        done = p.with_suffix(".harvested.json")
        if done.exists():
            done.unlink()
        total -= size
        freed += size
    return freed
