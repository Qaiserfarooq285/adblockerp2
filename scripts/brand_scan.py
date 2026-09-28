#!/usr/bin/env python3
"""The live brand scan: read every board about once a second.

The board model only finds the brands it was trained on, and the per-board
OCR check in render_engine.py only reads boards the model already found -- so
a betting brand the model has never seen is never even looked at.  This scan
closes that gap without retraining:

    frame (copy, ~1/s) -> OCR over the whole picture -> every text read is
        hide     (Brand Library / built-in betting brand)  -> a board to paint:
                 text quads on one rail are merged, grown to the board's
                 edges (v2_annotate.grow_band) and along the rail to where
                 the artwork ends, never past another sponsor (extend_along)
        keep     (a sponsor that must stay)                -> vetoes painting there
        unknown  (a sponsor nobody has decided about)      -> the operator's review list
    + picture matching for logos with no readable name (the operator's crops)

The results go back to the engine, which carries them from the scanned frame
to the present through the camera homographies and hands them to the tracker
(BoardTracker.inject).  Between scans the homography carries the board like
any other.  The same frames feed the training data: boards the model missed,
boards it wrongly flagged, and a steady trickle of what it already knows.

`scan_frame` is a pure function, also used offline by learn_retrain.py to
harvest labels from recorded matches.
"""
from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np

from brand_library import BrandLibrary, betting_hint, sponsor_like
from geometry import containment, convex_iou
from v2_annotate import add_margin, grow_band, group_quads, quad_metrics
from v2_brands import norm
from v2_render import canonical
from v2_synth import pitch_top_edge

MIN_TEXT_H = 7.0          # px; shorter OCR boxes are too small to trust


def make_ocr(threads: int = 2):
    """A RapidOCR instance of its own, so the scan never contends with the
    per-board OCR check for one engine; None if RapidOCR is missing."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR(intra_op_num_threads=int(threads), inter_op_num_threads=1)
    except Exception as exc:
        print(f"[scan] OCR unavailable ({type(exc).__name__}: {exc}); the brand scan is off")
        return None


def ocr_frame(engine, frame: np.ndarray, min_conf: float = 0.5) -> list[dict]:
    """[{quad, text, conf}] for every text line on the picture."""
    try:
        res, _ = engine(frame, use_cls=False)
    except Exception:
        return []
    out = []
    for r in res or []:
        try:
            quad = np.asarray(r[0], np.float32).reshape(4, 2)
            text, conf = str(r[1]), float(r[2])
        except (ValueError, TypeError, IndexError):
            continue
        if conf < min_conf or quad_metrics(quad)[3] < MIN_TEXT_H:
            continue
        out.append({"quad": quad, "text": text, "conf": conf})
    return out


def _excluded(pt, boxes, mask, w, h) -> bool:
    """Inside the score bug / broadcaster graphics?"""
    x, y = float(pt[0]), float(pt[1])
    if mask is not None and 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1] \
            and mask[int(y), int(x)]:
        return True
    return any(b[0] <= x / w <= b[2] and b[1] <= y / h <= b[3] for b in boxes)


def _clip(q: np.ndarray, w: int, h: int) -> np.ndarray:
    q = np.asarray(q, np.float32).copy()
    q[:, 0] = np.clip(q[:, 0], -0.02 * w, 1.02 * w)
    q[:, 1] = np.clip(q[:, 1], -0.02 * h, 1.02 * h)
    return q


def _codes(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int32) // 16
    return (lab[:, :, 0] << 16) | (lab[:, :, 1] << 8) | lab[:, :, 2]


def extend_along(frame: np.ndarray, quad: np.ndarray, blockers: list[np.ndarray],
                 max_grow: float = 2.0, px_per_col: float = 2.0) -> np.ndarray:
    """Follow the board along its rail, column by column, to where its artwork ends.

    The wordmark rarely spans the whole advert.  Like v2_render.extend_rail
    this keeps going while the rail still shows the board's own palette, but it
    finds the edge to within a couple of pixels instead of in quarter-length
    steps, and it never passes another sponsor's wordmark (`blockers`): the
    advert next door may share a dark background, and painting over a sponsor
    that must stay is the costlier mistake.
    """
    q = canonical(quad)
    top, bot = q[1] - q[0], q[2] - q[3]
    length = float(np.linalg.norm(top))
    thick = float(np.linalg.norm(q[3] - q[0]))
    if length < 12 or thick < 4:
        return q
    lo_u, hi_u = -max_grow, 1.0 + max_grow
    ncol = int(max(8, (hi_u - lo_u) * length / px_per_col))
    nrow = int(max(4, min(48, thick)))
    # sample the rail: x = along it (u), y = across it (0 top .. 1 bottom)
    u = np.linspace(lo_u, hi_u, ncol, dtype=np.float32)
    v = np.linspace(0.15, 0.85, nrow, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    tp = q[0][None, None, :] + U[..., None] * top[None, None, :]
    bp = q[3][None, None, :] + U[..., None] * bot[None, None, :]
    pts = tp + V[..., None] * (bp - tp)
    strip = cv2.remap(frame, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside = ((pts[..., 0] >= 0) & (pts[..., 0] < frame.shape[1]) &
              (pts[..., 1] >= 0) & (pts[..., 1] < frame.shape[0])).all(axis=0)
    codes = _codes(strip)
    core = (u >= 0) & (u <= 1)
    keys, counts = np.unique(codes[:, core].ravel(), return_counts=True)
    order = np.argsort(-counts)
    keep = counts[order].cumsum() <= 0.90 * counts.sum()
    keep[0] = True
    match = np.isin(codes, keys[order][keep]).mean(axis=0)
    match = np.convolve(match, np.ones(5) / 5.0, mode="same")
    thresh = 0.62 * float(match[core].mean())

    # where other sponsors' wordmarks sit along this rail
    stop_hi, stop_lo = hi_u, lo_u
    n = np.array([-top[1], top[0]], np.float32) / max(length, 1e-6)
    for b in blockers:
        bb = np.asarray(b, np.float32).reshape(-1, 2)
        rel = bb - q[0]
        across = rel @ n
        if across.max() < -thick or across.min() > 2 * thick:
            continue                                  # another rail
        ub = (rel @ top) / (length * length)
        if ub.min() >= 1.0:
            stop_hi = min(stop_hi, float(ub.min()) - 0.02)
        elif ub.max() <= 0.0:
            stop_lo = max(stop_lo, float(ub.max()) + 0.02)

    i0, i1 = int(np.argmax(core)), len(u) - 1 - int(np.argmax(core[::-1]))
    hi = i1
    while hi + 1 < len(u) and inside[hi + 1] and match[hi + 1] >= thresh and u[hi + 1] <= stop_hi:
        hi += 1
    lo = i0
    while lo - 1 >= 0 and inside[lo - 1] and match[lo - 1] >= thresh and u[lo - 1] >= stop_lo:
        lo -= 1
    a, b = float(min(u[lo], 0.0)), float(max(u[hi], 1.0))
    if a == 0.0 and b == 1.0:
        return q
    return canonical(np.float32([q[0] + top * a, q[0] + top * b, q[3] + bot * b, q[3] + bot * a]))


def _board_from(frame: np.ndarray, quads: list[np.ndarray], extend: bool,
                blockers: list[np.ndarray]) -> np.ndarray:
    """Text quads on one rail -> the board they are printed on."""
    pts = np.concatenate([q.reshape(4, 2) for q in quads]).astype(np.float32)
    merged = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
    board = grow_band(frame, merged)
    if extend:
        board = extend_along(frame, board, blockers)
    return canonical(add_margin(board, 0.03))


class TemplateBank:
    """Picture templates for hide brands that have no readable name."""

    def __init__(self, learn_root, per_brand: int = 3, max_total: int = 8):
        self.root = learn_root
        self.per_brand, self.max_total = per_brand, max_total
        self.version = -1
        self.items: list[tuple[str, np.ndarray]] = []

    def refresh(self, library: BrandLibrary) -> None:
        if library.version == self.version:
            return
        self.version = library.version
        items = []
        for b in library.picture_brands():
            for rel in b["examples"][-self.per_brand:]:
                img = cv2.imread(str(self.root / rel), cv2.IMREAD_GRAYSCALE)
                if img is None or min(img.shape) < 12:
                    continue
                items.append((b["id"], cv2.resize(img, None, fx=0.5, fy=0.5,
                                                  interpolation=cv2.INTER_AREA)))
        self.items = items[-self.max_total:]

    def match(self, frame: np.ndarray, roi: tuple[int, int] | None,
              thresh: float = 0.72) -> list[tuple[str, np.ndarray, float]]:
        """(brand, quad, score) for every confident hit, at half resolution."""
        if not self.items:
            return []
        g = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        y0 = 0
        if roi is not None:
            y0, y1 = max(0, roi[0] // 2), min(g.shape[0], roi[1] // 2)
            g = g[y0:y1]
        hits = []
        for bid, tpl in self.items:
            for s in (0.7, 1.0, 1.4):
                t = cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                th, tw = t.shape
                if th < 8 or tw < 12 or th >= g.shape[0] or tw >= g.shape[1]:
                    continue
                r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
                for _ in range(6):
                    _, mx, _, (x, y) = cv2.minMaxLoc(r)
                    if mx < thresh:
                        break
                    q = np.float32([[x, y], [x + tw, y], [x + tw, y + th], [x, y + th]])
                    q[:, 1] += y0
                    hits.append((bid, q * 2.0, float(mx)))
                    r[max(0, y - th // 2):y + th // 2 + 1, max(0, x - tw // 2):x + tw // 2 + 1] = -1
        hits.sort(key=lambda h: -h[2])
        kept = []
        for h in hits:
            if all(convex_iou(h[1], k[1]) < 0.3 for k in kept):
                kept.append(h)
        return kept


def scan_frame(frame: np.ndarray, library: BrandLibrary, engine,
               exclude_boxes=(), templates: TemplateBank | None = None,
               min_conf: float = 0.5, exclude_mask: np.ndarray | None = None) -> dict:
    """Read one frame.  Returns

        texts       [{quad, text, conf, cls, brand}]
        boards      [{quad, label: hide|keep|other, brand, texts, by, conf}]
        candidates  [{key, text, conf, quad, hint}]   unknown sponsors
    """
    h, w = frame.shape[:2]
    texts = []
    for r in ocr_frame(engine, frame, min_conf):
        if _excluded(r["quad"].mean(axis=0), exclude_boxes, exclude_mask, w, h):
            continue                         # score bug / broadcaster graphics
        r["cls"], r["brand"] = library.classify(r["text"])
        if r["cls"] != "ignore":
            texts.append(r)

    boards, candidates = [], []
    groups: dict[tuple, list[dict]] = {}
    for r in texts:
        key = ("hide", r["brand"]) if r["cls"] == "bet" else \
              ("keep", r["brand"]) if r["cls"] == "neg" else ("other", norm(r["text"]))
        groups.setdefault(key, []).append(r)
    blockers_all = [r["quad"] for r in texts if r["cls"] != "bet"]
    for (label, bid), rs in groups.items():
        quads = [r["quad"] for r in rs]
        for idx in group_quads(quads):
            members = [rs[k] for k in idx]
            board = _board_from(frame, [m["quad"] for m in members], extend=(label == "hide"),
                                blockers=blockers_all if label == "hide" else [])
            board = _clip(board, w, h)
            conf = max(m["conf"] for m in members)
            item = {"quad": board, "label": label, "brand": bid if label != "other" else None,
                    "texts": [m["text"] for m in members], "by": "ocr", "conf": conf}
            boards.append(item)
            if label == "other":
                best = max(members, key=lambda m: m["conf"])
                bx = best["quad"][:, 0]
                cut_off = bx.min() < 4 or bx.max() > w - 5      # a word the frame edge truncates
                if sponsor_like(best["text"], best["conf"]) and not cut_off:
                    candidates.append({"key": norm(best["text"]), "text": best["text"],
                                       "conf": best["conf"], "quad": board,
                                       "hint": betting_hint(best["text"])})

    if templates is not None:
        templates.refresh(library)
        if templates.items:
            edge = pitch_top_edge(frame)
            roi = None
            if edge is not None:
                roi = (int(max(0, edge.min() - 0.25 * h)), int(min(h, edge.max() + 0.35 * h)))
            for bid, q, score in templates.match(frame, roi):
                if any(b["label"] == "hide" and convex_iou(b["quad"], q) > 0.2 for b in boards):
                    continue                  # already found by name
                if any(b["label"] == "keep" and containment(q, b["quad"]) > 0.5 for b in boards):
                    continue                  # sits on a sponsor that must stay
                boards.append({"quad": canonical(add_margin(q, 0.04)), "label": "hide",
                               "brand": bid, "texts": [], "by": "picture", "conf": score})
    return {"texts": texts, "boards": boards, "candidates": candidates}


def crop_around(frame: np.ndarray, quad: np.ndarray, pad: float = 0.25, max_w: int = 640):
    """The board plus some context, for the operator's review list."""
    h, w = frame.shape[:2]
    q = np.asarray(quad, np.float32)
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    px, py = pad * (x1 - x0), max(pad * (y1 - y0), 12.0)
    x0, x1 = int(max(0, x0 - px)), int(min(w, x1 + px))
    y0, y1 = int(max(0, y0 - py)), int(min(h, y1 + py))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    c = frame[y0:y1, x0:x1].copy()
    cv2.polylines(c, [np.round(q - [x0, y0]).astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    if c.shape[1] > max_w:
        c = cv2.resize(c, (max_w, int(c.shape[0] * max_w / c.shape[1])), interpolation=cv2.INTER_AREA)
    return c


class BrandScanner:
    """Runs scan_frame off the frame loop and turns its results into review
    entries and training samples.  One frame in flight at a time: the engine
    only submits when the scanner is idle."""

    def __init__(self, library: BrandLibrary, cfg: dict, reviews=None, samples=None,
                 learn_root=None, async_mode: bool = True):
        self.library = library
        self.cfg = cfg
        self.reviews, self.samples = reviews, samples
        self.exclude = cfg.get("exclude_boxes") or []
        self.engine = make_ocr(int(cfg.get("threads", 2)))
        self.available = self.engine is not None
        self.templates = TemplateBank(learn_root) if learn_root is not None else None
        self.async_mode = async_mode
        self.jobs: "queue.Queue" = queue.Queue(maxsize=1)
        self.results: "queue.Queue" = queue.Queue()
        self._busy = False
        self._last: dict[str, float] = {}
        self.stats = {"scans": 0, "ms": 0.0, "hide": 0, "keep": 0, "unknown": 0, "samples": 0}
        if self.available and async_mode:
            threading.Thread(target=self._run, name="brand-scan", daemon=True).start()

    def idle(self) -> bool:
        return self.available and not self._busy

    def submit(self, idx: int, frame: np.ndarray, tracks: list[dict],
               exclude_mask: np.ndarray | None = None) -> bool:
        """`frame` must be the caller's own copy of the unpainted picture;
        `tracks` the tracker's boards on that frame: [{tid, quad, verdict,
        model_bet, conf}]; `exclude_mask` the broadcaster graphics, if known."""
        if not self.idle():
            return False
        self._busy = True
        job = (idx, frame, tracks, exclude_mask)
        if not self.async_mode:
            self._do(job)
            return True
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            self._busy = False
            return False

    def poll(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            self._do(job)

    def _do(self, job) -> None:
        idx, frame, tracks, mask = job
        t0 = time.perf_counter()
        try:
            self.library.reload_if_changed()
            res = scan_frame(frame, self.library, self.engine, self.exclude, self.templates,
                             exclude_mask=mask)
        except Exception as exc:
            print(f"[scan] frame {idx} failed: {type(exc).__name__}: {exc}")
            res = {"texts": [], "boards": [], "candidates": []}
        res["idx"] = idx
        res["tracks"] = tracks
        res["ms"] = 1000.0 * (time.perf_counter() - t0)
        self.stats["scans"] += 1
        self.stats["ms"] = res["ms"] if self.stats["scans"] == 1 else 0.8 * self.stats["ms"] + 0.2 * res["ms"]
        for b in res["boards"]:
            if b["label"] in ("hide", "keep"):
                self.stats[b["label"]] += 1
        self.results.put(res)
        self._busy = False
        try:
            self._learn(frame, res, tracks)
        except Exception as exc:
            print(f"[scan] learning data for frame {idx} failed: {type(exc).__name__}: {exc}")

    def _due(self, key: str, gap: float) -> bool:
        now = time.monotonic()
        if now - self._last.get(key, -1e9) < gap:
            return False
        self._last[key] = now
        return True

    def _learn(self, frame: np.ndarray, res: dict, tracks: list[dict]) -> None:
        """Review entries for unknown sponsors; training samples where they
        teach the model something."""
        if self.reviews is not None:
            for c in res["candidates"]:
                self.stats["unknown"] += 1
                self.reviews.offer(c["key"], c["text"], c["conf"], crop_around(frame, c["quad"]),
                                   frame, c["quad"], hint=c["hint"])
            self.reviews.save()
        if self.samples is None:
            return
        model = [t for t in tracks if t.get("model_bet") and t.get("verdict") != "neg"]
        boards = [dict(b) for b in res["boards"]]
        reason = None
        # 1. a board to hide that the model did not find: its blind spot
        missed = [b for b in boards if b["label"] == "hide"
                  and not any(convex_iou(b["quad"], t["quad"]) > 0.2 for t in model)]
        if missed and any(self._due(f"missed:{b['brand']}", float(self.cfg.get("sample_missed_s", 6)))
                          for b in missed):
            reason = "missed"
        # 2. the model flagged a board that carries a sponsor to keep
        wrong = [t for t in model if any(b["label"] == "keep" and
                                         containment(b["quad"], t["quad"]) > 0.5 for b in boards)]
        if reason is None and wrong and self._due("wrong", float(self.cfg.get("sample_wrong_s", 10))):
            reason = "wrong"
        # 3. what it already gets right, so retraining does not forget it
        if reason is None and model and self._due("replay", float(self.cfg.get("sample_replay_s", 30))):
            reason = "replay"
        if reason is None:
            return
        labels = [b for b in boards if b["label"] in ("hide", "keep", "other")]
        for t in model:
            if any(convex_iou(t["quad"], b["quad"]) > 0.3 for b in labels):
                continue
            labels.append({"quad": t["quad"], "label": "hide", "brand": None, "texts": [],
                           "by": "model", "conf": float(t.get("conf", 0.0)),
                           "verdict": t.get("verdict")})
        source = "replay" if reason == "replay" else "scan"
        if self.samples.add(frame, [{k: v for k, v in b.items()} for b in labels], source,
                            meta={"reason": reason, "idx": int(res["idx"])}):
            self.stats["samples"] += 1

    def close(self) -> None:
        if self.async_mode and self.available:
            try:
                self.jobs.put_nowait(None)
            except queue.Full:
                pass
