#!/usr/bin/env python3
"""The operator page's API: teach the live pipeline which logos to hide.

Served by live_stream.py under <http root>/api/ and guarded by a password of
its own (OPERATOR_TOKEN, sent as the X-Operator-Token header), separate from
the viewer link -- whoever has it decides what goes on air.

    GET  state                   library, review list, learning data, models, job
    GET  frames?at=<ms>&span=<s> original frames around the moment the player shows
    GET  snap/<id>.jpg | snap/<id>_t.jpg    one such frame, full size / thumbnail
    GET  file/<path>             review crops and brand examples
    POST mark    {snap, box, action, name?}   "this logo": hide or keep it
    POST review  {id, action, name?}          decide an unknown sponsor (hide/keep/ignore)
    POST brand   {name, action}               add a brand by name (before kick-off)
    POST brand   {id, action?|name?|picture?|remove?}   edit or remove one
    POST unignore {text}
    POST train   {action: start|stop}         the retraining job (learn_retrain.py)
    POST model   {version}                    switch the board model now ("original" = back)
"""
from __future__ import annotations

import hmac
import json
import threading
import time
from urllib.parse import parse_qs

import cv2
import numpy as np

from brand_library import ACTIONS
from geometry import containment
from learn_store import dir_bytes, imwrite
from v2_brands import norm

MAX_BODY = 64 * 1024


def _json(code: int, obj) -> tuple[int, str, bytes]:
    return code, "application/json", json.dumps(obj, default=float).encode()


def _err(code: int, msg: str) -> tuple[int, str, bytes]:
    return _json(code, {"error": msg})


class OperatorAPI:
    def __init__(self, token: str, ld, library, reviews, samples, snapshots, clock,
                 get_engine, models=None, job=None, swap_model=None, min_review_count: int = 2):
        self.token = str(token)
        self.ld, self.library, self.reviews, self.samples = ld, library, reviews, samples
        self.snapshots, self.clock, self.get_engine = snapshots, clock, get_engine
        self.models, self.job, self.swap_model = models, job, swap_model
        self.min_review_count = min_review_count
        self._ocr = None
        self._ocr_lock = threading.Lock()
        self._logo_n = 0

    # -- plumbing ---------------------------------------------------------------

    def authorized(self, headers) -> bool:
        got = headers.get("X-Operator-Token") or ""
        return bool(self.token) and hmac.compare_digest(got.encode(), self.token.encode())

    def handle(self, method: str, path: str, query: str, body: bytes, headers):
        if not self.authorized(headers):
            return _err(401, "operator password required")
        try:
            if method == "GET":
                return self._get(path, parse_qs(query))
            if method == "POST":
                if len(body) > MAX_BODY:
                    return _err(413, "request too large")
                data = json.loads(body or b"{}")
                if not isinstance(data, dict):
                    return _err(400, "expected a JSON object")
                return self._post(path, data)
        except (ValueError, KeyError, TypeError) as exc:
            return _err(400, f"{type(exc).__name__}: {exc}")
        return _err(405, "method not allowed")

    def _read_texts(self, img: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        """(text, conf, quad in `img` pixels) -- OCR with its own engine
        (requests are rare, and the live OCR threads stay undisturbed)."""
        with self._ocr_lock:
            if self._ocr is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR
                    self._ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
                except Exception:
                    self._ocr = False
            if self._ocr is False:
                return []
            k = max(1.0, 64.0 / max(1, img.shape[0]))
            if k > 1.0:
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            try:
                res, _ = self._ocr(img, use_cls=False)
            except Exception:
                return []
        return [(str(r[1]), float(r[2]), np.asarray(r[0], np.float32).reshape(4, 2) / k)
                for r in (res or [])]

    # -- GET --------------------------------------------------------------------

    def _get(self, path: str, q: dict):
        if path == "state":
            return _json(200, self.state())
        if path == "frames":
            return _json(200, self.frames(q))
        if path.startswith("snap/") and path.endswith(".jpg"):
            name = path[5:-4]
            thumb = name.endswith("_t")
            it = self.snapshots.get(int(name[:-2] if thumb else name))
            if it is None:
                return _err(404, "that frame has left the buffer")
            return 200, "image/jpeg", it["thumb" if thumb else "jpg"]
        if path.startswith("file/"):
            rel = path[5:]
            if not rel.startswith(("review/", "examples/")) or not rel.endswith(".jpg"):
                return _err(403, "not served")
            p = self.ld.resolve(rel)
            if p is None or not p.is_file():
                return _err(404, "not found")
            return 200, "image/jpeg", p.read_bytes()
        if path == "job":
            return _json(200, self.job.status() if self.job else {})
        return _err(404, "unknown endpoint")

    def state(self) -> dict:
        eng = self.get_engine()
        rec = self.ld.root / "recordings"
        files = list(rec.glob("*.ts"))
        return {
            "time": time.time(),
            "library": self.library.snapshot(),
            "review": self.reviews.pending(self.min_review_count),
            "samples": self.samples.summary(),
            "recordings": {"files": len(files), "gb": round(dir_bytes(rec) / 1e9, 2)},
            "models": self.models.status() if self.models else None,
            "job": self.job.status() if self.job else None,
            "engine": eng.learning_status() if eng is not None else None,
            "snapshots": len(self.snapshots.items),
        }

    def frames(self, q: dict) -> dict:
        items = self.snapshots.listing()
        if not items:
            return {"frames": [], "center": None}
        at = q.get("at", [None])[0]
        span = float(q.get("span", ["6"])[0])
        t_center = None
        if at:
            t_center = self.clock.t_at(float(at) / 1000.0)
        if t_center is None:
            # No programme date from the player: the newest frames, minus the
            # few seconds HLS playback runs behind.
            t_center = items[-1]["t"] - 8.0
        near = [it for it in items if abs(it["t"] - t_center) <= span / 2]
        if not near:
            near = sorted(items, key=lambda it: abs(it["t"] - t_center))[:12]
            near.sort(key=lambda it: it["t"])
        best = min(near, key=lambda it: abs(it["t"] - t_center))
        return {"frames": near, "center": best["id"]}

    # -- POST -------------------------------------------------------------------

    def _post(self, path: str, d: dict):
        if path == "mark":
            return self.mark(d)
        if path == "review":
            return self.review(d)
        if path == "brand":
            return self.brand(d)
        if path == "unignore":
            self.library.unignore(str(d["text"]))
            return _json(200, {"ok": True})
        if path == "train":
            if self.job is None:
                return _err(501, "retraining is not available here")
            ok = self.job.start() if d.get("action") == "start" else self.job.stop()
            return _json(200, {"ok": ok, "job": self.job.status()})
        if path == "model":
            if self.models is None:
                return _err(501, "no model registry")
            cur = self.models.activate(d.get("version"))
            weights = cur["weights"] if cur else self.models.base
            swapped = self.swap_model(weights) if self.swap_model else False
            return _json(200, {"current": cur["version"] if cur else "original",
                               "switching_now": bool(swapped)})
        return _err(404, "unknown endpoint")

    def _after_decision(self) -> None:
        """Entries the library now decides leave the review list."""
        self.reviews.drop_matching(self.library.classify)
        self.reviews.save()

    def _save_example(self, brand_id: str, img: np.ndarray) -> str | None:
        p = self.ld.root / "examples" / brand_id / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.jpg"
        return self.ld.rel(p) if imwrite(p, img, 92) else None

    def mark(self, d: dict):
        """The operator drew a box on an original frame: hide or keep that logo."""
        action = d.get("action")
        if action not in ACTIONS:
            return _err(400, f"action must be one of {ACTIONS}")
        snap = self.snapshots.get(int(d["snap"]))
        if snap is None:
            return _err(410, "that frame has left the buffer; pick a newer one")
        frame = self.snapshots.frame(snap["id"])
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in d["box"])
        x0, x1 = sorted((max(0.0, min(w, x0)), max(0.0, min(w, x1))))
        y0, y1 = sorted((max(0.0, min(h, y0)), max(0.0, min(h, y1))))
        if x1 - x0 < 8 or y1 - y0 < 6:
            return _err(400, "draw a larger box around the logo")
        crop = frame[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))].copy()
        # Read a margin around the box too: a box drawn around one logo often
        # clips the word ("RIMATCH"), and the name should be the whole word.
        px, py = 0.35 * (x1 - x0), 0.25 * (y1 - y0)
        ex0, ey0 = int(max(0, x0 - px)), int(max(0, y0 - py))
        ex1, ey1 = int(min(w, x1 + px)), int(min(h, y1 + py))
        box_q = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) - [ex0, ey0]
        read = [(t, c) for t, c, q in self._read_texts(frame[ey0:ey1, ex0:ex1])
                if containment(q, box_q) >= 0.35]
        texts = [t for t, c in read if c >= 0.6 and len(norm(t)) >= 3]
        name = str(d.get("name") or "").strip()
        aliases = []
        for t in texts:
            cls, bid = self.library.classify(t)
            if cls == "ignore" and norm(t) in self.library.ignored:
                continue
            if bid not in (None, "builtin") and self.library.brands.get(bid, {}).get("name") != name:
                continue                       # a neighbouring brand bleeding into the box
            aliases.append(t)
        picture = not aliases
        if not name:
            if aliases:
                name = max(((t, c) for t, c in read if t in aliases), key=lambda tc: tc[1])[0]
            else:
                self._logo_n += 1
                name = f"Logo {len(self.library.brands) + self._logo_n}"
        b = self.library.upsert(name, action, aliases, picture=picture if action == "hide" else None,
                                source="operator", name_is_alias=bool(aliases) or bool(d.get("name")))
        ex = self._save_example(b["id"], crop)
        if ex:
            self.library.add_example(b["id"], ex)
        quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        self.samples.add(frame, [{"quad": quad, "label": action, "brand": b["id"], "texts": texts,
                                  "by": "operator"}], "operator",
                         meta={"snap": snap["id"], "idx": snap["idx"]})
        eng = self.get_engine()
        live = False
        if eng is not None:
            live = eng.mark(snap["idx"], quad, "bet" if action == "hide" else "neg", b["id"],
                            generation=snap["session"])
        self._after_decision()
        how = ("by its name" if aliases else "by its picture") if action == "hide" else ""
        return _json(200, {"brand": self.library.brands.get(b["id"], b), "read": read,
                           "matched": how, "applied_to_this_shot": live})

    def review(self, d: dict):
        action = d.get("action")
        if action not in (*ACTIONS, "ignore"):
            return _err(400, "action must be hide, keep or ignore")
        it = self.reviews.take(str(d["id"]))
        if it is None:
            return _err(404, "already decided")
        if action == "ignore":
            self.library.ignore(it["key"])
            self.reviews.finish(it)
            self._after_decision()
            return _json(200, {"ignored": it["text"]})
        name = str(d.get("name") or "").strip() or it["text"]
        b = self.library.upsert(name, action, [it["text"]], source="review")
        added = 0
        for f in it["frames"]:
            src = self.ld.resolve(f["file"])
            if src is None or not src.exists():
                continue
            if added == 0:
                img = cv2.imread(str(src))
                if img is not None:
                    q = np.asarray(f["quad"], np.float32)
                    x0, y0 = np.maximum(q.min(axis=0), 0).astype(int)
                    x1, y1 = q.max(axis=0).astype(int)
                    ex = self._save_example(b["id"], img[y0:y1, x0:x1])
                    if ex:
                        self.library.add_example(b["id"], ex)
            if self.samples.add(None, [{"quad": f["quad"], "label": action, "brand": b["id"],
                                        "texts": [f["text"]], "by": "operator"}], "review",
                                meta={"w": f["w"], "h": f["h"]}, image_path=src):
                added += 1
        self.reviews.finish(it)
        self._after_decision()
        return _json(200, {"brand": self.library.brands.get(b["id"], b), "samples": added})

    def brand(self, d: dict):
        if "id" in d:
            bid = str(d["id"])
            if d.get("remove"):
                ok = self.library.remove(bid)
                return _json(200 if ok else 404, {"removed": ok})
            b = self.library.edit(bid, action=d.get("action"), name=d.get("name"),
                                  picture=d.get("picture"), add_alias=d.get("add_alias"),
                                  remove_alias=d.get("remove_alias"))
            if b is None:
                return _err(404, "no such brand")
            self._after_decision()
            return _json(200, {"brand": b})
        name = str(d.get("name") or "").strip()
        if len(norm(name)) < 3:
            return _err(400, "type the name as it is written on the boards (3+ letters)")
        action = d.get("action", "hide")
        if action not in ACTIONS:
            return _err(400, f"action must be one of {ACTIONS}")
        b = self.library.upsert(name, action, source="typed")
        self._after_decision()
        return _json(200, {"brand": b})
