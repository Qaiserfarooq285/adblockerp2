#!/usr/bin/env python3
"""The per-frame betting-ad renderer, shared by files and live streams.

    analyse (frame t):   camera motion -> detect (every det_interval frames)
                         -> track (homography predict + Kalman correct)
                         -> schedule brand OCR for new boards
    render  (frame t-D): panels for that frame -> protect overlay/pitch/goal
                         -> feather -> subtract people -> composite

The two halves run D = delay_frames apart.  Every frame is analysed as it
arrives but painted D frames later, by which time the tracker already knows
what happened next.  That short look-ahead is what removes the remaining
visible artefacts of a purely causal tracker:

  * a board is painted from the first frame it was visible, not from the frame
    its detection count reached the confirmation threshold -- the earlier
    frames are back-filled through the inverse camera homographies;
  * when a board stops being detected (an LED rotating to another advert), the
    panel ends right after the last detection instead of coasting on;
  * OCR has D frames to answer before a board is first shown.

D frames of latency is nothing next to HLS delivery (~30 s end to end) and is
irrelevant for files.  With delay_frames=0 the engine degrades gracefully to a
causal tracker.

v2_render.py uses this for files (--tracker homography, the default) and
live_stream.py for live streams; both see exactly the same behaviour.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

from board_tracker import BoardTrack, BoardTracker, TrackerConfig
from geometry import (CameraMotionEstimator, Motion, OnlineOverlayEstimator,
                      quad_axes, visible_fraction, warp_quad)
from v2_render import (PAINT_COLOR_BGR, _crop_for_guard, _get_ocr, _ocr_texts,
                       canonical, crossing_structure, merge_overlapping, pitch_mask,
                       pitch_reference, quad_from_mask, quad_shape, strips_from_frame)


def parse_imgsz(v):
    """1920 -> 1920; '1088x1920' (h x w) -> [1088, 1920]; lists pass through.

    A TensorRT engine has a fixed input shape and Ultralytics letterboxes to
    exactly `imgsz`, so an engine exported at 1088x1920 must be run with that
    pair, not with 1920 (which would mean 1920x1920).
    """
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    s = str(v).lower().replace(" ", "")
    if "x" in s:
        h, w = s.split("x", 1)
        return [int(h), int(w)]
    return int(float(s))


def _device(v):
    v = str(v).strip()
    return None if v in ("", "auto", "None") else v


def parse_boxes(v) -> list[list[float]]:
    """'x0,y0,x1,y1;x0,y0,x1,y1' (fractions of the frame) -> list of boxes."""
    if not v:
        return []
    if isinstance(v, (list, tuple)):
        return [list(map(float, b)) for b in v]
    return [[float(x) for x in part.split(",")] for part in str(v).split(";") if part.strip()]


# ------------------------------------------------------------------ detection

class BoardDetector:
    """Ad-board detections as clean strip quads.

    Identical to the per-frame detection in v2_render.process_legacy -- the
    shape filter, mask-level NMS and strip fitting were tuned on this footage
    and are not touched here.  Two differences: masks come off the GPU as
    uint8 instead of float32 (a quarter of the transfer), and the OCR brand
    guard is no longer applied per frame; it runs once per *track* instead
    (see AdBlockEngine._ocr_schedule), which removes OCR flip-flop as a
    source of blinking and most of the OCR cost.
    """

    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        self.model = YOLO(cfg["ad_weights"], task="segment")
        self.imgsz = parse_imgsz(cfg["ad_imgsz"])
        self.conf = float(cfg["render_conf"])
        self.min_aspect = float(cfg["min_aspect"])
        self.max_thick_frac = float(cfg["max_thick_frac"])
        self.device = _device(cfg.get("device", ""))
        self.half = bool(int(cfg.get("half", 1)))

    def __call__(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        h, w = frame.shape[:2]
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False,
                                 retina_masks=True, device=self.device, half=self.half)[0]
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            return []
        masks = res.masks.data.gt(0.5).byte().cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        raw, cf = [], []
        for mk, c in zip(masks, confs):
            mm = mk if mk.shape == (h, w) else cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
            q = quad_from_mask(mm)
            if q is None:
                continue
            length, thick = quad_shape(q)
            if thick < 3 or length / thick < self.min_aspect or thick > self.max_thick_frac * h:
                continue
            raw.append(mm)
            cf.append(float(c))
        kept = merge_overlapping(raw, cf)
        conf_of = max((g["conf"] for g in kept), default=0.0)
        out = []
        for q in strips_from_frame([g["mask"] for g in kept], (h, w), min_aspect=self.min_aspect):
            length, thick = quad_shape(q)
            if thick < 3 or length / thick < self.min_aspect or thick > self.max_thick_frac * h:
                continue
            # Per-strip confidence (the legacy path gave every strip the frame's
            # max): the best proposal under the strip's centre.  It only gates
            # track *birth* (trk_birth_conf), never sustaining a known board.
            cx, cy = np.clip(q.mean(axis=0), [0, 0], [w - 1, h - 1]).astype(int)
            c = max((g["conf"] for g in kept if g["mask"][cy, cx]), default=conf_of)
            out.append((q, float(c)))
        return out


# --------------------------------------------------------------------- people

class PersonProtector:
    """Masks of people standing in front of a painted board.

    Runs on the frame being *composited*, restricted to a region around the
    panels about to be drawn: a person only matters where a panel is, the crop
    is upscaled so small far-side figures are found (the v1 pipeline measured
    +32% person pixels at the board from this), and nothing runs at all on
    frames without panels.  Masks from the last `person_union` frames are kept,
    advanced by the camera homography, so a player whose torso the model
    misses for one frame is still protected (people always win).
    """

    def __init__(self, cfg: dict, model=None):
        self.weights = str(cfg["person_weights"])
        if model is None:
            from ultralytics import YOLO
            model = YOLO(self.weights, task="segment")
        self.model = model
        self.conf = float(cfg["person_conf"])
        self.imgsz_cfg = parse_imgsz(cfg["person_imgsz"])
        self.fixed = self.weights.endswith(".engine") and not int(cfg.get("person_dynamic", 0))
        self.upscale = float(cfg["roi_upscale"])
        self.roi = bool(int(cfg["roi_person"]))
        self.union = max(1, int(cfg["person_hist"]))
        self.every = max(1, int(cfg["person_every"]))
        self.device = _device(cfg.get("device", ""))
        self.half = bool(int(cfg.get("half", 1)))
        d = int(cfg["person_dilate"])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)) if d > 0 else None
        self.age: np.ndarray | None = None     # frames since a person covered each pixel
        self._n = 0

    def skip(self) -> None:
        """No panels this frame: the history would go stale unwarped, drop it."""
        self.age = None

    def _search_box(self, quads, w, h):
        if not self.roi:
            return 0, 0, w, h
        pts = np.concatenate([np.asarray(q, np.float32).reshape(4, 2) for q in quads])
        thick = max(quad_axes(q)[4] for q in quads)
        # A person in front of a board has feet on the grass below it and a
        # body reaching well above it; the crop must hold the whole figure.
        pad_y = max(3.0 * thick, 0.12 * h)
        pad_x = 0.04 * w
        x0 = int(max(0, pts[:, 0].min() - pad_x))
        x1 = int(min(w, pts[:, 0].max() + pad_x))
        y0 = int(max(0, pts[:, 1].min() - pad_y))
        y1 = int(min(h, pts[:, 1].max() + pad_y))
        if (x1 - x0) * (y1 - y0) > 0.6 * w * h:
            return 0, 0, w, h
        return x0, y0, x1, y1

    def _imgsz(self, bw: int, bh: int):
        if self.fixed:
            return self.imgsz_cfg
        cap = self.imgsz_cfg if isinstance(self.imgsz_cfg, int) else max(self.imgsz_cfg)
        side = int(np.clip(max(bw, bh) * self.upscale, 320, cap))
        return int(np.ceil(side / 32.0) * 32)

    @staticmethod
    def _in_front(pts: np.ndarray, quads, slack: float = 6.0) -> bool:
        """Same test as v2_render.PersonMasker._in_front, on a polygon: a
        person in front stands on the pitch, so their lowest point falls below
        the board's bottom edge; someone behind the rail is cut off by it."""
        foot_y = float(pts[:, 1].max())
        low = pts[pts[:, 1] > foot_y - 5]
        cx = float(low[:, 0].mean()) if len(low) else float(pts[:, 0].mean())
        for q in quads:
            c = canonical(q)
            (x0, y0), (x1, y1) = c[3], c[2]
            if not (min(x0, x1) - 40 <= cx <= max(x0, x1) + 40):
                continue
            t = 0.0 if abs(x1 - x0) < 1e-3 else (cx - x0) / (x1 - x0)
            board_bottom = y0 + (y1 - y0) * float(np.clip(t, 0.0, 1.0))
            if foot_y > board_bottom - slack:
                return True
        return False

    def __call__(self, frame, quads, box, H, shot_start) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.age is None or shot_start or self.age.shape != (h, w):
            self.age = np.full((h, w), 255, np.uint8)
        elif H is not None:
            self.age = cv2.warpPerspective(self.age, H, (w, h), flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        cv2.add(self.age, 1, dst=self.age)             # saturates at 255
        self._n += 1
        if self._n % self.every == 0 or shot_start:
            rx0, ry0, rx1, ry1 = self._search_box(quads, w, h)
            crop = np.ascontiguousarray(frame[ry0:ry1, rx0:rx1])
            r = self.model.predict(crop, conf=self.conf, imgsz=self._imgsz(rx1 - rx0, ry1 - ry0),
                                   classes=[0], verbose=False, retina_masks=False,
                                   device=self.device, half=self.half)[0]
            if r.masks is not None:
                cur = np.zeros((ry1 - ry0, rx1 - rx0), np.uint8)
                off = np.float32([rx0, ry0])
                for poly in r.masks.xy:
                    if len(poly) < 3:
                        continue
                    if not self._in_front(np.asarray(poly, np.float32) + off, quads):
                        continue
                    cv2.fillPoly(cur, [np.round(poly).astype(np.int32)], 1)
                if cur.any():
                    if self.kernel is not None:
                        cur = cv2.dilate(cur, self.kernel)
                    self.age[ry0:ry1, rx0:rx1][cur > 0] = 0
        x0, y0, x1, y1 = box
        return self.age[y0:y1, x0:x1] < self.union


# ---------------------------------------------------- pitch / goal protection

class StaticProtection:
    """Pitch and goal-frame protection, steadied in world coordinates.

    Both are per-pixel decisions (turf colour, white structure crossing the
    board) and both wobble a little from frame to frame at their boundaries,
    which reads as speckle flicker on the panel.  They belong to the static
    world, so the decision is accumulated at half resolution, carried by the
    camera homography, and switched with hysteresis: a structure must be seen
    on two frames to open a hole and must vanish for two to close it.  People
    are NOT handled here -- they move on their own and always win at once.
    """

    def __init__(self, cfg: dict):
        self.avoid_pitch = bool(int(cfg["avoid_pitch"]))
        self.protect_goal = bool(int(cfg["protect_goal"]))
        self.ema = float(cfg["protect_ema"])
        self.on = float(cfg["protect_on"])
        self.off = float(cfg["protect_off"])
        self.acc: np.ndarray | None = None
        self.state: np.ndarray | None = None
        self._ref = None             # turf colour, measured on the whole frame
        self._ref_age = 10 ** 9

    def skip(self) -> None:
        self.acc = self.state = None

    def _turf(self, frame: np.ndarray, shot_start: bool):
        """Turf colour of this shot: whole-frame, subsampled, refreshed ~1/s.

        Measuring it inside the small crop around the boards would be both
        slower per pixel and wrong -- a crop that is mostly advert holds too
        little grass for pitch_mask's own "is there a pitch" test.
        """
        self._ref_age += 1
        if shot_start or self._ref_age >= 30:
            self._ref = pitch_reference(frame, stride=4)
            self._ref_age = 0
        return self._ref

    def __call__(self, frame, box, area_u8, H, shot_start, thick) -> np.ndarray | None:
        if not (self.avoid_pitch or self.protect_goal):
            return None
        h, w = frame.shape[:2]
        hw, hh = max(1, w // 2), max(1, h // 2)
        fresh = self.acc is None or shot_start or self.acc.shape != (hh, hw)
        if fresh:
            self.acc = np.zeros((hh, hw), np.float32)
            self.state = np.zeros((hh, hw), np.uint8)
        elif H is not None:
            S = np.diag([0.5, 0.5, 1.0])
            Hh = S @ H @ np.linalg.inv(S)
            self.acc = cv2.warpPerspective(self.acc, Hh, (hw, hh), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            self.state = cv2.warpPerspective(self.state, Hh, (hw, hh), flags=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        x0, y0, x1, y1 = box
        # goal posts must be seen standing above/below the board to count
        pad = int(1.5 * thick) + 4
        cx0, cy0, cx1, cy1 = x0, max(0, y0 - pad), x1, min(h, y1 + pad)
        ctx = frame[cy0:cy1, cx0:cx1]
        area_ctx = np.zeros(ctx.shape[:2], np.uint8)
        area_ctx[y0 - cy0:y1 - cy0, :] = area_u8
        cur = np.zeros(ctx.shape[:2], np.uint8)
        if self.avoid_pitch:
            ref = self._turf(frame, shot_start)
            if ref is not None:
                cur |= pitch_mask(ctx, ref=ref)
        if self.protect_goal:
            cur |= (crossing_structure(ctx, area_ctx) > 0).astype(np.uint8)
        cur = cur[y0 - cy0:y1 - cy0, :]

        hx0, hy0 = x0 // 2, y0 // 2
        hx1, hy1 = max(hx0 + 1, min(hw, (x1 + 1) // 2)), max(hy0 + 1, min(hh, (y1 + 1) // 2))
        cur_h = cv2.resize(cur.astype(np.float32), (hx1 - hx0, hy1 - hy0), interpolation=cv2.INTER_AREA)
        if fresh:
            self.acc[hy0:hy1, hx0:hx1] = cur_h
        else:
            self.acc *= (1.0 - self.ema)                 # decays where nothing is seen
            self.acc[hy0:hy1, hx0:hx1] += self.ema * cur_h
        a = self.acc[hy0:hy1, hx0:hx1]
        st = self.state[hy0:hy1, hx0:hx1]
        st[:] = ((a >= self.on) | ((st > 0) & (a >= self.off))).astype(np.uint8)
        return cv2.resize(st, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0


# ------------------------------------------------------------------------ OCR

def ocr_verdict(crop: np.ndarray) -> str:
    """'bet' | 'neg' | 'unknown' -- the same precedence as v2_render.is_betting_board."""
    texts = _ocr_texts(crop)
    if not texts:
        return "unknown"
    from v2_brands import is_betting, is_non_betting
    if any(is_betting(t) for t in texts):
        return "bet"
    if any(is_non_betting(t) for t in texts):
        return "neg"
    return "unknown"


class OcrWorker:
    """Brand OCR off the frame loop (live) or inline (files, deterministic)."""

    def __init__(self, async_mode: bool):
        self.available = _get_ocr() is not None
        self.async_mode = async_mode and self.available
        self.results: "queue.Queue[tuple[int, str]]" = queue.Queue()
        self.jobs: "queue.Queue" = queue.Queue(maxsize=16)
        self._t = None
        if self.async_mode:
            self._t = threading.Thread(target=self._run, name="ocr", daemon=True)
            self._t.start()

    def submit(self, tid: int, crop: np.ndarray) -> bool:
        if not self.available:
            return False
        if not self.async_mode:
            self.results.put((tid, ocr_verdict(crop)))
            return True
        try:
            self.jobs.put_nowait((tid, crop))
            return True
        except queue.Full:
            return False

    def poll(self) -> list[tuple[int, str]]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        while True:
            tid, crop = self.jobs.get()
            if tid is None:
                return
            try:
                v = ocr_verdict(crop)
            except Exception:
                v = "unknown"
            self.results.put((tid, v))

    def close(self) -> None:
        if self._t is not None:
            self.jobs.put((None, None))


# --------------------------------------------------------------------- engine

@dataclass
class Slot:
    idx: int
    frame: np.ndarray
    motion: Motion
    shot_start: bool
    det_ran: bool
    meta: object = None
    _hinv: np.ndarray | None = None

    def hinv(self) -> np.ndarray | None:
        if self._hinv is None and self.motion.H is not None:
            self._hinv = np.linalg.inv(self.motion.H)
        return self._hinv


@dataclass
class Panel:
    tid: int
    quad: np.ndarray
    alpha: float
    kind: str          # det | coast | backfill


class AdBlockEngine:
    def __init__(self, cfg: dict, width: int, height: int,
                 overlay_mask: np.ndarray | None = None, live: bool = False,
                 debug: bool = False, panel_log: str | None = None,
                 detector=None, person_model=None, fps: float = 30.0):
        """`detector` / `person_model` override the YOLO models (tests, dry runs)."""
        self.cfg = cfg
        self.w, self.h = int(width), int(height)
        self.debug = debug
        self.detector = detector if detector is not None else BoardDetector(cfg)
        self.people = PersonProtector(cfg, model=person_model)
        self.static = StaticProtection(cfg)
        boxes = parse_boxes(cfg.get("motion_exclude_boxes")) if live else None
        self.motion = CameraMotionEstimator(self.w, self.h, work_width=int(cfg["motion_width"]),
                                            cut_hist_hard=float(cfg["cut_hist_hard"]),
                                            cut_hist_soft=float(cfg["cut_hist_soft"]),
                                            exclude_boxes=boxes)
        self.tracker = BoardTracker(TrackerConfig.from_dict(cfg), self.w, self.h)
        self.protect_overlay = bool(int(cfg["protect_overlay"]))
        self.overlay = overlay_mask if self.protect_overlay else None
        self.overlay_est = None
        if self.overlay is not None:
            self.motion.set_static_mask(self.overlay)
        elif live and self.protect_overlay:
            self.overlay_est = OnlineOverlayEstimator(self.w, self.h)
        self.ocr = OcrWorker(bool(int(cfg["ocr_async"]))) if int(cfg["ocr_guard"]) else None
        if self.ocr is not None and not self.ocr.available:
            print("[engine] OCR brand guard unavailable (pip install rapidocr_onnxruntime); "
                  "boards are painted on model evidence alone")
        self.delay = max(0, int(cfg["delay_frames"]))
        self.det_interval = max(1, int(cfg["det_interval"]))
        self.buf: deque[Slot] = deque()
        self.idx = 0
        self._last_det = -10 ** 9
        self._force_cut = False
        self.stats: Counter = Counter()
        self.timing: dict[str, float] = {}
        self._log = open(panel_log, "w") if panel_log else None
        if self._log is not None:
            self._log.write(json.dumps({"header": True, "w": self.w, "h": self.h,
                                        "fps": float(fps), "tracker": "homography"}) + "\n")
        # Tracker state is shared between the analysis and render halves; the
        # lock is held only for the cheap bookkeeping, never for inference or
        # compositing.
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._async = False
        self._rt: threading.Thread | None = None
        self._flushing = False
        self._stop = False
        self.out: "queue.Queue[tuple[np.ndarray, object]]" | None = None
        self.max_backlog = 60
        # A frame is rendered up to delay + backlog frames after it was
        # analysed; its per-track quads must still be in history by then.
        tc = self.tracker.cfg
        tc.history = max(tc.history, self.delay + self.max_backlog + 16)

    # -- public API -----------------------------------------------------------

    def warmup(self) -> None:
        """First CUDA inference is slow (context, cuDNN autotune); pay it now."""
        dummy = np.zeros((self.h, self.w, 3), np.uint8)
        self.detector(dummy)
        self.people.model.predict(dummy[: self.h // 3], conf=0.5, classes=[0], verbose=False,
                                  imgsz=self.people._imgsz(self.w, self.h // 3),
                                  device=self.people.device, half=self.people.half)

    def start_async(self, out: "queue.Queue | None" = None) -> "queue.Queue":
        """Render on a background thread (live).

        push() then only analyses and returns []; finished frames arrive, in
        order, on `out` (a new bounded queue if not given).  Analysis (motion,
        board detector) and rendering (person model, compositing) overlap, so
        a frame costs about max(analyse, render) instead of their sum.
        """
        self.out = out if out is not None else queue.Queue(maxsize=64)
        self._async = True
        self._rt = threading.Thread(target=self._render_loop, name="render", daemon=True)
        self._rt.start()
        return self.out

    def push(self, frame: np.ndarray, meta=None) -> list[tuple[np.ndarray, object]]:
        """Analyse one frame; return the frames (with their meta) now ready."""
        slot = self._analyse(frame, meta)
        with self._cv:
            self.buf.append(slot)
            if self._async:
                self._cv.notify_all()
                # Bounded backlog: if rendering falls far behind, hold the
                # producer here rather than let buffered 1080p frames pile up.
                while len(self.buf) > self.delay + self.max_backlog and not self._stop:
                    self._cv.wait(0.1)
                return []
        out = []
        while len(self.buf) > self.delay:
            out.append(self._emit())
        return out

    def flush(self, timeout: float | None = None) -> list[tuple[np.ndarray, object]]:
        """Emit everything still buffered (sync: returned; async: onto .out).

        In async mode this waits for the render thread, which in turn waits
        for whoever drains .out -- `timeout` bounds that at shutdown.
        """
        if self._async:
            deadline = None if timeout is None else time.monotonic() + timeout
            with self._cv:
                self._flushing = True
                self._cv.notify_all()
                while self.buf and not self._stop:
                    if deadline is not None and time.monotonic() > deadline:
                        break
                    self._cv.wait(0.1)
                self._flushing = False
            return []
        out = []
        while self.buf:
            out.append(self._emit())
        return out

    def discontinuity(self) -> list[tuple[np.ndarray, object]]:
        """Input restarted (reconnect): emit what is buffered, forget geometry."""
        out = self.flush()
        with self._lock:
            self.motion.reset()
            self._force_cut = True
        return out

    def close(self) -> None:
        if self._rt is not None:
            with self._cv:
                self._stop = True
                self._cv.notify_all()
            self._rt.join(timeout=5)
            self._rt = None
        if self.ocr is not None:
            self.ocr.close()
        if self._log is not None:
            self._log.close()
            self._log = None

    def _render_loop(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not (len(self.buf) > self.delay or
                                              (self._flushing and self.buf)):
                    self._cv.wait(0.1)
                if self._stop:
                    return
            item = self._emit()
            self.out.put(item)                     # blocks if the consumer is slow
            with self._cv:
                self._cv.notify_all()

    def set_det_interval(self, n: int) -> None:
        self.det_interval = max(1, int(n))

    def timing_summary(self) -> str:
        return " ".join(f"{k}={v:.1f}ms" for k, v in sorted(self.timing.items()))

    # -- analysis (frame t) ---------------------------------------------------

    def _tick(self, key: str, t0: float) -> float:
        now = time.perf_counter()
        ms = 1000.0 * (now - t0)
        self.timing[key] = ms if key not in self.timing else 0.9 * self.timing[key] + 0.1 * ms
        return now

    def _analyse(self, frame: np.ndarray, meta) -> Slot:
        t = self.idx
        t0 = time.perf_counter()
        with self._lock:
            prev = [tr.quad for tr in self.tracker.tracks if tr.alive]
        motion = self.motion.estimate(frame, prev)
        with self._lock:
            shot_start = t == 0 or motion.cut or self._force_cut
            if motion.cut or self._force_cut:
                self.tracker.reset(t)
                self.stats["cuts"] += int(motion.cut)
                self._force_cut = False
        t0 = self._tick("motion", t0)

        run_det = shot_start or (t - self._last_det) >= self.det_interval
        dets = None
        if run_det:
            dets = self.detector(frame)
            self._last_det = t
            t0 = self._tick("detect", t0)

        with self._lock:
            if run_det:
                self.stats["det_runs"] += 1
                self.stats["dets"] += len(dets)
            self.tracker.step(t, motion, dets)
            self._ocr_schedule(frame, t)
            if self.overlay_est is not None:
                self.overlay_est.update(frame, motion)
                m = self.overlay_est.mask()
                if m is not None and m is not self.overlay:
                    self.overlay = m
                    self.motion.set_static_mask(m)
            self.idx = t + 1          # the render side reads this as "now"
        self._tick("track", t0)
        return Slot(t, frame, motion, shot_start, run_det, meta)

    def _ocr_schedule(self, frame: np.ndarray, t: int) -> None:
        if self.ocr is None or not self.ocr.available:
            return
        cfg = self.cfg
        retry, recheck = int(cfg["ocr_retry"]), int(cfg["ocr_recheck"])
        max_attempts = int(cfg["ocr_max_attempts"])
        self._ocr_apply(t, retry, recheck, max_attempts)
        for tr in self.tracker.tracks:
            # Only read a board on a frame where it was actually detected: a
            # coasting quad is a prediction, and OCR on a stale crop is noise.
            if not tr.alive or tr.ocr_pending or t < tr.ocr_next or tr.last_det_frame != t:
                continue
            crop = _crop_for_guard(frame, tr.quad)
            if crop is None:
                tr.ocr_next = t + retry
                continue
            # An explicit copy: an unscaled crop is a VIEW of this frame, which
            # gets painted D frames from now -- possibly before the OCR thread
            # reads it.
            if self.ocr.submit(tr.id, crop.copy()):
                tr.ocr_pending = True
                tr.ocr_attempts += 1
                self.stats["ocr_jobs"] += 1
        if not self.ocr.async_mode:
            self._ocr_apply(t, retry, recheck, max_attempts)

    def _ocr_apply(self, t: int, retry: int, recheck: int, max_attempts: int) -> None:
        for tid, verdict in self.ocr.poll():
            tr = self.tracker.find(tid)
            if tr is None:
                continue
            tr.ocr_pending = False
            if verdict == "bet":
                tr.verdict = "bet"
                tr.ocr_next = t + recheck
            elif verdict == "neg":
                if tr.verdict != "neg":
                    self.stats["suppressed"] += 1
                tr.verdict = "neg"
                # An LED board rotates its advert: the same physical board that
                # showed a non-betting sponsor may show Betano next, so a
                # suppressed board keeps being re-read.
                tr.ocr_next = t + retry
            else:
                if tr.verdict == "pending":
                    tr.verdict = "unknown"     # fail-open, as the per-frame guard was
                tr.ocr_next = t + (retry if tr.ocr_attempts < max_attempts else recheck)

    # -- render (frame t - D) -------------------------------------------------

    def _drawable(self, tr: BoardTrack, now: int) -> bool:
        if tr.confirmed_frame is None or tr.verdict == "neg":
            return False
        if tr.verdict in ("bet", "unknown") or self.ocr is None or not self.ocr.available:
            return True
        return (now - tr.birth_frame) >= int(self.cfg["ocr_wait"])

    def _backfill(self, tr: BoardTrack, s: int) -> np.ndarray | None:
        """This board's quad on frame s < birth, through inverse camera motion."""
        b = tr.birth_frame
        if b - s > int(self.cfg["backfill_frames"]):
            return None
        q = tr.history.get(b)
        if q is None:
            return None
        base = self.buf[0].idx
        for k in range(b, s, -1):
            i = k - base
            if i <= 0 or i >= len(self.buf):
                return None
            sl = self.buf[i]
            if sl.shot_start or sl.hinv() is None:
                return None                     # never back-fill across a cut
            q = warp_quad(sl.hinv(), q)
            if q is None:
                return None
        return q if visible_fraction(q, self.w, self.h) >= 0.02 else None

    def _panels(self, slot: Slot) -> list[Panel]:
        s, now = slot.idx, self.idx - 1
        fade_in = int(self.cfg["fade_in"])
        out = []
        for tr in self.tracker.all_for_render():
            if not self._drawable(tr, now):
                continue
            if tr.end_frame is not None and s > tr.end_frame:
                continue
            q = tr.history.get(s)
            kind = "det" if s in tr.det_frames else "coast"
            if q is None:
                if s >= tr.birth_frame:
                    continue
                q = self._backfill(tr, s)
                kind = "backfill"
                if q is None:
                    continue
            alpha = 1.0
            if tr.first_drawn is None:
                tr.first_drawn = s
                # Popping in over a board that has been in plain view reads as
                # a blink; one that arrives with the shot or slides in from the
                # frame edge does not.
                tr.fade_in = (not slot.shot_start) and visible_fraction(q, self.w, self.h) > 0.9
            if tr.fade_in and fade_in > 0:
                alpha = min(alpha, (s - tr.first_drawn + 1) / (fade_in + 1.0))
            if tr.fade_from is not None and tr.end_frame is not None and s > tr.fade_from:
                alpha = min(alpha, (tr.end_frame + 1 - s) / float(tr.end_frame + 1 - tr.fade_from))
            if alpha > 0.01:
                out.append(Panel(tr.id, q, float(alpha), kind))
                self.stats[f"panel_{kind}"] += 1
        return out

    def _emit(self) -> tuple[np.ndarray, object]:
        t0 = time.perf_counter()
        with self._lock:
            slot = self.buf[0]
            panels = self._panels(slot)
        frame = slot.frame
        if panels:
            frame = self._composite(frame, panels, slot)
            self.stats["painted"] += 1
        else:
            self.people.skip()
            self.static.skip()
        if self.debug:
            self._draw_debug(frame, slot, panels)
        if self._log is not None:
            self._log.write(json.dumps({
                "f": slot.idx, "cut": bool(slot.shot_start), "det": bool(slot.det_ran),
                "H": None if slot.motion.H is None else np.round(slot.motion.H, 8).ravel().tolist(),
                "p": [[p.tid, p.kind, round(p.alpha, 3), np.round(p.quad, 2).ravel().tolist()]
                      for p in panels]}) + "\n")
        with self._lock:
            self.stats["frames"] += 1
            self.buf.popleft()
            self.tracker.prune(self.buf[0].idx if self.buf else self.idx)
        self._tick("render", t0)
        return frame, slot.meta

    def _composite(self, frame: np.ndarray, panels: list[Panel], slot: Slot) -> np.ndarray:
        h, w = frame.shape[:2]
        feather = int(self.cfg["feather"])
        pts = np.concatenate([p.quad for p in panels])
        pad = feather + 3
        x0 = int(max(0, np.floor(pts[:, 0].min()) - pad))
        y0 = int(max(0, np.floor(pts[:, 1].min()) - pad))
        x1 = int(min(w, np.ceil(pts[:, 0].max()) + pad))
        y1 = int(min(h, np.ceil(pts[:, 1].max()) + pad))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        bw, bh = x1 - x0, y1 - y0

        # Sub-pixel, anti-aliased edges.  Integer corners make every edge jump
        # a whole pixel back and forth as the board moves by fractions of one,
        # which is visible as a crawling outline even on a perfectly tracked
        # panel.
        cover = np.zeros((bh, bw), np.float32)
        tmp = np.zeros((bh, bw), np.uint8)
        org = np.float32([x0, y0])
        for p in panels:
            tmp[:] = 0
            cv2.fillConvexPoly(tmp, np.round((p.quad - org) * 16).astype(np.int32), 255,
                               lineType=cv2.LINE_AA, shift=4)
            np.maximum(cover, tmp.astype(np.float32) * (p.alpha / 255.0), out=cover)

        if self.overlay is not None:
            cover[self.overlay[y0:y1, x0:x1] > 0] = 0
        thick = max(quad_axes(p.quad)[4] for p in panels)
        area_u8 = (cover > 0).astype(np.uint8) * 255
        prot = self.static(frame, (x0, y0, x1, y1), area_u8, slot.motion.H, slot.shot_start, thick)
        if prot is not None:
            cover[prot] = 0
        if feather > 1:
            cover = cv2.GaussianBlur(cover, (feather | 1, feather | 1), 0)
        people = self.people(frame, [p.quad for p in panels], (x0, y0, x1, y1),
                             slot.motion.H, slot.shot_start)
        cover[people] = 0                         # people always win, hard edge

        a = cover[:, :, None]
        sub = frame[y0:y1, x0:x1].astype(np.float32)
        sub += (np.float32(PAINT_COLOR_BGR) - sub) * a
        frame[y0:y1, x0:x1] = np.clip(sub + 0.5, 0, 255).astype(np.uint8)
        return frame

    def _draw_debug(self, frame: np.ndarray, slot: Slot, panels: list[Panel]) -> None:
        colours = {"det": (0, 255, 0), "coast": (0, 165, 255), "backfill": (255, 255, 0)}
        drawn = set()
        for p in panels:
            drawn.add(p.tid)
            q = np.round(p.quad).astype(np.int32)
            cv2.polylines(frame, [q], True, colours[p.kind], 2, cv2.LINE_AA)
            cv2.putText(frame, f"{p.tid}{p.kind[0]}", tuple(q[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colours[p.kind], 1, cv2.LINE_AA)
        for tr in self.tracker.all_for_render():
            q = tr.history.get(slot.idx)
            if tr.id in drawn or q is None:
                continue
            col = (0, 0, 255) if tr.verdict == "neg" else (160, 160, 160)
            cv2.polylines(frame, [np.round(q).astype(np.int32)], True, col, 1, cv2.LINE_AA)
        m = slot.motion
        hud = (f"f{slot.idx} {m.method} inl={m.inliers} shift={m.shift_px:.1f}px "
               f"res={m.residual_px:.2f} det={'Y' if slot.det_ran else '-'} "
               f"tracks={len(self.tracker.tracks)}{' CUT' if slot.shot_start else ''}")
        cv2.putText(frame, hud, (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud, (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
