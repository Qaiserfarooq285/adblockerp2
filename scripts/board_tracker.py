#!/usr/bin/env python3
"""Board tracks that ride the camera and are corrected by the detector.

Each track is one physical ad board.  Between detections it is carried by the
frame-to-frame homography from geometry.py -- for a rotating/zooming broadcast
camera that *is* the board's image motion -- and every detection then nudges
it through a scalar Kalman gain.  That split fixes the reported artefacts:

  moving box  The previous tracker EMA'd the *detection* (alpha 0.2), so it
              reproduced the detector's pixel jitter while nothing moved, and
              lagged behind every pan by several frames.  Here the motion comes
              from the camera model and the detector only corrects, so jitter is
              divided down by the gain and a pan causes no lag at all.

  blinking    The previous tracker dropped a board after 5 missed frames and
              re-created it from scratch.  Here a confirmed board coasts on the
              homography through detector dropouts (motion blur, a player
              crossing, a far-side board flickering around the threshold).  It
              coasts longer while the detector is plausibly blind (fast pan,
              board at the frame edge) and only briefly while the board is
              plainly in view -- which is also how an LED board rotating to a
              non-betting advert is let go quickly.

Corrections are split into two directions, because the detector is good at
one and bad at the other:

  across the rail  (top/bottom edge) -- reliable to a few pixels; a standard
                   gated Kalman update.
  along the rail   (the two ends)    -- unreliable: occlusion truncates them,
                   the frame edge censors them, and split/merge decisions move
                   them.  Growing is accepted quickly (hiding more of a betting
                   board is the safe direction), shrinking slowly, and an end
                   the detector reports *at the frame border* is treated as
                   "the board continues at least this far", never as its end.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from geometry import (Motion, align_quad, containment, convex_iou, is_convex,
                      polygon_area, quad_axes, visible_fraction, warp_quad)


@dataclass
class TrackerConfig:
    min_hits: int = 2            # detections needed to confirm a board...
    confirm_window: int = 4      # ...before this many detector runs pass without one
    birth_conf: float = 0.12     # new boards need this much; sustaining one needs less
    match_iou: float = 0.20      # track <-> detection association
    absorb_frac: float = 0.60    # a detection this much inside a track is a piece of it
    merge_iou: float = 0.55      # two tracks this similar are one board
    coast_short: int = 8         # detector runs a board may go unseen while plainly visible
    coast_long: int = 40         # ...while the detector is plausibly blind
    fast_pan_px: float = 14.0    # per-frame camera shift counted as a fast pan
    edge_margin_px: float = 24.0
    min_visible: float = 0.05    # a board this far off screen has left the shot
    q_base: float = 0.05         # px^2/frame model noise with a still camera
    q_pan: float = 0.03          # extra std (px) per px of camera shift
    q_residual: float = 0.5      # * (homography fit residual)^2
    q_unknown: float = 40.0      # px^2 when the frame's motion could not be measured
    max_unknown_motion: int = 3  # frames to dead-reckon on the last homography
    r_across: float = 6.0        # detector noise across the rail (px^2)
    r_along: float = 60.0        # detector noise along the rail (px^2)
    grow_gain: float = 0.30      # minimum gain for an end moving outward
    gate: float = 3.0            # innovation gate, in sigmas
    resync_after: int = 3        # consecutive gated updates before trusting the detector
    border_px: float = 6.0       # a detected corner this close to the frame edge is censored,
    border_frac: float = 0.012   # ...plus this fraction of the board's length (see correct())
    border_overshoot_px: float = 32.0   # ...and the track's end is placed this far past it
    end_tail: int = 3            # frames drawn (fading) after a lost board's last detection
    history: int = 90            # frames of per-frame quads kept for the delayed renderer

    @classmethod
    def from_dict(cls, d: dict) -> "TrackerConfig":
        """Pick `trk_<field>` keys out of the flat renderer config.

        Prefixed on purpose: the legacy tracker's `min_hits`/`max_hold` live in
        the same dict with different meanings and must not leak in here.
        """
        kw = {f: type(getattr(cls, f))(d[f"trk_{f}"])
              for f in cls.__dataclass_fields__ if f"trk_{f}" in d}
        return cls(**kw)


def _huber(z: np.ndarray, gate) -> np.ndarray:
    return np.minimum(1.0, np.asarray(gate) / np.maximum(np.abs(z), 1e-9))


class BoardTrack:
    def __init__(self, tid: int, quad: np.ndarray, conf: float, frame_idx: int,
                 cfg: TrackerConfig):
        self.id = tid
        self.quad = np.asarray(quad, np.float32).reshape(4, 2).copy()
        self.P_n = cfg.r_across            # corner variance across the rail
        self.P_d = cfg.r_along             # ...and along it
        self.conf = float(conf)
        self.state = "tentative"           # tentative | confirmed | dead
        self.hits = 1
        self.runs_since_hit = 0
        self.miss_budget = 0.0             # >= 1.0 means the coast allowance is spent
        self.disagree = 0
        # per corner: its position along the rail has only ever been inferred
        # from the frame border, never actually seen
        self.censored = np.zeros(4, bool)
        self.birth_frame = frame_idx
        self.last_det_frame = frame_idx
        self.confirmed_frame: int | None = None
        self.end_frame: int | None = None  # last frame this board may be drawn on
        self.end_reason = ""
        self.fade_from: int | None = None  # fade out after this frame (lost boards)
        # brand verdict, owned by the engine's OCR scheduler
        self.verdict = "pending"           # pending | bet | neg | unknown
        self.brand: str | None = None      # Brand Library id behind the verdict
        self.ocr_texts: list[str] = []     # what OCR last read on this board
        self.ocr_attempts = 0
        self.ocr_next = frame_idx
        self.ocr_pending = False
        # who found it: "model" (a detection) or "scan" / "picture" /
        # "operator" (BoardTracker.inject), and the model's class evidence:
        # votes[0] betting board, votes[1] any other ad board (2-class models)
        self.source = "model"
        self.votes = [0.0, 0.0]
        self.last_model_frame: int | None = None
        # an injected board is held this long without detector support
        self.anchor_until: int | None = None
        self.scan_misses = 0               # scans in a row that looked and did not see it
        # delayed-render bookkeeping
        self.history: dict[int, np.ndarray] = {frame_idx: self.quad.copy()}
        self.det_frames: set[int] = {frame_idx}
        self.first_drawn: int | None = None
        self.fade_in = False

    @property
    def alive(self) -> bool:
        return self.state != "dead"

    def vote(self, cls: int, conf: float, frame_idx: int) -> None:
        self.votes[1 if cls else 0] += float(conf)
        self.last_model_frame = frame_idx

    @property
    def model_bet(self) -> bool:
        """The model itself calls this a betting board (always, with a
        single-class model; by majority with a two-class one)."""
        return self.last_model_frame is not None and self.votes[0] >= self.votes[1]

    def correct(self, det: np.ndarray, conf: float, frame_idx: int,
                w: int, h: int, cfg: TrackerConfig) -> None:
        det = align_quad(det, self.quad)
        pred = self.quad
        c, d, n, _, _ = quad_axes(pred)
        innov = det - pred
        a_n = innov @ n
        a_d = innov @ d

        s_n = self.P_n + cfg.r_across
        s_d = self.P_d + cfg.r_along
        k_n = self.P_n / s_n
        k_d0 = self.P_d / s_d

        w_n = _huber(a_n / np.sqrt(s_n), cfg.gate)
        side = np.sign((pred - c) @ d)             # -1 at one end, +1 at the other
        grow = a_d * side > 0
        w_d = _huber(a_d / np.sqrt(s_d), np.where(grow, 2.0 * cfg.gate, cfg.gate))
        k_d = np.where(grow, max(2.0 * k_d0, cfg.grow_gain), k_d0)

        # v2_render.fit_strip trims the outer 0.5% of mask pixels at each end
        # of the rail, so a board cut off by the frame edge is reported up to
        # ~1% of its length INSIDE that edge (~20 px on a full-width rail).  A
        # fixed pixel margin took that for the board's real end and left a
        # sliver of advert at the edge of the picture, sliding with every pan.
        det_len = quad_axes(det)[3]
        b = cfg.border_px + cfg.border_frac * det_len
        at_edge = ((det[:, 0] <= b) | (det[:, 0] >= w - 1 - b) |
                   (det[:, 1] <= b) | (det[:, 1] >= h - 1 - b))
        # At the frame border the detector reports where the PICTURE ends, not
        # where the board does: it says "at least this far".  Extend to it at
        # once if we fall short (a sliver of advert at the screen edge otherwise
        # survives for a second), and overshoot a little: an end pinned exactly
        # to the border is pulled back on screen by the very next pan step and
        # re-snapped, a sawtooth at the edge of the picture.  Past the border
        # the homography carries it and nothing is drawn.  Never pull a track
        # back in from beyond the border on the detector's word.
        snap = at_edge & grow
        k_d = np.where(at_edge & ~grow, 0.0, k_d)
        k_d = np.where(snap, 1.0, k_d)
        w_d = np.where(at_edge, 1.0, w_d)
        overshoot = np.where(snap, side * cfg.border_overshoot_px, 0.0)
        # An end that was only ever inferred from the border and has now come
        # into view is measured for the first time: let it shrink quickly.
        revealed = self.censored & ~at_edge & ~grow
        k_d = np.where(revealed, max(k_d0, 0.5), k_d)
        w_d = np.where(revealed, 1.0, w_d)
        self.censored = np.where(snap, True, np.where(at_edge, self.censored, False))

        new = (pred + (k_n * w_n * a_n)[:, None] * n[None, :]
               + (k_d * w_d * a_d + overshoot)[:, None] * d[None, :]).astype(np.float32)

        inner = ~at_edge
        gated = float(w_n.min()) < 0.5 or (bool(inner.any()) and float(w_d[inner].min()) < 0.35)
        self.disagree = self.disagree + 1 if gated else 0
        if self.disagree >= cfg.resync_after or not is_convex(new) or polygon_area(new) < 20:
            # Repeated large disagreement means the PREDICTION is wrong (a long
            # coast through a zoom, a moving camera's parallax) -- believe the
            # detector outright rather than crawl toward it.
            new = det.astype(np.float32)
            self.P_n, self.P_d, self.disagree = cfg.r_across, cfg.r_along, 0
            self.censored = at_edge.copy()
        else:
            self.P_n = max(1e-3, (1.0 - k_n * float(w_n.mean())) * self.P_n)
            self.P_d = max(1e-3, (1.0 - k_d0 * float(w_d.mean())) * self.P_d)

        self.quad = new
        self.conf = 0.7 * self.conf + 0.3 * float(conf)
        self.hits += 1
        self.runs_since_hit = 0
        self.miss_budget = 0.0
        self.last_det_frame = frame_idx
        self.det_frames.add(frame_idx)

    def resync(self, det: np.ndarray, frame_idx: int, cfg: TrackerConfig) -> None:
        self.quad = align_quad(det, self.quad)
        self.P_n, self.P_d, self.disagree = cfg.r_across, cfg.r_along, 0
        self.hits += 1
        self.runs_since_hit = 0
        self.miss_budget = 0.0
        self.last_det_frame = frame_idx
        self.det_frames.add(frame_idx)


class BoardTracker:
    def __init__(self, cfg: TrackerConfig, width: int, height: int):
        self.cfg = cfg
        self.w, self.h = int(width), int(height)
        self.tracks: list[BoardTrack] = []
        self.graveyard: list[BoardTrack] = []   # dead, but frames may still be drawn
        self.events: Counter = Counter()
        self._next = 0
        self._last_H: np.ndarray | None = None
        self._unknown = 0

    # -- queries --------------------------------------------------------------

    def alive(self) -> list[BoardTrack]:
        return [t for t in self.tracks if t.alive]

    def all_for_render(self):
        return self.tracks + self.graveyard

    def find(self, tid: int) -> BoardTrack | None:
        for t in self.tracks + self.graveyard:
            if t.id == tid:
                return t
        return None

    def prune(self, oldest_frame: int) -> None:
        """Forget dead boards whose last drawable frame has been emitted."""
        self.graveyard = [t for t in self.graveyard
                          if t.end_frame is not None and t.end_frame >= oldest_frame]

    # -- lifecycle ------------------------------------------------------------

    def reset(self, frame_idx: int) -> None:
        """Shot change: every board ends on the last frame of the previous shot."""
        for t in self.tracks:
            if t.alive:
                self._kill(t, frame_idx - 1, "cut")
        self.tracks = []
        self._last_H = None
        self._unknown = 0

    def step(self, frame_idx: int, motion: Motion,
             dets: list[tuple[np.ndarray, float]] | None) -> None:
        """Advance every track to `frame_idx`.

        dets=None means the detector did not run on this frame (it runs every
        det_interval frames); dets=[] means it ran and found nothing.
        """
        self._predict(frame_idx, motion)
        if dets is not None:
            self._update(frame_idx, motion, dets)
        self._lifecycle(frame_idx)
        self._merge_duplicates(frame_idx)
        for t in self.tracks:
            if t.alive:
                t.history[frame_idx] = t.quad.copy()
        horizon = frame_idx - self.cfg.history
        for t in self.tracks:
            if t.history and min(t.history) < horizon:
                t.history = {k: v for k, v in t.history.items() if k >= horizon}
                t.det_frames = {k for k in t.det_frames if k >= horizon}
        self.tracks = [t for t in self.tracks if t.alive]

    def _predict(self, frame_idx: int, motion: Motion) -> None:
        cfg = self.cfg
        H, extra = motion.H, 0.0
        if H is None:
            if self._last_H is not None and self._unknown < cfg.max_unknown_motion:
                # A frame whose motion could not be measured (whip-pan blur) is
                # far more likely to continue the pan than to stop dead.
                H, extra = self._last_H, 0.25 * cfg.q_unknown
            else:
                extra = cfg.q_unknown
            self._unknown += 1
        else:
            self._unknown = 0
            self._last_H = H
        q = (cfg.q_base + (cfg.q_pan * motion.shift_px) ** 2
             + cfg.q_residual * motion.residual_px ** 2 + extra)
        for t in self.tracks:
            if H is not None:
                wq = warp_quad(H, t.quad)
                if wq is None or not is_convex(wq) or polygon_area(wq) < 20:
                    self._kill(t, frame_idx - 1, "degenerate")
                    continue
                t.quad = wq
                self._hold_censored(t)
            t.P_n += q
            t.P_d += q

    def _hold_censored(self, t: BoardTrack) -> None:
        """Keep ends known only as "somewhere past the screen edge" just past it.

        Such an end has never been seen, so carrying it with the scene is
        wrong: a pan walks it back on screen one step at a time and the next
        detection snaps it out again -- a sawtooth sliver of advert at the edge
        of the picture.  Hold it `border_overshoot_px` beyond the point where
        its rail edge leaves the frame until a detection actually reveals it.
        """
        if not t.censored.any():
            return
        q = t.quad
        ov = self.cfg.border_overshoot_px
        for i in np.flatnonzero(t.censored):
            j = (1, 0, 3, 2)[i]                 # the other end of the same long edge
            a, c = q[j].astype(np.float64), q[i].astype(np.float64)
            u = c - a
            length = float(np.linalg.norm(u))
            if length < 1e-6:
                continue
            u /= length
            s_lo, s_hi = -np.inf, np.inf       # the edge line's span inside the frame
            for k, lim in ((0, self.w), (1, self.h)):
                if abs(u[k]) < 1e-9:
                    if not 0 <= a[k] <= lim:
                        s_hi = -np.inf
                    continue
                s0, s1 = -a[k] / u[k], (lim - a[k]) / u[k]
                s_lo, s_hi = max(s_lo, min(s0, s1)), min(s_hi, max(s0, s1))
            if s_hi > s_lo and length < s_hi + ov:
                q[i] = (a + (s_hi + ov) * u).astype(np.float32)

    def _plainly_visible(self, t: BoardTrack, motion: Motion) -> bool:
        m = self.cfg.edge_margin_px
        q = t.quad
        inside = bool(((q[:, 0] >= m) & (q[:, 0] <= self.w - m) &
                       (q[:, 1] >= m) & (q[:, 1] <= self.h - m)).all())
        return inside and motion.ok and motion.shift_px < self.cfg.fast_pan_px

    def _update(self, frame_idx: int, motion: Motion, dets) -> None:
        cfg = self.cfg
        live = [t for t in self.tracks if t.alive]
        # (quad, conf) or (quad, conf, cls): cls 0 = betting board, 1 = any
        # other ad board (two-class models); single-class models give 0
        dets = [(d[0], d[1], int(d[2]) if len(d) > 2 else 0) for d in dets]
        pairs = []
        for i, t in enumerate(live):
            for j, (dq, _, _) in enumerate(dets):
                iou = convex_iou(t.quad, dq)
                if iou >= cfg.match_iou:
                    pairs.append((iou, i, j))
        pairs.sort(key=lambda p: -p[0])
        t_of: dict[int, int] = {}
        d_of: dict[int, int] = {}
        for _, i, j in pairs:
            if i not in t_of and j not in d_of:
                t_of[i], d_of[j] = j, i

        # One detection covering several tracks is one board seen whole again
        # (e.g. a player who split it has walked away): the detection is the
        # best evidence of its full extent, so the keeper takes it outright and
        # the pieces it swallowed end here.
        swallowed: dict[int, list[int]] = {}
        for i, t in enumerate(live):
            if i in t_of:
                continue
            for j, (dq, _, _) in enumerate(dets):
                if j in d_of and containment(t.quad, dq) >= cfg.absorb_frac:
                    swallowed.setdefault(j, []).append(i)
                    break
        for j, pieces in swallowed.items():
            keeper = live[d_of[j]]
            for i in pieces:
                self._inherit(keeper, live[i])
                self._kill(live[i], frame_idx - 1, "merged")
            keeper.resync(dets[j][0], frame_idx, cfg)
            keeper.conf = 0.7 * keeper.conf + 0.3 * float(dets[j][1])
            keeper.vote(dets[j][2], dets[j][1], frame_idx)

        for i, t in enumerate(live):
            if not t.alive:
                continue
            if i in t_of:
                j = t_of[i]
                if j not in swallowed:
                    dq, dc, dk = dets[j]
                    t.correct(dq, dc, frame_idx, self.w, self.h, cfg)
                    t.vote(dk, dc, frame_idx)
            else:
                if t.anchor_until is not None:
                    if frame_idx <= t.anchor_until:
                        continue                # held by the brand scan
                    # the scan stopped confirming it: let it end from here,
                    # fading, instead of from its last confirmation long ago
                    t.last_det_frame = max(t.last_det_frame, t.anchor_until)
                    t.anchor_until = None
                t.runs_since_hit += 1
                limit = cfg.coast_short if self._plainly_visible(t, motion) else cfg.coast_long
                t.miss_budget += 1.0 / max(1, limit)

        for j, (dq, dc, dk) in enumerate(dets):
            if j in d_of or dc < cfg.birth_conf:
                continue
            # A fragment of a board we already track (the detector split it
            # around a player) must not become a second, independent panel.
            if any(containment(dq, t.quad) >= cfg.absorb_frac for t in self.tracks if t.alive):
                continue
            self._next += 1
            t = BoardTrack(self._next, dq, dc, frame_idx, cfg)
            t.vote(dk, dc, frame_idx)
            self.tracks.append(t)
            self.events["born"] += 1

    def inject(self, frame_idx: int, quad: np.ndarray, verdict: str, brand: str | None,
               anchor_frames: int, source: str, conf: float = 0.9) -> BoardTrack | None:
        """A board named by the brand scan or marked by the operator.

        It is not a detection -- the model did not see it -- so it never
        counts as a detector hit.  Overlapping an existing board, it names
        that board (the model found it, the scan read it); otherwise a
        'bet' starts a confirmed board of its own.  A board only the scan
        sees is held for `anchor_frames` whatever the detector says, and the
        next scan renews the hold or lets it lapse; a board the model tracks
        keeps the model's own coasting rules.  Call after step() for the
        same frame.
        """
        cfg = self.cfg
        q = np.asarray(quad, np.float32).reshape(4, 2)
        if not is_convex(q) or polygon_area(q) < 20:
            return None
        best, best_score = None, 0.0
        for t in self.tracks:
            if not t.alive:
                continue
            iou = convex_iou(t.quad, q)
            cover = max(containment(q, t.quad), containment(t.quad, q))
            if iou >= cfg.match_iou or cover >= cfg.absorb_frac:
                score = max(iou, 0.8 * cover)
                if score > best_score:
                    best, best_score = t, score
        if best is not None:
            t = best
            if verdict in ("bet", "neg"):
                t.verdict = verdict
            if brand:
                t.brand = brand
            if verdict == "bet":
                if t.state == "tentative":           # named: no need to wait for more hits
                    t.state = "confirmed"
                    t.confirmed_frame = frame_idx
                model_led = (t.last_model_frame is not None and
                             frame_idx - t.last_model_frame <= 2 * cfg.coast_short)
                if not model_led:
                    # Only the scan sees this board: its reading is the best
                    # evidence of where the board is, and nothing else keeps
                    # it alive between readings.  A board the model does see
                    # keeps the model's own coasting rules.
                    t.anchor_until = max(t.anchor_until or -1, frame_idx + int(anchor_frames))
                    t.scan_misses = 0
                    t.correct(q, conf, frame_idx, self.w, self.h, cfg)
                    t.history[frame_idx] = t.quad.copy()
            elif verdict == "neg":
                t.anchor_until = None
            self.events[f"named_{verdict}"] += 1
            return t
        if verdict != "bet" or visible_fraction(q, self.w, self.h) < cfg.min_visible:
            return None
        self._next += 1
        t = BoardTrack(self._next, q, conf, frame_idx, cfg)
        t.state = "confirmed"
        t.confirmed_frame = frame_idx
        t.verdict, t.brand, t.source = "bet", brand, source
        t.anchor_until = frame_idx + int(anchor_frames)
        self.tracks.append(t)
        self.events["injected"] += 1
        return t

    def _lifecycle(self, frame_idx: int) -> None:
        cfg = self.cfg
        for t in self.tracks:
            if not t.alive:
                continue
            if visible_fraction(t.quad, self.w, self.h) < cfg.min_visible:
                self._kill(t, frame_idx - 1, "left frame")
                continue
            if t.state == "tentative":
                if t.hits >= cfg.min_hits:
                    t.state = "confirmed"
                    t.confirmed_frame = frame_idx
                    self.events["confirmed"] += 1
                elif t.runs_since_hit >= cfg.confirm_window:
                    t.state = "dead"            # never drawn: no graveyard
                    self.events["unconfirmed"] += 1
            elif t.miss_budget >= 1.0:
                end = min(frame_idx, t.last_det_frame + cfg.end_tail)
                self._kill(t, end, "lost")
                t.fade_from = t.last_det_frame

    def _merge_duplicates(self, frame_idx: int) -> None:
        live = [t for t in self.tracks if t.alive]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                if not (a.alive and b.alive):
                    continue
                a_in_b, b_in_a = containment(a.quad, b.quad), containment(b.quad, a.quad)
                if a_in_b > 0.85 or b_in_a > 0.85:
                    keep, drop = (b, a) if a_in_b >= b_in_a else (a, b)   # keep the larger
                elif convex_iou(a.quad, b.quad) > self.cfg.merge_iou:
                    keep, drop = (a, b) if (a.hits, -a.birth_frame) >= (b.hits, -b.birth_frame) else (b, a)
                else:
                    continue
                self._inherit(keep, drop)
                self._kill(drop, frame_idx - 1, "merged")

    @staticmethod
    def _inherit(keep: BoardTrack, drop: BoardTrack) -> None:
        """A surviving track takes over what the absorbed one had earned, so
        a merge can never demote a visible, verified panel to a tentative one."""
        if drop.confirmed_frame is not None and (keep.confirmed_frame is None or
                                                 drop.confirmed_frame < keep.confirmed_frame):
            keep.confirmed_frame = drop.confirmed_frame
            keep.state = "confirmed"
        keep.birth_frame = min(keep.birth_frame, drop.birth_frame)
        keep.hits = max(keep.hits, drop.hits)
        # Same precedence as v2_render.is_betting_board: a positive betting read
        # is specific evidence and is not vetoed by a neighbour's wordmark.
        rank = {"bet": 3, "neg": 2, "unknown": 1, "pending": 0}
        if rank[drop.verdict] > rank[keep.verdict]:
            keep.verdict = drop.verdict
            keep.brand = drop.brand or keep.brand
            keep.ocr_texts = drop.ocr_texts or keep.ocr_texts
        if keep.first_drawn is None:
            keep.first_drawn = drop.first_drawn
        keep.votes = [keep.votes[0] + drop.votes[0], keep.votes[1] + drop.votes[1]]
        if drop.last_model_frame is not None:
            keep.last_model_frame = max(keep.last_model_frame or -1, drop.last_model_frame)
        if drop.anchor_until is not None:
            keep.anchor_until = max(keep.anchor_until or -1, drop.anchor_until)

    def _kill(self, t: BoardTrack, end_frame: int, reason: str) -> None:
        if not t.alive:
            return
        was_drawable = t.confirmed_frame is not None
        t.state = "dead"
        t.end_frame = end_frame
        t.end_reason = reason
        self.events[reason] += 1
        if was_drawable:
            self.graveyard.append(t)
