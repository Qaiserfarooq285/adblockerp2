#!/usr/bin/env python3
"""Hide betting ads in football video.

Stage order, deliberately:
  detect -> fit quad -> track/smooth -> build fill -> subtract people -> composite

Design notes that matter, because earlier versions failed on each of them:

* **Quad, not raw mask.**  A per-frame segmentation mask wobbles at the pixel
  level even when the board is rock still.  Fitting a four-corner quad and
  smoothing the corners removes that wobble without decoupling the panel from
  the board.

* **Camera geometry, corrected by every detection** (`--tracker homography`,
  the default; see render_engine.py / board_tracker.py / geometry.py).  A
  full frame-to-frame homography -- fitted with the scoreboard and the LED
  boards masked out -- carries each board, and each detection corrects it
  through a Kalman gain, so the panel neither copies detector jitter nor lags
  a pan, and it coasts through detection dropouts instead of blinking.  The
  old optical-flow propagation drifted for specific, fixed reasons (4-DOF
  similarity, scoreboard and LED features in the fit, a fixed 3% correction);
  geometry.py lists them.  `--tracker legacy` keeps the previous
  detection-gated tracker for A/B comparison.

* **People are a hard subtraction, computed independently.**  Person masks come
  from a general COCO segmentation model, not from the ad model, and are
  unioned over several frames so a partially-segmented player is still fully
  protected.

* **Fill is built in rail space.**  A perimeter LED board is a physical object
  with nothing behind it, so the fill synthesises blank board surface sampled
  from the same rail rather than pretending to recover a hidden background.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ geometry

def canonical(quad: np.ndarray) -> np.ndarray:
    """Corners ordered along the board's long axis, as tl, tr, br, bl.

    Ordering by y-coordinate looks equivalent and is not: on a rail receding
    steeply into the distance, both ends of one short edge can outrank a long
    edge's corners, which silently swaps length with thickness.  Everything
    downstream then works across the rail instead of along it -- the fill
    samples its colour from grass and crowd rather than from the rail.  Anchor
    on the longest edge instead, which no amount of perspective can confuse.
    """
    q = np.asarray(quad, np.float32).reshape(4, 2)
    lengths = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    q = np.roll(q, -int(np.argmax(lengths)), axis=0)
    if q[0][0] > q[1][0]:                       # run the long edge left to right
        q = q[[1, 0, 3, 2]]
    if q[0][1] > q[3][1]:                       # make the long edge the top one
        q = q[[3, 2, 1, 0]]
    return q.astype(np.float32)


def quad_shape(quad: np.ndarray) -> tuple[float, float]:
    """(length along the rail, thickness across it)."""
    q = canonical(quad)
    return (float(np.linalg.norm(q[1] - q[0])), float(np.linalg.norm(q[3] - q[0])))


def merge_overlapping(masks: list[np.ndarray], confs: list[float],
                      iou_thr: float = 0.20, contain_thr: float = 0.50):
    """Mask-level NMS over board proposals.

    The model often proposes a rail twice: a confident piece covering the
    betting run, and a lower-confidence one that over-reaches along the same
    rail into the neighbouring sponsor.  Ultralytics' box NMS keeps both --
    their boxes overlap too little to suppress -- and painting both draws the
    offset union, which both thickens the panel and swallows an innocent
    sponsor.

    Fusing them is tempting and wrong: the union is exactly the over-reach.
    Keep the confident proposal and drop what overlaps it.  Leaving part of a
    rail showing on one frame is recoverable -- the tracker carries the board
    and other frames catch it -- whereas painting over a non-betting sponsor is
    the failure this whole project exists to avoid.
    """
    order = sorted(range(len(masks)), key=lambda i: -confs[i])
    kept: list[dict] = []
    for i in order:
        mi = masks[i]
        ai = int(mi.sum())
        if ai == 0:
            continue
        for g in kept:
            inter = int(np.count_nonzero(g["mask"] & mi))
            if inter == 0:
                continue
            union = int(np.count_nonzero(g["mask"] | mi))
            if not (inter / union > iou_thr or inter / ai > contain_thr):
                continue
            # Same rail split into overlapping pieces, or one proposal
            # over-reaching past the end of the betting run?  Fusing collinear
            # pieces recovers the whole board; fusing an over-reach swallows
            # the neighbouring sponsor.  Shape decides: joining collinear
            # pieces leaves the strip just as thin, while joining anything else
            # fattens it.
            fused = g["mask"] | mi
            fq = quad_from_mask(fused)
            if fq is not None:
                fl, ft = quad_shape(fq)
                gq = quad_from_mask(g["mask"])
                iq = quad_from_mask(mi)
                _, gt = quad_shape(gq) if gq is not None else (0.0, 0.0)
                _, it = quad_shape(iq) if iq is not None else (0.0, 0.0)
                # FIX: 1.35 → 1.15 — prevent thickness-bloat when an off-axis
                # mask fuses into a good rail, rotating the PCA axis and
                # inflating the perpendicular dimension.  1.15 still allows
                # collinear fusions (which never inflate) while rejecting
                # fusions that would grow the box into crowd / neighbours.
                if ft <= 1.15 * max(gt, it, 1.0) and fl / max(ft, 1e-6) >= 3.0:
                    g["mask"] = fused
                    g["conf"] = max(g["conf"], confs[i])
            break
        else:
            kept.append({"mask": mi, "conf": confs[i]})
    return kept


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None
    return canonical(cv2.boxPoints(cv2.minAreaRect(c)))


def fit_strip(mask: np.ndarray, along_pct: float = 0.5,
              across_pct: float = 2.5) -> np.ndarray | None:
    """Fit one parallelogram tightly to a board mask.

    minAreaRect is driven by extreme pixels, so a few stray mask pixels above
    the board make the rectangle taller than the board itself.  Fitting the
    principal axis and taking robust percentiles across it instead gives a
    strip that matches the board's real thickness, and it is a true
    parallelogram, so all four edges are straight.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    centred = pts - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    d = vt[0] / (np.linalg.norm(vt[0]) + 1e-6)
    n = np.array([-d[1], d[0]], np.float32)
    t = centred @ d
    p = centred @ n
    t0, t1 = np.percentile(t, [along_pct, 100 - along_pct])
    p0, p1 = np.percentile(p, [across_pct, 100 - across_pct])
    if t1 - t0 < 8 or p1 - p0 < 3:
        return None
    corners = np.float32([
        mean + d * t0 + n * p0, mean + d * t1 + n * p0,
        mean + d * t1 + n * p1, mean + d * t0 + n * p1])
    return canonical(corners)


def strips_from_frame(masks: list[np.ndarray], shape: tuple[int, int],
                      min_fill: float = 0.60, seg_len: float = 240.0,
                      max_seg: int = 10, min_aspect: float = 1.8) -> list[np.ndarray]:
    """Turn all board masks in a frame into clean, non-overlapping strips.

    Each input mask already survived merge_overlapping's shape-guarded,
    IoU/containment-gated fusion, so it is either a complete board or a
    genuine fragment (e.g. a player occluding the middle of one board splits
    it into two masks with zero overlap, which merge_overlapping's IoU test
    cannot catch -- it only fuses masks that actually intersect).

    A mask that already fits a clean, well-proportioned strip on its own is a
    complete board and must never be unioned with a neighbour before
    connected-components runs: two different sponsor boards sitting side by
    side on the same perimeter rail are just as collinear and pixel-adjacent
    as one occluded board's two halves, so proximity alone can't tell them
    apart -- doing so was silently fusing a betting board into its innocent
    neighbour (wrong-sized paint region, and a single OCR check on the fused
    crop that could reject the whole thing, leaving the real ad unpainted).
    Only masks that fail as standalone strips (real fragments) go through the
    gap-bridging recovery below, and only with each other.
    """
    h, w = shape
    if not masks:
        return []

    out: list[np.ndarray] = []
    fragments: list[np.ndarray] = []
    for m in masks:
        mu = (m > 0).astype(np.uint8)
        q = fit_strip(mu)
        if q is not None:
            length, thick = quad_shape(q)
            if thick >= 3 and length / thick >= min_aspect:
                out.append(q)
                continue
        fragments.append(mu)

    if not fragments:
        return out

    union = np.zeros((h, w), np.uint8)
    for m in fragments:
        union |= m
    # Bridge the small gaps the detector leaves between pieces of one rail.
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n_lbl, labels = cv2.connectedComponents(union, 8)
    for i in range(1, n_lbl):
        comp = (labels == i).astype(np.uint8)
        if int(comp.sum()) < 150:
            continue
        q = fit_strip(comp)
        if q is None:
            continue
        area = cv2.contourArea(q.astype(np.float32))
        if area > 0 and float(comp.sum()) / area >= min_fill:
            out.append(q)
            continue
        length, _ = quad_shape(q)
        k = int(np.clip(round(length / seg_len), 2, max_seg))
        top, bot = q[1] - q[0], q[2] - q[3]
        for j in range(k):
            t0, t1 = j / k, (j + 1) / k
            strip = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(strip, np.float32([
                q[0] + top * t0, q[0] + top * t1,
                q[3] + bot * t1, q[3] + bot * t0]).astype(np.int32), 1)
            piece = comp & strip
            pq = fit_strip(piece)
            if pq is not None:
                out.append(pq)
    return out


def quads_from_mask(mask: np.ndarray, min_fill: float = 0.60,
                    seg_len: float = 220.0, max_seg: int = 12) -> list[np.ndarray]:
    """One or more tight quads covering a board mask."""
    q = quad_from_mask(mask)
    if q is None:
        return []
    area = cv2.contourArea(q.astype(np.float32))
    if area <= 0:
        return []
    if float(mask.sum()) / area >= min_fill:
        return [q]

    length, _ = quad_shape(q)
    n = int(np.clip(round(length / seg_len), 2, max_seg))
    h, w = mask.shape[:2]
    top, bot = q[1] - q[0], q[2] - q[3]
    out = []
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        strip = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(strip, np.float32([
            q[0] + top * t0, q[0] + top * t1,
            q[3] + bot * t1, q[3] + bot * t0]).astype(np.int32), 1)
        piece = mask & strip
        if int(piece.sum()) < 80:
            continue
        pq = quad_from_mask(piece)
        if pq is not None:
            out.append(pq)
    return out or [q]


def quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape
    sc = 640.0 / max(h, w)
    ma = np.zeros((int(h * sc) + 2, int(w * sc) + 2), np.uint8)
    mb = ma.copy()
    cv2.fillConvexPoly(ma, (a * sc).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, (b * sc).astype(np.int32), 1)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    iou = inter / union if union else 0.0
    if iou < 0.10:
        ca = a.mean(axis=0)
        cb = b.mean(axis=0)
        dist = float(np.linalg.norm(ca - cb))
        if dist < 45.0:
            iou = max(iou, 0.35 * (1.0 - dist / 45.0))
    return iou


def quad_metrics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    q = np.asarray(q, np.float32).reshape(4, 2)
    centre = q.mean(axis=0)
    e1 = (q[1] - q[0]) + (q[2] - q[3])
    length = float(np.linalg.norm(e1) / 2.0)
    direction = e1 / (np.linalg.norm(e1) + 1e-6)
    e2 = (q[3] - q[0]) + (q[2] - q[1])
    thickness = float(np.linalg.norm(e2) / 2.0)
    return centre, direction, length, thickness


def unify_collinear_quads(quads: list[np.ndarray], shape: tuple[int, int]) -> list[np.ndarray]:
    """Merge live quads on the same physical rail into unified smooth quads."""
    if not quads:
        return []
    if len(quads) == 1:
        return [quads[0]]

    n = len(quads)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ca, da, la, ta = quad_metrics(quads[i])
        for j in range(i + 1, n):
            cb, db, lb, tb = quad_metrics(quads[j])
            if abs(float(np.dot(da, db))) < 0.88:
                continue
            delta = cb - ca
            perp = abs(float(delta[0] * -da[1] + delta[1] * da[0]))
            if perp > 1.3 * max(ta, tb):
                continue
            along = abs(float(np.dot(delta, da)))
            gap = along - (la + lb) / 2.0
            if gap < 2.5 * max(ta, tb):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[np.ndarray] = []
    h, w = shape
    for idxs in groups.values():
        if len(idxs) == 1:
            out.append(quads[idxs[0]])
            continue
        union_mask = np.zeros((h, w), np.uint8)
        max_thick = 0.0
        for k in idxs:
            cv2.fillConvexPoly(union_mask, canonical(quads[k]).astype(np.int32), 255)
            _, _, _, tk = quad_metrics(quads[k])
            max_thick = max(max_thick, tk)

        fq = fit_strip(union_mask)
        if fq is None:
            fq = quad_from_mask(union_mask)
        if fq is not None:
            fl, ft = quad_shape(fq)
            if max_thick > 0 and ft > 1.15 * max_thick:
                q = canonical(fq)
                top_mid = (q[0] + q[1]) / 2.0
                bot_mid = (q[2] + q[3]) / 2.0
                n_vec = bot_mid - top_mid
                curr_t = float(np.linalg.norm(n_vec))
                if curr_t > 1e-3:
                    scale = (1.15 * max_thick) / curr_t
                    centre = q.mean(axis=0)
                    q[0] = centre + (q[0] - centre) * scale
                    q[1] = centre + (q[1] - centre) * scale
                    q[2] = centre + (q[2] - centre) * scale
                    q[3] = centre + (q[3] - centre) * scale
                    fq = canonical(q)
            out.append(fq)
        else:
            for k in idxs:
                out.append(quads[k])
    return out


# -------------------------------------------------------------------- tracks

class Track:
    __slots__ = ("quad", "mask", "hits", "misses", "conf", "alive", "id", "velocity")
    _next = 0

    def __init__(self, quad: np.ndarray, conf: float, mask: np.ndarray | None = None):
        self.quad = quad.copy()
        self.mask = mask
        self.hits = 1
        self.misses = 0
        self.conf = conf
        self.alive = False
        self.velocity = np.zeros(2, np.float32)
        Track._next += 1
        self.id = Track._next

    def predicted_quad(self) -> np.ndarray:
        """This track's quad extrapolated one frame ahead by its own recent
        motion -- used only to match against this frame's detections, never
        to render.

        A board panning at 15-20 px/frame moves further between frames than
        its own last-frame quad has IoU left to match against: measured on
        real footage, a continuously-panning Betano board lost its track
        within ~3 frames of the pan speeding up, freezing the old track's
        position (a stale "ghost" box) while a chain of short-lived new
        tracks spawned for the same physical board -- every one of them
        still "live" and rendered, so painting the union smeared a single
        board's paint across a much wider, wrongly-placed region. Matching
        against where the board should be this frame, instead of where it
        was last frame, keeps one continuous track through the pan.
        """
        return self.quad + self.velocity

    def update(self, quad: np.ndarray, conf: float, alpha: float,
               mask: np.ndarray | None = None, max_step_frac: float = 0.15) -> None:
        """EMA-blend position/orientation, but clamp the SIZE change hard.

        EMA alone does not stop a single bad frame from being visible: it
        moves 1-alpha of the way toward whatever the detector just proposed,
        and when the frame-level split/merge decision flips (a board fit as
        one rectangle on frame N and two pieces on frame N+1), the "just
        proposed" shape can differ enormously from reality.  That single-frame
        pop is exactly the "randomly increase, decrease, jump" behaviour this
        clamp exists to remove: after the EMA blend, length and thickness are
        pulled back to within max_step_frac of last frame's value, so the
        rendered size can only ever drift toward the true boundary a bounded
        amount per frame, never leap to it.  Position and rotation are left to
        the EMA alone, since those legitimately need to track camera motion.
        """
        prev = self.quad
        blended = (alpha * quad + (1.0 - alpha) * prev).astype(np.float32)

        pc, pd, plen, pthk = quad_metrics(prev)
        bc, bd, blen, bthk = quad_metrics(blended)

        lo_len, hi_len = plen * (1 - max_step_frac), plen * (1 + max_step_frac)
        lo_thk, hi_thk = pthk * (1 - max_step_frac), pthk * (1 + max_step_frac)
        clen = float(np.clip(blen, lo_len, hi_len)) if plen > 1e-3 else blen
        cthk = float(np.clip(bthk, lo_thk, hi_thk)) if pthk > 1e-3 else bthk

        # Rebuild the quad from the (already-smoothed) centre/direction but
        # the clamped size, using the blended quad's own perpendicular so a
        # slight perspective skew survives rather than being forced square.
        bq = canonical(blended)
        raw_n = (bq[3] - bq[0]) + (bq[2] - bq[1])
        n = raw_n / (np.linalg.norm(raw_n) + 1e-6)
        d = bd
        self.quad = np.float32([
            bc - d * clen / 2 - n * cthk / 2,
            bc + d * clen / 2 - n * cthk / 2,
            bc + d * clen / 2 + n * cthk / 2,
            bc - d * clen / 2 + n * cthk / 2,
        ])
        # bc is already the EMA-blended centre, so this delta is the
        # (already-damped) observed motion this frame -- feeding it straight
        # into velocity keeps prediction responsive; smoothing it further
        # would just reintroduce the pan lag this exists to remove.
        self.velocity = (bc - pc).astype(np.float32)
        if mask is not None:
            self.mask = mask
        self.hits += 1
        self.misses = 0
        self.conf = conf

    def miss(self) -> None:
        """Dead-reckon through a missed frame instead of freezing.

        A track can still be rendered while held (misses <= max_hold), and a
        board mid-pan does not stop moving just because one frame's OCR
        guard or detector had a blip -- freezing here is what produced the
        stale "ghost" position a real re-detection then had to fight past.
        """
        self.quad = self.quad + self.velocity
        self.misses += 1


class Tracker:
    def __init__(self, iou_thr: float, min_hits: int, max_hold: int, alpha: float,
                 max_size_step: float = 0.15):
        self.iou_thr = iou_thr
        self.min_hits = min_hits
        self.max_hold = max_hold
        self.alpha = alpha
        self.max_size_step = max_size_step
        self.tracks: list[Track] = []

    def reset(self) -> None:
        """Drop every track.  Used on a shot change.

        Holding a track across a cut paints the previous shot's board onto
        whatever the new shot happens to show, which reads as the panel
        "carrying over" into unrelated frames.  A cut invalidates all geometry,
        so nothing should survive it.
        """
        self.tracks.clear()

    def step(self, dets: list[tuple], shape) -> list[Track]:
        used = set()
        for tr in self.tracks:
            best, bi = self.iou_thr, -1
            pred = tr.predicted_quad()
            for i, d in enumerate(dets):
                if i in used:
                    continue
                v = quad_iou(pred, d[0], shape)
                if v > best:
                    best, bi = v, i
            if bi >= 0:
                used.add(bi)
                d = dets[bi]
                tr.update(d[0], d[1], self.alpha, d[2] if len(d) > 2 else None,
                         self.max_size_step)
            else:
                tr.miss()
        for i, d in enumerate(dets):
            if i not in used:
                self.tracks.append(Track(d[0], d[1], d[2] if len(d) > 2 else None))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_hold]
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.alive = True
            if t.alive and t.misses <= self.max_hold:
                out.append(t)
        return out


# ----------------------------------------------------------------- rail fill

# Single, fixed paint colour for every board, every frame, every clip.
#
# Earlier versions sampled a colour from the rail's own neighbourhood each
# frame so the fill would read as "a blank LED board" rather than a flat
# rectangle.  That is explicitly not what is wanted: sampling varies with
# lighting, camera exposure and what happens to be next to the board, which
# is a source of the very flicker/colour-drift this project is trying to
# eliminate.  A single constant avoids that class of bug entirely -- there is
# no neighbourhood, no percentile, nothing that can differ frame to frame.
# Neutral dark charcoal: reads as a deliberate "painted over" mask rather
# than a censor-black box or a distracting colour.
PAINT_COLOR_BGR = (40, 40, 40)


def rail_fill(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Solid-colour patch sized to the quad.  Same colour every call, everywhere."""
    q = canonical(quad)
    width = float(np.linalg.norm(q[1] - q[0]))
    height = float(np.linalg.norm(q[3] - q[0]))
    if width < 8 or height < 4:
        return np.zeros((0, 0, 3), np.uint8)
    ph, pw = max(2, int(round(height))), max(2, int(round(width)))
    return np.full((ph, pw, 3), PAINT_COLOR_BGR, np.uint8)


def warp_fill(frame: np.ndarray, quad: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Project a rail-space patch back onto the frame, returning a full-frame BGR."""
    h, w = frame.shape[:2]
    ph, pw = patch.shape[:2]
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    M = cv2.getPerspectiveTransform(src, canonical(quad))
    return cv2.warpPerspective(patch, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_TRANSPARENT,
                               dst=np.zeros((h, w, 3), np.uint8))


def _palette(patch: np.ndarray, cover: float = 0.90) -> np.ndarray:
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    q = lab.astype(np.int32) // 16
    codes = ((q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]).ravel()
    keys, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)
    keep = counts[order].cumsum() <= cover * codes.size
    keep[0] = True
    return keys[order][keep]


def _strip(frame: np.ndarray, quad: np.ndarray, t0: float, t1: float) -> np.ndarray | None:
    """Sample the rail between two positions along its own axis (0..1 = quad)."""
    q = canonical(quad)
    top, bot = q[1] - q[0], q[2] - q[3]
    a = q[0] + top * t0
    b = q[0] + top * t1
    c = q[3] + bot * t1
    d = q[3] + bot * t0
    poly = np.float32([a, b, c, d])
    h, w = frame.shape[:2]
    # Sample whatever part of the strip is on screen.  Rejecting the whole
    # strip the moment a corner leaves the frame stops extension dead on
    # exactly the rails that run to the edge, which are the long ones we care
    # about.  Require most of it to be visible so the match stays meaningful.
    inside = ((poly[:, 0] > -4) & (poly[:, 0] < w + 4) &
              (poly[:, 1] > -4) & (poly[:, 1] < h + 4)).sum()
    if inside < 3:
        return None
    wd = int(round(np.linalg.norm(b - a)))
    ht = int(round(np.linalg.norm(d - a)))
    if wd < 6 or ht < 4:
        return None
    M = cv2.getPerspectiveTransform(
        poly, np.float32([[0, 0], [wd, 0], [wd, ht], [0, ht]]))
    return cv2.warpPerspective(frame, M, (wd, ht))


def extend_rail(frame: np.ndarray, quad: np.ndarray, mask: np.ndarray,
                step: float = 0.25, max_grow: float = 1.0,
                match_rel: float = 0.62) -> np.ndarray:
    """Follow a detected board along the rail while the surface still matches.

    The model truncates long runs: every proposal on a full-width rail comes
    back about the same length, so a Betano run continuing behind the goal is
    simply never predicted.  The board itself is the evidence -- step outward
    along the rail axis and keep going while the pixels still belong to the
    same palette, which stops at the neighbouring sponsor because its colours
    are different.

    Conservative on purpose: a high match threshold and a hard cap on growth,
    because over-running into an innocent sponsor is worse than stopping early.
    """
    q = canonical(quad)
    core = _strip(frame, q, 0.0, 1.0)
    if core is None or core.size == 0:
        return q
    pal = _palette(core)
    lab0 = cv2.cvtColor(core, cv2.COLOR_BGR2LAB).astype(np.int32) // 16
    c0 = (lab0[:, :, 0] << 16) | (lab0[:, :, 1] << 8) | lab0[:, :, 2]
    # Judge the extension against how well the board matches *itself*, not an
    # absolute number: a thick slanted rail includes net and crowd inside its
    # strip, so the ceiling is well below 1.0 and a fixed threshold never fires.
    match_min = match_rel * float(np.isin(c0, pal).mean())

    def matches(t0: float, t1: float) -> bool:
        s = _strip(frame, q, t0, t1)
        if s is None or s.size == 0:
            return False
        lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB).astype(np.int32) // 16
        codes = (lab[:, :, 0] << 16) | (lab[:, :, 1] << 8) | lab[:, :, 2]
        return float(np.isin(codes, pal).mean()) >= match_min

    lo, hi = 0.0, 1.0
    grown = 0.0
    while grown < max_grow and matches(hi, hi + step):
        hi += step
        grown += step
    while grown < max_grow and matches(lo - step, lo):
        lo -= step
        grown += step
    if lo == 0.0 and hi == 1.0:
        return q

    top, bot = q[1] - q[0], q[2] - q[3]
    return canonical(np.float32([
        q[0] + top * lo, q[0] + top * hi, q[3] + bot * hi, q[3] + bot * lo]))


def pitch_reference(frame: np.ndarray, stride: int = 1) -> np.ndarray | None:
    """Median HSV of the turf in `frame`, or None when no pitch is in shot.

    `stride` > 1 estimates it from a pixel subsample: the median of a tight
    colour cluster is unchanged, and on a live frame this is the difference
    between ~1 ms and ~10 ms.
    """
    hsv = cv2.cvtColor(frame[::stride, ::stride] if stride > 1 else frame, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    cand = (h_ >= 38) & (h_ <= 62) & (s_ > 70) & (v_ > 60)
    if int(cand.sum()) < 0.02 * hsv.shape[0] * hsv.shape[1]:
        return None                                     # no pitch in shot
    return np.median(hsv[cand], axis=0)


def pitch_mask(frame: np.ndarray, dh: int = 6, ds: int = 48, dv: int = 58,
               ref: np.ndarray | None = None) -> np.ndarray:
    """Pixels that are almost certainly turf.

    On steeply-slanted rails near the corner flag the segmentation over-reaches
    downward onto the pitch, and a panel lying on the grass is the single most
    obvious artefact a viewer can spot.  An ad board is never turf, so turf is
    always safe to exclude.

    The threshold is deliberately narrow.  Measured over this footage turf
    occupies H 47-50, S 114-146, V 125-142 -- a very tight cluster -- whereas
    the Kalshi wordmark, the one betting brand that is also green, spans H
    35-117 at up to S 246.  Matching turf tightly removes grass without ever
    sparing a Kalshi board.

    `ref` (from pitch_reference) lets a caller measure the turf colour once per
    shot on the whole frame and then mask only a small crop around the boards.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if ref is None:
        ref = pitch_reference(frame)
        if ref is None:
            return np.zeros(frame.shape[:2], np.uint8)  # no pitch in shot
    m = ((np.abs(h_.astype(np.int16) - int(ref[0])) <= dh) &
         (np.abs(s_.astype(np.int16) - int(ref[1])) <= ds) &
         (np.abs(v_.astype(np.int16) - int(ref[2])) <= dv)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def crossing_structure(frame: np.ndarray, area: np.ndarray,
                       v_min: int = 168, s_max: int = 74,
                       outside_frac: float = 0.35) -> np.ndarray:
    """White objects that pass *in front of* a board, e.g. goal posts and net.

    A plain "don't paint white" rule cannot work here: the PredictStreet rails
    are white lettering on black, so it would leave the ad perfectly readable.
    The distinction that does hold is geometric -- a goalpost, crossbar or net
    continues well beyond the board, while a glyph is entirely contained by it.
    So keep any white component that is substantially outside the paint area
    and drop the ones that are not.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= v_min) & (hsv[:, :, 1] <= s_max)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    # Per-component overlap in one pass (bincount) instead of one full-frame
    # comparison per component: identical result, and a board full of white
    # lettering has hundreds of components.
    total = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    overlap = np.bincount(labels[area > 0], minlength=n)[:n]
    ok = (total >= 60) & (overlap > 0) & ((total - overlap) >= outside_frac * total)
    ok[0] = False
    keep = ok[labels].astype(np.uint8)
    if keep.any():
        keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    return keep * 255


def static_overlay_mask(video: Path, samples: int = 40) -> np.ndarray | None:
    """Broadcast graphics burned into fixed screen positions (clock, logo bug).

    Sampled widely across the clip, the camera has moved a great deal, so any
    pixel that never changes is painted on by the broadcaster rather than part
    of the scene.  Cheap to compute once per clip and it needs no hard-coded
    rectangle, which would not survive a different broadcaster.
    """
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < samples * 2:
        cap.release()
        return None
    idxs = {int(round((i + 0.5) * n / samples)) - 1 for i in range(samples)}
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idxs:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        i += 1
    cap.release()
    if len(frames) < 8:
        return None
    std = np.std(np.stack(frames), axis=0)
    # Tighten until the mask is plausibly just graphics.  A clip whose camera
    # moves little leaves much of the scene static at a loose threshold, and
    # rejecting the whole mask then leaves the clock and score unprotected --
    # which is exactly what happened on clip02.  Only a genuinely burned-in
    # graphic survives the strictest pass.
    for thresh in (7.0, 4.0, 2.5, 1.5):
        m = (std < thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if m.mean() <= 0.12:
            return cv2.dilate(m, np.ones((9, 9), np.uint8), iterations=1) * 255
    return None


# --------------------------------------------------------------------- people

class PersonMasker:
    """Union-of-recent person masks, so a briefly mis-segmented player stays safe."""

    def __init__(self, model, conf: float, union: int, dilate: int, imgsz: int):
        self.model = model
        self.conf = conf
        self.hist: deque[np.ndarray] = deque(maxlen=max(1, union))
        self.kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
        self.imgsz = imgsz

    def reset(self) -> None:
        """Forget history at a shot change; the previous shot's people are gone."""
        self.hist.clear()

    @staticmethod
    def _in_front(pmask: np.ndarray, quads: list[np.ndarray], slack: float = 6.0) -> bool:
        """Is this person between the camera and the boards, or behind them?

        Only people on the pitch occlude a board.  Crowd, photographers and
        officials standing behind the rail are themselves occluded by it, and
        subtracting them punches notches out of the panel's top edge -- which
        is what stops it reading as a clean rectangle.

        A person in front stands on the pitch, so their feet fall below the
        board's lower edge; someone behind it is cut off by the board and their
        lowest visible pixel sits at or above that edge.
        """
        ys, xs = np.nonzero(pmask)
        if len(ys) == 0:
            return False
        foot_y = float(ys.max())
        cx = float(xs[ys > ys.max() - 5].mean()) if (ys > ys.max() - 5).any() else float(xs.mean())
        for q in quads:
            c = canonical(q)
            # Bottom edge of the board, evaluated at this person's x.
            (x0, y0), (x1, y1) = c[3], c[2]
            if not (min(x0, x1) - 40 <= cx <= max(x0, x1) + 40):
                continue
            t = 0.0 if abs(x1 - x0) < 1e-3 else (cx - x0) / (x1 - x0)
            board_bottom = y0 + (y1 - y0) * float(np.clip(t, 0.0, 1.0))
            if foot_y > board_bottom - slack:
                return True
        return False

    def __call__(self, frame: np.ndarray,
                 quads: list[np.ndarray] | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        cur = np.zeros((h, w), np.uint8)
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                               classes=[0], verbose=False, retina_masks=True)[0]
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                mm = (m > 0.5).astype(np.uint8)
                if mm.shape != (h, w):
                    mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                if quads and not self._in_front(mm, quads):
                    continue
                cur |= mm
        if self.kernel is not None:
            cur = cv2.dilate(cur, self.kernel, iterations=1)
        self.hist.append(cur)
        out = self.hist[0].copy()
        for m in list(self.hist)[1:]:
            out |= m
        # Return 0/255, not 0/1.  Callers combine this with 0/255 paint masks,
        # and a 0/1 mask silently degrades bitwise_not into a no-op
        # (bitwise_not(1) == 254, so 255 & 254 == 254 stays painted).
        return out * 255


# ----------------------------------------------------------------------- main

# ---------------------------------------------------------------------------
# OCR brand guard (FIX — Issue 3: false detections)
# ---------------------------------------------------------------------------

_OCR_ENGINE = None


def _get_ocr():
    """Lazy-load OCR engine once; RapidOCR init takes ~300 ms."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except Exception:
            _OCR_ENGINE = False  # disable on import failure
    return _OCR_ENGINE if _OCR_ENGINE is not False else None


def _ocr_texts(crop: np.ndarray) -> list[str]:
    """Run OCR on a crop; return list of text strings or empty list."""
    engine = _get_ocr()
    if engine is None or crop is None or crop.size == 0:
        return []
    try:
        res, _ = engine(crop)
    except Exception:
        return []
    if not res:
        return []
    return [r[1] for r in res if len(r) > 1 and isinstance(r[1], str)]


def _crop_for_guard(frame: np.ndarray, quad: np.ndarray,
                    min_h: int = 80) -> np.ndarray | None:
    """Axis-aligned crop around the quad, upscaled for legibility."""
    x0 = int(max(0, np.floor(quad[:, 0].min()) - 4))
    x1 = int(min(frame.shape[1], np.ceil(quad[:, 0].max()) + 4))
    y0 = int(max(0, np.floor(quad[:, 1].min()) - 4))
    y1 = int(min(frame.shape[0], np.ceil(quad[:, 1].max()) + 4))
    if x1 - x0 < 12 or y1 - y0 < 4:
        return None
    crop = frame[y0:y1, x0:x1]
    h = crop.shape[0]
    if h < min_h:
        scale = min_h / h
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                  max(1, int(h * scale))),
                          interpolation=cv2.INTER_CUBIC)
    return crop


_GUARD_CACHE: dict[tuple[int, int], tuple[int, bool]] = {}
_GUARD_FRAME = [0]
_GUARD_TTL = 20        # frames a cached verdict is trusted before re-checking
_GUARD_CELL = 40        # px grid cell for matching "the same board" position


def _guard_cache_key(quad: np.ndarray) -> tuple[int, int]:
    c = quad.reshape(4, 2).mean(axis=0)
    return (int(c[0]) // _GUARD_CELL, int(c[1]) // _GUARD_CELL)


def is_betting_board(frame: np.ndarray, quad: np.ndarray,
                     use_ocr: bool = True) -> bool:
    """Return False *only* when OCR positively identifies a non-betting brand.

    This guard is fail-open: if OCR returns nothing, or finds text that is
    neither a known betting brand nor a known non-betting brand, the model's
    own confidence is trusted and the detection is kept.  Only a positive
    non-betting match (e.g. 'KRAKEN', 'MARRIOTT') causes rejection.

    Kalshi green boards often produce no OCR output at small scales, so the
    fail-open behaviour ensures they are still painted by model confidence.

    Cached by rough on-screen position with a short TTL.  A physical board
    does not change brand between one frame and the next, so re-running OCR
    on it every single frame is pure cost with no accuracy benefit -- it was
    also a latent source of instability, since a board that happened to OCR
    cleanly on frame N and ambiguously on frame N+1 could flicker in and out
    purely from OCR noise rather than anything about the board changing.
    """
    if not use_ocr:
        return True
    _GUARD_FRAME[0] += 1
    now = _GUARD_FRAME[0]
    key = _guard_cache_key(quad)
    cached = _GUARD_CACHE.get(key)
    if cached is not None and now - cached[0] < _GUARD_TTL:
        return cached[1]

    crop = _crop_for_guard(frame, quad)
    if crop is None:
        verdict = True   # too small — trust the model
    else:
        texts = _ocr_texts(crop)
        if not texts:
            verdict = True   # OCR returned nothing — trust the model
        else:
            from v2_brands import is_non_betting, is_betting
            if any(is_betting(t) for t in texts):
                # A positive brand match on THIS quad's own crop is strong,
                # specific evidence for THIS board -- it must not be vetoed
                # by incidental bleed-over text from a touching neighbour
                # (the crop margin is only ~4px, see _crop_for_guard, so a
                # board sitting flush against the next one routinely picks
                # up a few of its letters). Measured case: a real board read
                # ['FIFA', 'Betano', 'Betano', 'Betano', 'Betano'] and was
                # being rejected on the single incidental 'FIFA' match despite
                # four unambiguous 'Betano' reads in the same crop.
                verdict = True
            elif any(is_non_betting(t) for t in texts):
                verdict = False
            else:
                verdict = True   # unknown text — trust the model
    _GUARD_CACHE[key] = (now, verdict)
    if len(_GUARD_CACHE) > 400:
        _GUARD_CACHE.clear()   # cheap bound; a fresh cache costs one extra OCR pass
    return verdict


def _reencode(tmp: Path, video: Path, out_path: Path) -> None:
    """Re-encode to H.264 so the result plays everywhere, keeping the original audio."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-i", str(video),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
           str(out_path)]
    if subprocess.run(cmd).returncode == 0:
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(out_path)


def process(video: Path, out_path: Path, cfg: dict, debug: bool = False,
            panel_log: str | None = None) -> dict:
    """Render with the homography tracker (render_engine.AdBlockEngine)."""
    if cfg["tracker"] == "legacy":
        return process_legacy(video, out_path, cfg, debug, panel_log)
    from render_engine import AdBlockEngine

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay = static_overlay_mask(video) if int(cfg["protect_overlay"]) else None
    if overlay is not None:
        print(f"[render] static broadcast overlay: {100 * overlay.mean() / 255:.2f}% of frame")
    engine = AdBlockEngine(cfg, w, h, overlay_mask=overlay, live=False,
                           debug=debug, panel_log=panel_log, fps=fps)

    tmp = out_path.with_suffix(".tmp.mp4")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for out, _ in engine.push(frame):
                writer.write(out)
        for out, _ in engine.flush():
            writer.write(out)
    finally:
        cap.release()
        writer.release()
        engine.close()
    _reencode(tmp, video, out_path)
    st = engine.stats
    dt = max(1e-6, time.time() - t0)
    print(f"[render] {st['frames']} frames in {dt:.0f}s ({st['frames'] / dt:.1f} fps); "
          f"{engine.timing_summary()}")
    return {"frames": st["frames"], "det_frames": st["det_runs"], "painted": st["painted"],
            "dets": st["dets"], "cuts": st["cuts"], "ocr_jobs": st["ocr_jobs"],
            "suppressed_non_betting": st["suppressed"],
            "panel_frames": {k[6:]: v for k, v in st.items() if k.startswith("panel_")},
            "track_events": dict(engine.tracker.events)}


def process_legacy(video: Path, out_path: Path, cfg: dict, debug: bool = False,
                   panel_log: str | None = None) -> dict:
    """The previous detection-gated tracker, kept for A/B comparison.

    With --panel-log it also records each frame's painted quads plus a camera
    homography measured independently (geometry.py), so
    scripts/stability_report.py can score both trackers on the same footage.
    """
    from ultralytics import YOLO
    from render_engine import parse_imgsz

    ad = YOLO(cfg["ad_weights"])
    person = YOLO(cfg["person_weights"])
    pm = PersonMasker(person, cfg["person_conf"], cfg["person_union"],
                      cfg["person_dilate"], parse_imgsz(cfg["person_imgsz"]))
    tracker = Tracker(cfg["track_iou"], cfg["min_hits"], cfg["max_hold"], cfg["ema_alpha"],
                      cfg["max_size_step"])
    overlay = static_overlay_mask(video) if cfg["protect_overlay"] else None
    if overlay is not None:
        print(f"[render] static broadcast overlay: {100 * overlay.mean() / 255:.2f}% of frame")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = out_path.with_suffix(".tmp.mp4")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    feather = int(cfg["feather"])
    stats = {"frames": 0, "det_frames": 0, "painted": 0, "dets": 0, "cuts": 0}
    prev_small = None
    ad_imgsz = parse_imgsz(cfg["ad_imgsz"])
    log = open(panel_log, "w") if panel_log else None
    meas = None
    if log is not None:
        from geometry import CameraMotionEstimator
        meas = CameraMotionEstimator(w, h)
        log.write(json.dumps({"header": True, "w": w, "h": h, "fps": fps,
                              "tracker": "legacy"}) + "\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stats["frames"] += 1
        motion = meas.estimate(frame) if meas is not None else None
        cut_now = False

        # Shot-change detection.  Highlight packages cut constantly, and a
        # track held across a cut lands on unrelated content.
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev_small is not None:
            if float(np.abs(small.astype(np.int16) - prev_small).mean()) > cfg["cut_delta"]:
                tracker.reset()
                pm.reset()
                stats["cuts"] += 1
                cut_now = True
        prev_small = small.astype(np.int16)

        res = ad.predict(frame, conf=cfg["render_conf"], imgsz=ad_imgsz,
                         verbose=False, retina_masks=True)[0]
        dets: list[tuple[np.ndarray, float]] = []
        if res.masks is not None:
            # Shape-filter *before* suppression.  A perimeter rail is a long
            # thin strip; squat blobs are the model hedging.  Filtering after
            # NMS lets a high-confidence blob suppress every good rail it
            # overlaps and then be rejected itself, painting nothing at all --
            # which is precisely what happened once the threshold was lowered.
            raw, confs = [], []
            for mk, cf in zip(res.masks.data.cpu().numpy(),
                               res.boxes.conf.cpu().numpy()):
                mm = (mk > 0.5).astype(np.uint8)
                if mm.shape != (h, w):
                    mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                q = quad_from_mask(mm)
                if q is None:
                    continue
                length, thick = quad_shape(q)
                if thick < 3 or length / thick < cfg["min_aspect"]:
                    continue
                if thick > cfg["max_thick_frac"] * h:
                    continue
                raw.append(mm)
                confs.append(float(cf))
            kept = merge_overlapping(raw, confs)
            conf_of = max((g["conf"] for g in kept), default=0.0)
            for q in strips_from_frame([g["mask"] for g in kept], (h, w),
                                       min_aspect=cfg["min_aspect"]):
                length, thick = quad_shape(q)
                if thick < 3 or length / thick < cfg["min_aspect"]:
                    continue
                if thick > cfg["max_thick_frac"] * h:
                    continue
                # OCR brand guard (FIX — Issue 3: false detections).
                # Fail-open: only reject when OCR positively names a
                # non-betting brand (e.g. Kraken, Marriott).  Unknown
                # text or no OCR output keeps the detection alive so
                # Kalshi green boards are never silently dropped here.
                if not is_betting_board(frame, q, use_ocr=bool(cfg.get("ocr_guard", 1))):
                    continue
                dets.append((q, conf_of, None))
        if dets:
            stats["det_frames"] += 1
            stats["dets"] += len(dets)

        live = tracker.step(dets, (h, w))

        if live:
            fill = np.zeros((h, w, 3), np.uint8)
            area = np.zeros((h, w), np.uint8)
            grow = np.ones((cfg["mask_grow"], cfg["mask_grow"]), np.uint8) \
                if cfg["mask_grow"] > 1 else None

            for tr in live:
                quad = tr.quad
                patch = rail_fill(frame, quad)
                if patch.size == 0:
                    continue
                fill = np.maximum(fill, warp_fill(frame, quad, patch))
                cv2.fillConvexPoly(area, canonical(quad).astype(np.int32), 255)

            if area.any() and cfg["avoid_pitch"]:
                area = cv2.bitwise_and(area, cv2.bitwise_not(pitch_mask(frame) * 255))
            if area.any() and overlay is not None:
                area = cv2.bitwise_and(area, cv2.bitwise_not(overlay))
            if area.any() and cfg["protect_goal"]:
                area = cv2.bitwise_and(area, cv2.bitwise_not(crossing_structure(frame, area)))
            if area.any():
                soft = cv2.GaussianBlur(area, (feather | 1, feather | 1), 0) \
                    if feather > 1 else area
                people = pm(frame, [t.quad for t in live])
                soft = cv2.bitwise_and(soft, cv2.bitwise_not(people))
                a = (soft.astype(np.float32) / 255.0)[:, :, None]
                frame = (frame * (1 - a) + fill * a).astype(np.uint8)
                stats["painted"] += 1

        if debug:
            for tr in live:
                cv2.polylines(frame, [tr.quad.astype(np.int32)], True, (0, 255, 255), 2)
        if log is not None:
            log.write(json.dumps({
                "f": stats["frames"] - 1, "cut": cut_now, "det": True,
                "H": None if motion is None or motion.H is None
                else np.round(motion.H, 8).ravel().tolist(),
                "p": [[t.id, "det" if t.misses == 0 else "coast", 1.0,
                       np.round(t.quad, 2).ravel().tolist()] for t in live]}) + "\n")
        writer.write(frame)

    cap.release()
    writer.release()
    if log is not None:
        log.close()
    _reencode(tmp, video, out_path)
    return stats


DEFAULTS = dict(
    # homography = render_engine.AdBlockEngine (camera-geometry tracker);
    # legacy = process_legacy (the previous detection-gated tracker)
    tracker="homography",
    ad_weights=str(ROOT / "models" / "ad_seg_best.pt"),
    person_weights="yolo11m-seg.pt",
    # "1920", or "HxW" for a fixed-shape TensorRT engine (e.g. "1088x1920")
    ad_imgsz="1920", person_imgsz="1280",
    render_conf=0.07, person_conf=0.15,
    person_dilate=9,
    device="",               # "" = first CUDA GPU if present; "0", "cpu", ...
    half=1,                  # FP16 inference on CUDA (ignored for .engine files)

    # ---- homography tracker (render_engine.py / board_tracker.py) ----
    det_interval=1,          # run the board detector every N frames; the
                             # homography carries boards in between (live: 2)
    delay_frames=15,         # paint frame t-D while analysing frame t: panels
                             # start on a board's first frame, end right after
                             # its last detection, and OCR has time to answer
    backfill_frames=15,      # how far before its first detection a board is painted
    fade_in=3,               # frames to fade in a board picked up mid-shot in plain view
    motion_width=960,        # working width for the homography fit
    cut_hist_hard=0.45,      # HSV-histogram correlation below this = hard cut
    cut_hist_soft=0.75,      # ...below this AND no motion fit = cut (dissolve/wipe)
    motion_exclude_boxes="0,0,0.30,0.16;0.70,0,1,0.16",   # live only, until the
                             # online overlay estimate exists: score bug corners
    trk_min_hits=2, trk_confirm_window=4, trk_birth_conf=0.12,
    trk_coast_short=8,       # detector runs a board may go unseen in plain view
    trk_coast_long=40,       # ...during fast pans / at the frame edge
    trk_fast_pan_px=14.0,
    trk_r_across=6.0, trk_r_along=60.0,   # detector noise (px^2): edges vs ends
    trk_q_base=0.05, trk_q_pan=0.03,      # camera-model noise
    trk_gate=3.0, trk_end_tail=3,
    roi_person=1,            # person model only in a region around the panels
    roi_upscale=1.5, person_every=1, person_dynamic=0,
    person_hist=3,           # frames of camera-warped person masks unioned
    protect_ema=0.5, protect_on=0.6, protect_off=0.3,   # pitch/goal hysteresis
    ocr_async=0,             # files: inline (deterministic); live: 1
    ocr_wait=8,              # frames a new board waits for its OCR verdict
    ocr_retry=30, ocr_recheck=300, ocr_max_attempts=6,

    # ---- legacy tracker only ----
    track_iou=0.30, min_hits=1,
    # Dead-reckoned holds (Track.miss) now keep a lost track moving with its
    # last velocity rather than freezing, so a shorter hold is enough to
    # bridge a real 1-3 frame miss (a player crossing, one bad OCR read)
    # while dropping a track faster once it's genuinely lost -- otherwise a
    # stale coasting track can keep coexisting with the fresh one that
    # replaces it, and painting the union of both smears the region.
    max_hold=5,
    ema_alpha=0.20,
    max_size_step=0.15,
    feather=5,
    # A perimeter board is never square; 1.0 let genuinely blob-shaped false
    # detections through on shape alone, leaning entirely on the OCR guard
    # below to catch them.  1.8 still passes every real board measured on this
    # footage (thinnest ones run well past 3:1) while rejecting blobs before
    # they ever reach OCR or the tracker.
    min_aspect=1.8, max_thick_frac=0.12, mask_grow=3, avoid_pitch=1,
    protect_goal=1, protect_overlay=1, extend_rail=0,
    cut_delta=22.0, person_union=5,
    ocr_guard=1,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--panel-log", default=None,
                    help="write per-frame panel quads + camera motion (JSONL) for "
                         "scripts/stability_report.py")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in DEFAULTS}
    src = Path(args.input)
    out = Path(args.output) if args.output else ROOT / "outputs" / f"{src.stem}_clean.mp4"
    stats = process(src, out, cfg, args.debug, args.panel_log)
    print(json.dumps({"input": src.name, "output": str(out), **stats}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
