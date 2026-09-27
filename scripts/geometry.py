#!/usr/bin/env python3
"""Camera geometry: how the picture moved between two consecutive frames.

A panel has to stay glued to a physical board while the camera pans, tilts
and zooms.  A broadcast camera on a head turns about (very nearly) its own
optical centre, and for a camera that only rotates and zooms, the image motion
of *every* scene point -- crowd, boards, pitch, at any depth -- is one and the
same 3x3 homography.  One homography per frame is therefore not an
approximation of how the board moved; for the main camera it *is* how the
board moved, perspective foreshortening included.

The first attempt at camera-motion propagation (process_video.py's
CameraMotion) drifted during pans, which is why v2_render.py dropped the idea.
Each cause of that drift is addressed here rather than repeated:

* It fitted a 4-DOF similarity (estimateAffinePartial2D).  A similarity cannot
  express the keystone change of a pan through a wide lens: its error is zero
  at the frame centre and largest at the frame edges -- exactly where the
  perimeter boards sit.  -> full 8-DOF homography.
* It took features everywhere, including the burned-in scoreboard, which never
  moves and so votes for "the camera is still", and the LED boards, whose
  animated content moves on its own.  -> both are masked out of the fit.
* Lucas-Kanade mismatches on motion-blurred frames went straight into the fit.
  -> forward-backward consistency check before RANSAC.
* Nothing corrected the propagated panel except 3%/frame toward the detector,
  which cannot undo a biased motion model.  -> board_tracker.py fuses this
  prediction with every detection through a Kalman gain, so model error is
  shed continuously instead of accumulating into a slide.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# MAGSAC++ is both more accurate and less threshold-sensitive than plain
# RANSAC; OpenCV < 4.5 lacks it, so fall back rather than fail.
_ROBUST = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


# ------------------------------------------------------------ quad helpers

def warp_quad(H: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Map a quad's corners through a homography; None if it degenerates."""
    q = np.asarray(quad, np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(q, np.asarray(H, np.float64)).reshape(-1, 2)
    if not np.all(np.isfinite(out)) or np.abs(out).max() > 1e6:
        return None
    return out.astype(np.float32)


def polygon_area(q: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(q, np.float32).reshape(-1, 1, 2))))


def is_convex(q: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(np.asarray(q, np.float32).reshape(-1, 1, 2)))


def _hull(q: np.ndarray) -> np.ndarray:
    return cv2.convexHull(np.asarray(q, np.float32).reshape(-1, 1, 2))


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    """Exact overlap of two (near-)convex polygons."""
    try:
        area, _ = cv2.intersectConvexConvex(_hull(a), _hull(b))
    except cv2.error:
        return 0.0
    return max(0.0, float(area))


def convex_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Exact IoU of two quads.

    v2_render.quad_iou rasterises at 640 px, which is fine once per frame but
    too coarse and too slow to run for every track x detection pair; a thin
    far-side rail is 2-3 px tall at that scale.
    """
    aa, ab = polygon_area(a), polygon_area(b)
    if aa <= 0 or ab <= 0:
        return 0.0
    inter = intersection_area(a, b)
    return inter / max(1e-6, aa + ab - inter)


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    ai = polygon_area(inner)
    return intersection_area(inner, outer) / ai if ai > 0 else 0.0


def visible_fraction(q: np.ndarray, w: int, h: int) -> float:
    """Fraction of a quad that is on screen."""
    frame = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return containment(q, frame)


def quad_axes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(centre, unit direction along the rail, unit normal, length, thickness).

    Works for any consistent corner order: the rail direction is taken from
    whichever pair of opposite edges is longer.
    """
    q = np.asarray(q, np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    a = (q[1] - q[0]) + (q[2] - q[3])
    b = (q[3] - q[0]) + (q[2] - q[1])
    la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if la >= lb:
        d, length, thick = a / max(la, 1e-6), la / 2.0, lb / 2.0
    else:
        d, length, thick = b / max(lb, 1e-6), lb / 2.0, la / 2.0
    n = np.array([-d[1], d[0]], np.float32)
    return c, d.astype(np.float32), n, length, thick


def align_quad(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Reorder q's corners to correspond one-to-one with ref's.

    canonical() orders corners by edge length and left/right, which is stable
    for a normal rail but can flip for a near-vertical one behind the goal.
    Corner correspondence must never depend on that, or a Kalman update would
    pull the top-left corner toward the bottom-right one.  Try every rotation
    of both windings and keep the closest.
    """
    q = np.asarray(q, np.float32).reshape(4, 2)
    ref = np.asarray(ref, np.float32).reshape(4, 2)
    best, best_q = np.inf, q
    for cand in (q, q[::-1]):
        for k in range(4):
            c = np.roll(cand, -k, axis=0)
            err = float(((c - ref) ** 2).sum())
            if err < best:
                best, best_q = err, c
    return best_q.astype(np.float32)


# ----------------------------------------------------------- camera motion

@dataclass
class Motion:
    """Camera motion from the previous frame to this one, at full resolution."""
    H: np.ndarray | None = None       # 3x3, previous frame -> this frame
    method: str = "none"              # homography | similarity | none
    inliers: int = 0
    tracked: int = 0
    shift_px: float = 0.0             # median inlier displacement (pan speed)
    residual_px: float = 0.0          # median inlier reprojection error
    cut: bool = False                 # hard scene change between the frames
    hist_corr: float = 1.0

    @property
    def ok(self) -> bool:
        return self.H is not None


class CameraMotionEstimator:
    """Frame-to-frame homography from sparse, masked, FB-checked LK flow."""

    WIN = (21, 21)
    LEVELS = 3     # 21 px window x 2^3 follows ~80 px/frame at work res (160 full res)

    def __init__(self, width: int, height: int, work_width: int = 960,
                 max_corners: int = 600, quality: float = 0.004,
                 min_distance: int = 8, fb_max_px: float = 1.0,
                 ransac_px: float = 1.2, min_inliers: int = 40,
                 cut_hist_hard: float = 0.45, cut_hist_soft: float = 0.75,
                 exclude_boxes: list | None = None):
        self.w, self.h = int(width), int(height)
        self.s = min(1.0, float(work_width) / float(self.w))
        self.ws = max(16, int(round(self.w * self.s)))
        self.hs = max(16, int(round(self.h * self.s)))
        self.max_corners = int(max_corners)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.fb_max = float(fb_max_px)
        self.ransac_px = float(ransac_px)
        self.min_inliers = int(min_inliers)
        self.cut_hard = float(cut_hist_hard)
        self.cut_soft = float(cut_hist_soft)

        base = np.full((self.hs, self.ws), 255, np.uint8)
        m = max(4, int(0.01 * self.ws))
        base[:m, :] = 0            # letterbox bars and encoder edge garbage
        base[-m:, :] = 0
        base[:, :m] = 0
        base[:, -m:] = 0
        # Fixed screen regions that usually carry graphics (score bug, channel
        # logo).  Only used to keep them out of the motion fit, never for
        # painting decisions, and superseded once an overlay mask is known.
        self._boxes = []
        for bx in exclude_boxes or []:
            x0, y0, x1, y1 = (float(v) for v in bx)
            self._boxes.append((int(x0 * self.ws), int(y0 * self.hs),
                                int(x1 * self.ws), int(y1 * self.hs)))
        self.base = base
        self._static_small: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.prev_hist: np.ndarray | None = None

    # -- public --------------------------------------------------------------

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_hist = None

    def set_static_mask(self, mask_full: np.ndarray | None) -> None:
        """Burned-in graphics (0/255, full res) to exclude from the fit."""
        if mask_full is None:
            self._static_small = None
            return
        small = cv2.resize(mask_full, (self.ws, self.hs), interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate((small > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
        self._static_small = np.where(small > 0, 0, 255).astype(np.uint8)

    def estimate(self, frame: np.ndarray, exclude_quads=()) -> Motion:
        """Motion from the previous frame passed in to this one.

        `exclude_quads` are the boards as currently tracked, in PREVIOUS-frame
        coordinates -- the frame the features are picked in.
        """
        small = cv2.resize(frame, (self.ws, self.hs), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = self._hist(small)
        m = Motion()
        if self.prev_hist is not None:
            m.hist_corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
        prev = self.prev_gray
        self.prev_gray, self.prev_hist = gray, hist
        if prev is None:
            return m                           # first frame: nothing to compare
        if m.hist_corr < self.cut_hard:
            m.cut = True                       # unmistakable hard cut
            return m

        mask = self._feature_mask(exclude_quads)
        p0 = self._features(prev, mask)
        if (p0 is None or len(p0) < 3 * self.min_inliers) and exclude_quads:
            # A close-up can be mostly board; losing the fit costs more than
            # letting RANSAC sort out some animated LED pixels.
            p0 = self._features(prev, self._feature_mask(()))
        if p0 is None or len(p0) < 12:
            m.cut = m.hist_corr < self.cut_soft
            return m

        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=self.WIN,
                                             maxLevel=self.LEVELS, criteria=crit)
        if p1 is None:
            m.cut = m.hist_corr < self.cut_soft
            return m
        p0b, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, winSize=self.WIN,
                                                maxLevel=self.LEVELS, criteria=crit)
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1) if p0b is not None \
            else np.full(len(p0), np.inf)
        good = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < self.fb_max)
        a = p0.reshape(-1, 2)[good].astype(np.float32)
        b = p1.reshape(-1, 2)[good].astype(np.float32)
        m.tracked = int(len(a))
        if len(a) < max(12, self.min_inliers // 2):
            # Few trackable points AND a colour change: a cut through a
            # dissolve or a graphic wipe.  Few points alone: whip-pan blur.
            m.cut = m.hist_corr < self.cut_soft
            return m

        H, method, inl = None, "homography", None
        try:
            H, inl = cv2.findHomography(a, b, _ROBUST, self.ransac_px,
                                        maxIters=2000, confidence=0.999)
        except cv2.error:
            H = None
        if not self._plausible(H, inl, self.min_inliers):
            method = "similarity"
            A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.5 * self.ransac_px,
                                                 maxIters=2000, confidence=0.995)
            H = None if A is None else np.vstack([A, [0.0, 0.0, 1.0]])
            if not self._plausible(H, inl, max(12, self.min_inliers // 2)):
                m.cut = m.hist_corr < self.cut_soft
                return m

        inl = inl.ravel().astype(bool)
        ai, bi = a[inl], b[inl]
        proj = cv2.perspectiveTransform(ai.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        m.residual_px = float(np.median(np.linalg.norm(proj - bi, axis=1))) / self.s
        m.shift_px = float(np.median(np.linalg.norm(bi - ai, axis=1))) / self.s
        S = np.diag([self.s, self.s, 1.0])
        Hf = np.linalg.inv(S) @ H @ S
        m.H = Hf / Hf[2, 2]
        m.method, m.inliers = method, int(inl.sum())
        return m

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _hist(small_bgr: np.ndarray) -> np.ndarray:
        tiny = cv2.resize(small_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _feature_mask(self, quads) -> np.ndarray:
        mask = self.base.copy()
        if self._static_small is not None:
            mask &= self._static_small
        else:
            for x0, y0, x1, y1 in self._boxes:
                mask[y0:y1, x0:x1] = 0
        for q in quads:
            pts = np.round(np.asarray(q, np.float32).reshape(4, 2) * self.s).astype(np.int32)
            cv2.fillConvexPoly(mask, pts, 0)
            cv2.polylines(mask, [pts], True, 0, thickness=7)   # small margin
        return mask

    def _features(self, gray: np.ndarray, mask: np.ndarray):
        return cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners,
                                       qualityLevel=self.quality,
                                       minDistance=self.min_distance,
                                       mask=mask, blockSize=7)

    def _plausible(self, H, inl, min_inl: int) -> bool:
        """Reject fits a broadcast camera cannot produce in 1/25-1/60 s.

        The inlier ratio matters as much as the count: when a whip pan moves
        the picture further than LK's pyramid can follow, most "tracked"
        points are garbage and RANSAC will still find *some* consensus among
        them -- a confident, plausible-looking, wrong homography.  A true fit
        explains most of the points; a spurious one only a small minority.
        """
        if H is None or inl is None or int(np.count_nonzero(inl)) < min_inl:
            return False
        if np.count_nonzero(inl) < 0.30 * len(inl):
            return False
        if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-9:
            return False
        c = np.float64([[0, 0], [self.ws, 0], [self.ws, self.hs], [0, self.hs]]).reshape(-1, 1, 2)
        wc = cv2.perspectiveTransform(c, H / H[2, 2]).reshape(-1, 2)
        if not np.all(np.isfinite(wc)) or not is_convex(wc):
            return False
        ratio = polygon_area(wc) / float(self.ws * self.hs)
        if not (0.75 <= ratio <= 1.33):       # >15% linear zoom in one frame
            return False
        if float(np.linalg.norm(wc - c.reshape(-1, 2), axis=1).max()) > 0.30 * self.ws:
            return False
        return True


# ------------------------------------------------ burned-in graphics, live

class OnlineOverlayEstimator:
    """Broadcast graphics (clock, score, channel bug) without the whole clip.

    v2_render.static_overlay_mask samples 40 frames spread over the entire
    file: any pixel that never changes while the camera moves a great deal is
    painted on by the broadcaster.  A live stream has no "entire file", so the
    same statistic runs over a rolling window of samples taken only after the
    camera has actually moved (or the shot changed) -- a long static wide shot
    would otherwise make the whole picture look like a graphic.
    """

    def __init__(self, width: int, height: int, window: int = 48,
                 min_samples: int = 16, min_move_px: float = 60.0,
                 recompute_every: int = 4, small=(320, 180)):
        self.w, self.h = int(width), int(height)
        self.small = small
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.min_move = float(min_move_px)
        self.recompute_every = int(recompute_every)
        self.samples: list[np.ndarray] = []
        self._moved = 1e9
        self._since = 0
        self._mask: np.ndarray | None = None

    def update(self, frame: np.ndarray, motion: Motion) -> None:
        self._moved += motion.shift_px if motion.ok else 0.0
        if motion.cut:
            self._moved = 1e9
        if self._moved < self.min_move:
            return
        self._moved = 0.0
        g = cv2.cvtColor(cv2.resize(frame, self.small, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.samples.append(g)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        self._since += 1
        if len(self.samples) >= self.min_samples and (self._mask is None or
                                                      self._since >= self.recompute_every):
            self._since = 0
            self._mask = self._compute()

    def mask(self) -> np.ndarray | None:
        return self._mask

    def _compute(self) -> np.ndarray | None:
        std = np.std(np.stack(self.samples), axis=0)
        # Same tightening ladder as the offline estimator: only a genuinely
        # burned-in graphic survives the strictest pass.
        for thresh in (7.0, 4.0, 2.5, 1.5):
            m = (std < thresh).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if m.mean() <= 0.12:
                full = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                return cv2.dilate(full, np.ones((9, 9), np.uint8), iterations=1) * 255
        return None
