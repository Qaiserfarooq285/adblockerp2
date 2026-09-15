#!/usr/bin/env python3
"""Turn the OCR survey into polygon labels for betting ad boards.

Why polygons and not boxes: the perimeter rails run at a strong perspective
slant, so an axis-aligned box around "ADI PREDICTSTREET" swallows large
triangles of grass, goal net and crowd.  A four-point polygon follows the rail
and is what the renderer needs anyway.

Pipeline per frame:
  1. classify each OCR quad as betting / non-betting (scripts/v2_brands.py)
  2. group betting quads that sit on the same physical rail
  3. merge each group into one oriented quad
  4. grow the quad perpendicular to the rail until the board surface ends
  5. write an Ultralytics segmentation label

Frames with no betting text become explicit background images, but only when
the survey clearly saw the rails -- see `frame_is_safe_negative`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v2_brands import classify  # noqa: E402

MIN_QUAD_H = 4.5        # px; OCR noise below this is unusable
MIN_QUAD_W = 10.0
MAX_GROW = 2.6          # cap band growth at this multiple of text height
MIN_LABEL_ASPECT = 3.0  # real boards in this footage run well past 3:1;
                        # grow_band's own budget can still leave a squat
                        # result on a generic dark palette, so enforce this
                        # directly rather than trusting the multiple alone


# ---------------------------------------------------------------- geometry --

def quad_metrics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (centre, unit direction along the long axis, length, thickness)."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    centre = q.mean(axis=0)
    # Long axis = mean of the two roughly-horizontal edges.
    e1 = (q[1] - q[0]) + (q[2] - q[3])
    length = float(np.linalg.norm(e1) / 2.0)
    direction = e1 / (np.linalg.norm(e1) + 1e-6)
    e2 = (q[3] - q[0]) + (q[2] - q[1])
    thickness = float(np.linalg.norm(e2) / 2.0)
    return centre, direction, length, thickness


def same_rail(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two OCR quads plausibly sit on one physical board run."""
    ca, da, la, ta = quad_metrics(a)
    cb, db, lb, tb = quad_metrics(b)
    if min(ta, tb) < MIN_QUAD_H:
        return False
    # Similar apparent height (perspective changes it only gradually).
    if max(ta, tb) / max(1e-6, min(ta, tb)) > 2.5:
        return False
    # Similar slant.
    if abs(float(np.dot(da, db))) < 0.88:
        return False
    # Offset perpendicular to the rail must be small ...
    delta = cb - ca
    perp = abs(float(delta[0] * -da[1] + delta[1] * da[0]))
    if perp > 1.4 * max(ta, tb):
        return False
    # ... and the along-rail gap must be a small multiple of the height.
    along = abs(float(np.dot(delta, da)))
    gap = along - (la + lb) / 2.0
    return gap < 3.0 * max(ta, tb)


def group_quads(quads: list[np.ndarray]) -> list[list[int]]:
    """Union-find over `same_rail` so a broken-up board becomes one group."""
    n = len(quads)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if same_rail(quads[i], quads[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def merge_quad(quads: list[np.ndarray]) -> np.ndarray:
    """Oriented bounding quad over several OCR quads."""
    pts = np.concatenate([np.asarray(q, np.float32).reshape(4, 2) for q in quads])
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.float32)


def order_quad(q: np.ndarray) -> np.ndarray:
    """Order corners tl, tr, br, bl with the long axis horizontal-ish."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    # Rotate the corner list so edge 0->1 is the longest.
    best, bi = -1.0, 0
    for i in range(4):
        d = np.linalg.norm(q[(i + 1) % 4] - q[i])
        if d > best:
            best, bi = d, i
    q = np.roll(q, -bi, axis=0)
    if q[0][0] > q[1][0]:                       # make it left-to-right
        q = q[[1, 0, 3, 2]]
    if q[0][1] > q[3][1]:                       # make row 0 the top edge
        q = q[[3, 2, 1, 0]]
    return q


# --------------------------------------------------------------- band grow --

def rectify(img: np.ndarray, quad: np.ndarray, pad_mult: float) -> tuple[np.ndarray, np.ndarray]:
    """Warp the rail to an axis-aligned patch with vertical headroom."""
    q = order_quad(quad)
    _, _, length, thickness = quad_metrics(q)
    W = max(16, int(round(length)))
    H = max(8, int(round(thickness)))
    pad = int(round(H * pad_mult))
    dst = np.float32([[0, pad], [W, pad], [W, pad + H], [0, pad + H]])
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    patch = cv2.warpPerspective(img, M, (W, H + 2 * pad), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    return patch, np.linalg.inv(M)


def grow_band(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Extend the OCR quad perpendicular to the rail onto the board surface.

    The wordmark rarely fills the board: Betano rails leave red margin above
    and below the glyphs.  We rectify the rail, describe the text rows by their
    colour distribution, and walk outward while rows still look like the same
    surface.
    """
    q = order_quad(quad)
    _, _, _, thickness = quad_metrics(q)
    if thickness < MIN_QUAD_H:
        return q
    patch, Minv = rectify(img, q, MAX_GROW)
    ph, pw = patch.shape[:2]
    pad = int(round(thickness * MAX_GROW))
    core = patch[pad:ph - pad]
    if core.size == 0:
        return q

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    quant = (lab.astype(np.int32) // 16)
    codes = (quant[:, :, 0] << 16) | (quant[:, :, 1] << 8) | quant[:, :, 2]

    # A board is a *palette*, not one colour: Betano rails are red plus white
    # glyphs, and picking only the modal colour lands on the glyphs (they win
    # the vote on tightly-cropped OCR quads), after which the plain-red margin
    # rows fail to match and the band never grows.  Keep every bin that carries
    # the bulk of the core instead.
    core_codes = codes[pad:ph - pad].ravel()
    keys, counts = np.unique(core_codes, return_counts=True)
    order = np.argsort(-counts)
    keep = counts[order].cumsum() <= 0.92 * core_codes.size
    keep[0] = True                                    # always keep the mode
    palette = keys[order][keep]

    match = np.isin(codes, palette).mean(axis=1).astype(np.float32)
    core_match = float(match[pad:ph - pad].mean())
    # Margin rows carry no glyphs, so they should be *more* uniform than the
    # core; demand that rather than merely matching it.
    thresh = max(0.55, min(0.85, core_match))

    top = pad
    while top - 1 >= 0 and match[top - 1] >= thresh:
        top -= 1
    bot = ph - pad - 1
    while bot + 1 < ph and match[bot + 1] >= thresh:
        bot += 1

    # MAX_GROW names "the multiple of text height this may grow to", but
    # nothing enforced that -- the walk could consume the whole search pad
    # on each side, i.e. grow to 1+2*MAX_GROW times the original thickness,
    # not MAX_GROW times. On a black-LED board (Kalshi, ADI PREDICTSTREET)
    # the palette is just black+white/green, generic enough to keep
    # matching straight into the crowd behind it in a wide shot -- measured
    # on real footage, a clean 1418x151px rail grew to 1418x550px, and the
    # model trained on that "learned" boards are nearly-square blobs, which
    # is exactly the shape the render-time aspect filter then (correctly)
    # rejects. Clamp growth to what the constant already claims to be.
    max_extra = thickness * (MAX_GROW - 1.0) / 2.0
    top = max(top, pad - max_extra)
    bot = min(bot, ph - pad - 1 + max_extra)

    # Belt and braces: even within that multiple-of-original-height budget,
    # a long rail (pw large) can still grow thick enough to read as a squat
    # blob rather than a strip -- real boards in this footage are
    # consistently well past 3:1 end to end, so treat anything thicker than
    # that as still-wrong growth and pull it back in, centred on the walk's
    # own result rather than the original OCR band (a curved rail's true
    # centreline isn't necessarily the raw text's).
    grown_h = bot - top + 1
    min_h = max(pw / MIN_LABEL_ASPECT, thickness)
    if grown_h > min_h:
        centre = (top + bot) / 2.0
        top = centre - min_h / 2.0
        bot = centre + min_h / 2.0

    if top >= pad and bot <= ph - pad - 1:
        return q

    src = np.float32([[0, top], [pw, top], [pw, bot + 1], [0, bot + 1]]).reshape(-1, 1, 2)
    grown = cv2.perspectiveTransform(src, Minv).reshape(4, 2)
    return order_quad(grown)


def split_long(quad: np.ndarray, max_len: float = 500.0) -> list[np.ndarray]:
    """Cut a very long rail into segments the detector can actually predict.

    Measured on the trained model, every proposal on a full-width rail comes
    back about 950px long regardless of confidence, so boards continuing past
    that -- the ones behind the goal -- are never predicted at all.  22% of
    these labels are longer than 600px and some span the whole frame, which is
    what taught that ceiling.

    Segments are safe for the renderer: it fuses collinear neighbours back into
    one panel, so the output is unchanged while the training target becomes
    something the mask head can represent.

    max_len alone can still produce a squat, wrong-shaped segment: a rail
    thick enough (e.g. one grown to the edge of grow_band's own budget on a
    hard scene) splits into pieces shorter than they are thick once cut every
    500px, which is exactly the blob shape this whole module exists to avoid
    labelling. Segment length is widened to whatever keeps pieces at least
    MIN_LABEL_ASPECT long relative to their own thickness, up to the 950px
    ceiling the model was actually measured against -- so a thick rail is cut
    into fewer, longer, correctly-proportioned pieces instead.
    """
    q = order_quad(quad)
    _, _, length, thickness = quad_metrics(q)
    eff_max_len = max_len
    if thickness > 1e-3:
        eff_max_len = min(950.0, max(max_len, thickness * MIN_LABEL_ASPECT))
    n = int(np.ceil(length / eff_max_len))
    if n <= 1:
        return [q]
    top, bot = q[1] - q[0], q[2] - q[3]
    out = []
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        out.append(order_quad(np.float32([
            q[0] + top * t0, q[0] + top * t1, q[3] + bot * t1, q[3] + bot * t0])))
    return out


def add_margin(quad: np.ndarray, frac: float = 0.06) -> np.ndarray:
    """Push the quad outward slightly in every direction.

    Under-coverage is the expensive error: a surviving rim of the ad is still
    legible after inpainting.  Overshooting by a few percent lands on the wall
    or grass either side, which the inpainter reconstructs anyway.
    """
    q = order_quad(quad).astype(np.float32)
    centre = q.mean(axis=0)
    return centre + (q - centre) * (1.0 + frac)


# ------------------------------------------------------------------ labels --

def frame_is_safe_negative(recs: list[dict]) -> bool:
    """A frame may serve as a background image only if OCR clearly worked.

    A frame with no betting text could mean 'no betting board' or 'OCR failed'.
    Requiring several confidently-read non-betting sponsors distinguishes the
    two, so we never teach the model that a real board is background.
    """
    negs = sum(1 for r in recs if classify(r["text"]) == "neg" and r["conf"] >= 0.6)
    others = sum(1 for r in recs if classify(r["text"]) == "other")
    return negs >= 2 or (negs >= 1 and others >= 4) or len(recs) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--ocr", default=str(ROOT / "data" / "ocr"))
    ap.add_argument("--out", default=str(ROOT / "data" / "auto_labels"))
    ap.add_argument("--viz", default=str(ROOT / "data" / "auto_labels" / "_viz"))
    ap.add_argument("--viz-every", type=int, default=40)
    args = ap.parse_args()

    out_root = Path(args.out)
    viz_root = Path(args.viz)
    out_root.mkdir(parents=True, exist_ok=True)
    viz_root.mkdir(parents=True, exist_ok=True)

    ocr_files = sorted(Path(args.ocr).glob("clip*/*.json"))
    stats = {"frames": 0, "pos": 0, "neg": 0, "skip": 0, "polys": 0, "grown": 0}
    manifest = []

    for i, jf in enumerate(ocr_files):
        rel = jf.relative_to(args.ocr).with_suffix("")
        img_path = Path(args.frames) / rel.with_suffix(".jpg")
        if not img_path.exists():
            continue
        recs = json.loads(jf.read_text())
        stats["frames"] += 1

        bet_quads = []
        for r in recs:
            q = np.asarray(r["quad"], np.float32).reshape(4, 2)
            _, _, length, thickness = quad_metrics(q)
            if thickness < MIN_QUAD_H or length < MIN_QUAD_W:
                continue
            if classify(r["text"]) == "bet":
                bet_quads.append(q)

        if not bet_quads:
            if frame_is_safe_negative(recs):
                stats["neg"] += 1
                manifest.append({"image": str(img_path), "label": [], "kind": "neg"})
            else:
                stats["skip"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            stats["skip"] += 1
            continue
        h, w = img.shape[:2]

        polys = []
        for idx in group_quads(bet_quads):
            merged = merge_quad([bet_quads[k] for k in idx])
            before = quad_metrics(merged)[3]
            grown = grow_band(img, merged)
            after = quad_metrics(grown)[3]
            if after > before * 1.08:
                stats["grown"] += 1
            grown = add_margin(grown)
            for seg in split_long(grown):
                seg[:, 0] = np.clip(seg[:, 0], 0, w - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, h - 1)
                polys.append(seg)

        stats["pos"] += 1
        stats["polys"] += len(polys)
        manifest.append({
            "image": str(img_path),
            "label": [p.round(2).tolist() for p in polys],
            "kind": "pos",
        })

        if i % args.viz_every == 0:
            vis = img.copy()
            for p in polys:
                cv2.polylines(vis, [p.astype(np.int32)], True, (0, 0, 255), 3)
            dst = viz_root / f"{rel.as_posix().replace('/', '_')}.jpg"
            cv2.imwrite(str(dst), vis)

    # A board that OCR missed on a single frame would otherwise be handed to
    # the model as background.  Rails persist for seconds at a time, so drop
    # any negative sitting within `guard` frames of a positive in the same clip.
    guard = 3
    pos_idx: dict[str, set[int]] = {}
    for m in manifest:
        p = Path(m["image"])
        if m["kind"] == "pos":
            pos_idx.setdefault(p.parent.name, set()).add(int(p.stem))
    dropped = 0
    for m in manifest:
        if m["kind"] != "neg":
            continue
        p = Path(m["image"])
        near = pos_idx.get(p.parent.name, ())
        n = int(p.stem)
        if any(n + d in near for d in range(-guard, guard + 1)):
            m["kind"] = "drop"
            dropped += 1
    stats["neg"] -= dropped
    stats["neg_dropped_near_pos"] = dropped
    manifest = [m for m in manifest if m["kind"] != "drop"]

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(stats, indent=2))
    print(f"[annotate] manifest -> {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
