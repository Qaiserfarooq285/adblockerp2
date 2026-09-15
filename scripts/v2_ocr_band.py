#!/usr/bin/env python3
"""Third OCR pass, targeted at the perimeter-board band specifically.

Root-caused from a real miss: in wide, elevated, far-distance shots (both
corners of the pitch visible, players small), the board strip is maybe
20-30px tall in a 1920x1080 frame -- both the whole-frame OCR pass and the
generic 3x2-grid tiled pass (v2_ocr_tiles.py) spread their resolution across
the whole frame and never zoom in enough to make that text legible, so
"Kalshi" / "ADI PREDICTSTREET" boards in this framing never got OCR'd, never
became a training label, and the trained model correctly never learned to
segment them tightly -- it produces low-aspect blobby masks that the render
pipeline's shape filter (correctly) rejects, leaving the board unpainted.

This pass locates the board band directly (pitch_top_edge, same heuristic
v2_synth.py already uses to know where boards sit) and tiles ALONG that band
only, upscaled aggressively per-tile so the text is actually legible, instead
of tiling the whole frame generically. Results merge into the same
data/ocr/<clip>/<frame>.json the other two passes write, deduplicated by IoU
against everything already there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from v2_synth import pitch_top_edge  # noqa: E402


def quad_box(q) -> tuple[float, float, float, float]:
    a = np.asarray(q, np.float32).reshape(-1, 2)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


_ENGINE = None
_ARGS = None


def _init(cfg) -> None:
    global _ENGINE, _ARGS
    from rapidocr_onnxruntime import RapidOCR
    _ENGINE = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    _ARGS = cfg


def _work(fp_str: str) -> int:
    return process_frame(Path(fp_str), _ENGINE, _ARGS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--ocr", default=str(ROOT / "data" / "ocr"))
    ap.add_argument("--n-tiles", type=int, default=10,
                    help="horizontal segments across the board band")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--above", type=float, default=0.16,
                    help="fraction of frame height to include above the pitch edge")
    ap.add_argument("--below", type=float, default=0.02,
                    help="fraction of frame height to include below the pitch edge")
    ap.add_argument("--target-band-h", type=int, default=140,
                    help="upscale each tile so the band is this many px tall")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    frames = sorted(Path(args.frames).rglob("*.jpg"))
    todo = [str(p) for p in frames
            if not (Path(args.ocr) / p.relative_to(args.frames).with_suffix(".band")).exists()]
    print(f"[band] {len(todo)} frames pending of {len(frames)}, workers={args.workers}",
          flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        added_total = done = 0
        with ctx.Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
            for added in pool.imap_unordered(_work, todo, chunksize=4):
                added_total += added
                done += 1
                if done % 200 == 0:
                    print(f"[band] {done}/{len(todo)} new_boxes={added_total}", flush=True)
        print(f"[band] done {done} frames, {added_total} new text boxes")
        return 0

    _init(args)
    added_total = sum(process_frame(Path(p), _ENGINE, args) for p in todo)
    print(f"[band] done {len(todo)} frames, {added_total} new text boxes")
    return 0


def process_frame(fp: Path, engine, args) -> int:
    rel = fp.relative_to(args.frames).with_suffix(".json")
    jf = Path(args.ocr) / rel
    marker = jf.with_suffix(".band")
    if marker.exists():
        return 0

    img = cv2.imread(str(fp))
    if img is None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("0")
        return 0
    H, W = img.shape[:2]

    try:
        recs = json.loads(jf.read_text()) if jf.exists() else []
    except Exception:
        recs = []
    boxes = [quad_box(r["quad"]) for r in recs]
    added = 0

    edge = pitch_top_edge(img)
    if edge is not None:
        tw = W / args.n_tiles
        ox = tw * args.overlap
        for c_i in range(args.n_tiles):
            x0 = int(max(0, c_i * tw - ox))
            x1 = int(min(W, (c_i + 1) * tw + ox))
            if x1 - x0 < 8:
                continue
            col_edge = edge[x0:x1]
            y_top = int(max(0, col_edge.min() - args.above * H))
            y_bot = int(min(H, col_edge.max() + args.below * H))
            if y_bot - y_top < 6:
                continue
            tile = img[y_top:y_bot, x0:x1]
            if tile.size == 0:
                continue
            band_h = y_bot - y_top
            scale = min(4.0, max(1.0, args.target_band_h / max(band_h, 1)))
            if scale != 1.0:
                tile = cv2.resize(tile, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            try:
                res, _ = engine(tile)
            except Exception:
                continue
            for item in (res or []):
                quad, text, conf = item[0], str(item[1]), float(item[2])
                if conf < args.min_conf:
                    continue
                q = np.asarray(quad, np.float32) / scale
                q[:, 0] += x0
                q[:, 1] += y_top
                bb = quad_box(q)
                if any(iou(bb, e) > 0.45 for e in boxes):
                    continue
                boxes.append(bb)
                recs.append({"text": text, "conf": round(conf, 3),
                             "quad": q.round(1).tolist(), "src": "band"})
                added += 1

    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(recs))
    marker.write_text(str(added))
    return added


if __name__ == "__main__":
    raise SystemExit(main())
