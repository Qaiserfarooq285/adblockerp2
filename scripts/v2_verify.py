#!/usr/bin/env python3
"""Measure the rendered clips instead of eyeballing them.

Four questions, each answered by counting rather than by looking:

  1. Did the betting brands actually disappear?      OCR before vs after
  2. Were the innocent sponsors left alone?          OCR before vs after
  3. Were people painted over?                       person mask vs changed px
  4. Does the panel sit still?                       jitter of the changed region

Writes outputs/verify_report.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Minimum per-pixel change that counts as paint rather than re-encode noise.
PAINT_DELTA = 45
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from v2_brands import classify  # noqa: E402
from v2_render import PAINT_COLOR_BGR  # noqa: E402

# H.264 is lossy, so a perfectly flat source region drifts a little on
# re-encode; this is how much drift still counts as "the fixed paint colour"
# rather than a genuine second colour.
COLOR_TOL = 18


def sample_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    k = min(k, n)
    return [int(round((i + 0.5) * n / k)) - 1 for i in range(k)]


def read_frames(path: Path, idxs: list[int]) -> dict[int, np.ndarray]:
    """Sequential read; seeking in these files returns damaged frames."""
    want = set(idxs)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    i = 0
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = fr
            want.discard(i)
        i += 1
    cap.release()
    return out


def ocr_counts(engine, frames: dict[int, np.ndarray]) -> tuple[int, int]:
    bet = neg = 0
    for fr in frames.values():
        try:
            res, _ = engine(fr)
        except Exception:
            continue
        for item in (res or []):
            if float(item[2]) < 0.5:
                continue
            c = classify(str(item[1]))
            if c == "bet":
                bet += 1
            elif c == "neg":
                neg += 1
    return bet, neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=str(ROOT / "data" / "raw_videos"))
    ap.add_argument("--clean", default=str(ROOT / "outputs" / "clips"))
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "verify_report.json"))
    args = ap.parse_args()

    from rapidocr_onnxruntime import RapidOCR
    from ultralytics import YOLO
    engine = RapidOCR()
    person = YOLO("yolo11m-seg.pt")

    origs = sorted(Path(args.orig).glob("*.mp4"))
    rows = []
    for i, ov in enumerate(origs, 1):
        cv_path = Path(args.clean) / f"clip{i:02d}_clean.mp4"
        if not cv_path.exists():
            continue
        cap = cv2.VideoCapture(str(ov))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idxs = sample_indices(n, args.samples)

        fo = read_frames(ov, idxs)
        fc = read_frames(cv_path, idxs)
        common = sorted(set(fo) & set(fc))
        if not common:
            continue

        bet_o, neg_o = ocr_counts(engine, {k: fo[k] for k in common})
        bet_c, neg_c = ocr_counts(engine, {k: fc[k] for k in common})

        # Which pixels the renderer actually changed, and whether any of them
        # landed on a person.
        person_hit = 0
        changed_px = 0
        solid_color_px = 0
        centres = []
        target = np.array(PAINT_COLOR_BGR, np.int16)
        for k in common:
            diff = cv2.absdiff(fo[k], fc[k]).max(axis=2)
            # The output is re-encoded, so every pixel moves a little and the
            # error concentrates on high-detail regions -- players and crowd,
            # precisely where a false "painted over a person" reading hurts.
            # Measured on an untouched frame of clip01: 14,258 pixels differ by
            # >18 but only 20 by >45.  Count paint, not the codec.
            changed = (diff > PAINT_DELTA).astype(np.uint8)
            changed_px += int(changed.sum())
            if changed.sum() < 200:
                continue
            ys, xs = np.where(changed > 0)
            centres.append((xs.mean(), ys.mean()))
            # Does the painted-over region actually hold the one fixed colour?
            painted_pixels = fc[k][changed > 0].astype(np.int16)
            close = np.all(np.abs(painted_pixels - target) <= COLOR_TOL, axis=1)
            solid_color_px += int(close.sum())
            r = person.predict(fo[k], conf=0.3, classes=[0], imgsz=960,
                               verbose=False, retina_masks=True)[0]
            pm = np.zeros(changed.shape, np.uint8)
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy():
                    mm = (m > 0.5).astype(np.uint8)
                    if mm.shape != pm.shape:
                        mm = cv2.resize(mm, (pm.shape[1], pm.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    pm |= mm
            person_hit += int((changed & pm).sum())

        overlap = person_hit / changed_px if changed_px else 0.0
        color_consistency = solid_color_px / changed_px if changed_px else None
        rows.append({
            "clip": f"clip{i:02d}",
            "frames_checked": len(common),
            "betting_text_before": bet_o,
            "betting_text_after": bet_c,
            "removed_pct": round(100 * (1 - bet_c / bet_o), 1) if bet_o else None,
            "other_sponsors_before": neg_o,
            "other_sponsors_after": neg_c,
            "kept_pct": round(100 * neg_c / neg_o, 1) if neg_o else None,
            "person_pixel_overlap_pct": round(100 * overlap, 3),
            "fill_color_consistency_pct": round(100 * color_consistency, 1)
            if color_consistency is not None else None,
        })
        print(f"  {rows[-1]['clip']}: betting {bet_o}->{bet_c}, "
              f"others {neg_o}->{neg_c}, person overlap {100 * overlap:.2f}%, "
              f"fill consistency {rows[-1]['fill_color_consistency_pct']}%", flush=True)

    tot_b_o = sum(r["betting_text_before"] for r in rows)
    tot_b_c = sum(r["betting_text_after"] for r in rows)
    tot_n_o = sum(r["other_sponsors_before"] for r in rows)
    tot_n_c = sum(r["other_sponsors_after"] for r in rows)
    consistencies = [r["fill_color_consistency_pct"] for r in rows
                     if r["fill_color_consistency_pct"] is not None]
    summary = {
        "clips": len(rows),
        "betting_text_removed_pct": round(100 * (1 - tot_b_c / tot_b_o), 1) if tot_b_o else None,
        "other_sponsors_kept_pct": round(100 * tot_n_c / tot_n_o, 1) if tot_n_o else None,
        "max_person_overlap_pct": max((r["person_pixel_overlap_pct"] for r in rows), default=0),
        "min_fill_color_consistency_pct": round(min(consistencies), 1) if consistencies else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    print("\n== summary ==")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
