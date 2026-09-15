#!/usr/bin/env python3
"""Recall broken down by exactly the cases the project has been failing on.

An overall mAP number hides which specific promise is still broken -- it was
mAP alone that made two earlier retraining rounds look plausible right up
until a frame-by-frame look showed Kalshi and behind-goal boards were still
being missed.  This measures the failure taxonomy directly: brand, board
size (small/medium/large -- large stands in for "close/zoomed/ground-filling"
since that is what a large board-mask area means in this footage), and a
side-by-side flag, all against the held-out validation fixtures so the score
is honest.

Writes outputs/eval_targeted.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v2_brands import classify, norm  # noqa: E402


def brand_of(poly, ocr_recs) -> str:
    P = np.asarray(poly, np.float32)
    for r in ocr_recs:
        if classify(r["text"]) != "bet":
            continue
        c = np.asarray(r["quad"], np.float32).mean(axis=0)
        if cv2.pointPolygonTest(P, (float(c[0]), float(c[1])), False) >= 0:
            t = norm(r["text"])
            if "alsh" in t:
                return "kalshi"
            if "dictstr" in t or "redict" in t:
                return "predict"
            if "etano" in t or "betan" in t:
                return "betano"
    return "other"


def poly_mask(poly, h, w) -> np.ndarray:
    z = np.zeros((h, w), np.uint8)
    cv2.fillPoly(z, [np.asarray(poly, np.int32)], 1)
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "auto_labels" / "manifest.json"))
    ap.add_argument("--weights", default=str(ROOT / "models" / "ad_seg_best.pt"))
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--sample", type=int, default=160)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_targeted.json"))
    args = ap.parse_args()

    val_fixtures = None
    val_yaml = ROOT / "data" / "dataset" / "data.yaml"
    # Restrict to the same held-out clips v2_dataset.py chose, recovered from
    # which frames actually ended up under images/val -- keeps this honest
    # against whatever split the current dataset build produced.
    val_dir = ROOT / "data" / "dataset" / "images" / "val"
    val_stems = {p.stem for p in val_dir.glob("*.jpg")} if val_dir.exists() else set()

    man = json.loads(Path(args.manifest).read_text())
    pos = [m for m in man if m["kind"] == "pos"]
    if val_stems:
        def in_val(m):
            src = Path(m["image"])
            stem = f"{src.parent.name}_{src.stem}"
            return stem in val_stems
        pos = [m for m in pos if in_val(m)]
    if not pos:
        print("[eval] no held-out frames matched the current val split; "
              "falling back to all real positive frames")
        pos = [m for m in json.loads(Path(args.manifest).read_text()) if m["kind"] == "pos"]

    import random
    random.seed(0)
    if len(pos) > args.sample:
        pos = random.sample(pos, args.sample)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    hit_brand, tot_brand = defaultdict(int), defaultdict(int)
    hit_size, tot_size = defaultdict(int), defaultdict(int)
    sbs_hit, sbs_tot = 0, 0

    for m in pos:
        img = cv2.imread(m["image"])
        if img is None:
            continue
        h, w = img.shape[:2]
        ocr_path = m["image"].replace("data/frames", "data/ocr").replace(".jpg", ".json")
        try:
            recs = json.loads(Path(ocr_path).read_text())
        except Exception:
            recs = []

        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False,
                          retina_masks=True)[0]
        dm = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            for k in r.masks.data.cpu().numpy():
                q = (k > 0.5).astype(np.uint8)
                if q.shape != (h, w):
                    q = cv2.resize(q, (w, h), interpolation=cv2.INTER_NEAREST)
                dm |= q

        n_boards = len(m["label"])
        for poly in m["label"]:
            gt = poly_mask(poly, h, w)
            area = int(gt.sum())
            if area < 50:
                continue
            ok = (gt & dm).sum() / area > 0.4

            b = brand_of(poly, recs)
            hit_brand[b] += int(ok)
            tot_brand[b] += 1

            size = "small(<8k)" if area < 8000 else ("medium(8-25k)" if area < 25000 else "large(>25k)")
            hit_size[size] += int(ok)
            tot_size[size] += 1

            if n_boards >= 2:
                sbs_hit += int(ok)
                sbs_tot += 1

    def pct(h, t):
        return round(100 * h / t, 1) if t else None

    report = {
        "n_frames": len(pos),
        "by_brand": {b: {"recall_pct": pct(hit_brand[b], tot_brand[b]), "n": tot_brand[b]}
                    for b in sorted(tot_brand)},
        "by_size": {s: {"recall_pct": pct(hit_size[s], tot_size[s]), "n": tot_size[s]}
                   for s in sorted(tot_size)},
        "side_by_side": {"recall_pct": pct(sbs_hit, sbs_tot), "n": sbs_tot},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"[eval] {len(pos)} held-out frames, weights={args.weights}")
    print("by brand:")
    for b, v in report["by_brand"].items():
        print(f"   {b:<10} {v['recall_pct']}%  (n={v['n']})")
    print("by board size:")
    for s, v in report["by_size"].items():
        print(f"   {s:<14} {v['recall_pct']}%  (n={v['n']})")
    print(f"side-by-side frames: {report['side_by_side']['recall_pct']}%  "
          f"(n={report['side_by_side']['n']})")
    print(f"[eval] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
