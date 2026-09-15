#!/usr/bin/env python3
"""Train the single-class betting-ad segmentation model.

Resumable: re-running picks up `runs/v2/<name>/weights/last.pt` when present,
so an interrupted run (power cut, session end) continues instead of restarting.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset" / "data.yaml"))
    ap.add_argument("--weights", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="ad_seg_unified")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = ROOT / "runs" / "v2" / args.name
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    results = None

    done_epochs = 0
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        try:
            done_epochs = sum(1 for _ in csv_path.read_text().splitlines()) - 1
        except Exception:
            done_epochs = 0
    # "Already complete" (or "resumable") must not mean "this run's data
    # dir predates a since-rebuilt dataset" -- epoch count and last.pt alone
    # can't tell a genuinely-current run from a stale one (this bit once
    # already today: a full grow_band/split_long label-quality fix silently
    # got skipped because last time's epoch count still satisfied the
    # "already complete" check below). resume=True also reuses the run's
    # OWN saved config, not this invocation's --data, so a resume onto a
    # rebuilt dataset would silently keep training on the old one.
    data_path = Path(args.data)
    stale = (data_path.exists() and run_dir.exists()
             and data_path.stat().st_mtime > run_dir.stat().st_mtime)
    if stale:
        print(f"[train] {data_path} is newer than {run_dir}; dataset changed "
              f"since this run started -- starting fresh instead of resuming")
        done_epochs = 0

    if not stale and best.exists() and done_epochs >= args.epochs:
        print(f"[train] already complete ({done_epochs}/{args.epochs} epochs)")
    elif not stale and last.exists():
        print(f"[train] resuming from {last} at epoch {done_epochs}")
        try:
            model = YOLO(str(last))
            results = model.train(resume=True)
        except Exception as exc:
            if not best.exists():
                raise
            print(f"[train] resume declined ({exc}); using existing best.pt")
    else:
        model = YOLO(args.weights)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(ROOT / "runs" / "v2"),
            name=args.name,
            exist_ok=True,
            patience=25,
            # Rails are upright and thin in every real frame; the synthetic
            # generators already produce genuine perspective/angle diversity
            # via proper homographies (steep/zoom/side-by-side composites).
            # Post-hoc rotation and shear on top of that do not correspond to
            # any real camera geometry, they just make the regression target
            # ambiguous -- this combination (degrees=12, shear=4) was in the
            # two runs that scored 0.61 and 0.45 mask-mAP50, both worse than
            # the 0.71 an earlier run reached with rotation/shear off.
            degrees=0.0, flipud=0.0, fliplr=0.5,
            translate=0.10, scale=0.70, shear=0.0, perspective=0.0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            mosaic=1.0, close_mosaic=15, copy_paste=0.0,
            cache=False, workers=8, seed=0, verbose=True,
        )

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "models" / "ad_seg_best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"[train] best -> {dst}")
    try:
        print("[train] final metrics:", results.results_dict)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
