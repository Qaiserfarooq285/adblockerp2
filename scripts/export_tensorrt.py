#!/usr/bin/env python3
"""Build TensorRT FP16 engines for the live renderer, on the GPU that runs them.

A TensorRT engine is compiled for one GPU architecture and one TensorRT
version, so it cannot be built on a laptop and shipped: this runs on the
RunPod pod itself.  deploy/runpod/start.sh calls it once and caches the result
on the persistent volume, keyed by GPU model.  Expect roughly 2-3x the PyTorch
FP16 throughput for the same weights.

    python scripts/export_tensorrt.py --out /workspace/models/engines \
        --env-file /workspace/models/engines/engines.env

The env file holds AD_WEIGHTS / AD_IMGSZ / PERSON_WEIGHTS / PERSON_IMGSZ lines
for live_stream.py.  The board model is exported at a fixed 1088x1920 (a 1080p
frame letterboxed to a multiple of 32) -- an engine's input shape is fixed, so
AD_IMGSZ must say exactly that.  The person model runs on variable-size crops
around the panels and is exported with dynamic shapes up to --person-imgsz.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(weights: str, imgsz, dynamic: bool, out_dir: Path, workspace: float) -> Path:
    from ultralytics import YOLO
    name = Path(weights).stem
    dst = out_dir / f"{name}.engine"
    src_w = Path(weights)
    if dst.exists() and (not src_w.exists() or dst.stat().st_mtime > src_w.stat().st_mtime):
        print(f"[trt] {dst} is current; skipping")
        return dst
    print(f"[trt] exporting {weights} imgsz={imgsz} dynamic={dynamic} (several minutes)...")
    from render_engine import precision                 # quantize=16 on Ultralytics >= 8.4
    built = Path(YOLO(weights).export(format="engine", **precision(True), imgsz=imgsz, dynamic=dynamic,
                                      batch=1, workspace=workspace, device=0, simplify=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(built), dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ad-weights", default=str(ROOT / "models" / "ad_seg_best.pt"))
    ap.add_argument("--ad-imgsz", default="1088x1920", help="HxW, fixed")
    ap.add_argument("--person-weights", default="yolo11m-seg.pt")
    ap.add_argument("--person-imgsz", type=int, default=1280, help="max side, dynamic")
    ap.add_argument("--out", default=str(ROOT / "models" / "engines"))
    ap.add_argument("--workspace", type=float, default=4.0, help="TensorRT builder memory, GB")
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("TensorRT export needs a CUDA GPU (run this on the RunPod pod)")
    gpu = re.sub(r"[^a-z0-9]+", "_", torch.cuda.get_device_name(0).lower()).strip("_")
    out_dir = Path(args.out) / gpu
    h, w = (int(v) for v in args.ad_imgsz.lower().split("x"))

    ad = export(args.ad_weights, [h, w], False, out_dir, args.workspace)
    person = export(args.person_weights, args.person_imgsz, True, out_dir, args.workspace)
    env = (f"AD_WEIGHTS={ad}\nAD_IMGSZ={h}x{w}\n"
           f"PERSON_WEIGHTS={person}\nPERSON_IMGSZ={args.person_imgsz}\nPERSON_DYNAMIC=1\n")
    print("[trt] done:\n" + env)
    if args.env_file:
        Path(args.env_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.env_file).write_text(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
