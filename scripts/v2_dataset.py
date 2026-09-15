#!/usr/bin/env python3
"""Assemble the Ultralytics segmentation dataset from the auto-labels.

Split policy: by *match*, never by frame.  Frames sampled at 2 fps from one
clip are near-duplicates, and clips from the same fixture share a stadium,
rail set and lighting.  Splitting either of those randomly would leak the
validation set into training and report a score the model has not earned.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["betting_ad"]


def match_key(video_name: str) -> str:
    """Fixture identity, e.g. 'argentina_vs_caboverde'."""
    stem = video_name.split(" _ ")[0].split(" FIFA")[0]
    stem = re.sub(r"\s+Highlights$", "", stem.strip())
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "auto_labels" / "manifest.json"))
    ap.add_argument("--videos", default=str(ROOT / "data" / "raw_videos"))
    ap.add_argument("--out", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--val-frac", type=float, default=0.22)
    ap.add_argument("--synth", default=str(ROOT / "data" / "synth" / "manifest.json"))
    ap.add_argument("--external", default=str(ROOT / "data" / "external" / "manifest.json"))
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())

    # clipNN -> fixture, recovered from the sorted video list used at extraction
    vids = sorted(p.name for p in Path(args.videos).glob("*.mp4"))
    clip_match = {f"clip{i + 1:02d}": match_key(v) for i, v in enumerate(vids)}

    by_match: dict[str, list[dict]] = defaultdict(list)
    for m in manifest:
        clip = Path(m["image"]).parent.name
        by_match[clip_match.get(clip, clip)].append(m)

    # Greedily fill the validation split with whole fixtures until it is big
    # enough, largest-first so the target is hit with the fewest fixtures.
    order = sorted(by_match, key=lambda k: -len(by_match[k]))
    total = len(manifest)
    val_matches, n_val = [], 0
    for k in reversed(order):                    # smallest first -> tighter fit
        if n_val >= args.val_frac * total:
            break
        val_matches.append(k)
        n_val += len(by_match[k])

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Synthetic frames go to training only.  Scoring against composites we
    # generated ourselves would measure the generator, not the detector.
    synth_path = Path(args.synth)
    if synth_path.exists():
        syn = json.loads(synth_path.read_text())
        by_match["__synthetic__"] = syn
        print(f"synthetic (train only): {len(syn)} images")

    # External (Roboflow) real broadcast images.  Also train-only, for the
    # same reason as synthetic: val measures generalisation to whole unseen
    # matches from OUR footage, and mixing in a different broadcaster/league
    # would change what that number means and make it incomparable with every
    # earlier training run's val score.
    ext_path = Path(args.external)
    if ext_path.exists():
        ext = json.loads(ext_path.read_text())
        by_match["__external__"] = ext
        print(f"external real footage (train only): {len(ext)} images")

    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}   # images, polys, negatives
    for mk, items in by_match.items():
        split = "val" if mk in val_matches else "train"
        for m in items:
            src = Path(m["image"])
            if mk in ("__synthetic__", "__external__"):
                # These all live under a directory literally named "images",
                # so parent.name alone collides across sources (both Roboflow
                # sets use identical Roboflow-style folder layouts).  A path
                # hash guarantees uniqueness instead of silently overwriting
                # one image's label file with another's.
                import hashlib
                h = hashlib.sha1(str(src).encode()).hexdigest()[:10]
                name = f"{mk.strip('_')}_{h}_{src.stem}"[:120]
            else:
                name = f"{src.parent.name}_{src.stem}"
            dst_img = out / "images" / split / f"{name}.jpg"
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst_img)

            img = cv2.imread(str(src))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = []
            for poly in m["label"]:
                p = np.asarray(poly, np.float32)
                p[:, 0] = np.clip(p[:, 0], 0, w - 1) / w
                p[:, 1] = np.clip(p[:, 1], 0, h - 1) / h
                if cv2.contourArea(p.astype(np.float32)) <= 0:
                    continue
                coords = " ".join(f"{v:.6f}" for v in p.reshape(-1))
                lines.append(f"0 {coords}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            counts[split][0] += 1
            counts[split][1] += len(lines)
            counts[split][2] += 1 if not lines else 0

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )

    print(f"val fixtures ({len(val_matches)}): {sorted(val_matches)}")
    print(f"train fixtures ({len(by_match) - len(val_matches)}): "
          f"{sorted(k for k in by_match if k not in val_matches)}")
    for s in ("train", "val"):
        n, p, neg = counts[s]
        print(f"{s:<6} images={n:<5} polygons={p:<5} background={neg}")
    print(f"[dataset] {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
