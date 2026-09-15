#!/usr/bin/env python3
"""Consolidate the two external Roboflow datasets into the unified training set.

Both live under `input/dataset /`:

  Superbet - Betano.v10i.yolov11   4 classes: betano, betano_b, superbet, superbet_s
                                    Real broadcast footage (Romanian Superliga).
                                    Brand-specific classes -- trusted directly.

  logo-betting.v1i.yolov11          1 class: "logo" (generic).
                                    Real broadcast footage (Premier League, Getty).
                                    The single class is NOT trustworthy as-is: a manual
                                    sample turned up "JD Sports" (a clothing retailer)
                                    labelled identically to "Betano" -- the same kind of
                                    scorebug/"powerplay" contamination that poisoned an
                                    earlier version of this project's training data.
                                    Every box is therefore OCR-verified against a known
                                    betting-brand vocabulary before being trusted; boxes
                                    that don't resolve to a recognised brand are dropped
                                    rather than guessed at.  This makes it slow (roughly
                                    one OCR call per candidate box), so progress is
                                    checkpointed -- see below.

Output: one manifest in the same {"image","label","kind"} shape v2_annotate.py produces,
so v2_dataset.py can fold it in without caring where it came from.

Resumable: this environment has restarted mid-run before, and the OCR-verified
logo-betting phase alone takes the better part of an hour.  The manifest is
rewritten to disk every CHECKPOINT_EVERY accepted images (not just once at the
end), and on startup any image path already present in an existing output
manifest is skipped -- so an interruption costs at most a few images of
progress, not the whole run.
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
from v2_brands import is_betting, is_non_betting  # noqa: E402

DATASET_ROOT = ROOT / "input" / "dataset "
CHECKPOINT_EVERY = 20

# Real gambling brands confirmed present in logo-betting.v1i by manual inspection,
# but outside this project's own PDF/OCR brand vocabulary (which is scoped to the
# Ontario operator register and the brands actually seen on our own footage).
# Extending recognition to these lets genuinely-betting boxes survive the OCR
# filter instead of being discarded as "unknown".
EXTRA_BETTING_BRANDS = (
    "debet", "8xbet", "sbotop", "kubet", "ky825", "bc.game", "bcgame",
    "kalyun", "dafabet", "fun88", "w88", "m88", "12bet", "188bet", "sbobet",
)


def yolo_to_quad(cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> np.ndarray:
    x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
    x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def load_split(ds_dir: Path, split: str) -> list[tuple[Path, Path]]:
    img_dir, lbl_dir = ds_dir / split / "images", ds_dir / split / "labels"
    if not img_dir.exists():
        return []
    pairs = []
    for lf in sorted(lbl_dir.glob("*.txt")):
        imf = img_dir / (lf.stem + ".jpg")
        if not imf.exists():
            imf = next(img_dir.glob(lf.stem + ".*"), None)
        if imf and imf.exists():
            pairs.append((imf, lf))
    return pairs


def brand_ok(text: str) -> bool:
    t = text.lower()
    if is_non_betting(text):
        return False
    if is_betting(text):
        return True
    return any(b in t.replace(" ", "") for b in EXTRA_BETTING_BRANDS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "external" / "manifest.json"))
    ap.add_argument("--max-superbet", type=int, default=1400)
    ap.add_argument("--max-logo-betting", type=int, default=500,
                    help="OCR-verified phase; kept well below the ~2000-image pool "
                         "so a single run finishes in a bounded time")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    done_images: set[str] = set()
    if out.exists():
        try:
            manifest = json.loads(out.read_text())
            done_images = {m["image"] for m in manifest}
            print(f"[consolidate] resuming: {len(manifest)} images already in {out}",
                  flush=True)
        except Exception:
            manifest = []

    # Images that were OCR-checked and REJECTED are not in the manifest (only
    # kept ones are), so without a separate record a resumed run would redo
    # every rejected image's OCR from scratch -- an hour of recompute for a
    # deterministic result that can never change. Track them here too.
    seen_path = out.parent / f"{out.stem}.seen.json"
    seen_images: set[str] = set(done_images)
    if seen_path.exists():
        try:
            seen_images |= set(json.loads(seen_path.read_text()))
        except Exception:
            pass

    def checkpoint() -> None:
        out.write_text(json.dumps(manifest, indent=1))
        seen_path.write_text(json.dumps(sorted(seen_images), indent=1))

    have_superbet = sum(1 for m in manifest if m.get("source") == "superbet_betano")
    have_logo = sum(1 for m in manifest if m.get("source") == "logo_betting")

    # --- Superbet - Betano: brand-specific classes, trusted directly ----------
    if have_superbet < args.max_superbet:
        ds = DATASET_ROOT / "Superbet - Betano.v10i.yolov11"
        pairs = load_split(ds, "train") + load_split(ds, "valid")
        rng = np.random.default_rng(0)
        rng.shuffle(pairs)
        added = 0
        for imf, lf in pairs:
            if have_superbet + added >= args.max_superbet:
                break
            if str(imf) in done_images:
                continue
            img = cv2.imread(str(imf))
            if img is None:
                continue
            h, w = img.shape[:2]
            polys = []
            for line in lf.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                q = yolo_to_quad(cx, cy, bw, bh, w, h)
                if (q[2] - q[0]).min() < 6:
                    continue
                polys.append(q.round(2).tolist())
            if polys:
                manifest.append({"image": str(imf), "label": polys, "kind": "pos",
                                 "source": "superbet_betano"})
                done_images.add(str(imf))
                added += 1
                if added % CHECKPOINT_EVERY == 0:
                    checkpoint()
        checkpoint()
        print(f"[consolidate] superbet_betano: {have_superbet + added} total "
              f"(+{added} this run)", flush=True)

    # --- logo-betting: generic class, OCR-verified per box --------------------
    if have_logo < args.max_logo_betting:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1)

        ds = DATASET_ROOT / "logo-betting.v1i.yolov11"
        pairs = load_split(ds, "train") + load_split(ds, "valid")
        rng = np.random.default_rng(0)
        rng.shuffle(pairs)
        kept_boxes = dropped_boxes = 0
        done = have_logo
        for imf, lf in pairs:
            if done >= args.max_logo_betting:
                break
            if str(imf) in seen_images:
                continue
            img = cv2.imread(str(imf))
            if img is None:
                continue
            h, w = img.shape[:2]
            polys = []
            for line in lf.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                q = yolo_to_quad(cx, cy, bw, bh, w, h)
                bw_px, bh_px = q[2] - q[0]
                if bw_px < 20 or bh_px < 8:
                    continue
                x0, y0, x1, y1 = q[0][0], q[0][1], q[2][0], q[2][1]
                crop = img[int(y0):int(y1), int(x0):int(x1)]
                if crop.size == 0:
                    continue
                scale = max(1.0, 96.0 / max(1, crop.shape[0]))
                if scale > 1.0:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)
                try:
                    res, _ = ocr(crop)
                except Exception:
                    res = None
                texts = [r[1] for r in (res or []) if float(r[2]) >= 0.5]
                if texts and any(brand_ok(t) for t in texts):
                    polys.append(q.round(2).tolist())
                    kept_boxes += 1
                else:
                    dropped_boxes += 1
            seen_images.add(str(imf))
            if polys:
                manifest.append({"image": str(imf), "label": polys, "kind": "pos",
                                 "source": "logo_betting"})
                done_images.add(str(imf))
                done += 1
            if len(seen_images) % CHECKPOINT_EVERY == 0:
                checkpoint()
                print(f"[consolidate] logo-betting: {done}/{args.max_logo_betting} "
                      f"images kept ({kept_boxes} boxes kept, "
                      f"{dropped_boxes} dropped)", flush=True)
        checkpoint()
        print(f"[consolidate] logo-betting: {done} total this run "
              f"({kept_boxes} boxes kept, {dropped_boxes} dropped)", flush=True)

    print(f"[consolidate] {len(manifest)} external images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
