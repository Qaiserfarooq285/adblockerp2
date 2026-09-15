#!/usr/bin/env python3
"""Targeted synthetic training data for the two measured weak spots.

Held-out recall broke down as:

    Betano 100%   PredictStreet 89%   Kalshi 77%
    small 86%     medium 92%          large/zoomed 60%

So the detector is not weak in general -- it is weak on Kalshi boards and on
boards that fill a lot of the frame.  Generic augmentation spends most of its
effort on cases already at 90-100%, so instead we synthesise those two cases
specifically.

Two generators, both built from real pixels so the composites stay plausible:

  replace  a real board in a real frame is repainted with another brand's
           artwork, keeping the true perspective, lighting and occlusion.
           Sampling is biased towards Kalshi to correct its under-representation.

  zoom     a board crop is laid across a frame at close-up scale, which is the
           view the detector currently misses four times in ten.
"""
from __future__ import annotations

import argparse
import json
import random
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


def rectify(img, quad, out_w=None, out_h=None):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    # order: longest edge first
    lens = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    q = np.roll(q, -int(np.argmax(lens)), axis=0)
    if q[0][0] > q[1][0]:
        q = q[[1, 0, 3, 2]]
    if q[0][1] > q[3][1]:
        q = q[[3, 2, 1, 0]]
    w = int(out_w or round(np.linalg.norm(q[1] - q[0])))
    h = int(out_h or round(np.linalg.norm(q[3] - q[0])))
    if w < 12 or h < 6:
        return None
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, M, (w, h))


def pitch_top_edge(img, min_run=60):
    """Per-column y of the top of the playing surface, or None if no pitch.

    Perimeter boards line the far edge of the grass, so this is where a
    synthetic board has to sit for the composite to be worth training on.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = ((h_ >= 38) & (h_ <= 62) & (s_ > 70) & (v_ > 60)).astype(np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    H, W = grass.shape
    if grass.mean() < 0.15:
        return None
    edge = np.full(W, np.nan, np.float32)
    for x in range(W):
        col = np.nonzero(grass[:, x])[0]
        if len(col) < min_run:
            continue
        edge[x] = col.min()
    if np.isnan(edge).mean() > 0.5:
        return None
    idx = np.arange(W)
    good = ~np.isnan(edge)
    edge = np.interp(idx, idx[good], edge[good])
    return cv2.GaussianBlur(edge.reshape(1, -1), (51, 1), 0).ravel()


def paste(dst, crop, quad, feather=3):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    h, w = crop.shape[:2]
    M = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), q)
    warped = cv2.warpPerspective(crop, M, (dst.shape[1], dst.shape[0]))
    m = np.zeros(dst.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 255)
    if feather > 1:
        m = cv2.GaussianBlur(m, (feather | 1, feather | 1), 0)
    a = (m.astype(np.float32) / 255.0)[:, :, None]
    return (dst * (1 - a) + warped * a).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "auto_labels" / "manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--n-replace", type=int, default=550)
    ap.add_argument("--n-zoom", type=int, default=400)
    ap.add_argument("--n-sidebyside", type=int, default=220)
    ap.add_argument("--n-steep", type=int, default=340)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from ultralytics import YOLO
    person_model = YOLO("yolo11m-seg.pt")
    _pcache: dict[str, np.ndarray] = {}

    def people_mask(path: str, img) -> np.ndarray:
        if path in _pcache:
            return _pcache[path]
        m = np.zeros(img.shape[:2], np.uint8)
        r = person_model.predict(img, conf=0.2, classes=[0], imgsz=960,
                                 verbose=False, retina_masks=True)[0]
        if r.masks is not None:
            for k in r.masks.data.cpu().numpy():
                mm = (k > 0.5).astype(np.uint8)
                if mm.shape != m.shape:
                    mm = cv2.resize(mm, (m.shape[1], m.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                m |= mm
        if len(_pcache) < 400:
            _pcache[path] = m
        return m

    man = json.loads(Path(args.manifest).read_text())
    pos = [m for m in man if m["kind"] == "pos"]
    neg = [m for m in man if m["kind"] == "neg"]

    lib: dict[str, list[np.ndarray]] = defaultdict(list)
    for m in pos:
        ocr_path = m["image"].replace("data/frames", "data/ocr").replace(".jpg", ".json")
        try:
            recs = json.loads(Path(ocr_path).read_text())
        except Exception:
            continue
        img = cv2.imread(m["image"])
        if img is None:
            continue
        for poly in m["label"]:
            b = brand_of(poly, recs)
            if b == "other":
                continue
            c = rectify(img, poly)
            if c is not None and c.shape[0] >= 12 and c.shape[1] >= 60:
                lib[b].append(c)
    counts = {k: len(v) for k, v in lib.items()}
    print(f"[synth] crop library: {counts}")
    if not lib:
        print("[synth] no crops; nothing to do")
        return 1

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    # Resumable: this environment has restarted mid-run before (power loss),
    # and this step is GPU-bound over hundreds of composites.  Load whatever
    # a previous attempt already finished, skip regenerating those exact
    # images, and checkpoint the manifest periodically rather than only once
    # at the very end -- so an interruption costs a handful of images, not
    # the whole run.
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done_names = {Path(m["image"]).stem for m in manifest}
    if done_names:
        print(f"[synth] resuming: {len(done_names)} images already generated")
    _since_checkpoint = [0]

    def checkpoint(force: bool = False) -> None:
        _since_checkpoint[0] += 1
        if force or _since_checkpoint[0] >= 25:
            manifest_path.write_text(json.dumps(manifest, indent=1))
            _since_checkpoint[0] = 0

    # Betano and Kalshi both get explicit priority (Betano: broadest real-world
    # visual variety to cover; Kalshi: the measured weak brand).  PredictStreet
    # was already at 89% real-footage recall, so it gets the remainder.
    weights = {"kalshi": 0.42, "betano": 0.38, "predict": 0.20}
    brands = [b for b in weights if lib.get(b)]

    def pick_crop():
        b = rng.choices(brands, [weights[b] for b in brands])[0]
        crop = rng.choice(lib[b])
        if b == "kalshi" and rng.random() < 0.5:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-12, 12)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.65, 1.25), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.60, 1.20), 0, 255)
            crop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Every brand gets brightness/contrast jitter -- floodlit night games,
        # overcast day matches, and different broadcasters' colour grading all
        # change how a board reads, and the crop library was harvested from a
        # narrower set of lighting conditions than the model has to work under.
        if rng.random() < 0.7:
            crop = crop.astype(np.float32)
            crop *= rng.uniform(0.65, 1.35)                       # brightness
            crop = (crop - 127.5) * rng.uniform(0.75, 1.30) + 127.5  # contrast
            crop = np.clip(crop, 0, 255).astype(np.uint8)
        return crop, b

    def broadcast_degrade(img, poly, rng):
        """Mild blur/noise inside one pasted region, mimicking encode artefacts.

        A perfectly crisp paste is easier to detect than a real broadcast
        board, which has gone through camera optics, motion, and video
        compression.  Applied after the paste, scoped to the board's own
        area only, so it never touches the rest of the frame.
        """
        x0, y0 = poly[:, 0].min(), poly[:, 1].min()
        x1, y1 = poly[:, 0].max(), poly[:, 1].max()
        x0i, y0i = max(0, int(x0)), max(0, int(y0))
        x1i, y1i = min(img.shape[1], int(x1) + 1), min(img.shape[0], int(y1) + 1)
        if x1i - x0i < 6 or y1i - y0i < 6:
            return img
        roi = img[y0i:y1i, x0i:x1i]
        if rng.random() < 0.5:
            k = rng.choice([3, 5])
            roi = cv2.GaussianBlur(roi, (k, k), 0)
        if rng.random() < 0.35:
            noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
                0, rng.uniform(3, 10), roi.shape).astype(np.float32)
            roi = np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img[y0i:y1i, x0i:x1i] = roi
        return img

    made = 0
    for i in range(args.n_replace):
        if f"rep_{i:05d}" in done_names:
            continue
        m = rng.choice(pos)
        img = cv2.imread(m["image"])
        if img is None or not m["label"]:
            continue
        img = img.copy()
        polys = [np.asarray(p, np.float32) for p in m["label"]]
        target = rng.choice(polys)
        crop, _ = pick_crop()
        q = np.asarray(target, np.float32).reshape(4, 2)
        tw = max(24, int(np.linalg.norm(q[1] - q[0])))
        th = max(8, int(np.linalg.norm(q[3] - q[0])))
        tile = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        orig = img.copy()
        img = paste(img, tile, q)
        img = broadcast_degrade(img, q, rng)
        pmask = people_mask(m["image"], orig)
        img[pmask > 0] = orig[pmask > 0]
        name = f"rep_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append({"image": str(out / "images" / f"{name}.jpg"),
                         "label": [p.round(2).tolist() for p in polys], "kind": "pos"})
        made += 1
        checkpoint()

    pool = pos + neg
    for i in range(args.n_zoom):
        if f"zoom_{i:05d}" in done_names:
            continue
        m = rng.choice(pool)
        img = cv2.imread(m["image"])
        if img is None:
            continue
        img = img.copy()
        H, W = img.shape[:2]
        crop, _ = pick_crop()
        edge = pitch_top_edge(img)
        if edge is None:
            continue
        cx = rng.uniform(0.2, 0.8) * W
        base = float(np.interp(cx, np.arange(W), edge))
        if not (0.10 * H < base < 0.80 * H):
            continue
        length = rng.uniform(0.55, 1.25) * W
        thick = rng.uniform(0.10, 0.24) * H
        cy = base - thick * rng.uniform(0.35, 0.55)
        x0 = int(np.clip(cx - length / 2, 0, W - 1))
        x1 = int(np.clip(cx + length / 2, 0, W - 1))
        if x1 - x0 > 20:
            ang = float(np.arctan2(edge[x1] - edge[x0], x1 - x0))
            ang = float(np.clip(ang, np.deg2rad(-14), np.deg2rad(14)))
        else:
            ang = np.deg2rad(rng.uniform(-6, 6))
        d = np.array([np.cos(ang), np.sin(ang)], np.float32)
        n = np.array([-d[1], d[0]], np.float32)
        k = rng.uniform(0.85, 1.15)
        c = np.array([cx, cy], np.float32)
        q = np.float32([
            c - d * length / 2 - n * thick / 2,
            c + d * length / 2 - n * thick * k / 2,
            c + d * length / 2 + n * thick * k / 2,
            c - d * length / 2 + n * thick / 2])
        if q[:, 0].min() < -length or q[:, 0].max() > W + length:
            continue
        qm = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(qm, q.astype(np.int32), 1)
        qarea = int(qm.sum())
        if qarea < 4000:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grass = ((hsv[:, :, 0] >= 38) & (hsv[:, :, 0] <= 62) &
                 (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)).astype(np.uint8)
        if float((grass & qm).sum()) / qarea > 0.40:
            continue
        pmask = people_mask(m["image"], img)
        if float((pmask & qm).sum()) / qarea > 0.25:
            continue

        tile = cv2.resize(crop, (max(32, int(length)), max(12, int(thick))),
                          interpolation=cv2.INTER_LINEAR)
        orig = img.copy()
        img = paste(img, tile, q)
        img = broadcast_degrade(img, q, rng)
        img[pmask > 0] = orig[pmask > 0]
        name = f"zoom_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        keep = [q.round(2).tolist()]
        for p in m.get("label", []):
            pp = np.asarray(p, np.float32)
            inside = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(inside, q.astype(np.int32), 1)
            om = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(om, pp.astype(np.int32), 1)
            if om.sum() and (om & inside).sum() / om.sum() < 0.25:
                keep.append(pp.round(2).tolist())
        manifest.append({"image": str(out / "images" / f"{name}.jpg"),
                         "label": keep, "kind": "pos"})
        made += 1
        checkpoint()

    # --- generator 3: steep perspective & multi-angle 3D board generator ---
    for i in range(args.n_steep):
        if f"steep_{i:05d}" in done_names:
            continue
        m = rng.choice(pool)
        img = cv2.imread(m["image"])
        if img is None:
            continue
        img = img.copy()
        H, W = img.shape[:2]
        crop, _ = pick_crop()
        edge = pitch_top_edge(img)
        if edge is None:
            continue
        cx = rng.uniform(0.12, 0.88) * W
        base = float(np.interp(cx, np.arange(W), edge))
        if not (0.12 * H < base < 0.88 * H):
            continue
        length = rng.uniform(0.25, 0.75) * W
        thick = rng.uniform(0.04, 0.18) * H
        cy = base - thick * rng.uniform(0.15, 0.55)
        ang = np.deg2rad(rng.choice([-1, 1]) * rng.uniform(15, 38))
        d = np.array([np.cos(ang), np.sin(ang)], np.float32)
        n = np.array([-d[1], d[0]], np.float32)
        k = rng.uniform(0.35, 0.75) if rng.random() < 0.5 else rng.uniform(1.3, 2.2)
        c = np.array([cx, cy], np.float32)
        q = np.float32([
            c - d * length / 2 - n * thick / 2,
            c + d * length / 2 - n * thick * k / 2,
            c + d * length / 2 + n * thick * k / 2,
            c - d * length / 2 + n * thick / 2])
        if q[:, 0].min() < -length or q[:, 0].max() > W + length:
            continue
        qm = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(qm, q.astype(np.int32), 1)
        qarea = int(qm.sum())
        if qarea < 2000:
            continue
        pmask = people_mask(m["image"], img)
        if float((pmask & qm).sum()) / qarea > 0.35:
            continue

        tile = cv2.resize(crop, (max(24, int(length)), max(10, int(thick))),
                          interpolation=cv2.INTER_LINEAR)
        orig = img.copy()
        img = paste(img, tile, q)
        img = broadcast_degrade(img, q, rng)
        img[pmask > 0] = orig[pmask > 0]
        name = f"steep_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        keep = [q.round(2).tolist()]
        manifest.append({"image": str(out / "images" / f"{name}.jpg"),
                         "label": keep, "kind": "pos"})
        made += 1
        checkpoint()

    # --- generator 4: two betting boards side by side ----------------------
    # Real rails often carry two different sponsors back to back (e.g. an
    # "ADI PREDICTSTREET" run immediately followed by "Kalshi").  Nothing so
    # far trains that configuration explicitly: each composite above places
    # exactly one board.  Two independently-labelled crops, placed with a
    # visible seam, teaches the model to resolve them as two objects rather
    # than a single wide blob -- which is what the renderer's shape filters
    # already assume when they reject anything that isn't rail-shaped.
    for i in range(args.n_sidebyside):
        if f"sbs_{i:05d}" in done_names:
            continue
        m = rng.choice(pool)
        img = cv2.imread(m["image"])
        if img is None:
            continue
        img = img.copy()
        H, W = img.shape[:2]
        edge = pitch_top_edge(img)
        if edge is None:
            continue
        cx = rng.uniform(0.25, 0.75) * W
        base = float(np.interp(cx, np.arange(W), edge))
        if not (0.12 * H < base < 0.82 * H):
            continue
        thick = rng.uniform(0.09, 0.20) * H
        cy = base - thick * rng.uniform(0.35, 0.55)
        each_len = rng.uniform(0.18, 0.40) * W
        gap = rng.uniform(0.0, 0.06) * W
        ang = np.deg2rad(rng.uniform(-10, 10))
        d = np.array([np.cos(ang), np.sin(ang)], np.float32)
        n = np.array([-d[1], d[0]], np.float32)
        c = np.array([cx, cy], np.float32)

        keep = []
        pmask = people_mask(m["image"], img)
        orig = img.copy()
        ok = True
        for side in (-1.0, 1.0):
            centre = c + d * side * (each_len / 2 + gap / 2)
            crop, _ = pick_crop()
            q = np.float32([
                centre - d * each_len / 2 - n * thick / 2,
                centre + d * each_len / 2 - n * thick / 2,
                centre + d * each_len / 2 + n * thick / 2,
                centre - d * each_len / 2 + n * thick / 2])
            if q[:, 0].min() < -each_len or q[:, 0].max() > W + each_len:
                ok = False
                break
            qm = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(qm, q.astype(np.int32), 1)
            qarea = int(qm.sum())
            if qarea < 1500 or float((pmask & qm).sum()) / max(qarea, 1) > 0.30:
                ok = False
                break
            tile = cv2.resize(crop, (max(24, int(each_len)), max(10, int(thick))),
                              interpolation=cv2.INTER_LINEAR)
            img = paste(img, tile, q)
            img = broadcast_degrade(img, q, rng)
            keep.append(q.round(2).tolist())
        if not ok or not keep:
            continue
        img[pmask > 0] = orig[pmask > 0]
        name = f"sbs_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append({"image": str(out / "images" / f"{name}.jpg"),
                         "label": keep, "kind": "pos"})
        made += 1
        checkpoint()

    checkpoint(force=True)
    print(f"[synth] wrote {made} new images this run, {len(manifest)} total -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
