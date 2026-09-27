#!/usr/bin/env python3
"""Score renders for the two artefacts viewers notice: a moving box and blinking.

  moving box  How far each panel moves RELATIVE TO THE SCENE between frames,
              i.e. after the camera's own motion is removed.  A panel glued to
              its board scores ~0 however fast the camera pans; one that copies
              detector jitter or lags behind a pan scores high.
  blinking    A painted region that disappears while still on screen and is
              painted again within --blink-frames: exactly what reads as a
              flash of the advert.

Both are computed from the per-frame panel log v2_render.py writes with
--panel-log.  The camera homography in the log is measured independently of
the tracker (geometry.py), so the two trackers are scored on equal terms:

    python scripts/v2_render.py --input clip.mp4 --tracker legacy --panel-log old.jsonl
    python scripts/v2_render.py --input clip.mp4 --panel-log new.jsonl
    python scripts/stability_report.py old.jsonl new.jsonl
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from geometry import align_quad, warp_quad


def load(path: str) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    head = rows[0] if rows and rows[0].get("header") else {}
    return head, [r for r in rows if not r.get("header")]


def _quads(row) -> dict:
    return {p[0]: np.float32(p[3]).reshape(4, 2) for p in row["p"] if p[2] >= 0.5}


def jitter(rows, w: int, h: int) -> np.ndarray:
    """Per panel per frame: mean on-screen corner displacement vs. the camera."""
    out, prev = [], {}
    for r in rows:
        cur = _quads(r)
        H = None if r["H"] is None else np.float64(r["H"]).reshape(3, 3)
        if not r["cut"] and H is not None:
            for tid, q in cur.items():
                if tid not in prev:
                    continue
                exp = warp_quad(H, prev[tid])
                if exp is None:
                    continue
                a = align_quad(q, exp)
                on = ((exp[:, 0] >= 0) & (exp[:, 0] < w) & (exp[:, 1] >= 0) & (exp[:, 1] < h) &
                      (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h))
                if on.any():
                    out.append(float(np.linalg.norm(a - exp, axis=1)[on].mean()))
        prev = cur
    return np.array(out)


def blinks(rows, w: int, h: int, max_gap: int, scale: float = 0.125) -> tuple[int, int]:
    """(regions that vanished while on screen, of which re-painted = blinks)."""
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    S = np.diag([scale, scale, 1.0])
    Si = np.linalg.inv(S)
    k = np.ones((3, 3), np.uint8)
    pending: list[tuple[np.ndarray, int]] = []
    prev = None
    vanished = blinked = 0
    for r in rows:
        mask = np.zeros((sh, sw), np.uint8)
        for q in _quads(r).values():
            cv2.fillConvexPoly(mask, np.round(q * scale * 16).astype(np.int32), 1, shift=4)
        H = None if r["H"] is None else np.float64(r["H"]).reshape(3, 3)
        if r["cut"] or H is None:
            pending, prev = [], mask
            continue
        Hs = S @ H @ Si
        carried = []
        for reg, age in pending:                    # ride the camera while pending
            reg = cv2.warpPerspective(reg, Hs, (sw, sh), flags=cv2.INTER_NEAREST)
            area = int(reg.sum())
            if area < 4:
                continue                            # left the picture: not a blink
            if int((reg & mask).sum()) >= 0.5 * area:
                blinked += 1
                continue
            if age + 1 <= max_gap:
                carried.append((reg, age + 1))
        pending = carried
        if prev is not None:
            gone = cv2.warpPerspective(prev, Hs, (sw, sh), flags=cv2.INTER_NEAREST) & (1 - mask)
            gone = cv2.morphologyEx(gone, cv2.MORPH_OPEN, k)   # ignore edge slivers
            n, lab, st, _ = cv2.connectedComponentsWithStats(gone, 8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] >= 12:
                    vanished += 1
                    pending.append(((lab == i).astype(np.uint8), 0))
        prev = mask
    return vanished, blinked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--blink-frames", type=int, default=15,
                    help="re-painted within this many frames counts as a blink")
    ap.add_argument("--json", default=None, help="also write the table as JSON")
    args = ap.parse_args()

    table = []
    for path in args.logs:
        head, rows = load(path)
        w, h = int(head.get("w", 1920)), int(head.get("h", 1080))
        fps = float(head.get("fps", 30.0)) or 30.0
        j = jitter(rows, w, h)
        v, b = blinks(rows, w, h, args.blink_frames)
        minutes = max(1e-9, len(rows) / fps / 60.0)
        painted = sum(1 for r in rows if r["p"])
        table.append({
            "log": path, "tracker": head.get("tracker", "?"), "frames": len(rows),
            "painted_frames_pct": round(100.0 * painted / max(1, len(rows)), 1),
            "motion_vs_scene_px_median": round(float(np.median(j)), 2) if len(j) else None,
            "motion_vs_scene_px_p90": round(float(np.percentile(j, 90)), 2) if len(j) else None,
            "motion_vs_scene_px_p99": round(float(np.percentile(j, 99)), 2) if len(j) else None,
            "vanish_events_per_min": round(v / minutes, 1),
            "blinks_per_min": round(b / minutes, 1),
        })
    keys = list(table[0].keys())[1:]
    width = max(len(k) for k in keys)
    print(" " * width + "".join(f"  {t['log'][-24:]:>24}" for t in table))
    for k in keys:
        print(f"{k:<{width}}" + "".join(f"  {str(t[k]):>24}" for t in table))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(table, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
