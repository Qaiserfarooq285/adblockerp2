#!/usr/bin/env python3
"""Football betting-ad blocker: end-to-end, resumable orchestrator.

Every stage writes a stamp once it succeeds, so re-running after a crash, a
power cut or a closed session continues from where it stopped rather than
redoing finished work.  Rendering is resumable at clip granularity.

    python scripts/v2_pipeline.py              # run everything still pending
    python scripts/v2_pipeline.py --force train render
    python scripts/v2_pipeline.py --only render
    python scripts/v2_pipeline.py --status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STAMPS = ROOT / "data" / ".stamps" / "v2"
STATE = ROOT / "logs" / "v2_state.json"
LOGS = ROOT / "logs"

STEPS: list[tuple[str, callable]] = []


def step(name: str):
    def deco(fn):
        STEPS.append((name, fn))
        return fn
    return deco


def stamp_path(name: str) -> Path:
    return STAMPS / f"{name}.done"


def is_done(name: str) -> bool:
    return stamp_path(name).exists()


def mark(name: str, info: dict | None = None) -> None:
    STAMPS.mkdir(parents=True, exist_ok=True)
    payload = {"at": datetime.now(timezone.utc).isoformat(), **(info or {})}
    stamp_path(name).write_text(json.dumps(payload, indent=1))


def run(cmd: list[str], log: str) -> None:
    """Run a child step, streaming into logs/<log>.log; raise on failure."""
    LOGS.mkdir(parents=True, exist_ok=True)
    dst = LOGS / f"{log}.log"
    print(f"    $ {' '.join(str(c) for c in cmd)}  -> {dst}", flush=True)
    with dst.open("a") as fh:
        fh.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        fh.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"step failed ({r.returncode}); see {dst}")


# Exactly the 20 clips this project is scoped to.  frames/ocr/annotate were
# all built against these 20 back when they were the only files present;
# `newclip.mp4` (380MB, added later, does not match this set's naming
# convention) sorts after them alphabetically and would silently become an
# unrequested 21st clip in render_rest if videos() picked up every *.mp4 in
# the directory.  Named exclusion rather than a bare slice[:20], so a
# genuinely new *addition* to the real 20 later is not silently dropped too.
_OUT_OF_SCOPE = {"newclip.mp4"}


def videos() -> list[Path]:
    return sorted(p for p in (ROOT / "data" / "raw_videos").glob("*.mp4")
                 if p.name not in _OUT_OF_SCOPE)


# --------------------------------------------------------------------- steps

@step("logos")
def s_logos():
    """Brand imagery + alias vocabulary from the gambling-sites PDF."""
    run([PY, ROOT / "scripts" / "pdf_extract_logos.py"], "v2_logos")


@step("frames")
def s_frames():
    """Sample every clip at 2 fps."""
    out = ROOT / "data" / "frames"
    out.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(videos(), 1):
        d = out / f"clip{i:02d}"
        if d.exists() and any(d.glob("*.jpg")):
            continue
        d.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-i", v, "-vf", "fps=2", "-q:v", "2",
             d / "%05d.jpg", "-y"], "v2_frames")


@step("ocr")
def s_ocr():
    """Whole-frame OCR sweep."""
    run([PY, ROOT / "scripts" / "v2_ocr_survey.py"], "v2_ocr")


@step("ocr_tiles")
def s_ocr_tiles():
    """Tiled OCR pass; recovers wordmarks the whole-frame pass cannot resolve."""
    run([PY, ROOT / "scripts" / "v2_ocr_tiles.py"], "v2_ocr_tiles")


@step("ocr_band")
def s_ocr_band():
    """Third OCR pass targeted at the perimeter-board band specifically.

    Catches wordmarks (Kalshi, ADI PREDICTSTREET) that are too small in wide,
    elevated, far-distance shots for the whole-frame and generic-tile passes
    to read -- see scripts/v2_ocr_band.py for the measured root cause.
    """
    run([PY, ROOT / "scripts" / "v2_ocr_band.py"], "v2_ocr_band")


@step("annotate")
def s_annotate():
    """OCR -> betting-strip polygons + vetted background frames."""
    run([PY, ROOT / "scripts" / "v2_annotate.py"], "v2_annotate")


@step("external")
def s_external():
    """OCR-verified real images from the two external Roboflow datasets."""
    run([PY, ROOT / "scripts" / "v2_consolidate_external.py"], "v2_consolidate")


@step("synth")
def s_synth():
    """Targeted synthetic data for the measured weak cases (Kalshi, close-ups)."""
    run([PY, ROOT / "scripts" / "v2_synth.py"], "v2_synth")


@step("dataset")
def s_dataset():
    """Fixture-disjoint train/val split."""
    run([PY, ROOT / "scripts" / "v2_dataset.py"], "v2_dataset")


@step("train")
def s_train():
    """Train the segmentation model (itself resumable from last.pt)."""
    run([PY, ROOT / "scripts" / "v2_train.py"], "v2_train")


@step("targeted_eval")
def s_targeted_eval():
    """Recall broken down by brand / board size / side-by-side, not just mAP.

    mAP alone hid two earlier regressions in this project; this is the check
    that would have caught them before 20 clips got rendered on a worse model.
    """
    run([PY, ROOT / "scripts" / "v2_eval_targeted.py"], "v2_eval_targeted")


class Gated(Exception):
    """Raised to park the pipeline until a human approves the sample clip."""


def render_clips(indices: list[int]) -> None:
    weights = ROOT / "models" / "ad_seg_best.pt"
    if not weights.exists():
        raise RuntimeError(f"missing {weights}; train must succeed first")
    weights_mtime = weights.stat().st_mtime
    outdir = ROOT / "outputs" / "clips"
    outdir.mkdir(parents=True, exist_ok=True)
    vids = videos()
    for i in indices:
        v = vids[i - 1]
        dst = outdir / f"clip{i:02d}_clean.mp4"
        if dst.exists() and dst.stat().st_size > 100_000:
            # "Resumable" must not mean "reuse a clip rendered by a since-
            # retrained model" -- that silently shipped 3 stale clips once
            # already when a retrain finished after render_first had already
            # run once. A clip older than the weights that would render it
            # now is stale, not finished.
            if dst.stat().st_mtime >= weights_mtime:
                print(f"    clip{i:02d} already rendered (current model)", flush=True)
                continue
            print(f"    clip{i:02d} predates current model weights; re-rendering",
                  flush=True)
        run([PY, ROOT / "scripts" / "v2_render.py", "--input", v,
             "--output", dst, "--ad-weights", weights], "v2_render")


@step("render_first")
def s_render_first():
    """Render one clip only, so the settings can be checked before the batch."""
    render_clips([1])
    run([PY, ROOT / "scripts" / "v2_preview.py", "--clip", "1"], "v2_preview")


@step("approval")
def s_approval():
    """Hold the remaining clips until logs/APPROVED exists.

    Rendering 20 clips is hours of GPU time; spending it before anyone has
    looked at the first result is how you discover a bad setting 19 clips too
    late.  The watchdog re-enters every 10 minutes, so approval is picked up
    without anyone restarting anything.
    """
    if not (LOGS / "APPROVED").exists():
        raise Gated("waiting for logs/APPROVED (review outputs/preview_clip01.mp4)")


@step("render_rest")
def s_render_rest():
    """Render the other 19 clips.  Resumable: finished outputs are skipped."""
    render_clips(list(range(2, len(videos()) + 1)))


@step("verify")
def s_verify():
    """Measure the rendered results and write a report."""
    run([PY, ROOT / "scripts" / "v2_verify.py"], "v2_verify")


# ---------------------------------------------------------------------- main

def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def acquire_lock():
    """Exclusive run lock.

    The cron watchdog and a human can both try to start the pipeline, and a
    process check is racy: two starts a second apart both see an idle machine,
    then two trainings share one GPU and OOM.  A held lock cannot race.
    Returns the open handle, which must stay alive for the process lifetime.
    """
    import fcntl
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = (LOGS / "v2_pipeline.lock").open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(f"{__import__('os').getpid()}\n")
    fh.flush()
    return fh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], help="re-run these steps")
    ap.add_argument("--only", nargs="*", default=[], help="run only these steps")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if not args.status:
        lock = acquire_lock()
        if lock is None:
            print("[pipeline] another run holds the lock; exiting", flush=True)
            return 0

    names = [n for n, _ in STEPS]
    if args.status:
        for n in names:
            s = stamp_path(n)
            when = json.loads(s.read_text())["at"] if s.exists() else "-"
            print(f"  {'OK ' if s.exists() else '   '} {n:<12} {when}")
        return 0

    for n in args.force:
        stamp_path(n).unlink(missing_ok=True)

    state = load_state()
    state["started"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    for name, fn in STEPS:
        if args.only and name not in args.only:
            continue
        if is_done(name) and name not in args.force:
            print(f"[skip] {name}", flush=True)
            continue
        print(f"[run ] {name}", flush=True)
        t0 = time.time()
        try:
            fn()
        except Gated as g:
            # Not a failure: park without stamping so the next watchdog tick
            # re-checks, and leave everything already finished intact.
            state[name] = {"status": "waiting", "detail": str(g)}
            save_state(state)
            print(f"[wait] {name}: {g}", flush=True)
            return 0
        except Exception as exc:
            state[name] = {"status": "failed", "error": str(exc)}
            state["failed_at"] = name
            save_state(state)
            print(f"[FAIL] {name}: {exc}", flush=True)
            traceback.print_exc()
            return 1
        dt = time.time() - t0
        mark(name, {"seconds": round(dt, 1)})
        state[name] = {"status": "ok", "seconds": round(dt, 1)}
        state.pop("failed_at", None)
        save_state(state)
        print(f"[done] {name} in {dt / 60:.1f} min", flush=True)

    state["finished"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print("[pipeline] all steps complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
