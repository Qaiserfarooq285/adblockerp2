#!/usr/bin/env python3
"""Train the next board-model version from what the live pipeline collected.

Runs on the GPU pod after a match -- from the operator page ("Train a new
model now"), automatically once the input has been idle (live.learn.auto_train:
idle), or by hand.  It never overwrites a model: every run makes a new version
under <learn_dir>/models/, and only a version that passes the test is put in use.

    harvest   the ORIGINAL recordings: every ~2 s a frame is read (OCR + the
              Brand Library, as in the live scan) and run through the current
              model.  Brands the operator added are labelled wherever they
              appear -- including in matches recorded before they were added.
    collect   all training frames (operator marks, review decisions, the live
              scan's finds, harvest), their labels re-resolved against the
              Brand Library as it is now
    synth     each hidden brand's logo crops painted onto real boards in real
              frames (perspective, lighting, blur), so one crop teaches
              hundreds of views
    dataset   split by recording session, never by frame: near-identical
              frames on both sides would flatter the score
    train     fine-tune the model in use (or the original) on old + new
              labels -- the model's own confident detections are part of the
              data, so it keeps what it knew
    test      old vs new model on the held-out sessions: boards to hide still
              found, sponsors to keep still untouched, new brands found now
    promote   only if the test passes; the pod loads it at its next start, or
              the operator switches to it at once

    python scripts/learn_retrain.py --learn-dir /workspace/learning --base /workspace/models/ad_seg_best.pt
    python scripts/learn_retrain.py --learn-dir learning --status
    python scripts/learn_retrain.py --learn-dir learning --activate v002    # or: original
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from brand_library import BrandLibrary  # noqa: E402
from geometry import containment, convex_iou  # noqa: E402
from learn_jobs import ModelRegistry  # noqa: E402
from learn_store import LearnDir, SampleStore, write_json  # noqa: E402

STATE: dict = {}
STATE_PATH: Path | None = None


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [retrain] {msg}", flush=True)


def stage(name: str, **kw) -> None:
    """Record the current stage (and its details) for the operator page."""
    STATE.update(stage=name, detail=kw, updated=time.time())
    if STATE_PATH is not None:
        write_json(STATE_PATH, STATE)
    say(name + (" " + json.dumps(kw, default=str) if kw else ""))


# ------------------------------------------------------------------ harvest

_W = {}


def _harvest_init(brands_file: str, threads: int) -> None:
    from brand_scan import make_ocr
    _W["lib"] = BrandLibrary(brands_file)
    _W["ocr"] = make_ocr(threads)


def _harvest_scan(jpg: bytes) -> dict:
    from brand_scan import scan_frame
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    res = scan_frame(frame, _W["lib"], _W["ocr"])
    return {"boards": [{**b, "quad": np.asarray(b["quad"]).round(1).tolist()} for b in res["boards"]]}


def harvest(ld: LearnDir, samples: SampleStore, library: BrandLibrary, detector, args) -> int:
    """Label the original recordings with the current library + model."""
    import multiprocessing as mp

    import av
    recs = sorted((ld.root / "recordings").glob("original-*.ts"))
    # the newest file may still be being written by the live pipeline
    recs = [p for p in recs if time.time() - p.stat().st_mtime > 60]
    todo = []
    for p in recs:
        done = p.with_suffix(".harvested.json")
        try:
            if json.loads(done.read_text()).get("library_version") == library.version:
                continue
        except (OSError, ValueError):
            pass
        todo.append(p)
    for p in todo:
        samples.remove_recording(p.name)       # re-read under the new library
    if not todo:
        say("harvest: no new recordings (or none since the last Brand Library change)")
        return 0
    ctx = mp.get_context("spawn")
    added = 0
    with ctx.Pool(args.workers, initializer=_harvest_init,
                  initargs=(str(library.path), 2)) as pool:
        for n, path in enumerate(todo, 1):
            stage("harvest", file=path.name, files_done=n - 1, files=len(todo))
            frames, jpgs = [], []
            try:
                c = av.open(str(path))
                vs = c.streams.video[0]
                vs.thread_type = "AUTO"
                step, next_t = 1.0 / args.harvest_fps, None
                for fr in c.decode(vs):
                    if fr.time is None:
                        continue
                    if next_t is not None and fr.time < next_t:
                        continue
                    next_t = fr.time + step
                    img = fr.to_ndarray(format="bgr24")
                    frames.append((float(fr.time), img))
                    jpgs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
                    if len(frames) >= args.harvest_max_per_file:
                        break
                c.close()
            except Exception as exc:
                say(f"harvest: cannot read {path.name} ({type(exc).__name__}: {exc}); skipped")
                continue
            kept = bg = 0
            for (t, img), res in zip(frames, pool.imap(_harvest_scan, jpgs, chunksize=2)):
                boards = res["boards"]
                model = detector(img) if detector is not None else []
                for q, conf, cls in model:
                    if cls == 0 and conf >= args.pseudo_conf and not any(
                            b["label"] == "keep" and containment(np.asarray(b["quad"], np.float32), q) > 0.5
                            for b in boards):
                        boards.append({"quad": q.round(1).tolist(), "label": "hide", "brand": None,
                                       "texts": [], "by": "model", "conf": float(conf)})
                if not boards:
                    bg += 1
                    if bg % 20:
                        continue                     # a few plain frames, as background
                samples.add(img, boards, "harvest",
                            meta={"recording": path.name, "t": round(t, 2),
                                  "library_version": library.version})
                kept += 1
            write_json(path.with_suffix(".harvested.json"),
                       {"library_version": library.version, "frames": len(frames), "samples": kept,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            added += kept
            say(f"harvest: {path.name}: {len(frames)} frames read, {kept} training frames")
    return added


# ------------------------------------------------------------------ collect

def resolve(board: dict, library: BrandLibrary) -> str | None:
    """A stored board's label under the library as it is now: hide | keep | other | None."""
    by = board.get("by")
    if by == "operator":
        b = library.brands.get(board.get("brand") or "")
        return b["action"] if b is not None else board.get("label")
    if by in ("ocr", "picture"):
        if board.get("texts"):
            v, _ = library.verdict(board["texts"])
            return {"bet": "hide", "neg": "keep"}.get(v, "other")
        b = library.brands.get(board.get("brand") or "")
        return b["action"] if b is not None else None
    if by == "model":
        return "hide" if board.get("verdict") != "neg" else None
    return board.get("label")


def _vetoed(q, boards) -> bool:
    """The model called it betting, but a sponsor to keep was read on it and
    nothing betting was: the model was wrong there."""
    keep_on = any(l == "keep" and by != "model" and containment(q2, q) > 0.5 for q2, l, _, by in boards)
    bet_on = any(l == "hide" and by != "model" and convex_iou(q2, q) > 0.2 for q2, l, _, by in boards)
    return keep_on and not bet_on


def collect(ld: LearnDir, library: BrandLibrary) -> list[dict]:
    """Every sample with its boards resolved: [{image, w, h, group, boards: [(quad, label, kind)]}]
    kind: 'new' for a brand the operator added, 'known' otherwise (for the test)."""
    out = []
    for jf in sorted((ld.root / "samples").glob("*/*.json")):
        try:
            rec = json.loads(jf.read_text())
        except (OSError, ValueError):
            continue
        img = jf.with_suffix(".jpg")
        if not img.exists():
            continue
        boards = []
        for b in rec.get("boards", []):
            lab = resolve(b, library)
            if lab is None:
                continue
            q = np.asarray(b["quad"], np.float32).reshape(4, 2)
            brand = b.get("brand")
            kind = "new" if brand not in (None, "builtin") and brand in library.brands else "known"
            boards.append((q, lab, kind, b.get("by")))
        final = []
        for i, (q, lab, kind, by) in enumerate(boards):
            if by == "model" and _vetoed(q, boards):
                continue
            # the operator's word overrides whatever else was said about that spot
            if any(j != i and by2 == "operator" and l2 != lab and containment(q, q2) > 0.6
                   for j, (q2, l2, _, by2) in enumerate(boards)):
                continue
            dup = next((k for k, (q2, l2, _) in enumerate(final)
                        if l2 == lab and convex_iou(q, q2) > 0.5), None)
            if dup is not None:
                if by == "model":                    # the model's mask is the tighter outline
                    final[dup] = (q, lab, final[dup][2] if final[dup][2] == "new" else kind)
                continue
            final.append((q, lab, kind))
        meta = rec.get("meta") or {}
        day = jf.parent.name
        group = meta.get("recording") or f"{day}-{rec.get('time', '')[11:13]}"   # a recording, or an hour
        out.append({"image": str(img), "w": rec["w"], "h": rec["h"], "group": group,
                    "source": rec.get("source"), "boards": final})
    return out


# --------------------------------------------------------------------- synth

def _paste(dst: np.ndarray, tile: np.ndarray, quad: np.ndarray) -> np.ndarray:
    h, w = tile.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad.astype(np.float32))
    warped = cv2.warpPerspective(tile, M, (dst.shape[1], dst.shape[0]))
    m = np.zeros(dst.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 255)
    m = cv2.GaussianBlur(m, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    return (dst * (1 - m) + warped * m).astype(np.uint8)


def _tile_logo(logo: np.ndarray, length: int, thick: int, rng: random.Random) -> np.ndarray:
    """An LED board repeats its artwork: the logo scaled to the board's
    height, repeated along it on the logo's own background colour."""
    scale = thick / max(1, logo.shape[0])
    lw = max(4, int(round(logo.shape[1] * scale)))
    logo = cv2.resize(logo, (lw, thick), interpolation=cv2.INTER_AREA)
    border = np.concatenate([logo[0], logo[-1], logo[:, 0], logo[:, -1]])
    bg = np.median(border, axis=0).astype(np.uint8)
    tile = np.empty((thick, length, 3), np.uint8)
    tile[:] = bg
    gap = int(lw * rng.uniform(0.15, 0.6))
    x = int(rng.uniform(0, gap + lw)) - lw
    while x < length:
        a, b = max(0, x), min(length, x + lw)
        if b > a:
            tile[:, a:b] = logo[:, a - x:b - x]
        x += lw + gap
    return tile


def synth(items: list[dict], ld: LearnDir, library: BrandLibrary, out_dir: Path, per_brand: int,
          people=None, seed: int = 0) -> list[dict]:
    """Paint each hidden brand's crops onto real boards; train split only."""
    rng = random.Random(seed)
    brands = [b for b in library.brands.values() if b["action"] == "hide" and b.get("examples")]
    hosts = [(it, q) for it in items for q, lab, _ in it["boards"]
             if np.linalg.norm(q[1] - q[0]) >= 120 and np.linalg.norm(q[3] - q[0]) >= 14]
    if not brands or not hosts:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for b in brands:
        logos = [cv2.imread(str(ld.root / rel)) for rel in b["examples"]]
        logos = [x for x in logos if x is not None and min(x.shape[:2]) >= 8]
        if not logos:
            continue
        for k in range(per_brand):
            it, q = rng.choice(hosts)
            img = cv2.imread(it["image"])
            if img is None:
                continue
            qc = q.astype(np.float32)
            length = int(np.linalg.norm(qc[1] - qc[0]))
            thick = int(np.linalg.norm(qc[3] - qc[0]))
            logo = rng.choice(logos).astype(np.float32)
            logo = np.clip((logo * rng.uniform(0.7, 1.3) - 127.5) * rng.uniform(0.8, 1.25) + 127.5, 0, 255)
            tile = _tile_logo(logo.astype(np.uint8), max(8, length), max(4, thick), rng)
            if rng.random() < 0.5:
                tile = cv2.GaussianBlur(tile, (rng.choice([3, 5]),) * 2, 0)
            orig = img.copy()
            img = _paste(img, tile, qc)
            if people is not None:
                pm = people(orig)
                img[pm > 0] = orig[pm > 0]
            name = out_dir / f"synth_{b['id']}_{k:04d}.jpg"
            cv2.imwrite(str(name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            boards = [(q2, lab, kind) for q2, lab, kind in it["boards"] if convex_iou(q2, qc) < 0.3]
            boards.append((qc, "hide", "new"))
            made.append({"image": str(name), "w": it["w"], "h": it["h"], "group": "__synth__",
                         "source": "synth", "boards": boards})
    return made


def person_masker(weights: str | None, device):
    if not weights:
        return None
    try:
        from ultralytics import YOLO
        m = YOLO(weights)
    except Exception as exc:
        say(f"synth: no person model ({type(exc).__name__}); pasted boards may cover people")
        return None

    def masks(img: np.ndarray) -> np.ndarray:
        out = np.zeros(img.shape[:2], np.uint8)
        r = m.predict(img, conf=0.25, classes=[0], imgsz=960, verbose=False, device=device)[0]
        if r.masks is not None:
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(out, [np.round(poly).astype(np.int32)], 1)
        return out
    return masks


# ------------------------------------------------------------------- dataset

def build_dataset(items: list[dict], synth_items: list[dict], out: Path, two_class: bool,
                  val_frac: float, seed: int = 0) -> tuple[Path, list[dict], dict]:
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    val_keys, n_val = set(), 0
    if len(keys) >= 2:
        for k in keys:
            if n_val >= val_frac * len(items):
                break
            val_keys.add(k)
            n_val += len(groups[k])
        if len(val_keys) == len(keys):
            val_keys.discard(keys[-1])
    val = [it for k in val_keys for it in groups[k]]
    if not val_keys and len(items) >= 10:
        # one session only: hold out its last stretch in time, not random frames
        items_sorted = sorted(items, key=lambda it: it["image"])
        cut = int(len(items_sorted) * (1 - val_frac))
        val = items_sorted[cut:]

    def has_new(it):
        return any(lab == "hide" and kind == "new" for _, lab, kind in it["boards"])
    if not any(has_new(it) for it in val):
        # The test must see the new brands, or it cannot tell whether they were
        # learned: hold out the last stretch of the frames that show them.
        vset = {id(it) for it in val}
        new_items = sorted((it for it in items if has_new(it) and id(it) not in vset),
                           key=lambda it: it["image"])
        if len(new_items) >= 5:
            val = val + new_items[-max(1, int(len(new_items) * 0.25)):]
    val_ids = {id(it) for it in val}
    counts = Counter()
    for it in items + synth_items:
        split = "val" if id(it) in val_ids else "train"
        src = Path(it["image"])
        name = f"{src.parent.name}_{src.stem}"
        dst = out / "images" / split / f"{name}.jpg"
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
        lines = []
        w, h = float(it["w"]), float(it["h"])
        for q, lab, _ in it["boards"]:
            if lab == "hide":
                cls = 0
            elif two_class:
                cls = 1
            else:
                continue                            # single class: other adverts are background
            p = np.asarray(q, np.float32).copy()
            p[:, 0] = np.clip(p[:, 0], 0, w - 1) / w
            p[:, 1] = np.clip(p[:, 1], 0, h - 1) / h
            if cv2.contourArea(p) <= 1e-6:
                continue
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)))
            counts[f"{split}_{'hide' if cls == 0 else 'other'}"] += 1
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        counts[f"{split}_images"] += 1
    names = ["betting_ad", "other_ad"] if two_class else ["betting_ad"]
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                         f"nc: {len(names)}\nnames: {names}\n")
    return yaml_path, val, dict(counts)


# ---------------------------------------------------------------------- test

def evaluate(detector, val: list[dict]) -> dict:
    """What matters on air, per board:
        found      a board to hide is >= 50% covered by the model's betting masks
        false      a sponsor to keep is >= 30% covered
        spill      betting-mask area outside every board to hide, % of the picture
    """
    hit = Counter()
    tot = Counter()
    spill = []
    for it in val:
        img = cv2.imread(it["image"])
        if img is None:
            continue
        h, w = img.shape[:2]
        pred = np.zeros((h, w), np.uint8)
        for q, conf, cls in detector(img):
            if cls == 0:
                cv2.fillConvexPoly(pred, np.round(q).astype(np.int32), 1)
        hide_mask = np.zeros((h, w), np.uint8)
        for q, lab, kind in it["boards"]:
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 1)
            area = max(1, int(m.sum()))
            cover = float((pred & m).sum()) / area
            if lab == "hide":
                tot[f"hide_{kind}"] += 1
                hit[f"hide_{kind}"] += int(cover >= 0.5)
                hide_mask |= m
            elif lab == "keep":
                tot["keep"] += 1
                hit["keep"] += int(cover >= 0.3)
        spill.append(float((pred & (1 - hide_mask)).sum()) / (h * w))

    def rate(k):
        return None if tot[k] == 0 else round(hit[k] / tot[k], 4)
    return {"recall_known": rate("hide_known"), "recall_new": rate("hide_new"),
            "false_keep": rate("keep"), "spill_pct": round(100 * float(np.mean(spill)) if spill else 0.0, 4),
            "boards": dict(tot), "images": len(val)}


def gate(base: dict, cand: dict) -> tuple[bool, str]:
    """Promote only if nothing that worked got worse, and something got better."""
    def v(d, k):
        return d[k] if d[k] is not None else 0.0
    if cand["images"] == 0:
        return False, "no held-out frames to test on yet: collect more matches"
    if base["recall_known"] is not None and v(cand, "recall_known") < v(base, "recall_known") - 0.03:
        return False, (f"hides fewer known betting boards ({v(cand, 'recall_known'):.0%} vs "
                       f"{v(base, 'recall_known'):.0%})")
    if base["false_keep"] is not None and v(cand, "false_keep") > v(base, "false_keep") + 0.02:
        return False, (f"covers more sponsors that must stay ({v(cand, 'false_keep'):.0%} vs "
                       f"{v(base, 'false_keep'):.0%})")
    if cand["spill_pct"] > base["spill_pct"] * 1.25 + 0.2:
        return False, f"paints more outside the boards ({cand['spill_pct']:.2f}% vs {base['spill_pct']:.2f}%)"
    if cand["recall_new"] is not None:
        if v(cand, "recall_new") >= v(base, "recall_new") + 0.05:
            return True, (f"finds the new brands on {v(cand, 'recall_new'):.0%} of their boards "
                          f"(was {v(base, 'recall_new'):.0%}); known brands and sponsors held")
        return False, (f"no real gain on the new brands ({v(cand, 'recall_new'):.0%} vs "
                       f"{v(base, 'recall_new'):.0%})")
    if v(cand, "recall_known") > v(base, "recall_known") + 0.01:
        return True, "hides more of the known betting boards; sponsors held"
    return False, "no measurable improvement"


# ---------------------------------------------------------------------- main

def _free_gpu() -> None:
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def detector_for(weights: str, args):
    from render_engine import BoardDetector
    from v2_render import DEFAULTS
    cfg = dict(DEFAULTS)
    cfg.update(ad_weights=weights, ad_imgsz=str(args.eval_imgsz), device=args.device or "",
               half=0 if (args.device or "").lower() in ("cpu", "mps") else 1, render_conf=0.25)
    return BoardDetector(cfg)


def main() -> int:
    global STATE_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learn-dir", default=None, help="default: LEARN_DIR / the RunPod volume / <repo>/learning")
    ap.add_argument("--base", default=None, help="weights to start from (default: the model in use)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--eval-imgsz", type=int, default=1920)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="0, cpu, mps (default: auto)")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--harvest-fps", type=float, default=0.5)
    ap.add_argument("--harvest-max-per-file", type=int, default=600)
    ap.add_argument("--pseudo-conf", type=float, default=0.35,
                    help="the model's own detections at least this confident become labels")
    ap.add_argument("--no-harvest", action="store_true")
    ap.add_argument("--synth", type=int, default=250, help="synthetic frames per hidden brand")
    ap.add_argument("--person-weights", default=os.environ.get("PERSON_WEIGHTS_PT", "yolo11m-seg.pt"))
    ap.add_argument("--classes", choices=["auto", "1", "2"], default="auto",
                    help="2 = also learn 'any other advert' (needs plenty of data; auto decides)")
    ap.add_argument("--freeze", type=int, default=-1,
                    help="backbone layers kept fixed (-1: 10 while the data is small, else 0)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--promote", choices=["auto", "never", "force"], default="auto")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--activate", default=None, help="put a version in use (or 'original') and exit")
    args = ap.parse_args()

    from learn_store import default_learn_dir
    ld = LearnDir(args.learn_dir or default_learn_dir())
    base_original = os.environ.get("AD_WEIGHTS_BASE") or str(ROOT / "models" / "ad_seg_best.pt")
    reg = ModelRegistry(ld.root / "models", base_original)
    if args.status:
        print(json.dumps({"models": reg.status(), "samples": SampleStore(ld).summary(),
                          "state": json.loads((ld.root / "jobs" / "retrain.json").read_text())
                          if (ld.root / "jobs" / "retrain.json").exists() else None}, indent=1))
        return 0
    if args.activate:
        cur = reg.activate(args.activate)
        print(f"in use from the next start: {cur['version'] if cur else 'the original model'}")
        return 0

    jobs = ld.root / "jobs"
    lock = jobs / "retrain.lock"
    if lock.exists():
        try:
            os.kill(int(lock.read_text().strip() or 0), 0)
            say("another retraining run is in progress; not starting a second one")
            return 1
        except (OSError, ValueError):
            pass                                     # stale lock: its process is gone
    lock.write_text(str(os.getpid()))
    STATE_PATH = jobs / "retrain.json"
    STATE.clear()
    STATE.update(pid=os.getpid(), started=time.time())

    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    base = args.base or reg.current_weights()
    version = reg.next_version()
    STATE["version"] = version
    try:
        if not Path(base).is_file() or not base.endswith(".pt"):
            raise SystemExit(f"base weights {base!r} not found (a .pt file is needed to train from)")
        library = BrandLibrary(ld.brands_file)
        samples = SampleStore(ld)
        n_before = samples.summary()["total"]
        write_json(jobs / "last_train.json", {"samples_total": n_before, "started": time.time(),
                                              "version": version})
        say(f"{version}: starting from {base}; library v{library.version} with "
            f"{sum(b['action'] == 'hide' for b in library.brands.values())} hidden and "
            f"{sum(b['action'] == 'keep' for b in library.brands.values())} kept brands; "
            f"{n_before} training frames")

        base_det = detector_for(base, args)
        if not args.no_harvest:
            stage("harvest")
            harvest(ld, samples, library, base_det, args)

        stage("collect")
        items = [it for it in collect(ld, library) if it["boards"] or it["source"] == "harvest"]
        n_hide = sum(1 for it in items for _, lab, _ in it["boards"] if lab == "hide")
        n_other = sum(1 for it in items for _, lab, _ in it["boards"] if lab != "hide")
        say(f"collect: {len(items)} frames, {n_hide} boards to hide, {n_other} other adverts")
        if len(items) < args.min_samples:
            STATE["result"] = f"only {len(items)} training frames so far; {args.min_samples} are needed"
            stage("done")
            return 0

        two = args.classes == "2" or (args.classes == "auto" and len(items) >= 3000 and n_other >= 1000)
        work = ld.root / "datasets" / version
        stage("synth")
        dev = args.device if args.device not in (None, "") else None
        syn = synth(items, ld, library, work / "synth", args.synth,
                    people=person_masker(args.person_weights, dev))
        say(f"synth: {len(syn)} composites")

        stage("dataset")
        yaml_path, val, counts = build_dataset(items, syn, work / "yolo", two, args.val_frac)
        say(f"dataset: {counts}; classes: {'betting_ad + other_ad' if two else 'betting_ad'}")

        stage("train", epochs=args.epochs)
        del base_det
        _free_gpu()
        from ultralytics import YOLO
        model = YOLO(base)

        def on_epoch(trainer):
            STATE["epoch"] = int(trainer.epoch) + 1
            if STATE_PATH is not None:
                write_json(STATE_PATH, STATE)
        model.add_callback("on_fit_epoch_end", on_epoch)
        small = len(items) + len(syn) < 1500
        model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=dev, project=str((ld.root / "runs").resolve()), name=version, exist_ok=True,
                    patience=max(5, args.epochs // 3), lr0=0.002, lrf=0.1, warmup_epochs=1,
                    # little data: adapt the head, keep the backbone
                    freeze=args.freeze if args.freeze >= 0 else (10 if small else 0),
                    degrees=0.0, flipud=0.0, fliplr=0.5, translate=0.10, scale=0.5, shear=0.0,
                    perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=1.0,
                    close_mosaic=max(1, args.epochs // 5), workers=min(8, args.workers), seed=0,
                    verbose=False, plots=False)
        # Ultralytics decides the final folder (it re-roots relative paths)
        run_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", ld.root / "runs" / version))
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            best = run_dir / "weights" / "last.pt"
        if not best.exists():
            raise RuntimeError(f"training left no weights in {run_dir}")
        vdir = ld.root / "models" / version
        vdir.mkdir(parents=True, exist_ok=True)
        weights = vdir / f"ad_seg_{version}.pt"
        shutil.copy2(best, weights)

        stage("test", frames=len(val))
        del model
        _free_gpu()
        m_base = evaluate(detector_for(base, args), val)
        m_cand = evaluate(detector_for(str(weights), args), val)
        passed, reason = gate(m_base, m_cand)
        promote = args.promote == "force" or (args.promote == "auto" and passed)
        report = {"version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "base": base, "library_version": library.version, "classes": 2 if two else 1,
                  "data": {"frames": len(items), "synthetic": len(syn), **counts},
                  "test": {"base": m_base, "candidate": m_cand},
                  "gate": {"passed": passed, "reason": reason}, "promoted": promote,
                  "summary": (f"known {m_cand['recall_known']} (was {m_base['recall_known']}), "
                              f"new {m_cand['recall_new']} (was {m_base['recall_new']}), "
                              f"sponsors covered {m_cand['false_keep']} (was {m_base['false_keep']})")}
        write_json(vdir / "report.json", report)
        if promote:
            reg.activate(version)
        say(f"{version}: {'PROMOTED' if promote else 'kept aside'} -- {reason}")
        say(f"test: base {m_base}")
        say(f"test: new  {m_cand}")
        STATE["result"] = (f"{version} is in use from the next start (or switch now on the operator page)"
                           if promote else f"{version} not promoted: {reason}")
        stage("done", promoted=promote)
        # keep the last three datasets and runs; frames and models stay
        for sub in ("datasets", "runs"):
            olds = sorted((ld.root / sub).glob("v[0-9]*"))[:-3]
            for d in olds:
                shutil.rmtree(d, ignore_errors=True)
        return 0
    except KeyboardInterrupt:
        stage("stopped")
        return 130
    except SystemExit as exc:
        stage("failed", error=str(exc))
        raise
    except Exception as exc:
        stage("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        STATE["finished"] = time.time()
        if STATE_PATH is not None:
            write_json(STATE_PATH, STATE)
        # Everything collected up to now counts as trained on (the harvest's own
        # frames included), so an idle pod does not retrain on the same data.
        write_json(jobs / "last_train.json", {**(json.loads((jobs / "last_train.json").read_text())
                                                  if (jobs / "last_train.json").exists() else {}),
                                              "samples_total": SampleStore(ld).summary()["total"],
                                              "finished": time.time(), "stage": STATE.get("stage")})
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
