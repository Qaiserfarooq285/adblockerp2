#!/usr/bin/env python3
"""Model versions and the retraining job, as the live pipeline sees them.

    <learn_dir>/models/
      v001/ad_seg_v001.pt      a version learn_retrain.py produced
      v001/report.json         what it was trained on and how it scored
      current.json             {"version": "v001", "weights": ".../ad_seg_v001.pt"}
                               the version in use; absent = the original model

deploy/runpod/start.sh loads the current version when the pod starts; the
operator page can also switch to a version at once (the board model is
swapped between two frames) or roll back to the original.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class ModelRegistry:
    def __init__(self, models_dir: Path, base_weights: str):
        self.dir = Path(models_dir)
        self.base = str(base_weights)

    def current(self) -> dict | None:
        cur = _read(self.dir / "current.json")
        if cur and Path(cur.get("weights", "")).is_file():
            return cur
        return None

    def current_weights(self) -> str:
        cur = self.current()
        return cur["weights"] if cur else self.base

    def versions(self) -> list[dict]:
        out = []
        for d in sorted(self.dir.glob("v[0-9]*")):
            rep = _read(d / "report.json") or {}
            w = d / f"ad_seg_{d.name}.pt"
            if w.is_file():
                out.append({"version": d.name, "weights": str(w),
                            "created": rep.get("created"), "promoted": rep.get("promoted"),
                            "passed": rep.get("gate", {}).get("passed"),
                            "verdict": rep.get("gate", {}).get("reason"),
                            "summary": rep.get("summary")})
        return out

    def next_version(self) -> str:
        n = max((int(d.name[1:]) for d in self.dir.glob("v[0-9]*") if d.name[1:].isdigit()),
                default=0)
        return f"v{n + 1:03d}"

    def activate(self, version: str | None) -> dict | None:
        """Make `version` current (None = back to the original model)."""
        path = self.dir / "current.json"
        if version in (None, "", "base", "original"):
            if path.exists():
                path.unlink()
            return None
        w = self.dir / version / f"ad_seg_{version}.pt"
        if not w.is_file():
            raise FileNotFoundError(f"no weights for {version}")
        cur = {"version": version, "weights": str(w),
               "activated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cur, indent=1))
        tmp.replace(path)
        return cur

    def status(self) -> dict:
        cur = self.current()
        return {"original": self.base, "current": cur["version"] if cur else "original",
                "versions": self.versions()}


class RetrainJob:
    """learn_retrain.py as a background process of the live pipeline."""

    def __init__(self, learn_dir: Path, base_weights_fn):
        self.ld = Path(learn_dir)
        self.jobs = self.ld / "jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.base_weights_fn = base_weights_fn
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.started: float | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, extra: list[str] | None = None) -> bool:
        if self.running():
            return False
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.log_path = self.jobs / f"retrain-{stamp}.log"
        cmd = [sys.executable, str(ROOT / "scripts" / "learn_retrain.py"),
               "--learn-dir", str(self.ld), "--base", self.base_weights_fn(), *(extra or [])]
        log = open(self.log_path, "w")
        # Lowest CPU priority: the live stream keeps the machine first.
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                     preexec_fn=(lambda: os.nice(10)) if hasattr(os, "nice") else None)
        self.started = time.time()
        return True

    def stop(self) -> bool:
        if not self.running():
            return False
        self.proc.send_signal(signal.SIGTERM)
        return True

    def status(self) -> dict:
        state = _read(self.jobs / "retrain.json") or {}
        tail = []
        log = self.log_path
        if log is None:
            logs = sorted(self.jobs.glob("retrain-*.log"))
            log = logs[-1] if logs else None
        if log is not None and log.exists():
            try:
                tail = log.read_text(errors="replace").splitlines()[-15:]
            except OSError:
                pass
        return {"running": self.running(), "started": self.started, "state": state,
                "exit": None if self.proc is None else self.proc.poll(), "log": tail}
