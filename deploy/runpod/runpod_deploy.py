#!/usr/bin/env python3
"""Create / start / stop / inspect the live ad-blocker pod on RunPod.

Standard library only (runs on any laptop).  Settings and the API key come
from deploy/runpod/.env -- copy .env.example, paste your RunPod API key
(runpod.io -> Settings -> API Keys), fill in the rest.  .env is git-ignored.

    python deploy/runpod/runpod_deploy.py create             # custom image (Dockerfile)
    python deploy/runpod/runpod_deploy.py create --bootstrap # stock image, clones the repo
    python deploy/runpod/runpod_deploy.py status <pod_id>    # state, cost, URLs, live stats
    python deploy/runpod/runpod_deploy.py stop   <pod_id>    # stop billing for GPU time
    python deploy/runpod/runpod_deploy.py start  <pod_id>
    python deploy/runpod/runpod_deploy.py delete <pod_id>    # terminate (asks first)
    python deploy/runpod/runpod_deploy.py list
    python deploy/runpod/runpod_deploy.py create --dry-run   # print the request, send nothing

REST API: https://docs.runpod.io/api-reference (POST /v1/pods etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
HERE = Path(__file__).resolve().parent
PLACEHOLDER = "PASTE_YOUR_RUNPOD_API_KEY_HERE"
BOOTSTRAP_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
HTTP_PORT = 8080

# Environment variables forwarded to the pod (only those that are set).
POD_ENV = ("INPUT_URL", "OUTPUT_URL", "HTTP_TOKEN", "AD_WEIGHTS_URL", "USE_TENSORRT",
           "DET_INTERVAL", "ENCODER", "VIDEO_BITRATE", "DEVICE", "LIVE_CONFIG",
           "REPO_URL", "REPO_REF")


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0]          # trailing " # comment"; a bare # (URLs) is kept
            env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k in env or k == "RUNPOD_API_KEY"})
    return env


def api(method: str, path: str, key: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"RunPod API {method} {path}: HTTP {e.code} {e.read().decode(errors='replace')}")


BOOTSTRAP = r"""set -e
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg git curl ca-certificates libgl1 libglib2.0-0 openssh-server >/dev/null
if [ -d /workspace/app/.git ]; then git -C /workspace/app fetch -q origin "$REPO_REF" && git -C /workspace/app checkout -q FETCH_HEAD;
else git clone -q --depth 1 -b "$REPO_REF" "$REPO_URL" /workspace/app; fi
cd /workspace/app && pip install -q -r requirements-live.txt && (pip install -q -r requirements-trt.txt || true)
mkdir -p models && (cd models && python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')")
APP_DIR=/workspace/app exec bash deploy/runpod/start.sh"""


def pod_body(env: dict, bootstrap: bool) -> dict:
    gpus = [g.strip() for g in env.get("GPU_TYPES", "NVIDIA GeForce RTX 4090").split(",") if g.strip()]
    pod_env = {k: env[k] for k in POD_ENV if env.get(k)}
    pod_env.setdefault("OUTPUT_URL", "hls:///workspace/live/out")
    body = {
        "name": env.get("POD_NAME", "betting-ad-blocker-live"),
        "cloudType": env.get("CLOUD_TYPE", "SECURE"),
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpus,
        "gpuTypePriority": "custom",           # try them in the order listed
        "containerDiskInGb": int(env.get("CONTAINER_DISK_GB", 40)),
        "volumeInGb": int(env.get("VOLUME_GB", 30)),
        "volumeMountPath": "/workspace",
        "ports": [f"{HTTP_PORT}/http", "22/tcp"],
        "minVCPUPerGPU": int(env.get("MIN_VCPU", 8)),
        "minRAMPerGPU": int(env.get("MIN_RAM_GB", 24)),
        "env": pod_env,
    }
    if env.get("NETWORK_VOLUME_ID"):
        body["networkVolumeId"] = env["NETWORK_VOLUME_ID"]
        body.pop("volumeInGb")
    if env.get("REGISTRY_AUTH_ID"):
        body["containerRegistryAuthId"] = env["REGISTRY_AUTH_ID"]
    if bootstrap:
        body["imageName"] = BOOTSTRAP_IMAGE
        pod_env.setdefault("REPO_URL", "https://github.com/Qaiserfarooq285/adblockerp2.git")
        pod_env.setdefault("REPO_REF", "main")
        body["dockerEntrypoint"] = ["bash", "-c"]
        body["dockerStartCmd"] = [BOOTSTRAP]
    else:
        image = env.get("IMAGE", "")
        if not image or "YOUR_" in image:
            raise SystemExit("set IMAGE in deploy/runpod/.env (the image you built and pushed), "
                             "or use --bootstrap to run from the stock PyTorch image")
        body["imageName"] = image
    return body


def urls(pod_id: str, env: dict) -> dict:
    tok = env.get("HTTP_TOKEN", "").strip("/")
    base = f"https://{pod_id}-{HTTP_PORT}.proxy.runpod.net" + (f"/{tok}" if tok else "")
    return {"player": base + "/", "hls": base + "/out/index.m3u8", "status": base + "/status.json"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["create", "status", "start", "stop", "delete", "list"])
    ap.add_argument("pod_id", nargs="?")
    ap.add_argument("--env-file", default=str(HERE / ".env"))
    ap.add_argument("--bootstrap", action="store_true",
                    help="stock PyTorch image; the pod clones REPO_URL@REPO_REF and installs at start")
    ap.add_argument("--dry-run", action="store_true", help="print the request, do not call RunPod")
    ap.add_argument("--yes", action="store_true", help="do not ask before delete")
    args = ap.parse_args()

    env = load_env(Path(args.env_file))
    key = env.get("RUNPOD_API_KEY", "")
    if args.command != "create" and not args.pod_id and args.command != "list":
        raise SystemExit(f"{args.command} needs a pod id (see: list)")

    if args.command == "create":
        body = pod_body(env, args.bootstrap)
        if args.dry_run:
            shown = json.loads(json.dumps(body))
            for k in ("HTTP_TOKEN", "AD_WEIGHTS_URL"):
                if k in shown.get("env", {}):
                    shown["env"][k] = "***"
            print(json.dumps({"POST": API + "/pods", "body": shown}, indent=1))
            return 0
    if not key or key == PLACEHOLDER:
        raise SystemExit(f"No RunPod API key: put RUNPOD_API_KEY=... in {args.env_file} "
                         f"(copy .env.example) or export RUNPOD_API_KEY")

    if args.command == "create":
        pod = api("POST", "/pods", key, body)
        pid = pod.get("id")
        print(f"pod created: {pid}  ({pod.get('costPerHr', '?')} credits/h, "
              f"GPU {pod.get('gpu', {}).get('displayName', gpus_str(body))})")
        for k, v in urls(pid, env).items():
            print(f"  {k:7s} {v}")
        print("First start: image pull + TensorRT build take several minutes; "
              f"check progress with: status {pid}")
    elif args.command == "list":
        pods = api("GET", "/pods", key)
        for p in pods if isinstance(pods, list) else pods.get("pods", []):
            print(f"{p.get('id')}  {p.get('desiredStatus', '?'):9s}  {p.get('name')}  "
                  f"{p.get('costPerHr', '?')}/h  {p.get('image', p.get('imageName', ''))}")
    elif args.command == "status":
        p = api("GET", f"/pods/{args.pod_id}", key)
        print(f"{p.get('id')}  {p.get('desiredStatus')}  {p.get('costPerHr', '?')}/h  "
              f"last started {p.get('lastStartedAt', '-')}")
        u = urls(args.pod_id, env)
        for k, v in u.items():
            print(f"  {k:7s} {v}")
        try:
            with urllib.request.urlopen(u["status"], timeout=10) as r:
                print("pipeline:", json.dumps(json.loads(r.read()), indent=1))
        except Exception as exc:
            print(f"pipeline status not reachable yet ({type(exc).__name__})")
    elif args.command in ("start", "stop"):
        api("POST", f"/pods/{args.pod_id}/{args.command}", key)
        print(f"{args.command} requested for {args.pod_id}")
    elif args.command == "delete":
        if not args.yes and input(f"Terminate pod {args.pod_id}? Its volume is deleted. [y/N] ").lower() != "y":
            print("cancelled")
            return 1
        api("DELETE", f"/pods/{args.pod_id}", key)
        print(f"terminated {args.pod_id}")
    return 0


def gpus_str(body: dict) -> str:
    return "/".join(body.get("gpuTypeIds", []))


if __name__ == "__main__":
    sys.exit(main())
