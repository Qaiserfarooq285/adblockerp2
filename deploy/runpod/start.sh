#!/usr/bin/env bash
# Container entrypoint on the RunPod pod:
#   1. optional SSH, 2. board model in place (a retrained version if one is in
#   use), 3. TensorRT engines (built once per GPU model and model version,
#   cached on the volume), 4. exec the live pipeline.
# Every setting comes from environment variables set on the pod (see
# deploy/runpod/.env.example); nothing here needs editing.
set -euo pipefail

APP=${APP_DIR:-/app}
VOL=${VOLUME_DIR:-/workspace}
MODELS=${MODELS_DIR:-$VOL/models}
# brand library, training frames, original recordings, model versions
export LEARN_DIR=${LEARN_DIR:-$VOL/learning}
cd "$APP"
mkdir -p "$MODELS" "$VOL/live" "$LEARN_DIR"

# 1. SSH: RunPod injects PUBLIC_KEY when an SSH key is set on the account.
if [ -n "${PUBLIC_KEY:-}" ] && command -v sshd >/dev/null 2>&1; then
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    mkdir -p /run/sshd && ssh-keygen -A >/dev/null 2>&1 || true
    /usr/sbin/sshd || echo "[start] sshd failed to start (continuing)"
fi

# 2. The trained board model: already on the volume, baked into the image,
#    or downloaded once from AD_WEIGHTS_URL.
AD_PT="$MODELS/ad_seg_best.pt"
if [ ! -s "$AD_PT" ]; then
    if [ -s "$APP/models/ad_seg_best.pt" ]; then
        cp "$APP/models/ad_seg_best.pt" "$AD_PT"
    elif [ -n "${AD_WEIGHTS_URL:-}" ]; then
        echo "[start] downloading board model from AD_WEIGHTS_URL"
        curl -fL --retry 5 --retry-delay 3 -o "$AD_PT.part" "$AD_WEIGHTS_URL" && mv "$AD_PT.part" "$AD_PT"
    else
        echo "[start] ERROR: no board model. Upload ad_seg_best.pt to $MODELS/ (volume) or set"
        echo "[start]        AD_WEIGHTS_URL on the pod, then restart it. Idling so you can connect."
        exec sleep infinity
    fi
fi
PERSON_PT="$APP/models/yolo11m-seg.pt"
[ -s "$PERSON_PT" ] || PERSON_PT="yolo11m-seg.pt"      # Ultralytics downloads it
# The original stays the fallback ("back to the original model") and what the
# first retraining starts from; a version the retraining job promoted (or the
# operator chose) is used instead when there is one.
export AD_WEIGHTS_BASE="$AD_PT"
export PERSON_WEIGHTS_PT="$PERSON_PT"
CURRENT="$LEARN_DIR/models/current.json"
if [ -s "$CURRENT" ]; then
    V_PT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("weights",""))' "$CURRENT" 2>/dev/null || true)
    if [ -n "$V_PT" ] && [ -s "$V_PT" ]; then
        echo "[start] board model: retrained version $(basename "$V_PT")"
        AD_PT="$V_PT"
    else
        echo "[start] $CURRENT names missing weights -- using the original model"
    fi
fi
export AD_WEIGHTS="${AD_WEIGHTS:-$AD_PT}"
export PERSON_WEIGHTS="${PERSON_WEIGHTS:-$PERSON_PT}"

# 3. TensorRT: 2-3x faster inference.  Engines are specific to the GPU model,
#    so they are built here, once, and kept on the volume.
if [ "${USE_TENSORRT:-1}" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
          | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')
    ENV_FILE="$MODELS/engines/$GPU-$(basename "$AD_PT" .pt).env"
    if python scripts/export_tensorrt.py --ad-weights "$AD_PT" --person-weights "$PERSON_PT" \
            --out "$MODELS/engines" --env-file "$ENV_FILE"; then
        set -a; . "$ENV_FILE"; set +a
        # where engines for other model versions go (a live switch uses one if built)
        export TRT_ENGINE_DIR="$(dirname "$AD_WEIGHTS")"
        echo "[start] using TensorRT engines for $GPU"
    else
        echo "[start] TensorRT export failed -- continuing with PyTorch FP16 weights"
    fi
fi

# 4. Go.  INPUT_URL / OUTPUT_URL / HTTP_TOKEN / ... are read by live_stream.py.
exec python scripts/live_stream.py --config "${LIVE_CONFIG:-config.yaml}"
