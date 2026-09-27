# Live betting-ad blocker for a RunPod GPU pod.
#
#   docker build -f deploy/runpod/Dockerfile -t YOUR_REGISTRY/betting-ad-blocker-live:1.0 .
#   docker push YOUR_REGISTRY/betting-ad-blocker-live:1.0
#
# If models/ad_seg_best.pt exists at build time it is baked into the image --
# push that image to a PRIVATE registry only.  Otherwise the pod downloads it
# from AD_WEIGHTS_URL on first start (see start.sh).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG INSTALL_TENSORRT=1
# `video` exposes NVENC/NVDEC inside the container; without it h264_nvenc is
# listed by ffmpeg but fails at runtime (live_stream.py probes and falls back).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    YOLO_CONFIG_DIR=/workspace/.ultralytics

# ffmpeg: stream ingest/egress (Ubuntu 22.04's build has h264_nvenc, libx264,
# libsrt and TLS).  libgl1/libglib: OpenCV.  openssh-server: optional SSH.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates libgl1 libglib2.0-0 openssh-server \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-live.txt requirements-trt.txt ./
RUN pip install -r requirements-live.txt \
    && if [ "$INSTALL_TENSORRT" = "1" ]; then pip install -r requirements-trt.txt; fi

COPY . .
# Stock COCO person model, baked in so a pod start needs no model download.
RUN mkdir -p /app/models && cd /app/models \
    && python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')" \
    && chmod +x /app/deploy/runpod/start.sh

EXPOSE 8080
ENTRYPOINT ["/app/deploy/runpod/start.sh"]
