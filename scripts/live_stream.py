#!/usr/bin/env python3
"""Live betting-ad blocking: stream in -> GPU renderer -> stream out.

    ingest   ffmpeg (HLS / SRT / RTMP / HTTP-TS / file; network + reconnects)
               --MPEG-TS pipe-->  PyAV demux + decode, timestamps kept
    engine   render_engine.AdBlockEngine in async mode: analysis (camera motion,
             board detector, tracker) and rendering (person model, compositing)
             run on separate threads, D frames apart
    egress   PyAV writes raw video + PCM (NUT) --pipe--> ffmpeg encodes
             (h264_nvenc on the GPU, or libx264) and publishes: an HLS
             directory, RTMP(S), SRT, UDP, or a file.  The same encode also
             records the match (tee), and a second stream shows ORIGINAL |
             BETTING ADS HIDDEN side by side, both halves from one frame
    http     serves the player page (processed / side-by-side views),
             status.json and the recordings on --http-port (RunPod exposes it
             as https://<POD_ID>-<PORT>.proxy.runpod.net)
    learn    the operator page (operator.html + /api/, own password): mark a
             logo on a paused frame, decide new sponsors the brand scan found,
             switch model versions.  Decisions apply live within a second; the
             original feed and training frames are kept under LEARN_DIR for
             learn_retrain.py, which trains the next model version

Audio is decoded and re-muxed with its ORIGINAL timestamps, mapped onto the
same output timeline as the video, so lip-sync survives processing latency,
dropped frames and reconnects.  Nothing here knows about betting ads; all of
that lives in render_engine.py, shared with the offline renderer.

    python scripts/live_stream.py                                  # config.yaml `live:`
    python scripts/live_stream.py --input https://.../index.m3u8 --output hls:///workspace/live/out
    python scripts/live_stream.py --input match.mp4 --output rtmp://host/app/key --loop

Environment variables override config.yaml (the RunPod container sets them):
INPUT_URL, OUTPUT_URL, HTTP_PORT, HTTP_TOKEN, ENCODER, VIDEO_BITRATE,
AD_WEIGHTS, PERSON_WEIGHTS, AD_IMGSZ, PERSON_IMGSZ, PERSON_DYNAMIC,
DET_INTERVAL, DEVICE, and for learning LEARN, LEARN_DIR, OPERATOR_TOKEN, SCAN,
RECORD_ORIGINAL, AUTO_TRAIN.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from fractions import Fraction
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

LIVE_DEFAULTS = dict(
    input="https://karostream.tulix.tv/FIFA/index.m3u8",
    output="hls:///workspace/live/out",
    http_port=8080,
    http_token="",
    encoder="auto",              # auto | nvenc | x264
    video_bitrate="6M",
    audio_bitrate="160k",
    gop_seconds=2.0,
    hls_segment_seconds=2,
    hls_list_size=8,
    hls_live_start_index=-3,
    loop=False,
    stall_timeout_s=20.0,
    max_backoff_s=30.0,
    max_queue_seconds=4.0,
    adaptive=True,
    max_det_interval=4,
    status_every_s=10.0,
    http_root="",                # served on http_port; "" = the HLS output's parent
                                 # directory, or /workspace/live for other outputs
    record=True,                 # also save the processed match under record_dir
    record_dir="",               # "" = <http_root>/recordings
    compare=True,                # also publish original | processed side by side
    compare_output="",           # "" = hls://<http_root>/compare
    compare_bitrate="4M",
    engine={},
    learn={},                    # teaching it new brands: LEARN_DEFAULTS
)
LEARN_DEFAULTS = dict(
    enabled=True,
    dir="",                      # "" = LEARN_DIR, else /workspace/learning (RunPod volume),
                                 # else <repo>/learning
    operator_token="",           # the operator page's password; "" = no operator page
    scan=True,                   # read every board ~1/s, so brands the model has not
                                 # learned are hidden as soon as the library names them
    scan_threads=2,
    record_original=True,        # keep the original feed (stream copy) to retrain from
    original_max_gb=15.0,        # the oldest original recordings go beyond this
    samples=True,                # save training frames automatically
    snapshot_seconds=90,         # how far back the operator can pick a frame to mark
    auto_train="idle",           # off | idle: retrain once the input has been down idle_minutes |
                                 # daily@HH:MM: every day at that time, UTC (shares the GPU
                                 # with a live stream if one is running)
    idle_minutes=20,
    min_new_samples=30,          # ...and at least this many new training frames exist
)
# Engine settings that differ from the offline renderer's defaults.  The longer
# look-ahead lets a brand read by the scan (~0.5-1 s per frame) still be
# painted from the first frame of the shot, through the back-fill.
LIVE_ENGINE_DEFAULTS = dict(det_interval=2, ocr_async=1, delay_frames=36, backfill_frames=36)

ENV_LIVE = {"INPUT_URL": "input", "OUTPUT_URL": "output", "HTTP_PORT": "http_port",
            "HTTP_TOKEN": "http_token", "ENCODER": "encoder", "VIDEO_BITRATE": "video_bitrate",
            "HTTP_ROOT": "http_root", "RECORD": "record", "RECORD_DIR": "record_dir",
            "COMPARE": "compare", "COMPARE_OUTPUT": "compare_output"}
ENV_LEARN = {"LEARN": "enabled", "LEARN_DIR": "dir", "OPERATOR_TOKEN": "operator_token",
             "SCAN": "scan", "RECORD_ORIGINAL": "record_original", "AUTO_TRAIN": "auto_train"}
ENV_ENGINE = {"AD_WEIGHTS": "ad_weights", "PERSON_WEIGHTS": "person_weights",
              "AD_IMGSZ": "ad_imgsz", "PERSON_IMGSZ": "person_imgsz",
              "PERSON_DYNAMIC": "person_dynamic", "DET_INTERVAL": "det_interval",
              "DEVICE": "device"}

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_STOP = threading.Event()


def log(tag: str, msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


# --------------------------------------------------------------------- config

def _coerce(value, like):
    if isinstance(like, bool):
        return str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(float(value))
    if isinstance(like, float):
        return float(value)
    return value


def load_settings(args) -> tuple[dict, dict]:
    from v2_render import DEFAULTS as RENDER_DEFAULTS
    live = {k: (dict(v) if isinstance(v, dict) else v) for k, v in LIVE_DEFAULTS.items()}
    cfg_path = Path(args.config) if args.config else ROOT / "config.yaml"
    if cfg_path.exists():
        live.update((yaml.safe_load(cfg_path.read_text()) or {}).get("live", {}) or {})
    for env, key in ENV_LIVE.items():
        if os.environ.get(env):
            live[key] = _coerce(os.environ[env], LIVE_DEFAULTS[key])
    for key in ("input", "output", "http_port", "encoder", "video_bitrate"):
        v = getattr(args, key)
        if v is not None:
            live[key] = v
    if args.loop:
        live["loop"] = True

    eng = dict(RENDER_DEFAULTS)
    eng.update(LIVE_ENGINE_DEFAULTS)
    for k, v in (live.get("engine") or {}).items():
        if k not in RENDER_DEFAULTS:
            raise SystemExit(f"config live.engine.{k}: unknown renderer setting")
        eng[k] = _coerce(v, RENDER_DEFAULTS[k])
    for env, key in ENV_ENGINE.items():
        if os.environ.get(env):
            eng[key] = _coerce(os.environ[env], RENDER_DEFAULTS[key])

    learn = dict(LEARN_DEFAULTS)
    for k, v in (live.get("learn") or {}).items():
        if k not in LEARN_DEFAULTS:
            raise SystemExit(f"config live.learn.{k}: unknown setting")
        learn[k] = _coerce(v, LEARN_DEFAULTS[k]) if v is not None else LEARN_DEFAULTS[k]
    for env, key in ENV_LEARN.items():
        if os.environ.get(env):
            learn[key] = _coerce(os.environ[env], LEARN_DEFAULTS[key])
    live["learn"] = learn

    kind, hls_dir = output_target(live["output"])
    if not live["http_root"]:
        live["http_root"] = str(Path(hls_dir).parent) if kind == "hls" else "/workspace/live"
    if not live["record_dir"]:
        live["record_dir"] = str(Path(live["http_root"]) / "recordings")
    if not live["compare_output"]:
        live["compare_output"] = "hls://" + str(Path(live["http_root"]) / "compare")
    return live, eng


# --------------------------------------------------------------- ffmpeg bits

def _drain(proc: subprocess.Popen, tag: str, tail: deque) -> None:
    for raw in iter(proc.stderr.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if line:
            tail.append(line)
            log(tag, line)


def ffmpeg_major() -> int:
    try:
        out = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout
        v = out.split()[2].lstrip("n")
        return int(v.split(".")[0]) if v[0].isdigit() else 99   # git builds: assume new
    except Exception:
        return 4


NVENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "cbr", "-bf", "0"]
NVENC_ARGS_OLD = ["-c:v", "h264_nvenc", "-preset", "llhq", "-rc", "cbr", "-bf", "0"]
X264_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"]


def pick_encoder(pref: str) -> list[str]:
    """Probe that NVENC really works in this container (listed != usable:
    the driver's `video` capability may be missing), else fall back to x264."""
    if pref not in ("x264", "libx264"):
        for args in (NVENC_ARGS, NVENC_ARGS_OLD):
            probe = [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                     "color=c=black:s=256x144:r=30", "-frames:v", "5", *args, "-f", "null", "-"]
            if subprocess.run(probe, capture_output=True).returncode == 0:
                log("egress", f"encoder: NVENC ({' '.join(args[3:5])})")
                return args
        if pref == "nvenc":
            log("egress", "WARNING: NVENC requested but not usable here -- falling back to libx264")
    log("egress", "encoder: libx264 (CPU)")
    return X264_ARGS


def ingest_cmd(url: str, s: dict, record_dir: str | None = None) -> list[str]:
    """ffmpeg reading the input; with `record_dir` it also saves the ORIGINAL
    feed there (stream copy, 15-minute files) for the retraining job -- via the
    tee muxer with onfail=ignore, so a full disk never stops the live input."""
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "warning"]
    is_file = Path(url).exists()
    if is_file:
        cmd += ["-re"] + (["-stream_loop", "-1"] if s["loop"] else [])
    elif url.startswith(("http://", "https://")):
        cmd += ["-user_agent", "betting-ad-blocker/1.0"]
        if ".m3u8" in url:
            cmd += ["-live_start_index", str(int(s["hls_live_start_index"]))]
        else:   # raw MPEG-TS over HTTP (e.g. Flussonic /mpegts): let ffmpeg reconnect
            cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    cmd += ["-fflags", "+genpts+discardcorrupt", "-i", url,
            "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy"]
    if record_dir:
        pattern = _tee_escape(str(Path(record_dir) / "original-%Y%m%d-%H%M%S.ts"))
        cmd += ["-f", "tee", "[f=mpegts]pipe:1|[f=segment:segment_time=900:strftime=1:"
                             f"segment_format=mpegts:onfail=ignore]{pattern}"]
    else:
        cmd += ["-f", "mpegts", "pipe:1"]
    return cmd


def output_target(output: str) -> tuple[str, str | None]:
    """(kind, hls_dir) for an output spec."""
    if output.startswith("hls://"):
        return "hls", output[len("hls://"):]
    if output.endswith(".m3u8"):
        return "hls", str(Path(output).parent)
    for scheme, kind in (("rtmp://", "flv"), ("rtmps://", "flv"), ("srt://", "mpegts"),
                         ("udp://", "mpegts"), ("rtp://", "rtp_mpegts")):
        if output.startswith(scheme):
            return kind, None
    return "file", None


def served_file(path: str | None, root: Path | None) -> str | None:
    """`path` as a URL path under the HTTP root, or None when it is not served."""
    if not path or root is None:
        return None
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def served_path(target: str, root: Path | None) -> str | None:
    """The playlist of an HLS output spec, as served; None for RTMP/SRT/files."""
    kind, d = output_target(target)
    return served_file(str(Path(d) / "index.m3u8"), root) if kind == "hls" else None


def _tee_escape(v: str) -> str:
    """Escape a value inside a tee muxer slave's [option=value:...] list."""
    return v.replace("\\", "\\\\").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")


def muxer_for(target: str, s: dict) -> tuple[str, dict, str]:
    """(ffmpeg format, muxer options, destination) for one output spec."""
    kind, hls_dir = output_target(target)
    if kind == "hls":
        Path(hls_dir).mkdir(parents=True, exist_ok=True)
        return "hls", {
            "hls_time": int(s["hls_segment_seconds"]),
            "hls_list_size": int(s["hls_list_size"]),
            "hls_flags": "delete_segments+independent_segments+omit_endlist"
                         "+program_date_time+discont_start",
            "hls_start_number_source": "epoch",
            "hls_segment_filename": str(Path(hls_dir) / "seg_%d.ts"),
        }, str(Path(hls_dir) / "index.m3u8")
    if kind == "flv":
        return "flv", {"flvflags": "no_duration_filesize"}, target
    if kind in ("mpegts", "rtp_mpegts"):
        return kind, {}, target
    fmt = {".ts": "mpegts", ".mkv": "matroska", ".flv": "flv"}.get(Path(target).suffix.lower(), "mp4")
    # fragmented mp4: playable while it is still being written
    return fmt, ({"movflags": "+frag_keyframe+empty_moov+default_base_moof"} if fmt == "mp4" else {}), target


def egress_cmd(s: dict, enc: list[str], fps: float, has_audio: bool,
               compare: bool, record: str | None) -> list[str]:
    """One ffmpeg process: the processed stream (+ its recording) and,
    optionally, the side-by-side stream.

    The recording rides on the processed stream's own encode through the tee
    muxer -- one encode, two destinations -- and is marked onfail=ignore, so
    a full disk or a bad path can never take the live output down with it.
    """
    gop = max(1, int(round(fps * float(s["gop_seconds"]))))
    seg = int(s["hls_segment_seconds"])
    passthrough = ["-fps_mode", "passthrough"] if ffmpeg_major() >= 5 else ["-vsync", "0"]

    def encode(vmap: str, br: str, global_header: bool) -> list[str]:
        a = ["-map", vmap] + (["-map", "0:a:0"] if has_audio else [])
        a += enc + ["-b:v", br, "-maxrate", br, "-bufsize", f"{2 * int(_bits(br))}",
                    "-g", str(gop), "-keyint_min", str(gop), "-pix_fmt", "yuv420p",
                    "-profile:v", "high", "-force_key_frames", f"expr:gte(t,n_forced*{seg})"]
        a += ["-sc_threshold", "0"] if "libx264" in enc else ["-forced-idr", "1"]
        a += passthrough
        if has_audio:
            a += ["-c:a", "aac", "-b:a", str(s["audio_bitrate"]), "-ar", "48000", "-ac", "2",
                  "-af", "aresample=async=1000"]
        if global_header:
            # FLV/MP4 need codec headers up front; ffmpeg sets that itself for
            # a plain output but cannot see through the tee muxer.
            a += ["-flags:v", "+global_header", "-flags:a", "+global_header"]
        return a

    def plain(fmt: str, opts: dict, dst: str) -> list[str]:
        a = []
        for k, v in opts.items():
            a += [f"-{k}", str(v)]
        return a + ["-f", fmt, "-y", dst]

    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "warning", "-f", "nut", "-i", "pipe:0"]
    fmt, opts, dst = muxer_for(s["output"], s)
    if record:
        main = "[" + ":".join([f"f={fmt}"] + [f"{k}={_tee_escape(str(v))}" for k, v in opts.items()]) + "]"
        cmd += encode("0:v:0", str(s["video_bitrate"]), fmt in ("flv", "mp4", "matroska"))
        cmd += ["-f", "tee", f"{main}{dst}|[f=mpegts:onfail=ignore]{record}"]
    else:
        cmd += encode("0:v:0", str(s["video_bitrate"]), False) + plain(fmt, opts, dst)
    if compare:
        cmd += encode("0:v:1", str(s["compare_bitrate"]), False) + plain(*muxer_for(s["compare_output"], s))
    return cmd


def half_size(w: int, h: int) -> tuple[int, int]:
    """Each half of the side-by-side picture: half size, even (yuv420p)."""
    return max(2, (w // 2) & ~1), max(2, (h // 2) & ~1)


def side_by_side(original_half: np.ndarray | None, processed: np.ndarray,
                 canvas: np.ndarray) -> np.ndarray:
    """ORIGINAL | BETTING ADS HIDDEN, both halves from the same instant.

    Built from one frame's before and after, never from two streams, so the
    halves cannot drift apart the way two separate players do.  A missing
    original leaves its half black rather than silently showing the
    processed picture twice.
    """
    h2, w2 = canvas.shape[0], canvas.shape[1] // 2
    if original_half is None:
        canvas[:, :w2] = 0
    else:
        if original_half.shape[:2] != (h2, w2):
            original_half = cv2.resize(original_half, (w2, h2), interpolation=cv2.INTER_AREA)
        canvas[:, :w2] = original_half
    canvas[:, w2:] = cv2.resize(processed, (w2, h2), interpolation=cv2.INTER_AREA)
    canvas[:, w2 - 1:w2 + 1] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x0, label in ((0, "ORIGINAL"), (w2, "BETTING ADS HIDDEN")):
        (tw, th), base = cv2.getTextSize(label, font, 0.8, 2)
        cx = x0 + w2 // 2
        cv2.rectangle(canvas, (cx - tw // 2 - 12, 0), (cx + tw // 2 + 12, th + base + 14), (0, 0, 0), -1)
        cv2.putText(canvas, label, (cx - tw // 2, th + 8), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _bits(rate: str) -> float:
    r = str(rate).strip().lower()
    mult = {"k": 1e3, "m": 1e6, "g": 1e9}.get(r[-1], 1.0)
    return float(r[:-1] if r[-1] in "kmg" else r) * mult


# ------------------------------------------------------------------ timeline

class Timeline:
    """Maps each input session's timestamps onto one continuous output clock.

    Every reconnect restarts the input clock at an arbitrary value; an HLS
    discontinuity can jump it within a session.  Video re-anchors the mapping
    (continuing from the last output frame); audio reuses it, so both move
    together and stay in sync.
    """

    def __init__(self):
        self.offset: float | None = None
        self.last_v: float | None = None
        self.frame_dur = 1001 / 30000

    def new_session(self, fps: float) -> None:
        self.offset = None
        self.frame_dur = 1.0 / fps if fps > 0 else self.frame_dur

    def video(self, t_in: float) -> float:
        expect = None if self.last_v is None else self.last_v + self.frame_dur
        if self.offset is None or (expect is not None and abs(t_in + self.offset - expect) > 5.0):
            self.offset = (0.0 if expect is None else expect) - t_in
        t = t_in + self.offset
        if self.last_v is not None and t <= self.last_v:
            t = self.last_v + 1e-3
        self.last_v = t
        return t

    def audio(self, t_in: float) -> float:
        if self.offset is None:
            # Audio usually decodes before the first video frame of a session;
            # anchor on it rather than drop it (the video then lands a few ms
            # later on the same mapping, well inside the 5 s re-anchor band).
            self.offset = (0.0 if self.last_v is None else self.last_v + self.frame_dur) - t_in
        return t_in + self.offset


# -------------------------------------------------------------------- ingest

class Ingest(threading.Thread):
    def __init__(self, s: dict, frames_q: queue.Queue, audio_q: queue.Queue, stats: dict,
                 record_dir: str | None = None):
        super().__init__(name="ingest", daemon=True)
        self.s, self.frames_q, self.audio_q, self.stats = s, frames_q, audio_q, stats
        self.url = s["input"]
        self.is_file = Path(self.url).exists()
        # a file input is its own original; only a live feed is recorded
        self.record_dir = None if self.is_file else record_dir
        self.timeline = Timeline()
        self.last_packet = time.monotonic()
        self.tail: deque = deque(maxlen=20)

    def run(self) -> None:
        backoff = 1.0
        while not _STOP.is_set():
            ok = self._session()
            if _STOP.is_set():
                break
            if self.is_file and not self.s["loop"]:
                self.frames_q.put(("eof",))
                return
            self.stats["reconnects"] += 1
            wait = 1.0 if ok else backoff
            log("ingest", f"input ended ({'clean' if ok else 'error'}); reconnecting in {wait:.0f}s")
            _STOP.wait(wait)
            backoff = 1.0 if ok else min(2 * backoff, float(self.s["max_backoff_s"]))

    def _put_frame(self, item) -> None:
        try:
            self.frames_q.put_nowait(item)
        except queue.Full:
            # Falling behind real time: drop this frame rather than block the
            # network reader (a blocked reader stalls the socket and loses the
            # stream).  Dropping the newest, not the oldest, keeps latency just
            # as bounded and can never discard a queued "session" marker.
            self.stats["dropped"] += 1

    def _session(self) -> bool:
        import av
        cmd = ingest_cmd(self.url, self.s, self.record_dir)
        log("ingest", "starting: " + " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=_drain, args=(proc, "ingest-ffmpeg", self.tail), daemon=True).start()
        self.last_packet = time.monotonic()

        def watchdog():
            while proc.poll() is None and not _STOP.is_set():
                if time.monotonic() - self.last_packet > float(self.s["stall_timeout_s"]):
                    log("ingest", f"no data for {self.s['stall_timeout_s']}s -- restarting input")
                    proc.kill()
                    return
                time.sleep(1.0)
            if _STOP.is_set() and proc.poll() is None:
                proc.kill()
        threading.Thread(target=watchdog, daemon=True).start()

        container = None
        clean = False
        try:
            container = av.open(proc.stdout, format="mpegts",
                                options={"probesize": "1500000", "analyzeduration": "3000000"})
            if not container.streams.video:
                raise RuntimeError("input has no video stream")
            vs = container.streams.video[0]
            vs.thread_type = "AUTO"
            ast = container.streams.audio[0] if container.streams.audio else None
            rate = vs.average_rate or vs.guessed_rate or Fraction(30000, 1001)
            fps = float(rate) if 5 <= float(rate) <= 120 else 30000 / 1001
            info = dict(width=vs.codec_context.width, height=vs.codec_context.height,
                        fps=fps, has_audio=ast is not None)
            log("ingest", f"connected: {info['width']}x{info['height']} @ {fps:.3f} fps, "
                          f"audio={'yes' if ast is not None else 'no'}")
            self.timeline.new_session(fps)
            self.frames_q.put(("session", info))
            res = None
            if ast is not None:
                import av.audio.resampler as avr
                res = avr.AudioResampler(format="s16", layout="stereo", rate=48000)
            streams = [vs] + ([ast] if ast is not None else [])
            for packet in container.demux(*streams):
                if _STOP.is_set():
                    break
                self.last_packet = time.monotonic()
                try:
                    decoded = packet.decode()
                except Exception:
                    self.stats["decode_errors"] += 1
                    continue
                if packet.stream is vs:
                    for fr in decoded:
                        if fr.time is None:
                            continue
                        img = self._bgr(fr)
                        # The side-by-side stream needs the frame as it was
                        # BEFORE painting; the renderer paints in place, so the
                        # half-size copy is taken here, on this thread.
                        small = (cv2.resize(img, half_size(img.shape[1], img.shape[0]),
                                            interpolation=cv2.INTER_AREA)
                                 if self.s["compare"] else None)
                        self.stats["frames_in"] += 1
                        self._put_frame(("frame", img, self.timeline.video(fr.time),
                                         time.monotonic(), small))
                else:
                    for fr in decoded:
                        if fr.time is None:
                            continue
                        t = self.timeline.audio(fr.time)
                        for chunk in res.resample(fr):
                            pcm = chunk.to_ndarray()
                            try:
                                self.audio_q.put_nowait((pcm, t))
                            except queue.Full:
                                self.stats["audio_dropped"] += 1
                            t += pcm.shape[1] / 2 / 48000.0
            clean = not _STOP.is_set()
        except Exception as exc:
            log("ingest", f"error: {type(exc).__name__}: {exc}")
        finally:
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            if proc.returncode not in (0, None) and not _STOP.is_set():
                clean = False
        return clean

    @staticmethod
    def _bgr(fr) -> np.ndarray:
        if fr.format.name == "yuv420p" and fr.width % 2 == 0 and fr.height % 2 == 0:
            return cv2.cvtColor(fr.to_ndarray(), cv2.COLOR_YUV2BGR_I420)   # fast path
        return fr.to_ndarray(format="bgr24")


# -------------------------------------------------------------------- egress

class _Pipe:
    """Write-only file object for PyAV: no seek(), so the muxer streams."""

    def __init__(self, f):
        self.f = f

    def write(self, b) -> int:
        self.f.write(b)
        return len(b)

    def flush(self) -> None:
        self.f.flush()


class Egress(threading.Thread):
    """Frames (in order, from the engine) + audio -> NUT -> ffmpeg -> output."""

    def __init__(self, s: dict, video_q: queue.Queue, audio_q: queue.Queue, stats: dict,
                 clock=None):
        super().__init__(name="egress", daemon=True)
        self.s, self.video_q, self.audio_q, self.stats = s, video_q, audio_q, stats
        self.clock = clock                  # learn_store.OutputClock: player time -> frame
        self.enc = pick_encoder(str(s["encoder"]))
        self.params = None                  # (w, h, fps, has_audio), set by "reconfig"
        self.compare = bool(s["compare"])
        self.recording: str | None = None   # file the current session records into
        self.pending_audio: deque = deque()
        self.tail: deque = deque(maxlen=20)
        self.latency = deque(maxlen=300)

    def _recording_path(self) -> str | None:
        """A new file per output session: a restart never overwrites a match."""
        if not self.s["record"]:
            return None
        try:
            d = Path(self.s["record_dir"])
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log("egress", f"recording disabled: cannot create {self.s['record_dir']} ({exc})")
            return None
        return str(d / time.strftime("match-%Y%m%d-%H%M%SZ.ts", time.gmtime()))

    def run(self) -> None:
        """Runs until it receives the None sentinel -- never merely because a
        stop was requested, so shutdown can always drain the engine."""
        backoff = 1.0
        while True:
            if self.params is None:
                item = self.video_q.get()
                if item is None:
                    return
                if isinstance(item, dict):
                    self.params = (item["width"], item["height"], item["fps"], item["has_audio"])
                continue
            try:
                if self._session():
                    return                               # end of stream
                backoff = 1.0                            # new geometry: restart at once
            except Exception as exc:
                if _STOP.is_set() and self.video_q.empty():
                    return
                self.stats["egress_restarts"] += 1
                log("egress", f"output error ({type(exc).__name__}: {exc}); restarting in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(2 * backoff, float(self.s["max_backoff_s"]))

    def _session(self) -> bool:
        """One ffmpeg output process.  True at end of stream; False when the
        input geometry changed and the output must be restarted with it."""
        import av
        w, h, fps, has_audio = self.params
        self.recording = self._recording_path()
        cmd = egress_cmd(self.s, self.enc, fps, has_audio, self.compare, self.recording)
        log("egress", "starting: " + " ".join(cmd))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=_drain, args=(proc, "egress-ffmpeg", self.tail), daemon=True).start()
        out = av.open(_Pipe(proc.stdin), mode="w", format="nut")
        rate = Fraction(fps).limit_denominator(1001)
        vs = out.add_stream("rawvideo", rate=rate)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        vs.codec_context.time_base = Fraction(1, 90000)
        vc = canvas = None
        if self.compare:                    # second video stream: original | processed
            w2, h2 = half_size(w, h)
            vc = out.add_stream("rawvideo", rate=rate)
            vc.width, vc.height, vc.pix_fmt = 2 * w2, h2, "yuv420p"
            vc.codec_context.time_base = Fraction(1, 90000)
            canvas = np.zeros((h2, 2 * w2, 3), np.uint8)
        ast = None
        if has_audio:
            ast = out.add_stream("pcm_s16le", rate=48000)
            ast.codec_context.layout = "stereo"
            ast.codec_context.format = "s16"
        last_v = last_a_end = -1.0
        done = True
        try:
            while True:
                item = self._get()
                if item is None:
                    break                                # end of stream
                if item == "tick":
                    if proc.poll() is not None:
                        raise RuntimeError(f"ffmpeg exited with {proc.returncode}")
                    continue
                if isinstance(item, dict):               # new input session
                    new = (item["width"], item["height"], item["fps"], item["has_audio"])
                    if new != self.params:
                        self.params = new
                        done = False
                        break
                    continue
                frame, meta = item
                t, t_recv = meta[0], meta[1]
                if t <= last_v:
                    t = last_v + 1e-3
                if last_v < 0 and self.clock is not None:
                    self.clock.start(t)              # the HLS programme date starts here
                # Audio up to just past this frame; the muxer interleaves.
                if ast is not None:
                    last_a_end = self._write_audio(out, ast, t + 0.25, last_a_end)
                pts = int(round(t * 90000))
                for stream, img in ((vs, frame), (vc, None)):
                    if stream is None:
                        continue
                    if img is None:                  # the side-by-side picture, same pts
                        img = side_by_side(meta[2] if len(meta) > 2 else None, frame, canvas)
                    vf = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420),
                                                    format="yuv420p")
                    vf.pts = pts
                    vf.time_base = Fraction(1, 90000)
                    for p in stream.encode(vf):
                        out.mux(p)
                last_v = t
                self.stats["frames_out"] += 1
                self.latency.append(time.monotonic() - t_recv)
                if proc.poll() is not None:
                    raise RuntimeError(f"ffmpeg exited with {proc.returncode}")
        finally:
            try:
                for stream in (vs, vc):
                    if stream is not None:
                        for p in stream.encode(None):
                            out.mux(p)
                out.close()
            except Exception:
                pass
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        return done

    def _get(self):
        try:
            return self.video_q.get(timeout=1.0)
        except queue.Empty:
            return "tick"

    def _write_audio(self, out, ast, until: float, last_end: float) -> float:
        import av
        while True:
            try:
                self.pending_audio.append(self.audio_q.get_nowait())
            except queue.Empty:
                break
        while self.pending_audio and self.pending_audio[0][1] <= until:
            pcm, t = self.pending_audio.popleft()
            n = pcm.shape[1] // 2
            if t + n / 48000.0 <= last_end + 1e-3:
                continue                                 # overlaps what is written
            af = av.AudioFrame.from_ndarray(pcm, format="s16", layout="stereo")
            af.sample_rate = 48000
            af.pts = int(round(t * 48000))
            af.time_base = Fraction(1, 48000)
            for p in ast.encode(af):
                out.mux(p)
            last_end = t + n / 48000.0
        # never let audio pile up without bound if video stalls
        while len(self.pending_audio) > 3000:
            self.pending_audio.popleft()
        return last_end


# ---------------------------------------------------------------------- http

class _Handler(SimpleHTTPRequestHandler):
    token = ""
    api = None                              # operator_api.OperatorAPI, when enabled

    def log_message(self, *a) -> None:      # quiet: HLS players poll constantly
        pass

    def end_headers(self) -> None:
        if not getattr(self, "_is_api", False):
            self.send_header("Access-Control-Allow-Origin", "*")
        if self.path.split("?")[0].endswith((".m3u8", ".json")):
            self.send_header("Cache-Control", "no-cache, no-store")
        super().end_headers()

    def _api_route(self) -> tuple[str | None, str]:
        """('state', query) for <token>/api/state?..., else (None, '')."""
        path, _, query = self.path.partition("?")
        if self.token:
            prefix = f"/{self.token}"
            if not path.startswith(prefix + "/"):
                return None, ""
            path = path[len(prefix):]
        return (path[5:], query) if path.startswith("/api/") else (None, "")

    def do_GET(self) -> None:
        sub, query = self._api_route()
        if sub is None:
            return super().do_GET()
        self._api("GET", sub, query)

    def do_POST(self) -> None:
        sub, query = self._api_route()
        if sub is None:
            self.send_error(404)
            return
        self._api("POST", sub, query)

    def _api(self, method: str, sub: str, query: str) -> None:
        self._is_api = True
        if self.api is None:
            code, ctype, body = 404, "application/json", \
                b'{"error": "the operator page is off: set OPERATOR_TOKEN on the pod"}'
        else:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 256 * 1024:
                code, ctype, body = 413, "application/json", b'{"error": "request too large"}'
            else:
                data = self.rfile.read(n) if n > 0 else b""
                try:
                    code, ctype, body = self.api.handle(method, sub, query, data, self.headers)
                except Exception as exc:              # never let a bad request kill the server
                    log("api", f"{method} {sub}: {type(exc).__name__}: {exc}")
                    code, ctype, body = 500, "application/json", b'{"error": "internal error"}'
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path):
        if str(path).endswith(".m3u8"):
            return "application/vnd.apple.mpegurl"
        if str(path).endswith(".ts"):
            return "video/mp2t"
        return super().guess_type(path)

    def translate_path(self, path):
        if self.token:
            prefix = f"/{self.token}"
            if not (path == prefix or path.startswith(prefix + "/")):
                return "/nonexistent"
            path = path[len(prefix):] or "/"
        return super().translate_path(path)


class _Server(ThreadingHTTPServer):
    # The default listen backlog is 5: a few HLS players, or the operator
    # page fetching a strip of frames, overflow it and connections are reset.
    request_queue_size = 128
    daemon_threads = True


def serve_http(root: Path, port: int, token: str, api=None) -> None:
    handler = functools.partial(_Handler, directory=str(root))
    _Handler.token = token.strip("/")
    _Handler.api = api
    httpd = _Server(("0.0.0.0", port), handler)
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    pod = os.environ.get("RUNPOD_POD_ID")
    base = f"https://{pod}-{port}.proxy.runpod.net" if pod else f"http://<host>:{port}"
    pre = f"/{_Handler.token}" if _Handler.token else ""
    log("http", f"serving {root} -- player page: {base}{pre}/"
                + (f" -- operator page: {base}{pre}/operator.html" if api is not None else ""))


# --------------------------------------------------------------------- learn

def runtime_weights(pt: str) -> str:
    """The TensorRT engine start.sh built for these weights on this GPU, if any."""
    d = os.environ.get("TRT_ENGINE_DIR")
    if d:
        eng = Path(d) / (Path(pt).stem + ".engine")
        if eng.is_file():
            return str(eng)
    return pt


class Learning:
    """The Brand Library, the brand scan, training data, model versions and the
    operator page's API -- everything that lets the pipeline learn new brands."""

    def __init__(self, s: dict, ecfg: dict):
        from brand_library import BrandLibrary
        from brand_scan import BrandScanner
        from learn_jobs import ModelRegistry, RetrainJob
        from learn_store import (LearnDir, OutputClock, ReviewQueue, SampleStore, Snapshots,
                                 default_learn_dir)
        from render_engine import parse_boxes
        L = self.cfg = s["learn"]
        self.ld = LearnDir(L["dir"] or default_learn_dir())
        self.library = BrandLibrary(self.ld.brands_file)
        self.reviews = ReviewQueue(self.ld)
        self.samples = SampleStore(self.ld)
        self.snapshots = Snapshots(seconds=float(L["snapshot_seconds"]))
        self.clock = OutputClock()
        self.record_dir = str(self.ld.root / "recordings") if L["record_original"] else None
        # the ORIGINAL model (start.sh passes it when AD_WEIGHTS is a version or an engine)
        self.models = ModelRegistry(self.ld.root / "models",
                                    os.environ.get("AD_WEIGHTS_BASE") or ecfg["ad_weights"])
        self.job = RetrainJob(self.ld.root, self.models.current_weights)
        self.scanner = None
        if L["scan"]:
            sc = BrandScanner(self.library, {"threads": L["scan_threads"],
                                             "exclude_boxes": parse_boxes(ecfg.get("motion_exclude_boxes"))},
                              reviews=self.reviews, samples=self.samples if L["samples"] else None,
                              learn_root=self.ld.root)
            self.scanner = sc if sc.available else None
        self.engine = None
        self.api = None
        if L["operator_token"]:
            from operator_api import OperatorAPI
            self.api = OperatorAPI(L["operator_token"], self.ld, self.library, self.reviews,
                                   self.samples, self.snapshots, self.clock, lambda: self.engine,
                                   self.models, self.job, self.swap_model)
        self._last_prune = 0.0
        self._last_auto = 0.0
        log("learn", f"learning data in {self.ld.root}: {len(self.library.brands)} brands in the "
                     f"library, {self.samples.summary()['total']} training frames; brand scan "
                     f"{'on' if self.scanner else 'off'}; operator page "
                     f"{'on' if self.api else 'off (set OPERATOR_TOKEN)'}; model in use: "
                     f"{self.models.status()['current']}")

    def swap_model(self, weights: str) -> bool:
        """Load another board model beside the running one, then switch the
        engine over between two frames: no restart, no gap in the stream."""
        eng = self.engine
        if eng is None:
            return False

        def work():
            try:
                from render_engine import BoardDetector
                cfg = dict(eng.cfg)
                cfg["ad_weights"] = runtime_weights(weights)
                det = BoardDetector(cfg)
                det(np.zeros((eng.h, eng.w, 3), np.uint8))          # warm up off the frame loop
                eng.detector = det
                log("learn", f"board model switched to {cfg['ad_weights']}")
            except Exception as exc:
                log("learn", f"model switch failed ({type(exc).__name__}: {exc}); "
                             f"the current model stays")
        threading.Thread(target=work, name="model-swap", daemon=True).start()
        return True

    def tick(self, idle_s: float) -> None:
        """Housekeeping, from the main loop's status tick."""
        from learn_store import prune_recordings
        now = time.monotonic()
        if self.record_dir and now - self._last_prune > 600:
            self._last_prune = now
            freed = prune_recordings(Path(self.record_dir), float(self.cfg["original_max_gb"]))
            if freed:
                log("learn", f"deleted {freed / 1e9:.1f} GB of the oldest original recordings")
        mode = str(self.cfg["auto_train"]).strip().lower()
        due = False
        if mode == "idle":
            due = idle_s > 60 * float(self.cfg["idle_minutes"]) and now - self._last_auto > 3600
        elif mode.startswith("daily@"):
            try:
                hh, mm = (int(x) for x in mode[6:].split(":"))
            except ValueError:
                hh = mm = -1
            utc = time.gmtime()
            today = time.strftime("%Y%m%d", utc)
            due = (0 <= hh < 24 and (utc.tm_hour, utc.tm_min) >= (hh, mm)
                   and getattr(self, "_daily_done", None) != today)
            if due:
                self._daily_done = today
        if due and not self.job.running():
            last = {}
            try:
                last = json.loads((self.ld.root / "jobs" / "last_train.json").read_text())
            except (OSError, ValueError):
                pass
            new = self.samples.summary()["total"] - int(last.get("samples_total", 0))
            if new >= int(self.cfg["min_new_samples"]):
                self._last_auto = now
                log("learn", f"{'input idle for %.0f min' % (idle_s / 60) if mode == 'idle' else mode}"
                             f" and {new} new training frames: starting a retraining run")
                self.job.start()

    def summary(self) -> dict:
        return {"brands": len(self.library.brands), "review": len(self.reviews.pending()),
                "samples": self.samples.summary()["total"],
                "model": self.models.status()["current"], "training": self.job.running()}

    def close(self) -> None:
        if self.scanner is not None:
            self.scanner.close()
        self.reviews.save()


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML with a `live:` section (default: config.yaml)")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None,
                    help="hls://<dir> | rtmp(s)://... | srt://... | udp://... | out.mp4/.ts")
    ap.add_argument("--http-port", type=int, default=None, dest="http_port")
    ap.add_argument("--encoder", default=None, choices=["auto", "nvenc", "x264"])
    ap.add_argument("--video-bitrate", default=None, dest="video_bitrate")
    ap.add_argument("--loop", action="store_true", help="loop a local file input forever")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (benchmark)")
    ap.add_argument("--debug", action="store_true", help="draw track state on the output")
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args()

    s, ecfg = load_settings(args)
    if args.print_config:
        print(json.dumps({"live": s, "engine": ecfg}, indent=1, default=str))
        return 0

    from render_engine import AdBlockEngine
    learn = Learning(s, ecfg) if s["learn"]["enabled"] else None
    http_root: Path | None = Path(s["http_root"])
    try:
        http_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log("http", f"cannot create {http_root} ({exc}): no player page or status file")
        http_root = None
    if http_root is not None:
        player = ROOT / "deploy" / "web" / "player.html"
        if player.exists():
            shutil.copy(player, http_root / "index.html")
        operator = ROOT / "deploy" / "web" / "operator.html"
        if learn is not None and learn.api is not None and operator.exists():
            shutil.copy(operator, http_root / "operator.html")
        if int(s["http_port"]) > 0:
            serve_http(http_root, int(s["http_port"]), str(s.get("http_token") or ""),
                       api=learn.api if learn is not None else None)
    links = {"stream": served_path(s["output"], http_root),
             "compare": served_path(s["compare_output"], http_root) if s["compare"] else None,
             "operator": bool(learn is not None and learn.api is not None)}

    def write_status(st: dict) -> None:
        if http_root is None:
            return
        tmp = http_root / "status.json.tmp"
        tmp.write_text(json.dumps(st, indent=1, default=float))
        tmp.replace(http_root / "status.json")

    write_status(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), input=s["input"],
                      output=s["output"], fps_in=0, fps_out=0, **links, recording=None))

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: _STOP.set())

    stats = {k: 0 for k in ("frames_in", "frames_out", "dropped", "audio_dropped", "reconnects",
                            "decode_errors", "egress_restarts")}
    cap = max(8, int(30 * float(s["max_queue_seconds"])))
    frames_q: queue.Queue = queue.Queue(maxsize=cap)
    audio_q: queue.Queue = queue.Queue(maxsize=20000)
    video_q: queue.Queue = queue.Queue(maxsize=64)          # engine -> egress
    ingest = Ingest(s, frames_q, audio_q, stats,
                    record_dir=learn.record_dir if learn is not None else None)
    egress = Egress(s, video_q, audio_q, stats, clock=learn.clock if learn is not None else None)
    ingest.start()
    egress.start()

    engine = None
    geometry = None
    generation = 0
    last_frame_at = time.monotonic()
    base_det = cur_det = int(ecfg["det_interval"])
    last_adapt = last_status = t_start = time.monotonic()
    last_in = last_out = 0
    try:
        while not _STOP.is_set():
            if args.seconds and time.monotonic() - t_start > args.seconds:
                break
            try:
                item = frames_q.get(timeout=1.0)
            except queue.Empty:
                item = None
            if item is not None and item[0] == "eof":
                break
            if item is not None and item[0] == "session":
                info = item[1]
                g = (info["width"], info["height"])
                if engine is not None and g != geometry:
                    engine.flush()
                    engine.close()
                    engine = None
                if engine is None:
                    log("engine", f"loading models for {g[0]}x{g[1]} "
                                  f"(ad={ecfg['ad_weights']} person={ecfg['person_weights']})")
                    engine = AdBlockEngine(ecfg, g[0], g[1], live=True, debug=args.debug,
                                           fps=info["fps"],
                                           library=learn.library if learn is not None else None,
                                           scanner=learn.scanner if learn is not None else None)
                    generation += 1
                    engine.generation = generation
                    if learn is not None:
                        learn.engine = engine
                    engine.warmup()
                    engine.start_async(video_q)
                    geometry = g
                else:
                    engine.discontinuity()
                video_q.put(dict(info))                 # (re)configures the egress
                continue
            if item is not None and engine is not None:
                _, img, t_out, t_recv, small = item
                last_frame_at = time.monotonic()
                if learn is not None:
                    # the ORIGINAL picture, for the operator to mark logos on
                    learn.snapshots.offer(engine.idx, img, t_out, engine.generation)
                engine.push(img, meta=(t_out, t_recv, small))

            now = time.monotonic()
            if engine is not None and s["adaptive"] and now - last_adapt > 1.0:
                fill = frames_q.qsize() / cap
                if fill > 0.5 and cur_det < int(s["max_det_interval"]):
                    cur_det += 1
                    engine.set_det_interval(cur_det)
                    log("adapt", f"falling behind (input queue {fill:.0%}): detector every {cur_det} frames")
                    last_adapt = now
                elif fill < 0.1 and cur_det > base_det and now - last_adapt > 10.0:
                    cur_det -= 1
                    engine.set_det_interval(cur_det)
                    log("adapt", f"caught up: detector every {cur_det} frames")
                    last_adapt = now
            if now - last_status >= float(s["status_every_s"]):
                dt = now - last_status
                fin, fout = stats["frames_in"], stats["frames_out"]
                lat = float(np.median(egress.latency)) if egress.latency else 0.0
                st = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), input=s["input"],
                          output=s["output"], fps_in=round((fin - last_in) / dt, 2),
                          fps_out=round((fout - last_out) / dt, 2), queue=frames_q.qsize(),
                          det_interval=cur_det, processing_latency_s=round(lat, 2),
                          engine_ms=(engine.timing if engine else {}), **stats,
                          tracks=(len(engine.tracker.tracks) if engine else 0), **links,
                          recording=served_file(egress.recording, http_root),
                          learning=learn.summary() if learn is not None else None)
                log("status", f"in {st['fps_in']} fps, out {st['fps_out']} fps, queue {st['queue']}, "
                              f"det/{cur_det}, latency {lat:.2f}s, dropped {stats['dropped']}, "
                              f"reconnects {stats['reconnects']} | "
                              + (engine.timing_summary() if engine else "no input yet"))
                write_status(st)
                last_status, last_in, last_out = now, fin, fout
                if learn is not None:
                    learn.tick(now - last_frame_at)
    finally:
        # Order matters: stop the input, let the engine drain through the
        # still-running egress, then end the output with the sentinel.
        _STOP.set()
        if engine is not None:
            engine.flush(timeout=15)
            engine.close()
        if learn is not None:
            learn.close()
        video_q.put(None)
        egress.join(timeout=20)
        ingest.join(timeout=5)
        log("main", f"stopped: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
