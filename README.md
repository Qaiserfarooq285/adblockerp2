# Betting Ad Blocker

An offline computer-vision pipeline that finds betting/gambling advertising boards in football broadcast video and hides them with a flat, board-matched fill — without covering players, referees, the goal frame, or any non-betting sponsor on the same rail.

![Clean rail example](docs/images/example_clean_rail.jpg)
*Betano hidden on both perimeter rails. `kraken`, `kraken.com`, `MARRIOTT BONVOY`, `Be Active`, and `FIFA` on the same physical strip are left untouched — brand discrimination, not just "bright rectangle" detection.*

## What it does

1. **Detects** betting boards in broadcast video with a segmentation model trained specifically on this footage (not a generic ad-blocker — it distinguishes `Betano` / `Kalshi` / `PredictStreet` from co-located sponsors like Kraken, Marriott, Visa, FIFA, and scoreboard graphics).
2. **Fits** a clean quadrilateral per board from the segmentation mask — one rectangle per physical rail, not a stack of overlapping boxes.
3. **Tracks** boards with the camera's own geometry: a frame-to-frame homography (scoreboard and LED boards masked out of the fit) carries each panel, and every detection corrects it through a Kalman gain. The panel neither copies detector jitter nor lags a pan, and it holds position through detection dropouts instead of blinking. A 0.5 s look-ahead lets a panel start on a board's first visible frame and end right after its last detection.
4. **Fills** each board with a colour sampled from the rail itself, so the result reads as a blank/unlit board rather than a black box.
5. **Protects** everything that must stay visible: players and referees (hard subtraction from an independent person model, restricted to people standing *in front of* the board — crowd and photographers behind it don't get carved out of the panel), the goal frame and net, the live scoreboard/clock overlay, and the pitch itself.

## Example: goal-area protection

![Goal protection example](docs/images/example_goal_protection.jpg)
*Near-side Betano rail hidden with a clean, board-matched rectangle; the goalpost and net stay fully visible. The far-side board behind the goal is a known limitation — see below.*

## Pipeline

```
input video
   │
   ▼
frame extraction (2 fps)
   │
   ▼
OCR sweep (whole-frame + tiled high-res pass)
   │
   ▼
brand classification  →  betting polygon labels + hard negatives
   │
   ▼
targeted synthetic data (repaint + close-up composites, weighted to weak brands)
   │
   ▼
train YOLO11-seg (fixture-disjoint train/val split — no frame leakage)
   │
   ▼
render: detect → merge/clean masks → fit rectangle per rail → track (camera homography + Kalman) → fill → protect → composite
   │
   ▼
verify: OCR before/after, person-pixel overlap, sponsor preservation, panel stability
```

The same renderer runs on files (`v2_render.py`) and on live streams (`live_stream.py`); see [Live streaming on RunPod](#live-streaming-on-runpod).

Every stage is a standalone script under `scripts/`, orchestrated end-to-end by `scripts/v2_pipeline.py`, which is resumable — a crash, reboot, or manual stop picks back up from the last completed stage rather than restarting.

## Current results

Measured on the full 20-clip evaluation set (not a curated sample), by re-running OCR on the actual rendered output:

| Metric | Result |
|---|---|
| Betting ad text removed | **73.5%** |
| Non-betting sponsors preserved | **99.8%** |
| Player/referee pixels incorrectly painted | **0.81% max, ~0% on most clips** |

### Known limitations

This is stated plainly rather than glossed over:

- **Boards seen from a steep angle behind the goal** are detected less reliably than boards on the near touchline — the model under-predicts very long or heavily foreshortened rails.
- **`Kalshi`'s green-on-black wordmark** is the weakest brand class (fewer real training examples than `Betano`), and is still missed on a meaningful fraction of frames despite targeted synthetic oversampling.
- Both are training-data problems, not renderer bugs — the fill/tracking/protection logic behaves correctly whenever the board is detected at all. The straightforward next step is harvesting more real crops of exactly these two cases (extreme angle, low-contrast green) rather than further renderer tuning.

## Repository layout

```
scripts/          all pipeline stages: extraction, OCR, labelling, synthetic data,
                   training, rendering, verification
notebooks/         interactive workflow entry point
config.yaml         pipeline configuration
```

Large generated artifacts (`data/`, `models/`, `runs/`, `outputs/`, `logs/`, `input/`) are intentionally not committed — they are multi-GB of raw video, extracted frames, and trained weights that are reproducible by running the pipeline, not by cloning it.

## Key scripts

| Script | Purpose |
|---|---|
| `v2_pipeline.py` | End-to-end orchestrator with resume-from-last-stage and a run lock to prevent concurrent execution |
| `v2_ocr_survey.py` / `v2_ocr_tiles.py` | Whole-frame and tiled high-resolution OCR passes |
| `v2_brands.py` | Betting vs. non-betting brand classification from OCR text |
| `v2_annotate.py` | OCR text → betting-strip polygon labels, with hard-negative frames |
| `v2_synth.py` | Targeted synthetic training data (brand repaint + close-up composites) for measured weak spots |
| `v2_dataset.py` | Fixture-disjoint train/val split (no near-duplicate frame leakage) |
| `v2_train.py` | Trains the YOLO11-seg detector, resumable from `last.pt` |
| `v2_render.py` | The file renderer CLI: detection → clean rectangle fitting → tracking → fill → protection → composite (`--tracker legacy` keeps the previous tracker for A/B) |
| `render_engine.py` | The per-frame engine shared by files and live streams: analysis and a delayed render, async on live |
| `board_tracker.py` | Board tracks carried by the camera homography and corrected by each detection (Kalman, across/along the rail) |
| `geometry.py` | Frame-to-frame homography (masked, FB-checked LK + MAGSAC), cut detection, live overlay estimate |
| `live_stream.py` | Live pipeline: HLS/SRT/RTMP in → engine → NVENC/x264 → HLS/RTMP/SRT out, with A/V sync and reconnects |
| `brand_library.py` | The Brand Library: brands to hide / keep, edited live on the operator page, overriding the built-in lists |
| `brand_scan.py` | Reads every board ~1/s, so brands the model has not learned are hidden as soon as the library names them |
| `operator_api.py` | The operator page's API: mark a logo on a paused frame, decide new sponsors, switch model versions |
| `learn_retrain.py` | Trains the next model version from the collected frames and recordings, tests it, promotes it only if it passes |
| `export_tensorrt.py` | Builds TensorRT FP16 engines on the GPU that will run them |
| `stability_report.py` | Scores a render for panel movement relative to the scene and blinks (old vs. new tracker) |
| `v2_verify.py` | Measures the actual rendered output against the original (OCR + person-overlap), rather than trusting the renderer's own claims |

## Setup

```bash
pip install -r requirements.txt
```

Requires `ffmpeg` on `PATH` and a CUDA-capable GPU for training/inference.

## Usage

```bash
# Run the whole pipeline (resumable)
python scripts/v2_pipeline.py

# Check progress at any time
python scripts/v2_pipeline.py --status

# Render a single video with an existing model
python scripts/v2_render.py --input path/to/video.mp4 --output out.mp4

# Same clip with the previous tracker, then compare stability
python scripts/v2_render.py --input clip.mp4 --output old.mp4 --tracker legacy --panel-log old.jsonl
python scripts/v2_render.py --input clip.mp4 --output new.mp4 --panel-log new.jsonl
python scripts/stability_report.py old.jsonl new.jsonl
```

`--debug` draws the track state on the output: green = detected this frame, orange = carried by the camera homography, cyan = back-filled before the first detection, red outline = suppressed as a non-betting sponsor.

## Live streaming on RunPod

`scripts/live_stream.py` runs the same renderer on a live stream:

```
Karo HLS ─► ffmpeg ingest ─► decode ─► analyse (homography · detector · tracker)
                                          │   0.5 s look-ahead
                                          ▼
          HLS/RTMP/SRT ◄─ NVENC ◄─ NUT pipe ◄─ render (person model · protect · composite)
```

Analysis and rendering run on separate threads. The detector runs on every 2nd frame (the homography carries boards in between) and runs less often automatically if the pod falls behind. Audio keeps its original timestamps, so lip-sync survives processing delay, dropped frames and reconnects. The input reconnects on its own after a stall or a dropped connection.

**Run it anywhere with a GPU:**

```bash
pip install -r requirements-live.txt       # plus ffmpeg on PATH
python scripts/live_stream.py              # uses the `live:` section of config.yaml
python scripts/live_stream.py --input match.mp4 --loop --output hls:///tmp/live/out   # rehearse on a file
```

**On RunPod** (`deploy/runpod/`):

1. `cp deploy/runpod/.env.example deploy/runpod/.env` and paste your RunPod API key into it (the file is git-ignored).
2. Either build and push the image (`docker build -f deploy/runpod/Dockerfile -t <registry>/betting-ad-blocker-live:1.0 . && docker push …`, then set `IMAGE`), or skip that and use `--bootstrap`, which starts the stock PyTorch image and clones this repo on the pod.
3. Make the board model reachable: bake `models/ad_seg_best.pt` into a private image, set `AD_WEIGHTS_URL`, or upload it to `/workspace/models/` on the pod's volume.
4. `python deploy/runpod/runpod_deploy.py create` (or `create --bootstrap`). The first start builds TensorRT engines for that GPU (a few minutes; cached on the volume).
5. Open `https://<POD_ID>-8080.proxy.runpod.net/<HTTP_TOKEN>/`. The page has two views: **Processed** (the stream viewers get) and **Side by side** (ORIGINAL | BETTING ADS HIDDEN, both halves cut from the same frame so they never drift apart), plus live pipeline stats and a download link for the recording of the match so far. `status <pod_id>` prints the same stats; `stop <pod_id>` stops GPU billing.

Every processed match is also saved on the pod's volume (`/workspace/live/recordings/`, about 2.7 GB per hour at 6 Mbps) from the same encode as the live stream, and a failed recording never interrupts the live output. With `OUTPUT_URL` set to an RTMP/SRT address (for example Tulix's ingest), the page still serves the side-by-side view and the recordings. `RECORD=0` / `COMPARE=0` turn either off.

Pick a GPU with a hardware encoder (NVENC): RTX 4090, L40S, RTX A6000, L4. A100 and H100 have none; there the encoder falls back to CPU x264.

## Teaching it new brands (operator page)

The model hides the brands it was trained on. A betting brand it has never seen is taught on the operator page, with no labelling work, in two speeds:

```
DURING THE MATCH (about a second, no training)
  every board ─► read its name (OCR, ~1/s) + compare its picture ─► Brand Library ─► hide / keep
                                                                  └─ unknown sponsor ─► page asks: Hide | Keep
  operator pauses, picks the frame, draws a box ──────────────────────────────────► added to the library

AFTER THE MATCH (automatic)
  original recordings + saved frames ─► labelled with the library ─► synthetic views of each new logo
  ─► fine-tune the model in use ─► test against it ─► new version, used only if it passes
```

**Set it up:** put `OPERATOR_TOKEN=<a long password>` in `deploy/runpod/.env` (it becomes the page's password, separate from the viewer link, because it controls what goes on air). The page is at `https://<POD_ID>-8080.proxy.runpod.net/<HTTP_TOKEN>/operator.html`.

**On the page:**
- **New sponsors seen**: names the scan read on the boards that are in no list yet, with a picture. *Hide*, *Keep* or *Not a sponsor*; the decision applies at once and is remembered.
- **Pause & mark a logo**: freezes the player, shows the original frames around that moment, and you drag a box around the logo. *Hide this logo* or *Keep it*. The name is read from the box; a logo with no readable name is matched by its picture instead.
- **Brand list**: add brands by name before kick-off (tournament sponsors are usually known in advance), switch hide/keep, remove.
- **Learning**: training frames collected, the model in use, *Train a new model now*, and every version with *Use this model* / *Back to the original model* (switched between two frames, no restart).

**What happens underneath:**
- `brand_library.py` keeps the decisions in `<LEARN_DIR>/brands.json`; they override the built-in lists (`v2_brands.py`), and every tracked board is re-decided the moment the library changes.
- `brand_scan.py` reads the whole picture about once a second off the frame loop. A board carrying a hidden brand is found from the text, grown to the edges of its artwork along the rail (never over another sponsor's name), carried to the present through the camera homographies, and tracked like any other; the next reading refreshes it, and a board the scan stops seeing ends within two readings (an LED rotating to another advert). The live look-ahead is 1.2 s, so a brand read ~1 s into a shot is still painted from the shot's first frame.
- The same frames become training data (`<LEARN_DIR>/samples/`): boards the model missed, boards it flagged but that carry a sponsor to keep, and what it already gets right, plus the original feed (`recordings/`, ~1.5 GB/h, oldest deleted beyond 15 GB).
- `learn_retrain.py` runs after the match (`AUTO_TRAIN=idle`: once the input has been down 20 minutes; `daily@04:00`; or the button). It re-reads the original recordings with today's library, so a brand added today is also labelled in earlier matches; paints each new logo onto real boards for synthetic views; fine-tunes the model in use on old and new labels together (the model's own confident detections are part of the data, so it does not forget Betano); and tests old against new on held-out sessions. A version is promoted only if it hides at least as many known betting boards, covers no more sponsors that must stay, and finds the new brands clearly better. Nothing is overwritten: every version is kept in `<LEARN_DIR>/models/`, and `start.sh` loads the promoted one at the next start.

```bash
python scripts/learn_retrain.py --learn-dir /workspace/learning            # by hand, on the pod
python scripts/learn_retrain.py --learn-dir /workspace/learning --status
python scripts/learn_retrain.py --learn-dir /workspace/learning --activate original   # roll back
python scripts/v2_render.py --input clip.mp4 --brands learning/brands.json --scan 1   # files, same rules
```

**Limits.** The first time an unregistered brand appears, it is on air until someone clicks (the review list surfaces it within seconds). Until a retrained version has learned a brand, it can show for a fraction of a second after a camera cut while the scan reads it. Logos with no readable name rely on picture matching, which is weaker than reading a name; a few marks from different shots help, and the retraining makes them reliable. Retraining shares the GPU with a live stream if one is running: best after the match.

## Design notes

A few decisions that mattered enough to be worth stating:

- **Camera geometry, fixed properly rather than removed.** Early versions carried the panel with camera-motion estimation and drifted during pans, so a detection-gated tracker replaced it. That tracker copied detector jitter (the panel wobbled with nothing moving), lagged behind pans, and dropped boards after five missed frames (blinking). The drift had specific causes, and each is now fixed:
  - a 4-DOF similarity transform, where the correct model is a full homography;
  - scoreboard and LED-board features included in the fit, where they are now masked out;
  - no forward-backward check on the tracked features, which is now applied;
  - a fixed 3% correction toward detections, replaced by a Kalman gain.

  An end that runs off-screen is held just past the frame edge instead of being carried with the scene. `--tracker legacy` keeps the previous behaviour for comparison.
- **Person protection is directional.** Subtracting *every* detected person from the paint area punches notches out of the board wherever a crowd member or photographer stands behind the rail. Only people whose feet fall below the board's bottom edge — i.e. people standing on the pitch, in front of the board — are protected.
- **Verification re-derives the result from the output video**, not from the renderer's internal bookkeeping. The removed/kept percentages above come from running OCR on the actual rendered frames and diffing against the source, specifically to avoid the renderer's own success claims being taken on faith.
