# learning/ — what the ad blocker learns from

This folder holds everything the pipeline collects to learn new betting brands.
On the RunPod pod it lives on the persistent volume at `/workspace/learning`
(`LEARN_DIR`); on a laptop run it is this folder. Only this README is in git:
the data is large and belongs to the deployment, not the code.

```
learning/
  brands.json          the Brand Library: brands to hide / keep, editable live
                       on the operator page (scripts/brand_library.py)
  review/              sponsors the live scan read but nobody has decided about yet
  examples/<brand>/    logo crops per brand: picture matching + training artwork
  samples/<day>/       training frames: the ORIGINAL picture (.jpg) + board labels (.json)
  recordings/          the original feed, 15-minute MPEG-TS files, for re-harvesting
  models/              model versions made by scripts/learn_retrain.py
    v001/ad_seg_v001.pt, v001/report.json
    current.json       the version in use (absent = the original ad_seg_best.pt)
  datasets/, runs/     the last three training runs' datasets and logs
  jobs/                retraining job state and logs
```

How it fills up, with no manual labelling:

1. **During a match** the operator marks logos on the operator page, or answers
   "New sponsor: … Hide / Keep?". Each decision goes into `brands.json` and takes
   effect in the live stream within about a second; the frames it was made on
   become training samples.
2. **The live scan** saves frames where it found a hidden brand the model missed,
   frames where the model flagged a sponsor that must stay, and a steady trickle
   of what the model already gets right.
3. **After the match** `scripts/learn_retrain.py` re-reads the original
   recordings with the current Brand Library (so a brand added today is also
   labelled in last week's matches), adds synthetic views of each new logo,
   fine-tunes the model in use, and tests it against that model. Only a version
   that passes becomes `current.json`; every version stays, so rolling back is
   one click (or `learn_retrain.py --activate original`).

The original model the project was trained on is never modified.
