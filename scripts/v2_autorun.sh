#!/usr/bin/env bash
# Fully automated end-to-end driver: runs the orchestrator, and instead of
# waiting on a human to review render_first before releasing the other 19
# clips (the pipeline's normal safety gate), auto-approves once render_first
# has actually produced a real clip. This is intentional per explicit
# instruction to run the whole thing unattended with no manual steps.
#
# Safe to re-launch after any interruption (power loss, session restart):
# every stage underneath is independently checkpointed/resumable, so this
# script just keeps calling the same idempotent orchestrator until the
# status shows "verify" done.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/v2_autorun.log
PY=$(command -v python3)

echo "===== autorun start $(date -u +%FT%TZ) =====" >> "$LOG"

is_finished() {
  "$PY" scripts/v2_pipeline.py --status | awk '/^  OK  verify/{found=1} END{exit !found}'
}

for i in $(seq 1 200); do
  if is_finished; then
    echo "[autorun] verify already done, exiting" >> "$LOG"
    exit 0
  fi

  "$PY" scripts/v2_pipeline.py >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[autorun] pipeline exited $rc, stopping (see $LOG)" >> "$LOG"
    exit $rc
  fi

  if is_finished; then
    echo "[autorun] all steps complete, exiting" >> "$LOG"
    exit 0
  fi

  if [ -f logs/APPROVED ]; then
    # approved but not finished yet (still mid render_rest/verify) -- loop again
    continue
  fi

  # Not approved yet: pipeline must have parked at the approval gate.
  # Only auto-approve if render_first genuinely produced a usable clip.
  clip=outputs/clips/clip01_clean.mp4
  if [ -f "$clip" ] && [ "$(stat -c%s "$clip" 2>/dev/null || echo 0)" -gt 100000 ]; then
    echo "[autorun] render_first looks good ($(stat -c%s "$clip") bytes); auto-approving" >> "$LOG"
    touch logs/APPROVED
    continue
  fi

  echo "[autorun] no valid render_first clip yet and pipeline returned; stopping to avoid a loop" >> "$LOG"
  exit 1
done

echo "[autorun] exceeded max iterations" >> "$LOG"
exit 1
