#!/usr/bin/env bash
# Wait for a running `run_v2.sh train` (PID $1) and, if it finished cleanly,
# run sanity then eval.   usage: nohup bash sc_sbi/after_train.sh <train_pid> > log &
cd "$(dirname "$0")/.."
LOG=trained_model/sc_sbi/logs/${RUN:-v2_giants}.log
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] train exited"
if tr '\r' '\n' < "$LOG" | grep -qE "Traceback|RESOURCE_EXHAUSTED" || ! grep -q "done ->" "$LOG"; then
  echo "[$(date +%T)] train did NOT finish cleanly -> not running sanity/eval"; exit 1
fi
tr '\r' '\n' < "$LOG" | grep -E "non-finite|timing "
echo "[$(date +%T)] starting sanity"; bash sc_sbi/run_v2.sh sanity || { echo "sanity FAILED"; exit 1; }
echo "[$(date +%T)] starting eval";   bash sc_sbi/run_v2.sh eval   || { echo "eval FAILED"; exit 1; }
echo "[$(date +%T)] ALL DONE"
