#!/usr/bin/env bash
# Wait for the running v3 pipeline (PID $1); if it printed "v3 ALL DONE", start v3_logTeff
# (log-Teff parametrisation; physical Teff prior unchanged) -- but only once the go-file
# exists (touch trained_model/sc_sbi/logs/GO_v3_logTeff after checking the v3 SC diagnostics,
# which this script runs first: sc_sbi/sc_diagnostics.py -> logs/sc_diagnostics_v3_giants.log).
# Until then GPU $2 is held with hold_gpu.py so it is not lost; the card is never shared:
# if someone else is on it, autocvd waits for a completely free GPU instead.
#   usage: nohup bash sc_sbi/after_v3.sh <run_v3_pid> <gpu> > log 2>&1 &
cd "$(dirname "$0")/.."
LOGS=trained_model/sc_sbi/logs
GO=$LOGS/GO_v3_logTeff
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$2")
gpu_busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; }

while kill -0 "$1" 2>/dev/null; do sleep 2; done
echo "[$(date +%T)] v3 pipeline exited"
grep -q "v3 ALL DONE" $LOGS/run_v3.nohup || { echo "[$(date +%T)] v3 did NOT finish -> stop"; exit 1; }

# SC diagnostics on the finished run (GPU to itself): is SC learning, is lambda sensible?
if gpu_busy; then unset CUDA_VISIBLE_DEVICES; else export CUDA_VISIBLE_DEVICES=$2; fi
.venv/bin/python sc_sbi/sc_diagnostics.py trained_model/sc_sbi/v3_giants \
    --models amortized amortized_cont amortized_sc --ref amortized_cont > $LOGS/sc_diagnostics_v3_giants.log 2>&1
echo "[$(date +%T)] SC diagnostics done (exit $?) -> $LOGS/sc_diagnostics_v3_giants.log"
unset CUDA_VISIBLE_DEVICES

holder=
if [ ! -e "$GO" ] && ! gpu_busy; then
  CUDA_VISIBLE_DEVICES=$2 .venv/bin/python sc_sbi/hold_gpu.py & holder=$!
  echo "[$(date +%T)] holding GPU $2 (pid $holder) until $GO exists"
fi
until [ -e "$GO" ]; do sleep 5; done
if [ -n "$holder" ]; then
  kill "$holder"; while kill -0 "$holder" 2>/dev/null; do sleep 1; done
  while gpu_busy; do sleep 1; done          # our memory released
fi
if gpu_busy; then
  echo "[$(date +%T)] GPU $2 is taken by someone else -> autocvd will wait for a free GPU"
  unset CUDA_VISIBLE_DEVICES
else
  echo "[$(date +%T)] starting v3_logTeff on GPU $2"
  export CUDA_VISIBLE_DEVICES=$2
fi
RUN=v3_logTeff EXTRA="--log-params vmic Teff" bash sc_sbi/run_v3.sh
