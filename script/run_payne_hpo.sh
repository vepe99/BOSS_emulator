#!/usr/bin/env bash
# Launch a 3-GPU Optuna hyper-parameter search for the Payne MLP emulator.
#
# Each worker is a separate process that joins the SAME shared study via
# JournalStorage and trains its trials on its own GPU. GPUs are chosen by
# autocvd (lab policy) at import time, so we STAGGER the launches: a worker
# claims and allocates its GPU (JAX grabs ~75% of memory in claim_gpu) before
# the next worker's autocvd runs, otherwise two workers can race for the same
# still-idle device.
#
# Usage:
#     bash script/run_payne_hpo.sh                 # 3 workers, 40 trials each
#     N_WORKERS=3 TRIALS_PER_WORKER=40 EPOCHS=1000 bash script/run_payne_hpo.sh
#
# Logs + all run artefacts land under trained_model/payne_hpo/. The study is
# resumable: re-running this script adds more trials to the same journal.
# To stop early, kill the workers (Ctrl-C / kill); completed trials persist.
set -euo pipefail

cd "$(dirname "$0")/.."

# uv cache must live on the data volume (home is over quota).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/export/data/vgiusepp/.uv_cache}"

N_WORKERS="${N_WORKERS:-3}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-40}"
EPOCHS="${EPOCHS:-1000}"
CKPT_EVERY="${CKPT_EVERY:-250}"
STUDY_NAME="${STUDY_NAME:-payne_hpo}"
STAGGER="${STAGGER:-45}"

OUTDIR="trained_model/payne_hpo"
mkdir -p "$OUTDIR"

echo "[hpo] launching $N_WORKERS workers x $TRIALS_PER_WORKER trials "\
"(epochs=$EPOCHS) study=$STUDY_NAME"
echo "[hpo] artefacts + logs -> $OUTDIR/"

pids=()
for i in $(seq 0 $((N_WORKERS - 1))); do
    log="$OUTDIR/worker_${i}.log"
    echo "[hpo] starting worker $i -> $log"
    nohup uv run python script/tune_payne.py \
        --study-name "$STUDY_NAME" \
        --n-trials "$TRIALS_PER_WORKER" \
        --epochs "$EPOCHS" \
        --ckpt-every "$CKPT_EVERY" \
        --outdir "$OUTDIR" \
        >"$log" 2>&1 &
    pids+=($!)
    # Stagger so this worker claims its GPU before the next one's autocvd runs
    # (skip the wait after the last worker).
    if [ "$i" -lt "$((N_WORKERS - 1))" ]; then
        sleep "$STAGGER"
    fi
done

echo "[hpo] worker PIDs: ${pids[*]}"
echo "[hpo] tail logs with:  tail -f $OUTDIR/worker_*.log"
echo "[hpo] waiting for workers to finish (Ctrl-C to detach; workers keep running under nohup)"
wait
echo "[hpo] all workers finished."
