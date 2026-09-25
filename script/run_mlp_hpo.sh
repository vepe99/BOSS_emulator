#!/usr/bin/env bash
# Launch a multi-GPU Optuna hyper-parameter search for the MLP emulator.
#
# Each worker is a separate process that joins the SAME shared study via
# JournalStorage and trains its trials on its own GPU. GPUs are chosen by
# autocvd (lab policy) at import time, so we STAGGER the launches: a worker
# claims and allocates its GPU (JAX grabs ~75% of memory in claim_gpu) before
# the next worker's autocvd runs, otherwise two workers can race for the same
# still-idle device.
#
# Usage:
#     bash script/run_mlp_hpo.sh                   # 3 workers, 40 trials each
#     N_WORKERS=3 TRIALS_PER_WORKER=40 EPOCHS=1000 bash script/run_mlp_hpo.sh
#     OUTDIR=trained_model/combined_hpo STUDY_NAME=combined_hpo bash script/run_mlp_hpo.sh
#
# Point BOSS_CACHE_DIR at the preprocessed cache if it is not under
# preprocessed_data/. Logs + run artefacts land under trained_model/mlp_hpo/.
# The study is resumable: re-running adds more trials to the same journal.
set -euo pipefail

cd "$(dirname "$0")/.."

N_WORKERS="${N_WORKERS:-3}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-40}"
EPOCHS="${EPOCHS:-1000}"
CKPT_EVERY="${CKPT_EVERY:-250}"
STUDY_NAME="${STUDY_NAME:-mlp_hpo}"
STAGGER="${STAGGER:-45}"
PYTHON="${PYTHON:-.venv/bin/python}"
# Abundance inputs. Default = abundance-free (6 stellar labels); set
# ABUNDANCES="Fe C ..." for the legacy 27-input variant.
ABUNDANCES="${ABUNDANCES:-}"

OUTDIR="${OUTDIR:-trained_model/mlp_hpo}"
mkdir -p "$OUTDIR"

echo "[hpo] launching $N_WORKERS workers x $TRIALS_PER_WORKER trials "\
"(epochs=$EPOCHS) study=$STUDY_NAME"
echo "[hpo] artefacts + logs -> $OUTDIR/"

pids=()
for i in $(seq 0 $((N_WORKERS - 1))); do
    log="$OUTDIR/worker_${i}.log"
    echo "[hpo] starting worker $i -> $log"
    nohup "$PYTHON" script/tune_mlp.py \
        --study-name "$STUDY_NAME" \
        --n-trials "$TRIALS_PER_WORKER" \
        --epochs "$EPOCHS" \
        --ckpt-every "$CKPT_EVERY" \
        --abundances $ABUNDANCES \
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
