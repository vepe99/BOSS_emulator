#!/usr/bin/env bash
# End-to-end pipeline on the BOSS-Grid-Combined grid (random-sampled, 650k rows):
#   1. forward emulator  -- trial-77 MLP config, 3000 epochs from scratch
#                           (the recipe that won on the factorial grid)
#   2. forward evaluation (eval_mlp)
#   3. amortized inverse nets (clean + SNR 10 noise-augmented)
#   4. gradient-descent inversion: 4096-target diagnostic + full test set
#
# Resumable: each stage is skipped if its output already exists.
# Usage:  nohup script/run_combined.sh > trained_model/combined/pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export BOSS_RAW_PATH=/export/scratch/lbranca/SDSS5-BOSS/BOSS-Grid-Combined/korg_grid.h5
export BOSS_CACHE_DIR=${BOSS_CACHE_DIR:-/export/scratch/lbranca/BOSS_emulator/preprocessed_data_combined}
PY=.venv/bin/python
OUT=${OUT:-trained_model/combined}
PLOTS=${PLOTS:-plots/combined}
# Forward-model hyper-parameters (defaults = the factorial-grid trial-77 recipe).
# Override for the HPO winner, e.g.
#   HIDDEN="3238 3238 3238 3238 3238" ACT=gelu DROPOUT=0.00075 LR=1.4e-4 WD=2.6e-5 BATCH=4096
HIDDEN=${HIDDEN:-"1281 1281 1281 1281 1281 1281 1281 1281"}
ACT=${ACT:-silu}
DROPOUT=${DROPOUT:-0.0008585251227289674}
LR=${LR:-0.00023195651422037596}
WD=${WD:-9.616904240621662e-05}
BATCH=${BATCH:-4096}
EPOCHS=${EPOCHS:-3000}
mkdir -p "$OUT" "$PLOTS"

# GPU: autocvd, default mode = block until a GPU is COMPLETELY free (lab rule:
# never run on a GPU another user's job is using). Picked once here and pinned
# via CUDA_VISIBLE_DEVICES so every stage below uses the same device.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[pipeline] $(date)  waiting for a free GPU (autocvd) ..."
    export CUDA_VISIBLE_DEVICES=$($PY -c "from autocvd import autocvd; print(autocvd(set_env=False, progress=False)[0])" 2>/dev/null)
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi
echo "[pipeline] $(date)  GPU=$CUDA_VISIBLE_DEVICES"

# 0. wait for the cache build (started separately) to finish
until [ -f "$BOSS_CACHE_DIR/korg_grid_train.h5" ] && ! pgrep -f "data.build_cache" >/dev/null; do
    echo "[pipeline] waiting for cache build ..."; sleep 60
done
ls -la "$BOSS_CACHE_DIR"

# 1. forward emulator
RUN=$(ls -d $OUT/mlp_*/ 2>/dev/null | sort | tail -1)
if [ -z "$RUN" ] || [ ! -f "$RUN/final.pkl" ]; then
    echo "[pipeline] $(date)  stage 1: forward training"
    echo "[pipeline] hidden=$HIDDEN act=$ACT dropout=$DROPOUT lr=$LR wd=$WD batch=$BATCH epochs=$EPOCHS cache=$BOSS_CACHE_DIR"
    $PY -m boss_emulator.training --model mlp \
        --labels Teff logg m_H alpha_m carbon_m vmic \
        --hidden $HIDDEN --activation "$ACT" \
        --dropout "$DROPOUT" \
        --lr "$LR" --weight-decay "$WD" \
        --batch-size "$BATCH" --epochs "$EPOCHS" --ckpt-every 500 \
        --outdir "$OUT" || exit 1
    RUN=$(ls -d $OUT/mlp_*/ | sort | tail -1)
fi
RUN=${RUN%/}
echo "[pipeline] forward run: $RUN"

# 2. forward evaluation
if [ ! -f "$PLOTS/eval/metrics.json" ] && [ -z "$(ls $PLOTS/eval 2>/dev/null)" ]; then
    echo "[pipeline] $(date)  stage 2: eval_mlp"
    $PY script/eval_mlp.py "$RUN" --n-worst 10 --outdir "$PLOTS/eval" || echo "[pipeline] eval failed"
fi

# 3. inverse nets
INV=$OUT/inverse_net
mkdir -p "$INV"
if [ ! -f "$INV/inv.pkl" ]; then
    echo "[pipeline] $(date)  stage 3a: inverse net (clean)"
    $PY script/inv_model/train_inverse_net.py --epochs 300 --out "$INV/inv.pkl" || exit 1
fi
if [ ! -f "$INV/inv_snr10.pkl" ]; then
    echo "[pipeline] $(date)  stage 3b: inverse net (SNR 10 augmented)"
    $PY script/inv_model/train_inverse_net.py --epochs 300 --snr 10 --out "$INV/inv_snr10.pkl" || exit 1
fi

# 4. inversion
if [ -z "$(ls $PLOTS/inverse/korg_snr100_* 2>/dev/null)" ]; then
    echo "[pipeline] $(date)  stage 4a: gradient inversion, 4096 Korg targets, SNR 100"
    $PY script/inv_model/run_gradient.py --run "$RUN" --obs korg --snr 100 \
        --n-targets 4096 --init amortized --amortized-net "$INV/inv.pkl" \
        --n-restarts 8 --tag korg_snr100 --outdir "$PLOTS/inverse" || echo "[pipeline] 4a failed"
fi
if [ -z "$(ls $PLOTS/inverse/emu_noiseless_* 2>/dev/null)" ]; then
    echo "[pipeline] $(date)  stage 4b: self-consistency control (emulator obs, noiseless)"
    $PY script/inv_model/run_gradient.py --run "$RUN" --obs emulator --noiseless \
        --n-targets 4096 --init amortized --amortized-net "$INV/inv.pkl" \
        --n-restarts 8 --tag emu_noiseless --outdir "$PLOTS/inverse" || echo "[pipeline] 4b failed"
fi
if [ -z "$(ls $PLOTS/inverse_full/* 2>/dev/null)" ]; then
    echo "[pipeline] $(date)  stage 4c: full test-set inversion"
    $PY script/inv_model/run_gradient_full.py --run "$RUN" --obs korg --snr 100 \
        --init amortized --amortized-net "$INV/inv.pkl" --chunk-size 8192 \
        --outdir "$PLOTS/inverse_full" || echo "[pipeline] 4c failed"
fi
echo "[pipeline] $(date)  ALL DONE"
