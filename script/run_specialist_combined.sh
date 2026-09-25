#!/usr/bin/env bash
# Cool-star specialist + gate on top of the frozen BOSS-Grid-Combined winner.
# Waits for the combined_best pipeline to finish (so we never share its GPU),
# then runs two arms back to back on one autocvd-selected free GPU:
#   A) plain specialist (Teff<=5000 rows, warm-started)
#   B) same + 3x loss weight on logg<=2 rows (the cool-giant corner)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export BOSS_RAW_PATH=/export/scratch/lbranca/SDSS5-BOSS/BOSS-Grid-Combined/korg_grid.h5
export BOSS_CACHE_DIR=/export/scratch/lbranca/BOSS_emulator/preprocessed_data_combined_v2
PY=.venv/bin/python
GEN=trained_model/combined_best/mlp_1790330838
OUT=trained_model/combined_specialist
mkdir -p "$OUT"

until grep -q "ALL DONE" trained_model/combined_best/pipeline.log; do
    echo "[spec-run] $(date +%H:%M) waiting for the combined_best pipeline ..."; sleep 60
done

echo "[spec-run] $(date)  waiting for a free GPU (autocvd) ..."
export CUDA_VISIBLE_DEVICES=$($PY -c "from autocvd import autocvd; print(autocvd(set_env=False, progress=False)[0])" 2>/dev/null)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "[spec-run] $(date)  GPU=$CUDA_VISIBLE_DEVICES"

COMMON="--generalist $GEN --region-teff-max 5000 --gate-lo 3800 --gate-hi 4800 \
        --epochs 600 --lr 5e-5 --weight-decay 2.6e-5 --batch-size 4096"

echo "[spec-run] $(date)  arm A: plain specialist"
$PY script/exp_specialist_gate.py $COMMON --outdir "$OUT/plain" || echo "[spec-run] arm A failed"
echo "[spec-run] $(date)  arm B: logg<=2 rows weighted x3"
$PY script/exp_specialist_gate.py $COMMON --logg-max 2 --logg-weight 3 --outdir "$OUT/loggw" || echo "[spec-run] arm B failed"
echo "[spec-run] $(date)  ALL DONE"
