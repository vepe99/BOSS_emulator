#!/usr/bin/env bash
# DIAGNOSTIC: train emulators from scratch on the pathological region only, to
# see whether the cool-giant tail is intrinsically unlearnable or just starved
# of capacity/attention in the full model. Not for production.
#   A) Teff<=4000 & logg<=2 (47.7k rows), winner config
#   B) Teff<=4000 & logg<=1 (24.6k rows), winner config
#   C) as A but a small 3x1024 net (over-fitting vs capacity check)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=trained_model/region_test
mkdir -p "$OUT"

echo "[region-run] $(date)  waiting for a free GPU (autocvd) ..."
export CUDA_VISIBLE_DEVICES=$($PY -c "from autocvd import autocvd; print(autocvd(set_env=False, progress=False)[0])" 2>/dev/null)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "[region-run] $(date)  GPU=$CUDA_VISIBLE_DEVICES"

run_arm () {  # name cache hidden
    local name=$1 cache=$2; shift 2
    export BOSS_CACHE_DIR=/export/scratch/lbranca/BOSS_emulator/$cache
    echo "[region-run] $(date)  arm $name: cache=$cache hidden=$*"
    $PY -m boss_emulator.training --model mlp \
        --labels Teff logg m_H alpha_m carbon_m vmic \
        --hidden "$@" --activation gelu --dropout 0.0007451075028774108 \
        --lr 0.0001396945542025476 --weight-decay 2.597639598258449e-05 \
        --batch-size 1024 --epochs 3000 --ckpt-every 1000 \
        --outdir "$OUT/$name" || { echo "[region-run] arm $name train failed"; return; }
    local run; run=$(ls -d $OUT/$name/mlp_*/ | sort | tail -1); run=${run%/}
    $PY script/eval_mlp.py "$run" --n-worst 6 --outdir "plots/region_test/$name" \
        || echo "[region-run] arm $name eval failed"
}
run_arm A_coolgiants preprocessed_data_region_coolgiants 3238 3238 3238 3238 3238
run_arm B_corner     preprocessed_data_region_corner     3238 3238 3238 3238 3238
run_arm C_coolgiants_small preprocessed_data_region_coolgiants 1024 1024 1024
echo "[region-run] $(date)  ALL DONE"
