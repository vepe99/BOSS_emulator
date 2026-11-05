#!/usr/bin/env bash
# v4 pipeline: v3 SC-SBI with the *real* per-star BOSS LSF.
#   forward model: Korg -> Vedant's LSF (LSF_vedant.py, JAX port) with the WDISP of a real
#   DR20 spec-lite star -> random Chebyshev tilt -> that same star's noise (ivar/mask);
#   posterior conditioned on log S/N + WDISP PCA coefficients;
#   phase A (NPE) -> control (no SC) -> phase B (SC on SC_N spec-lite spectra, own WDISP);
#   evaluation (recovery, SBC, calibration, coverage, chi2): Korg test + BOSS test twice --
#   eval/ = spec-lite visits with their own WDISP, eval_clam/ = boss_clam coadds (WDISP of
#   the star's visit) -- both vs BOSS-CLAM labels.
#
#   bash sc_sbi/run_v4.sh            # bank (if missing) -> check -> train -> eval
#   bash sc_sbi/run_v4.sh bank|check|train|eval
#   SMOKE=1 bash sc_sbi/run_v4.sh    # CPU, 3000-file bank, tiny training + eval
#
# GPU: autocvd picks a free one (train/eval), unless CUDA_VISIBLE_DEVICES is set.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=${UV_CACHE_DIR:-/export/data/vgiusepp/.uv_cache}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}   # 11 GB cards; autocvd picks a free GPU
PY="uv run -q --extra sbi --with astropy python"
SMOKE=${SMOKE:-0}
UNL_BATCH=${UNL_BATCH:-64}
SC_N=${SC_N:-5000}
EXTRA=${EXTRA:-}                     # extra train.py flags
LOGS=trained_model/sc_sbi/logs; mkdir -p "$LOGS"
if [ "$SMOKE" = 1 ]; then
  export JAX_PLATFORMS=cpu
  RUN=${RUN:-v4_smoke}; BANK=${BANK:-sc_sbi/cache/speclite_bank_smoke.npz}
  BANK_ARGS="--max-files 3000"; TRAIN_ARGS="--smoke"
  EVAL_ARGS="--n-korg 200 --n-boss 200 --n-samples 50 --n-repeat-stars 0"; CLAM_ARGS=""
else
  RUN=${RUN:-v4_vedant}; BANK=${BANK:-sc_sbi/cache/speclite_bank_v6_2_1.npz}
  BANK_ARGS=""; TRAIN_ARGS=""
  EVAL_ARGS="--n-korg 5000 --n-boss 5000 --n-repeat-stars 0 --n-samples 200"
  CLAM_ARGS="--n-repeat-stars 500"   # boss_clam has repeat visits (later flag wins)
fi

bank() {
  [ -f "$BANK" ] && { echo "bank exists: $BANK"; return; }
  JAX_PLATFORMS=cpu $PY sc_sbi/build_wdisp_bank.py --out "$BANK" $BANK_ARGS 2>&1 | tee "$LOGS/${RUN}_bank.log"
}
check() {
  JAX_PLATFORMS=cpu $PY sc_sbi/check_forward_v4.py --bank "$BANK" 2>&1 | tee "$LOGS/${RUN}_check.log"
}
train() {
  $PY sc_sbi/train.py --preset v4 --wdisp-bank "$BANK" --sc-n "$SC_N" --unlabeled-batch "$UNL_BATCH" \
      $TRAIN_ARGS $EXTRA --tag "$RUN" 2>&1 | tee "$LOGS/$RUN.log"
}
evaluate() {   # Korg test + (a) spec-lite bank test -> eval/, (b) boss_clam coadd test -> eval_clam/
  $PY sc_sbi/evaluate.py "trained_model/sc_sbi/$RUN" $EVAL_ARGS 2>&1 | tee "$LOGS/eval_$RUN.log"
  $PY sc_sbi/evaluate.py "trained_model/sc_sbi/$RUN" --boss-source clam $EVAL_ARGS $CLAM_ARGS 2>&1 \
      | tee "$LOGS/eval_clam_$RUN.log"
}

case "${1:-all}" in
  bank)  bank ;;
  check) check ;;
  train) train ;;
  eval)  evaluate ;;
  all)   bank; check; train; evaluate; echo "[$(date +%T)] v4 ALL DONE ($RUN)" ;;
  *) echo "usage: $0 {bank|check|train|eval|all}"; exit 1 ;;
esac
