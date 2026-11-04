#!/usr/bin/env bash
# v3 pipeline: v2 SC + L. Branca's forward-model fixes (Gaussian LSF with lsf_px inferred,
# profiled Chebyshev-6 continuum in the SC likelihood + random tilts in the simulations,
# ivar weights, no s_m), MLP summary, SC on CLAM giants with <= 30% masked pixels.
#
#   bash sc_sbi/run_v3.sh            # train -> sanity -> eval, one job (no idle GPU gap)
#   bash sc_sbi/run_v3.sh train|sanity|eval
#
# GPU: autocvd picks a free one, unless CUDA_VISIBLE_DEVICES is set by the caller.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=${RUN:-v3_giants}
UNL_BATCH=${UNL_BATCH:-64}
EXTRA=${EXTRA:-}            # extra train.py flags, e.g. EXTRA="--log-params vmic Teff"
LOGS=trained_model/sc_sbi/logs; mkdir -p "$LOGS"

train() {
  $PY sc_sbi/train.py --preset v3 --unlabeled-batch "$UNL_BATCH" $EXTRA --tag "$RUN" 2>&1 | tee "$LOGS/$RUN.log"
}
sanity() {
  $PY sc_sbi/train.py --preset v3 --unlabeled-batch "$UNL_BATCH" $EXTRA --init-run "trained_model/sc_sbi/$RUN" \
      --sc-unlabeled korg --no-control --tag "${RUN}_sanity_korg" 2>&1 | tee "$LOGS/${RUN}_sanity_korg.log"
}
evaluate() {
  for r in "$RUN" "${RUN}_sanity_korg"; do
    d=trained_model/sc_sbi/$r
    [ -f "$d/config.json" ] || continue
    $PY sc_sbi/evaluate.py "$d" --n-korg 5000 --n-boss 5000 --n-repeat-stars 500 \
        --n-samples 200 2>&1 | tee "$LOGS/eval_$r.log"
  done
}

case "${1:-all}" in
  train)  train ;;
  sanity) sanity ;;
  eval)   evaluate ;;
  all)    train; sanity; evaluate; echo "[$(date +%T)] v3 ALL DONE" ;;
  *) echo "usage: $0 {train|sanity|eval|all}"; exit 1 ;;
esac
