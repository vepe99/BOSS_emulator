#!/usr/bin/env bash
# v2 pipeline: Phase 1 (mismatch diagnostics) -> Phase 2+4 (train) -> Phase 5 (evaluate).
# GPUs are claimed automatically by autocvd (imported via boss_emulator.training).
#
#   bash sc_sbi/run_v2.sh diagnose     # Phase 1 only (~minutes on GPU) -> look at the plots first
#   bash sc_sbi/run_v2.sh train        # v2 base + control + SC   (~9-10 h)
#   bash sc_sbi/run_v2.sh sanity       # SC on noisy Korg, re-using the v2 base (needs `train` phase A)
#   bash sc_sbi/run_v2.sh resid        # residual-augmentation variant (full retrain)
#   bash sc_sbi/run_v2.sh eval         # evaluate every finished run
#   bash sc_sbi/run_v2.sh all          # diagnose + train + sanity + eval
#
# Old code: sc_sbi/legacy_v1/{train,evaluate}.py (frozen), or the new code with --preset v1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MM_TAG=${MM_TAG:-v1}
MM=trained_model/sc_sbi/mismatch/$MM_TAG/mismatch.npz
RUN=${RUN:-v2_giants}
UNL_BATCH=${UNL_BATCH:-64}   # unlabeled spectra per SC step (v1: 32)
NET=${NET:-'{}'}           # --net JSON overrides, e.g. '{"summary":"tst"}'
LOGS=trained_model/sc_sbi/logs; mkdir -p "$LOGS"

diagnose() {
  $PY sc_sbi/diagnose_mismatch.py --n-stars 4000 --steps 500 --n-random-inits 2 \
      --sm-iters 2 --tag "$MM_TAG" 2>&1 | tee "$LOGS/diagnose_$MM_TAG.log"
}
train() {
  $PY sc_sbi/train.py --preset v2 --mismatch "$MM" --unlabeled-batch "$UNL_BATCH" --net "$NET" --tag "$RUN" 2>&1 | tee "$LOGS/$RUN.log"
}
sanity() {
  $PY sc_sbi/train.py --preset v2 --mismatch "$MM" --unlabeled-batch "$UNL_BATCH" --net "$NET" --init-run "trained_model/sc_sbi/$RUN" \
      --sc-unlabeled korg --no-control --tag "${RUN}_sanity_korg" 2>&1 | tee "$LOGS/${RUN}_sanity_korg.log"
}
resid() {
  $PY sc_sbi/train.py --preset v2 --mismatch "$MM" --unlabeled-batch "$UNL_BATCH" --net "$NET" --noise-mode residual \
      --tag "${RUN}_resid" 2>&1 | tee "$LOGS/${RUN}_resid.log"
}
evaluate() {
  for r in "$RUN" "${RUN}_sanity_korg" "${RUN}_resid" full; do
    d=trained_model/sc_sbi/$r
    [ -f "$d/config.json" ] || continue
    ls "$d"/*.weights.h5 >/dev/null 2>&1 || continue
    $PY sc_sbi/evaluate.py "$d" --n-korg 5000 --n-boss 5000 --n-repeat-stars 500 \
        --n-samples 200 2>&1 | tee "$LOGS/eval_$r.log"
  done
}

case "${1:-all}" in
  diagnose) diagnose ;;
  train)    train ;;
  sanity)   sanity ;;
  resid)    resid ;;
  eval)     evaluate ;;
  all)      diagnose; train; sanity; evaluate ;;
  *) echo "usage: $0 {diagnose|train|sanity|resid|eval|all}"; exit 1 ;;
esac
