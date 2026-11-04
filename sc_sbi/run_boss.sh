#!/usr/bin/env bash
# NPE trained purely on BOSS-CLAM (sc_sbi/train_boss.py); autocvd picks a free GPU.
#   bash sc_sbi/run_boss.sh [extra train_boss.py flags]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-boss_only}
LOGS=trained_model/sc_sbi/logs; mkdir -p "$LOGS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python sc_sbi/train_boss.py --tag "$RUN" "$@" 2>&1 | tee "$LOGS/$RUN.log"
