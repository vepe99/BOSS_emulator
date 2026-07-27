"""Regenerate the full test-set metric set for one or more trained runs.

For each run directory (containing ``best.pkl`` + ``split_idx.npz``), streams
the held-out test set once and writes ``reconstruction_stats_full.json`` with
RMSE percentiles plus the scale-relative metrics (R^2, standardized RMSE, MAE,
SNR-equivalent). See ``training.full_reconstruction_metrics``.

    UV_CACHE_DIR=... uv run python script/eval_full_metrics.py RUNDIR [RUNDIR...]

autocvd claims a free GPU at import (lab policy). The dataset is loaded once
and reused across all supplied runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint


def main(run_dirs):
    training.claim_gpu()
    base = training.build_parser().parse_args([])
    base.model = "payne"
    base.labels = list(data_mod.CORE_LABELS)
    base.abundances = list(data_mod.ABUNDANCE_ELEMENTS)
    base.val_frac = 0.1; base.test_frac = 0.1; base.seed = 0
    ds, Xn, Yn = training.prepare_data(base)

    for rd in run_dirs:
        rd = Path(rd)
        sp = rd / "split_idx.npz"
        test_idx = (np.load(sp)["test_idx"] if sp.exists() else ds.test_idx)
        model, _ = load_checkpoint(rd / "best.pkl")
        metrics, _ = training.full_reconstruction_metrics(
            model, Xn, ds.Y, Yn, test_idx, ds.y_std, ds.y_mean, 8192)
        (rd / "reconstruction_stats_full.json").write_text(
            json.dumps({"test_set": metrics}, indent=2))
        print(f"[full] {rd.name}: R2={metrics['r2']:.6f} "
              f"std_rmse={metrics['standardized_rmse']:.5f} "
              f"med={metrics['per_spectrum_rmse_median']:.5f} "
              f"mae={metrics['mae_mean']:.5f} "
              f"snr~{metrics['snr_equivalent_median']:.0f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: eval_full_metrics.py RUNDIR [RUNDIR ...]")
    main(sys.argv[1:])
