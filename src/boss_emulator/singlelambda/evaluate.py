"""Evaluate a trained single-lambda emulator: reconstruct + score spectra.

The coordinate-query model has a two-argument forward ``model(x, coords)``, so
it cannot use :func:`boss_emulator.predict.predict` (which calls ``model(x)``).
This module supplies :func:`reconstruct`, the drop-in equivalent that queries
every pixel position in chunks to rebuild full de-normalised spectra, and then
reuses :func:`boss_emulator.predict.reconstruction_metrics` so the reported
numbers match the other emulators.

Usage (from the project root)::

    uv run python -m boss_emulator.singlelambda.evaluate \
        --run trained_model/singlelambda_1700000000 --test
"""

from __future__ import annotations

# --- GPU selection MUST happen before importing jax (see boss_emulator.training).
import os

if not (
    os.environ.get("BOSS_NO_AUTOCVD")
    or os.environ.get("CUDA_VISIBLE_DEVICES")
    or os.environ.get("JAX_PLATFORMS", "").lower() == "cpu"
):
    from autocvd import autocvd

    autocvd(num_gpus=1)

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from boss_emulator import data as data_mod
from boss_emulator.emulator import load_checkpoint
from boss_emulator.predict import reconstruction_metrics
from boss_emulator.singlelambda.training import make_coord_grid


def reconstruct(model, payload, X: np.ndarray, eval_batch_size: int = 8192,
                pixel_chunk: int = 1024) -> np.ndarray:
    """Reconstruct full de-normalised spectra for raw label array ``X``.

    The single-lambda equivalent of :func:`boss_emulator.predict.predict`:
    positions are queried in ``pixel_chunk`` slices for each row-batch, so
    activation memory is bounded over both spectra and pixels. Returns a
    ``(n, n_pixels)`` float32 array of predicted flux.
    """
    x_mean, x_std = payload["x_mean"], payload["x_std"]
    y_mean, y_std = payload["y_mean"], payload["y_std"]
    n_pix = int(payload["n_pixels"])
    coord_grid = make_coord_grid(n_pix)
    out = np.empty((X.shape[0], n_pix), dtype="float32")
    for start in range(0, X.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, X.shape[0])
        xn = (X[start:stop] - x_mean) / x_std
        xb = jnp.asarray(xn.astype("float32"))
        for c0 in range(0, n_pix, pixel_chunk):
            c1 = min(c0 + pixel_chunk, n_pix)
            cb = jnp.asarray(coord_grid[c0:c1])
            yn = np.asarray(model(xb, cb))               # (rows, chunk)
            out[start:stop, c0:c1] = yn * y_std[c0:c1] + y_mean[c0:c1]
    return out


def whole_dataset_error_stats(model, payload, ds, eval_batch_size=8192,
                              pixel_chunk=1024):
    """Stream reconstructions over every row and accumulate error statistics.

    Mirrors the helper in ``script/train_and_evaluate_deeponet.py`` but calls
    :func:`reconstruct` instead of ``predict``. Returns a JSON-serialisable
    summary plus (per-pixel RMSE, per-spectrum RMSE) arrays for plotting.
    """
    n, P = ds.X.shape[0], ds.n_pixels
    ss_res = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    ss_tot = 0.0
    y_bar = float(ds.Y.mean())
    per_pixel_sq = np.zeros(P, dtype=np.float64)
    per_spectrum_rmse = np.empty(n, dtype=np.float32)

    for start in range(0, n, eval_batch_size):
        stop = min(start + eval_batch_size, n)
        pred = reconstruct(model, payload, ds.X[start:stop],
                           eval_batch_size=eval_batch_size,
                           pixel_chunk=pixel_chunk)
        resid = pred - ds.Y[start:stop]
        ss_res += float((resid ** 2).sum())
        abs_sum += float(np.abs(resid).sum())
        max_abs = max(max_abs, float(np.abs(resid).max()))
        ss_tot += float(((ds.Y[start:stop] - y_bar) ** 2).sum())
        per_pixel_sq += (resid ** 2).sum(axis=0)
        per_spectrum_rmse[start:stop] = np.sqrt((resid ** 2).mean(axis=1))

    n_elem = n * P
    per_pixel_rmse = np.sqrt(per_pixel_sq / n)
    summary = {
        "n_spectra": int(n),
        "n_pixels": int(P),
        "rmse": float(np.sqrt(ss_res / n_elem)),
        "mae": float(abs_sum / n_elem),
        "max_abs_error": float(max_abs),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "per_spectrum_rmse_mean": float(per_spectrum_rmse.mean()),
        "per_spectrum_rmse_median": float(np.median(per_spectrum_rmse)),
        "per_spectrum_rmse_p90": float(np.percentile(per_spectrum_rmse, 90)),
        "per_spectrum_rmse_p99": float(np.percentile(per_spectrum_rmse, 99)),
        "per_spectrum_rmse_max": float(per_spectrum_rmse.max()),
    }
    return summary, per_pixel_rmse, per_spectrum_rmse


def evaluate_test(model, payload, run_dir: Path, eval_batch_size: int = 8192,
                  pixel_chunk: int = 1024) -> dict:
    """Rebuild the held-out test split, reconstruct it, and report metrics.

    Mirrors :func:`boss_emulator.predict.evaluate_test` (which the two-arg
    forward precludes reusing directly).
    """
    cfg = payload.get("data_config")
    if cfg is None:
        raise ValueError(
            "checkpoint has no data_config; retrain to enable --test")
    print(f"[eval] rebuilding dataset to recover the test split: {cfg}")
    ds = data_mod.load_dataset(
        core_labels=cfg["core_labels"],
        abundance_elements=cfg["abundance_elements"],
        target=cfg["target"],
        val_frac=cfg["val_frac"],
        test_frac=cfg["test_frac"],
        seed=cfg["seed"],
        max_rows=cfg["max_rows"],
        standardize_y=cfg["standardize_y"],
    )
    test = ds.test_idx
    print(f"[eval] test set: {test.size:,} spectra")
    Y_pred = reconstruct(model, payload, ds.X[test],
                         eval_batch_size=eval_batch_size, pixel_chunk=pixel_chunk)
    metrics = reconstruction_metrics(Y_pred, ds.Y[test])
    summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
    print("[eval] test reconstruction metrics:")
    for k, v in summary.items():
        print(f"        {k:26s} {v}")
    out = run_dir if run_dir.is_dir() else run_dir.parent
    (out / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote test_metrics.json -> {out}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained single-lambda emulator on its test set.")
    p.add_argument("--run", required=True,
                   help="run directory under trained_model/ or a .pkl path")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best")
    p.add_argument("--test", action="store_true",
                   help="evaluate reconstruction metrics on the held-out test set")
    p.add_argument("--eval-batch-size", type=int, default=8192,
                   dest="eval_batch_size")
    p.add_argument("--pixel-chunk", type=int, default=1024, dest="pixel_chunk")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run = Path(args.run)
    ckpt = run if run.is_file() else run / f"{args.checkpoint}.pkl"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint {ckpt}")
    model, payload = load_checkpoint(ckpt)
    print(f"[eval] loaded {ckpt}  model={payload['model']}  "
          f"features={payload['feature_names']}  n_pixels={payload['n_pixels']}")
    if not args.test:
        raise SystemExit("pass --test to evaluate the held-out test set")
    evaluate_test(model, payload, run, args.eval_batch_size, args.pixel_chunk)


if __name__ == "__main__":
    main()
