"""Train a Payne emulator and evaluate its spectral reconstruction.

This is a thin driver over the ``boss_emulator`` package:

1. Build the preprocessed cache if needed and train a Payne-style MLP
   (``boss_emulator.training``).
2. Reload the best checkpoint and evaluate reconstruction:
   * plot the true-vs-predicted reconstruction of 5 random test-set spectra;
   * accumulate reconstruction-error statistics over the whole dataset
     (streamed in batches) and, separately, over the held-out test set.

All artefacts are written into the run directory under ``trained_model/``.

GPU
---
Run this on a node with a free GPU. Importing ``boss_emulator.training``
triggers ``autocvd(num_gpus=1)``, which selects a free GPU and sets
``CUDA_VISIBLE_DEVICES`` before JAX initialises -- no manual device choice
needed. (For a CPU dry run, set ``JAX_PLATFORMS=cpu`` to skip autocvd.)

Example
-------
    uv run python script/train_and_evaluate_payne.py \
        --epochs 100 --batch-size 2048 --hidden 1024 1024 1024 \
        --abundances Fe Mg C
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Importing training selects a GPU via autocvd (unless JAX_PLATFORMS=cpu).
from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint
from boss_emulator.predict import predict, reconstruction_metrics


# =========================================================================
# Evaluation helpers
# =========================================================================

def whole_dataset_error_stats(model, payload, ds, batch_size=8192):
    """Stream predictions over every row and accumulate error statistics.

    Returns a JSON-serialisable summary dict plus (per-pixel RMSE,
    per-spectrum RMSE) arrays for plotting. Only one batch of predictions is
    held in memory at a time, so this scales to the full ~2.25M-row set.
    """
    n, P = ds.X.shape[0], ds.n_pixels
    ss_res = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    ss_tot = 0.0
    y_bar = float(ds.Y.mean())
    per_pixel_sq = np.zeros(P, dtype=np.float64)
    per_spectrum_rmse = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        pred = predict(model, payload, ds.X[start:stop], batch_size=batch_size)
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


def plot_reconstructions(pred, true, rows, path):
    """Plot true vs predicted flux and the residual for each selected row."""
    pix = np.arange(pred.shape[1])
    n = len(rows)
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(14, 2.6 * n), sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n, "hspace": 0.05})
    for i, r in enumerate(rows):
        top, bot = axes[2 * i], axes[2 * i + 1]
        top.plot(pix, true[r], lw=0.6, color="#333333", label="true")
        top.plot(pix, pred[r], lw=0.6, color="#C44E52", alpha=0.8, label="pred")
        rmse = float(np.sqrt(((pred[r] - true[r]) ** 2).mean()))
        top.set_ylabel("flux")
        top.text(0.005, 0.9, f"test row {int(r)}  RMSE={rmse:.3e}",
                 transform=top.transAxes, fontsize=8, va="top")
        if i == 0:
            top.legend(loc="lower right", fontsize=8)
        bot.plot(pix, pred[r] - true[r], lw=0.5, color="#4C72B0")
        bot.axhline(0, color="k", lw=0.4)
        bot.set_ylabel("resid")
    axes[-1].set_xlabel("pixel index")
    axes[0].set_title("Payne reconstruction of 5 test-set spectra")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_error_distributions(per_pixel_rmse, per_spectrum_rmse, path):
    """Per-pixel RMSE curve and per-spectrum RMSE histogram side by side."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4.5))
    ax0.plot(np.arange(per_pixel_rmse.size), per_pixel_rmse, lw=0.7,
             color="#C44E52")
    ax0.set_xlabel("pixel index")
    ax0.set_ylabel("RMSE")
    ax0.set_title("Per-pixel reconstruction RMSE (whole dataset)")

    ax1.hist(per_spectrum_rmse, bins=100, color="#4C72B0")
    ax1.set_yscale("log")
    ax1.set_xlabel("per-spectrum RMSE")
    ax1.set_ylabel("count")
    ax1.set_title("Per-spectrum RMSE distribution (whole dataset)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Payne emulator and evaluate reconstruction.")
    # Training hyper-parameters (forwarded to boss_emulator.training).
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2048, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="*", default=[1024, 1024, 1024],
                   help="Payne hidden layer widths")
    p.add_argument("--labels", nargs="+", default=list(data_mod.CORE_LABELS))
    # Default to the curated abundance elements (CURATED_ELEMENTS minus the
    # fixed H/He, i.e. data.ABUNDANCE_ELEMENTS). Pass --abundances explicitly
    # (or an empty list) to override.
    p.add_argument("--abundances", nargs="*",
                   default=list(data_mod.ABUNDANCE_ELEMENTS),
                   help="abundance elements to add as inputs "
                        "(default: the curated ABUNDANCE_ELEMENTS)")
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    p.add_argument("--test-frac", type=float, default=0.1, dest="test_frac")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None, dest="max_rows",
                   help="use only N rows (quick end-to-end dry run)")
    p.add_argument("--eval-batch-size", type=int, default=8192,
                   dest="eval_batch_size")
    p.add_argument("--outdir",
                   default=str(data_mod.PROJECT_ROOT / "trained_model"))
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Train (build a fully-populated training args namespace with defaults,
    #    then override with our choices) -> returns the run directory.
    train_args = training.build_parser().parse_args([])
    train_args.model = "payne"
    train_args.epochs = args.epochs
    train_args.batch_size = args.batch_size
    train_args.lr = args.lr
    train_args.hidden = args.hidden
    train_args.labels = args.labels
    train_args.abundances = args.abundances
    train_args.val_frac = args.val_frac
    train_args.test_frac = args.test_frac
    train_args.seed = args.seed
    train_args.max_rows = args.max_rows
    train_args.outdir = args.outdir

    run_dir = training.train(train_args)

    # 2. Reload the best checkpoint and rebuild the exact same dataset/split.
    model, payload = load_checkpoint(run_dir / "best.pkl")
    cfg = payload["data_config"]
    print(f"\n[eval] rebuilding dataset for evaluation: {cfg}")
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

    # 3. Reconstruction of 5 random test-set spectra.
    rng = np.random.default_rng(args.seed)
    n_show = min(5, ds.test_idx.size)
    sample = rng.choice(ds.test_idx, size=n_show, replace=False)
    pred_samples = predict(model, payload, ds.X[sample],
                           batch_size=args.eval_batch_size)
    plot_reconstructions(pred_samples, ds.Y[sample], np.arange(n_show),
                         run_dir / "reconstruction_examples.png")
    print(f"[eval] wrote reconstruction_examples.png "
          f"(test rows {sample.tolist()})")

    # 4. Reconstruction-error statistics over the whole dataset.
    print("[eval] computing reconstruction error over the whole dataset ...")
    whole, per_pixel_rmse, per_spectrum_rmse = whole_dataset_error_stats(
        model, payload, ds, batch_size=args.eval_batch_size)

    # Test-set-only stats (honest generalisation number).
    test_pred = predict(model, payload, ds.X[ds.test_idx],
                        batch_size=args.eval_batch_size)
    test_metrics = reconstruction_metrics(test_pred, ds.Y[ds.test_idx])
    test_summary = {k: v for k, v in test_metrics.items()
                    if not k.startswith("_")}

    report = {"whole_dataset": whole, "test_set": test_summary}
    (run_dir / "reconstruction_stats.json").write_text(
        json.dumps(report, indent=2))
    plot_error_distributions(per_pixel_rmse, per_spectrum_rmse,
                             run_dir / "reconstruction_error_distributions.png")

    print("\n=== reconstruction error (whole dataset) ===")
    for k, v in whole.items():
        print(f"  {k:26s} {v}")
    print("=== reconstruction error (test set only) ===")
    for k, v in test_summary.items():
        print(f"  {k:26s} {v}")
    print(f"\n[eval] wrote reconstruction_stats.json + plots -> {run_dir}")


if __name__ == "__main__":
    main()
