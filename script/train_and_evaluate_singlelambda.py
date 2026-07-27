"""Train the single-lambda emulator and evaluate its spectral reconstruction.

A thin driver over ``boss_emulator.singlelambda``, mirroring
``train_and_evaluate_deeponet.py`` / ``_transformer.py`` but for the
coordinate-query cross-attention model:

1. Build the preprocessed cache if needed and train
   :class:`~boss_emulator.singlelambda.emulator_singlelambda.SpectraFormerSingleLambda`
   -- keys/values come from the stellar labels and abundances, and the query
   from a single pixel position (Fourier features). Each step trains on a
   random subsample of ``--positions-per-step`` positions (the "rearranged"
   training set); a full spectrum is reconstructed by querying every position.
2. Reload the best checkpoint and evaluate reconstruction:
   * plot true-vs-predicted reconstruction of 5 random test-set spectra;
   * accumulate reconstruction-error statistics over the whole dataset and,
     separately, over the held-out test set.

All artefacts are written into the run directory under ``trained_model/``.

GPU
---
Importing ``boss_emulator.training`` triggers ``autocvd(num_gpus=1)`` (lab
policy), selecting a free GPU before JAX initialises. ``main`` calls
``training.claim_gpu()`` first so the device is held before the CPU-heavy data
load. For a CPU dry run set ``JAX_PLATFORMS=cpu`` (skips autocvd).

Example
-------
    uv run python script/train_and_evaluate_singlelambda.py \
        --epochs 1000 --batch-size 512 --positions-per-step 512 \
        --d-model 256 --n-heads 8 --n-blocks 4 --mlp-dim 512 --n-fourier 16
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
from boss_emulator.predict import reconstruction_metrics
from boss_emulator.singlelambda import training as sl_training
from boss_emulator.singlelambda.evaluate import reconstruct, whole_dataset_error_stats


# =========================================================================
# Plotting (mirrors the deeponet driver; titles say "single-lambda")
# =========================================================================

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
    axes[0].set_title("Single-lambda reconstruction of 5 test-set spectra")
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
        description="Train a single-lambda emulator and evaluate reconstruction.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512, dest="batch_size",
                   help="number of spectra per step (B)")
    p.add_argument("--positions-per-step", type=int, default=512,
                   dest="positions_per_step",
                   help="number of pixel positions sampled per step (K)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   dest="weight_decay", help="AdamW weight decay")
    p.add_argument("--d-model", type=int, default=256, dest="d_model")
    p.add_argument("--n-heads", type=int, default=8, dest="n_heads")
    p.add_argument("--n-blocks", type=int, default=4, dest="n_blocks")
    p.add_argument("--mlp-dim", type=int, default=512, dest="mlp_dim")
    p.add_argument("--n-fourier", type=int, default=16, dest="n_fourier",
                   help="number of Fourier frequency bands for the position")
    p.add_argument("--activation", default="gelu",
                   choices=["gelu", "silu", "relu", "tanh"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ckpt-every", type=int, default=0, dest="ckpt_every",
                   help="also snapshot the model every N epochs")
    p.add_argument("--labels", nargs="+", default=list(data_mod.CORE_LABELS))
    p.add_argument("--abundances", nargs="*",
                   default=list(data_mod.ABUNDANCE_ELEMENTS),
                   help="abundance elements to add as inputs "
                        "(default: the curated ABUNDANCE_ELEMENTS)")
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    p.add_argument("--test-frac", type=float, default=0.1, dest="test_frac")
    p.add_argument("--val-subsample", type=int, default=None,
                   dest="val_subsample",
                   help="validate on a fixed random subset of N val spectra "
                        "each epoch (default: full val split)")
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

    # 0. Claim the autocvd-selected GPU up front, before the CPU-heavy load.
    training.claim_gpu()

    # 1. Build a fully-populated training args namespace (root defaults), then
    #    override with our choices. The single-lambda train loop and the reused
    #    checkpoint helper read these attributes; n_fourier / positions_per_step
    #    are not in the root parser, so set them explicitly.
    train_args = training.build_parser().parse_args([])
    train_args.model = "singlelambda"
    train_args.hidden = None
    train_args.epochs = args.epochs
    train_args.batch_size = args.batch_size
    train_args.positions_per_step = args.positions_per_step
    train_args.lr = args.lr
    train_args.weight_decay = args.weight_decay
    train_args.d_model = args.d_model
    train_args.n_heads = args.n_heads
    train_args.n_blocks = args.n_blocks
    train_args.mlp_dim = args.mlp_dim
    train_args.n_fourier = args.n_fourier
    train_args.activation = args.activation
    train_args.dropout = args.dropout
    train_args.ckpt_every = args.ckpt_every
    train_args.labels = args.labels
    train_args.abundances = args.abundances
    train_args.val_frac = args.val_frac
    train_args.test_frac = args.test_frac
    train_args.val_subsample = args.val_subsample
    train_args.seed = args.seed
    train_args.max_rows = args.max_rows
    train_args.eval_batch_size = args.eval_batch_size
    train_args.outdir = args.outdir

    run_dir, ds, _best_obj = sl_training.train(train_args)

    # 2. Reload the best checkpoint (reuse the dataset + split from training).
    model, payload = load_checkpoint(run_dir / "best.pkl")

    # 3. Reconstruction of 5 random test-set spectra.
    rng = np.random.default_rng(args.seed)
    n_show = min(5, ds.test_idx.size)
    sample = rng.choice(ds.test_idx, size=n_show, replace=False)
    pred_samples = reconstruct(model, payload, ds.X[sample],
                               eval_batch_size=args.eval_batch_size)
    plot_reconstructions(pred_samples, ds.Y[sample], np.arange(n_show),
                         run_dir / "reconstruction_examples.png")
    print(f"[eval] wrote reconstruction_examples.png "
          f"(test rows {sample.tolist()})")

    # 4. Reconstruction-error statistics over the whole dataset.
    print("[eval] computing reconstruction error over the whole dataset ...")
    whole, per_pixel_rmse, per_spectrum_rmse = whole_dataset_error_stats(
        model, payload, ds, eval_batch_size=args.eval_batch_size)

    # Test-set-only stats (honest generalisation number).
    test_pred = reconstruct(model, payload, ds.X[ds.test_idx],
                            eval_batch_size=args.eval_batch_size)
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
