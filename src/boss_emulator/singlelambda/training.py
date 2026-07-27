"""Training loop for the coordinate-query ("single wavelength") emulator.

This mirrors :mod:`boss_emulator.training` but for the two-argument
``model(x, coords)`` forward: each step trains on a subsample of pixel
positions rather than the whole spectrum. A batch of ``B`` spectra is paired
with ``K = positions_per_step`` randomly sampled pixel positions (shared across
the batch and re-drawn every step); the target is the standardised flux at
those pixels. Validation reconstructs full spectra by querying every position
in chunks, and reports the same per-spectrum **median** reconstruction RMSE (in
physical flux units) as the other three emulators, so the numbers are directly
comparable.

The dataset load, checkpoint format, optimiser schedule and Optuna hooks are
reused from :mod:`boss_emulator.training`; only the coords-aware train/eval
steps and the position sampling are new here.

Usage (from the project root)::

    uv run python -m boss_emulator.singlelambda.training --epochs 1000
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
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from boss_emulator import data as data_mod
from boss_emulator import training as root_training
from boss_emulator.data import ABUNDANCE_ELEMENTS, CORE_LABELS
from boss_emulator.emulator import build_model


# =========================================================================
# Train / eval steps (coords-aware)
# =========================================================================

@nnx.jit
def train_step(model, optimizer, x, coords, y):
    def loss_fn(m):
        pred = m(x, coords)
        return jnp.mean((pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _eval_chunk(model, x, coords, y, y_std_chunk):
    """Per-(row-batch, pixel-chunk) pieces for both metrics, one forward pass.

    Returns ``(sq_sum, persp_sq_partial)`` where ``sq_sum`` is the summed
    *standardised* squared error over this chunk (for the mean-MSE monitor) and
    ``persp_sq_partial`` is the per-row sum over *this chunk's* pixels of the
    *physical* squared residual (scaled back to flux units by the per-pixel
    ``y_std_chunk``). Summing ``persp_sq_partial`` across all chunks of a
    spectrum and taking ``sqrt(total / n_pixels)`` gives its physical RMSE --
    matching :func:`boss_emulator.training.evaluate`.
    """
    pred = model(x, coords)
    d = pred - y
    sq_sum = jnp.sum(d ** 2)
    persp_sq = jnp.sum((d * y_std_chunk) ** 2, axis=1)
    return sq_sum, persp_sq


def make_coord_grid(n_pixels: int) -> np.ndarray:
    """Normalised pixel coordinates in ``[-1, 1]`` (``2*i/(n_pixels-1)-1``).

    The same evenly-spaced grid the DeepONet trunk uses; there is no physical
    wavelength array in the data, so position is the pixel index normalised.
    """
    return np.linspace(-1.0, 1.0, n_pixels).astype("float32")


def evaluate(model, Xn, Yn, idx, y_std_dev, coord_grid,
             row_batch=4096, pixel_chunk=1024):
    """Reconstruct full spectra for ``idx`` and return metrics.

    Queries every position (in ``pixel_chunk`` slices) for each row-batch,
    bounding activation memory over both spectra and pixels. Returns
    ``(val_mse, per_spectrum_rmse)`` with the exact semantics of
    :func:`boss_emulator.training.evaluate`: the mean standardised MSE and a
    float32 array of per-spectrum physical RMSE (one value per row in ``idx``
    order). The median of that array is the optimised objective.
    """
    n_pix = Yn.shape[1]
    sq_total = 0.0
    persp = np.empty(idx.size, dtype=np.float32)
    write = 0
    for start in range(0, idx.size, row_batch):
        rows = idx[start:start + row_batch]
        xb = jnp.asarray(Xn[rows])
        persp_sq = np.zeros(rows.size, dtype=np.float64)
        for c0 in range(0, n_pix, pixel_chunk):
            c1 = min(c0 + pixel_chunk, n_pix)
            yb = jnp.asarray(Yn[rows, c0:c1])
            cb = jnp.asarray(coord_grid[c0:c1])
            ys = y_std_dev[c0:c1]
            sq_sum, psq = _eval_chunk(model, xb, cb, yb, ys)
            sq_total += float(sq_sum)
            persp_sq += np.asarray(psq, dtype=np.float64)
        persp[write:write + rows.size] = np.sqrt(persp_sq / n_pix)
        write += rows.size
    val_mse = sq_total / (idx.size * n_pix)
    return val_mse, persp


# =========================================================================
# Training loop
# =========================================================================

def train(args, prepared=None, trial=None) -> tuple[Path, "data_mod.Dataset", float]:
    """Train the single-lambda emulator; return ``(run_dir, dataset, best_obj)``.

    Mirrors :func:`boss_emulator.training.train` (same optimiser schedule,
    checkpointing, Optuna hooks and objective) but trains the two-argument
    ``model(x, coords)`` on a subsample of ``args.positions_per_step`` pixel
    positions per step. ``prepared`` may be a ``(ds, Xn, Yn)`` tuple from
    :func:`boss_emulator.training.prepare_data` to skip the dataset load.
    """
    # 1-3. Dataset + standardised host arrays (load once, or reuse if given).
    if prepared is None:
        ds, Xn, Yn = root_training.prepare_data(args)
    else:
        ds, Xn, Yn = prepared

    # 4. Model + optimizer (AdamW with warmup-cosine decay).
    model = build_model(
        "singlelambda", ds.n_features, ds.n_pixels, seed=args.seed,
        d_model=args.d_model, n_heads=args.n_heads, n_blocks=args.n_blocks,
        mlp_dim=args.mlp_dim, n_fourier=args.n_fourier,
        activation=args.activation, dropout=args.dropout)
    model.train()  # enable dropout during training (no-op when dropout=0)
    y_std_dev = jnp.asarray(ds.y_std)  # per-pixel std, for physical RMSE

    steps_per_epoch = max(1, ds.train_idx.size // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1, peak_value=args.lr,
        warmup_steps=min(500, total_steps // 20 + 1),
        decay_steps=total_steps, end_value=args.lr * 0.01,
    )
    optimizer = nnx.Optimizer(
        model, optax.adamw(schedule, weight_decay=args.weight_decay),
        wrt=nnx.Param)

    # 5. Output directory for this run.
    outdir = Path(args.outdir) / f"singlelambda_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[train] writing run to {outdir}")
    np.savez(outdir / "split_idx.npz",
             train_idx=ds.train_idx, val_idx=ds.val_idx, test_idx=ds.test_idx)

    # 6. Epoch loop. Objective = per-spectrum median reconstruction RMSE on the
    #    validation set (physical flux units); val_mse (standardised) is logged.
    coord_grid = make_coord_grid(ds.n_pixels)
    K = int(args.positions_per_step)
    # Validation reconstructs full spectra, so its activation memory scales with
    # row_batch * pixel_chunk (independent of K). Cap it well below a large
    # training batch so a big --batch-size can't OOM the per-epoch val pass.
    eval_row_batch = min(int(args.batch_size), 4096)
    # Optionally validate on a fixed random subset of the validation split each
    # epoch: full validation reconstructs every one of ~225k spectra x 4479
    # pixels, which costs about as much as a training pass. A fixed seeded
    # subset keeps the per-epoch objective stable (so best.pkl selection stays
    # consistent) while cutting the per-epoch cost ~proportionally. The final
    # test-set evaluation (in the driver) always uses the full test split.
    val_subsample = getattr(args, "val_subsample", None)
    if val_subsample and val_subsample < ds.val_idx.size:
        val_eval_idx = np.sort(np.random.default_rng(args.seed).choice(
            ds.val_idx, size=int(val_subsample), replace=False))
        print(f"[train] validating on a fixed random subset of "
              f"{val_eval_idx.size:,}/{ds.val_idx.size:,} val spectra")
    else:
        val_eval_idx = ds.val_idx
    ckpt_every = int(getattr(args, "ckpt_every", 0) or 0)
    best_obj = np.inf
    history = []
    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            # Per-epoch RNG for column sampling (deterministic, reproducible).
            col_rng = np.random.default_rng(args.seed + epoch)
            train_losses = []
            for batch in data_mod.iter_batches(
                    ds.train_idx, args.batch_size, seed=args.seed + epoch):
                # Sample K pixel positions, shared across the batch, re-drawn
                # each step -- this is the "rearranged" training set.
                cols = col_rng.integers(0, ds.n_pixels, size=K)
                xb = jnp.asarray(Xn[batch])            # (B, n_features)
                yb = jnp.asarray(Yn[batch][:, cols])   # (B, K)
                cb = jnp.asarray(coord_grid[cols])     # (K,)
                loss = train_step(model, optimizer, xb, cb, yb)
                train_losses.append(float(loss))

            # Validation (full-spectrum reconstruction). Eval mode disables
            # dropout; back to train mode for the next epoch.
            model.eval()
            va, persp = evaluate(model, Xn, Yn, val_eval_idx, y_std_dev,
                                 coord_grid, row_batch=eval_row_batch)
            model.train()
            obj = float(np.median(persp)) if persp.size else float("nan")

            tr = float(np.mean(train_losses))
            dt = time.time() - t0
            print(f"[train] epoch {epoch + 1:4d}/{args.epochs}  "
                  f"train_mse={tr:.4e}  val_mse={va:.4e}  "
                  f"val_medRMSE={obj:.4e}  ({dt:.1f}s)", flush=True)
            history.append({"epoch": epoch + 1, "train_mse": tr,
                            "val_mse": float(va), "val_median_rmse": obj})

            # Best checkpoint on the objective.
            if obj < best_obj:
                best_obj = obj
                root_training.save_checkpoint(outdir / "best.pkl", model, ds, args)

            # Periodic snapshots "while training" (0 disables).
            if ckpt_every and (epoch + 1) % ckpt_every == 0:
                root_training.save_checkpoint(
                    outdir / f"epoch_{epoch + 1:04d}.pkl", model, ds, args)

            # Optuna: report and possibly prune.
            if trial is not None:
                trial.report(obj, step=epoch)
                if trial.should_prune():
                    root_training.save_checkpoint(
                        outdir / "final.pkl", model, ds, args)
                    (outdir / "history.json").write_text(
                        json.dumps(history, indent=2))
                    root_training.plot_loss(history, outdir / "loss_curve.png")
                    print(f"[train] pruned at epoch {epoch + 1}; "
                          f"best val_medRMSE={best_obj:.4e}  -> {outdir}",
                          flush=True)
                    import optuna
                    raise optuna.TrialPruned()
    finally:
        (outdir / "config.json").write_text(
            json.dumps(vars(args), indent=2, default=str))

    # Always save the final state too.
    root_training.save_checkpoint(outdir / "final.pkl", model, ds, args)
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    root_training.plot_loss(history, outdir / "loss_curve.png")
    print(f"[train] done. best val_medRMSE={best_obj:.4e}  -> {outdir}")
    return outdir, ds, float(best_obj)


# =========================================================================
# CLI
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the single-lambda cross-attention emulator.")
    p.add_argument("--labels", nargs="+", default=list(CORE_LABELS),
                   metavar="LABEL")
    p.add_argument("--abundances", nargs="*", default=list(ABUNDANCE_ELEMENTS),
                   metavar="ELEM")
    p.add_argument("--target", default="norm_flux")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512, dest="batch_size",
                   help="number of spectra per step (B)")
    p.add_argument("--positions-per-step", type=int, default=512,
                   dest="positions_per_step",
                   help="number of pixel positions sampled per step (K)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   dest="weight_decay")
    p.add_argument("--d-model", type=int, default=256, dest="d_model")
    p.add_argument("--n-heads", type=int, default=8, dest="n_heads")
    p.add_argument("--n-blocks", type=int, default=4, dest="n_blocks")
    p.add_argument("--mlp-dim", type=int, default=512, dest="mlp_dim")
    p.add_argument("--n-fourier", type=int, default=16, dest="n_fourier",
                   help="number of Fourier frequency bands for the position")
    p.add_argument("--activation", default="gelu",
                   choices=["gelu", "silu", "relu", "tanh"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ckpt-every", type=int, default=0, dest="ckpt_every")
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    p.add_argument("--test-frac", type=float, default=0.1, dest="test_frac")
    p.add_argument("--val-subsample", type=int, default=None,
                   dest="val_subsample",
                   help="validate on a fixed random subset of N val spectra "
                        "each epoch (default: full val split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None, dest="max_rows")
    p.add_argument("--no-standardize-y", action="store_true",
                   dest="no_standardize_y")
    p.add_argument("--rebuild-cache", action="store_true", dest="rebuild_cache")
    p.add_argument("--eval-batch-size", type=int, default=8192,
                   dest="eval_batch_size")
    p.add_argument("--outdir",
                   default=str(data_mod.PROJECT_ROOT / "trained_model"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    # The reused checkpoint helper reads args.model; set it for this model.
    args.model = "singlelambda"
    args.hidden = None
    root_training.claim_gpu()
    train(args)


if __name__ == "__main__":
    main()
