"""Training pipeline for the BOSS/Korg spectral emulator.

Usage (from the project root)::

    uv run python -m boss_emulator.training --model payne --epochs 50
    uv run python -m boss_emulator.training --model deeponet \
        --labels Teff logg m_H alpha_m carbon_m vmic --abundances Fe C

On first run the ~40 GB float32 cache is built under ``preprocessed_data/``;
subsequent runs just load it. A free GPU is selected automatically with
``autocvd`` (per lab policy) before JAX is imported.
"""

from __future__ import annotations

# --- GPU selection MUST happen before importing jax ----------------------
# autocvd picks a free GPU and sets CUDA_VISIBLE_DEVICES (lab policy). It is
# skipped when the device is already chosen (CUDA_VISIBLE_DEVICES set), when
# CPU is forced (JAX_PLATFORMS=cpu), or when BOSS_NO_AUTOCVD is set -- so the
# module can be imported for CPU tests without blocking on a free GPU.
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
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from boss_emulator import data as data_mod
from boss_emulator.data import ABUNDANCE_ELEMENTS, CORE_LABELS
from boss_emulator.emulator import build_model


# =========================================================================
# GPU claim
# =========================================================================

def claim_gpu() -> None:
    """Force JAX to initialise on the autocvd-selected GPU *now*.

    ``autocvd`` (run at import, above) only sets ``CUDA_VISIBLE_DEVICES``;
    JAX would not otherwise touch the device until the first train step --
    which is *after* the minutes-long CPU dataset load. Touching the device
    here initialises the CUDA context and claims the free GPU up front, so it
    can't be grabbed by another process during that load, and surfaces the
    device choice immediately. No-op cost is negligible.
    """
    backend = jax.default_backend()
    devices = jax.devices()
    # A trivial op forces the backend to initialise and allocate the device.
    jax.device_put(0.0).block_until_ready()
    print(f"[gpu] JAX backend={backend}  devices={devices}", flush=True)


# =========================================================================
# Train / eval steps
# =========================================================================

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        pred = m(x)
        return jnp.mean((pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y) ** 2)


# =========================================================================
# Training loop
# =========================================================================

def train(args) -> Path:
    # 1. Ensure the preprocessed cache exists (built once).
    data_mod.build_cache(rebuild=args.rebuild_cache)

    # 2. Assemble the dataset (selected labels + target, standardised).
    print(f"[train] loading dataset: labels={args.labels} "
          f"abundances={args.abundances or 'none'}")
    ds = data_mod.load_dataset(
        core_labels=args.labels,
        abundance_elements=args.abundances,
        target=args.target,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        max_rows=args.max_rows,
        standardize_y=not args.no_standardize_y,
    )
    print(f"[train] features ({ds.n_features}): {ds.feature_names}")
    print(f"[train] n_train={ds.train_idx.size:,}  n_val={ds.val_idx.size:,}  "
          f"n_test={ds.test_idx.size:,}  n_pixels={ds.n_pixels}")

    # 3. Standardise once, keep everything on host as float32; batches are
    #    sliced and shipped to the device per step.
    Xn = ds.normalize_x(ds.X).astype("float32")
    Yn = ds.normalize_y(ds.Y).astype("float32")

    # 4. Model + optimizer (Adam with cosine decay).
    model_kwargs = {}
    if args.hidden:
        if args.model == "payne":
            model_kwargs["hidden"] = tuple(args.hidden)
        else:
            model_kwargs["branch_hidden"] = tuple(args.hidden)
    if args.model == "deeponet":
        model_kwargs["latent"] = args.latent
    if args.dropout:
        model_kwargs["dropout"] = args.dropout
    model = build_model(
        args.model, ds.n_features, ds.n_pixels, seed=args.seed, **model_kwargs)
    model.train()  # enable dropout during training (no-op when dropout=0)

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
    outdir = Path(args.outdir) / f"{args.model}_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[train] writing run to {outdir}")

    # Persist the exact split so the held-out test set is reproducible for
    # predict.py (also reconstructible from data_config, saved below).
    np.savez(outdir / "split_idx.npz",
             train_idx=ds.train_idx, val_idx=ds.val_idx, test_idx=ds.test_idx)

    # 6. Epoch loop.
    best_val = np.inf
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_losses = []
        for batch in data_mod.iter_batches(
                ds.train_idx, args.batch_size, seed=args.seed + epoch):
            xb = jnp.asarray(Xn[batch])
            yb = jnp.asarray(Yn[batch])
            loss = train_step(model, optimizer, xb, yb)
            train_losses.append(float(loss))

        # Validation (batched to bound device memory). Switch to eval mode so
        # dropout is disabled, then back to train mode for the next epoch.
        model.eval()
        val_losses = []
        for batch in data_mod.iter_batches(
                ds.val_idx, args.batch_size, shuffle=False):
            xb = jnp.asarray(Xn[batch])
            yb = jnp.asarray(Yn[batch])
            val_losses.append(float(eval_step(model, xb, yb)))
        model.train()

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        dt = time.time() - t0
        print(f"[train] epoch {epoch + 1:3d}/{args.epochs}  "
              f"train_mse={tr:.4e}  val_mse={va:.4e}  ({dt:.1f}s)")
        history.append({"epoch": epoch + 1, "train_mse": tr, "val_mse": va})

        if va < best_val:
            best_val = va
            save_checkpoint(outdir / "best.pkl", model, ds, args)

    # Always save the final state too.
    save_checkpoint(outdir / "final.pkl", model, ds, args)
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    plot_loss(history, outdir / "loss_curve.png")
    print(f"[train] done. best val_mse={best_val:.4e}  -> {outdir}")
    return outdir


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the train/validation MSE evolution as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, [h["train_mse"] for h in history], label="train", lw=1.5)
    ax.plot(ep, [h["val_mse"] for h in history], label="val", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised flux)")
    ax.set_yscale("log")
    ax.set_title("Emulator training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(path: Path, model, ds, args) -> None:
    """Persist model parameters + normalisation stats + minimal config.

    The pure-Python/NumPy pytree of parameters is pickled together with the
    standardisation statistics needed to reproduce preprocessing and to
    de-normalise predictions. Reload with :func:`load_checkpoint`.
    """
    # Persist only the learnable Param state. Non-Param state (dropout RNG
    # streams -- typed PRNG keys that np.asarray can't convert -- and the
    # DeepONet fixed pixel coords) is reconstructed deterministically at
    # build time, so it need not be pickled.
    _, param_state, _ = nnx.split(model, nnx.Param, ...)
    params = jax.tree.map(lambda x: np.asarray(x), nnx.to_pure_dict(param_state))
    payload = {
        "model": args.model,
        "model_kwargs": {"latent": args.latent} if args.model == "deeponet" else {},
        "hidden": args.hidden,
        "dropout": args.dropout,
        "n_features": ds.n_features,
        "n_pixels": ds.n_pixels,
        "feature_names": ds.feature_names,
        "params": params,
        "x_mean": ds.x_mean, "x_std": ds.x_std,
        "y_mean": ds.y_mean, "y_std": ds.y_std,
        "seed": args.seed,
        # Everything needed to reconstruct the exact same dataset split
        # (and thus the held-out test set) in predict.py.
        "data_config": {
            "core_labels": list(args.labels),
            "abundance_elements": list(args.abundances),
            "target": args.target,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "max_rows": args.max_rows,
            "standardize_y": not args.no_standardize_y,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


# =========================================================================
# CLI
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a BOSS spectral emulator.")
    p.add_argument("--model", choices=["payne", "deeponet"], default="payne")
    p.add_argument("--labels", nargs="+", default=list(CORE_LABELS),
                   metavar="LABEL",
                   help=f"core labels to use (subset of {CORE_LABELS})")
    p.add_argument("--abundances", nargs="*", default=[], metavar="ELEM",
                   help="element symbols whose A(X) to add as inputs "
                        "(e.g. Fe C); redundant with m_H/alpha_m/carbon_m")
    p.add_argument("--target", default="norm_flux")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   dest="weight_decay", help="AdamW weight decay")
    p.add_argument("--hidden", type=int, nargs="*", default=None,
                   help="hidden layer widths (payne: whole net; "
                        "deeponet: branch net)")
    p.add_argument("--latent", type=int, default=256,
                   help="DeepONet latent/basis dimension")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout rate on hidden activations (payne: whole "
                        "net; deeponet: branch net). 0.0 disables it")
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac",
                   help="fraction held out for validation (loss monitoring)")
    p.add_argument("--test-frac", type=float, default=0.1, dest="test_frac",
                   help="fraction held out for the test set (final metrics)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None, dest="max_rows",
                   help="use only N random rows (smoke tests)")
    p.add_argument("--no-standardize-y", action="store_true",
                   dest="no_standardize_y",
                   help="do not per-pixel standardise the target")
    p.add_argument("--rebuild-cache", action="store_true", dest="rebuild_cache")
    p.add_argument("--outdir", default=str(data_mod.PROJECT_ROOT / "trained_model"))
    p.add_argument("--list-abundances", action="store_true",
                   dest="list_abundances",
                   help="print the selectable abundance elements and exit")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.list_abundances:
        print("Selectable abundance elements:")
        print(" ".join(ABUNDANCE_ELEMENTS))
        return
    train(args)


if __name__ == "__main__":
    main()
