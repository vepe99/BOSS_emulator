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


@nnx.jit
def _eval_components(model, x, y, y_std):
    """Per-batch pieces for both metrics, in one forward pass.

    Returns ``(sq_sum, persp_sq)`` where ``sq_sum`` is the summed *standardised*
    squared error (for the mean-MSE monitor) and ``persp_sq`` is the per-row
    sum over pixels of the *physical* squared residual -- residuals scaled back
    to flux units by the per-pixel ``y_std`` (mean/std cancels in the
    difference, so only ``y_std`` is needed). ``sqrt(persp_sq / n_pixels)`` is
    then the per-spectrum reconstruction RMSE in physical flux units.
    """
    pred = model(x)
    d = pred - y
    sq_sum = jnp.sum(d ** 2)
    persp_sq = jnp.sum((d * y_std) ** 2, axis=1)
    return sq_sum, persp_sq


def evaluate(model, Xn, Yn, idx, y_std_dev, batch_size):
    """Stream ``idx`` through the model and return metrics.

    Returns ``(val_mse, per_spectrum_rmse)``: the mean standardised MSE over
    all selected elements, and a float32 array of per-spectrum physical RMSE
    (one value per row in ``idx`` order). The median of that array is the
    objective this project optimises (robust to tail spectra).
    """
    n_pix = Yn.shape[1]
    sq_total = 0.0
    persp = np.empty(idx.size, dtype=np.float32)
    write = 0
    for batch in data_mod.iter_batches(idx, batch_size, shuffle=False):
        xb = jnp.asarray(Xn[batch])
        yb = jnp.asarray(Yn[batch])
        sq_sum, persp_sq = _eval_components(model, xb, yb, y_std_dev)
        sq_total += float(sq_sum)
        k = batch.size
        persp[write:write + k] = np.sqrt(np.asarray(persp_sq) / n_pix)
        write += k
    val_mse = sq_total / (idx.size * n_pix)
    return val_mse, persp


def full_reconstruction_metrics(model, Xn, Y, Yn, idx, y_std, y_mean,
                                batch_size):
    """Rich test-set metrics from a single streaming pass over ``idx``.

    Complements the bare RMSE with *scale-relative* statistics so an absolute
    error in continuum-normalised flux can be interpreted:

    * ``standardized_rmse`` -- RMSE divided by the per-pixel flux std, i.e. the
      error as a fraction of the natural pixel-to-pixel spectral variation
      (0.007 = 0.7% of it), not of the trivially-~1 continuum level.
    * ``r2`` -- physical coefficient of determination (fraction of test flux
      variance captured), computed against the per-pixel test-set mean.
    * ``mae_*`` -- mean/median absolute error (robust companion to RMSE, which
      the cool carbon-star tail inflates).
    * ``snr_equivalent_*`` -- 1 / per-spectrum RMSE: the observational SNR at
      which emulator error would start to matter.

    ``Y`` is the *physical* target array (continuum-normalised flux), ``Yn`` its
    standardised counterpart, ``y_std``/``y_mean`` the per-pixel stats used to
    de-standardise predictions. Returns ``(metrics_dict, per_spectrum_rmse)``.
    Caller must have put the model in eval mode.
    """
    n_pix = Yn.shape[1]
    ybar = Y[idx].mean(axis=0)                      # per-pixel test mean
    ss_tot = float(((Y[idx] - ybar) ** 2).sum())    # total variance (physical)
    ys = jnp.asarray(y_std)
    ym = jnp.asarray(y_mean)

    ss_res = 0.0            # physical squared-residual sum  -> R^2
    ss_res_std = 0.0        # standardised squared-residual sum -> std MSE
    sum_abs = 0.0          # physical absolute-residual sum -> MAE
    persp = np.empty(idx.size, np.float32)          # per-spectrum physical RMSE
    permae = np.empty(idx.size, np.float32)         # per-spectrum physical MAE
    write = 0
    for batch in data_mod.iter_batches(idx, batch_size, shuffle=False):
        predn = model(jnp.asarray(Xn[batch]))
        d_std = predn - jnp.asarray(Yn[batch])
        d = predn * ys + ym - jnp.asarray(Y[batch])     # physical residual
        ss_res += float(jnp.sum(d ** 2))
        ss_res_std += float(jnp.sum(d_std ** 2))
        sum_abs += float(jnp.sum(jnp.abs(d)))
        persp[write:write + batch.size] = np.sqrt(
            np.asarray(jnp.mean(d ** 2, axis=1)))
        permae[write:write + batch.size] = np.asarray(jnp.mean(jnp.abs(d), 1))
        write += batch.size

    n = idx.size
    std_mse = ss_res_std / (n * n_pix)
    metrics = {
        "n": int(n),
        "n_pixels": int(n_pix),
        "test_mse_standardized": std_mse,
        "standardized_rmse": float(np.sqrt(std_mse)),
        "r2": 1.0 - ss_res / ss_tot,
        "per_spectrum_rmse_median": float(np.median(persp)),
        "per_spectrum_rmse_mean": float(persp.mean()),
        "per_spectrum_rmse_p90": float(np.percentile(persp, 90)),
        "per_spectrum_rmse_p99": float(np.percentile(persp, 99)),
        "per_spectrum_rmse_max": float(persp.max()),
        "mae_mean": sum_abs / (n * n_pix),
        "per_spectrum_mae_median": float(np.median(permae)),
        "snr_equivalent_median": float(1.0 / np.median(persp)),
        "snr_equivalent_mean": float(1.0 / persp.mean()),
    }
    return metrics, persp


# =========================================================================
# Data preparation
# =========================================================================

def prepare_data(args) -> tuple["data_mod.Dataset", np.ndarray, np.ndarray]:
    """Load the dataset and produce standardised host arrays.

    Split out from :func:`train` so a long-running process (e.g. a
    hyper-parameter search) can load the ~38 GB cache *once* and reuse the
    result across many trials instead of paying the load per run.
    """
    data_mod.build_cache(rebuild=args.rebuild_cache)
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
    Xn = ds.normalize_x(ds.X).astype("float32")
    Yn = ds.normalize_y(ds.Y).astype("float32")
    return ds, Xn, Yn


def prepare_test_data(args) -> tuple["data_mod.Dataset", np.ndarray, np.ndarray]:
    """:func:`prepare_data` restricted to the test split -- for evaluation.

    Same signature and return shape, but the arrays hold only the ~10% of rows
    an evaluation actually reads, so this costs ~4 GB and seconds instead of
    ~76 GB and minutes. The returned Dataset is locally numbered; see
    :func:`data_mod.load_test_dataset`. Requires the same ``args`` fields, so
    the split matches the runs being evaluated.
    """
    data_mod.build_cache(rebuild=args.rebuild_cache)
    print(f"[eval] loading test split: labels={args.labels} "
          f"abundances={args.abundances or 'none'}")
    ds = data_mod.load_test_dataset(
        core_labels=args.labels,
        abundance_elements=args.abundances,
        target=args.target,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        max_rows=args.max_rows,
        standardize_y=not args.no_standardize_y,
    )
    print(f"[eval] features ({ds.n_features}): {ds.feature_names}")
    Xn = ds.normalize_x(ds.X).astype("float32")
    Yn = ds.normalize_y(ds.Y).astype("float32")
    return ds, Xn, Yn


# =========================================================================
# Training loop
# =========================================================================

def train(args, prepared=None, trial=None) -> tuple[Path, "data_mod.Dataset", float]:
    """Train an emulator; return ``(run_dir, dataset, best_objective)``.

    ``prepared`` may be a ``(ds, Xn, Yn)`` tuple from :func:`prepare_data` to
    skip the (expensive) dataset load -- used by the hyper-parameter search so
    the ~38 GB cache is loaded once per worker and reused across trials.

    ``trial`` is an optional Optuna ``Trial``: the per-epoch validation
    objective (per-spectrum median RMSE, physical units) is reported to it, and
    ``optuna.TrialPruned`` is raised when the pruner says so. The best
    checkpoint is selected on that same objective.
    """
    # 1-3. Dataset + standardised host arrays (load once, or reuse if given).
    if prepared is None:
        ds, Xn, Yn = prepare_data(args)
    else:
        ds, Xn, Yn = prepared

    # Stage the train AND validation splits resident on the GPU so every epoch
    # -- both the training steps and the validation pass -- gathers its batches
    # on-device instead of re-copying data host->device each epoch (~32 GB for
    # train, ~4 GB for val). That per-epoch host round-trip, not compute, was
    # the bottleneck (the loop was host-starved, which is also why batch size
    # did not help). Batches are formed from shuffled *positions* into these
    # resident arrays; with the same seed this reproduces the exact same batches
    # as indexing the global rows, so behaviour is unchanged -- only faster.
    # The host copies Xn/Yn are kept only for the end-of-trial test-set metrics
    # (full_reconstruction_metrics), which run once, not per epoch.
    Xtr = jax.device_put(Xn[ds.train_idx])
    Ytr = jax.device_put(Yn[ds.train_idx])
    train_pos = np.arange(ds.train_idx.size)
    Xva = jax.device_put(Xn[ds.val_idx])
    Yva = jax.device_put(Yn[ds.val_idx])
    val_pos = np.arange(ds.val_idx.size)

    # 4. Model + optimizer (Adam with cosine decay).
    model_kwargs = {}
    if args.hidden:
        if args.model == "payne":
            model_kwargs["hidden"] = tuple(args.hidden)
        else:
            model_kwargs["branch_hidden"] = tuple(args.hidden)
    if args.model == "payne":
        model_kwargs["activation"] = args.activation
        model_kwargs["residual"] = args.residual
        model_kwargs["layernorm"] = args.layernorm
    if args.model == "deeponet":
        model_kwargs["latent"] = args.latent
    if args.model == "transformer":
        model_kwargs["d_model"] = args.d_model
        model_kwargs["n_heads"] = args.n_heads
        model_kwargs["n_blocks"] = args.n_blocks
        model_kwargs["mlp_dim"] = args.mlp_dim
    if args.dropout:
        model_kwargs["dropout"] = args.dropout
    model = build_model(
        args.model, ds.n_features, ds.n_pixels, seed=args.seed, **model_kwargs)
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
    outdir = Path(args.outdir) / f"{args.model}_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[train] writing run to {outdir}")

    # Persist the exact split so the held-out test set is reproducible for
    # predict.py (also reconstructible from data_config, saved below).
    np.savez(outdir / "split_idx.npz",
             train_idx=ds.train_idx, val_idx=ds.val_idx, test_idx=ds.test_idx)

    # 6. Epoch loop. The objective (best-checkpoint + Optuna metric) is the
    #    per-spectrum median reconstruction RMSE on the validation set, in
    #    physical flux units; ``val_mse`` (standardised) is kept for logging.
    ckpt_every = int(getattr(args, "ckpt_every", 0) or 0)
    best_obj = np.inf
    history = []
    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_losses = []
            for batch in data_mod.iter_batches(
                    train_pos, args.batch_size, seed=args.seed + epoch):
                # Positions index the device-resident train split; permuting
                # positions with the same seed reproduces the exact same
                # batches as permuting the global row indices did.
                loss = train_step(model, optimizer, Xtr[batch], Ytr[batch])
                train_losses.append(loss)  # device scalar; reduced once/epoch

            # Validation (batched, device-resident). Switch to eval mode so
            # dropout is disabled, then back to train mode for the next epoch.
            model.eval()
            va, persp = evaluate(model, Xva, Yva, val_pos, y_std_dev,
                                 args.batch_size)
            model.train()
            obj = float(np.median(persp)) if persp.size else float("nan")

            tr = float(jnp.mean(jnp.stack(train_losses)))
            dt = time.time() - t0
            print(f"[train] epoch {epoch + 1:4d}/{args.epochs}  "
                  f"train_mse={tr:.4e}  val_mse={va:.4e}  "
                  f"val_medRMSE={obj:.4e}  ({dt:.1f}s)", flush=True)
            history.append({"epoch": epoch + 1, "train_mse": tr,
                            "val_mse": float(va), "val_median_rmse": obj})

            # Best checkpoint on the objective.
            if obj < best_obj:
                best_obj = obj
                save_checkpoint(outdir / "best.pkl", model, ds, args)

            # Periodic snapshots "while training" (0 disables).
            if ckpt_every and (epoch + 1) % ckpt_every == 0:
                save_checkpoint(outdir / f"epoch_{epoch + 1:04d}.pkl",
                                model, ds, args)

            # Optuna: report and possibly prune (kills unpromising trials).
            if trial is not None:
                trial.report(obj, step=epoch)
                # Keep the history/curve for the pruned run before bailing out.
                if trial.should_prune():
                    save_checkpoint(outdir / "final.pkl", model, ds, args)
                    (outdir / "history.json").write_text(
                        json.dumps(history, indent=2))
                    plot_loss(history, outdir / "loss_curve.png")
                    print(f"[train] pruned at epoch {epoch + 1}; "
                          f"best val_medRMSE={best_obj:.4e}  -> {outdir}",
                          flush=True)
                    import optuna
                    raise optuna.TrialPruned()
    finally:
        # Persist config regardless of how the loop ends (incl. pruning).
        (outdir / "config.json").write_text(
            json.dumps(vars(args), indent=2, default=str))

    # Always save the final state too.
    save_checkpoint(outdir / "final.pkl", model, ds, args)
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    plot_loss(history, outdir / "loss_curve.png")
    print(f"[train] done. best val_medRMSE={best_obj:.4e}  -> {outdir}")
    # Return the assembled dataset too, so callers that go on to evaluate can
    # reuse it instead of re-loading the ~38 GB cache a second time.
    return outdir, ds, float(best_obj)


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


def _persisted_model_kwargs(args) -> dict:
    """Model-specific constructor kwargs to store in the checkpoint.

    Excludes ``dropout`` and ``hidden``, which ``load_checkpoint`` restores
    from their own dedicated payload fields.
    """
    if args.model == "payne":
        return {
            "activation": args.activation,
            "residual": args.residual,
            "layernorm": args.layernorm,
        }
    if args.model == "deeponet":
        return {"latent": args.latent}
    if args.model == "transformer":
        return {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_blocks": args.n_blocks,
            "mlp_dim": args.mlp_dim,
        }
    if args.model == "singlelambda":
        return {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_blocks": args.n_blocks,
            "mlp_dim": args.mlp_dim,
            "n_fourier": args.n_fourier,
        }
    return {}


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
        "model_kwargs": _persisted_model_kwargs(args),
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
    p.add_argument("--model", choices=["payne", "deeponet", "transformer"],
                   default="payne")
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
    p.add_argument("--d-model", type=int, default=256, dest="d_model",
                   help="transformer token/embedding width")
    p.add_argument("--n-heads", type=int, default=8, dest="n_heads",
                   help="transformer number of attention heads")
    p.add_argument("--n-blocks", type=int, default=4, dest="n_blocks",
                   help="transformer number of cross-attention blocks")
    p.add_argument("--mlp-dim", type=int, default=512, dest="mlp_dim",
                   help="transformer feed-forward hidden width")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="dropout rate on hidden activations (payne: whole "
                        "net; deeponet: branch net). 0.0 disables it")
    p.add_argument("--activation", default="gelu",
                   choices=["gelu", "silu", "relu", "tanh"],
                   help="payne hidden-layer activation")
    p.add_argument("--residual", action="store_true",
                   help="payne: residual/skip connections between "
                        "equal-width hidden layers")
    p.add_argument("--layernorm", action="store_true",
                   help="payne: pre-activation LayerNorm on hidden layers")
    p.add_argument("--ckpt-every", type=int, default=0, dest="ckpt_every",
                   help="also snapshot the model every N epochs "
                        "(0 = only best.pkl/final.pkl)")
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
