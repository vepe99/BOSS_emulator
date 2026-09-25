"""Experiment: residual-weighted POST-TRAINING of the generalist emulator.

Question
--------
The best generalist (HPO trial 77) is not overfitting -- train and validation
MSE are equal at epoch 1000 and the loss was still falling -- so its error is a
fixed-capacity budget spent unevenly: the median spectrum is emulated to
7e-4 while the cool/carbon-rich tail sits at 3e-3 (p99) with grid-cliff
outliers at 0.2. Can a short fine-tune that *up-weights* the spectra the model
currently gets wrong move that budget toward the tail -- and at what cost to
the bulk? And is a longer plain continuation, or a smoother input
parameterisation of Teff, a better use of the same compute?

Four arms, all evaluated in flux units on the same held-out test set as
every other number in this project (median / p90 / p99 / max per-spectrum
RMSE, plus the cool-star regimes):

``control``
    Plain MSE, warm-started from trial 77's ``best.pkl``, fresh low-LR
    warmup-cosine schedule. Measures what the extra epochs alone buy, so the
    weighted arms are credited only with what the WEIGHTS add.
``batch_w``
    Same, but each batch is re-weighted on the fly by the model's *current*
    per-spectrum flux RMSE: ``w_i = clip(r_i / median_batch(r), 1, cap)``
    (stop-gradient), loss ``sum(w_i * mse_i) / sum(w_i)``. Bulk spectra keep
    weight 1, hard ones get up to ``cap``x. Weights track the model every
    step, so nothing is over-shot.
``epoch_w``
    Same weighting, but ``r_i`` comes from a full pass over the training set
    recomputed every ``--reweight-every`` epochs (global median, boosting-
    style). Lagged weights, same cap.
``scratch``
    Plain Teff input, trained FROM SCRATCH with trial 77's hyper-parameters for
    ``--scratch-epochs`` epochs: "just train it longer" under one long cosine
    schedule (warm restarts of a converged model are a worse use of epochs).
``logteff``
    A structural alternative: the network takes ``log10(Teff)`` instead of
    ``Teff``. Spectra change far faster per Kelvin at 3000 K than at 7000 K,
    yet a linear-standardised Teff compresses the cool end of the input space.
    Since the input changes, this arm trains FROM SCRATCH with trial 77's
    hyper-parameters for ``--scratch-epochs`` epochs (1000 = a like-for-like
    re-run of trial 77 with one input remapped).

Design notes
------------
* Weights use the residual in FLUX units (the metric we report), capped so the
  few hundred under-resolved-grid spectra (RMSE ~0.2, ~300x the median) cannot
  dominate the gradient; forcing the net onto those cliffs would also degrade
  interpolation between grid nodes, which the inverse pipeline depends on.
* Best checkpoint per arm is selected on the validation MEAN per-spectrum
  RMSE (not the median): the median would ignore exactly the tail the weights
  target, the mean balances bulk and tail. Every arm uses the same rule.
* Memory: this node's GPUs are 40 GB, the float32 training targets are 32 GB.
  The targets are staged on-device as float16 RAW FLUX (16 GB) and standardised
  per batch on the device. Flux in [0, 1.5] has a float16 quantisation noise
  of ~1e-4 rms at most (smaller in deep bands where flux -> 0), ~0.02% of the
  MSE floor -- negligible against a 7e-4 model error. The test evaluation uses
  the exact float32 truth, so it is directly comparable to earlier numbers.

Wavelength-space arms (weights over PIXELS, not spectra)
--------------------------------------------------------
A diagnostic of the 3000-epoch model showed the aggregate per-pixel flux
residual is nearly flat in wavelength (max/median 2.7), yet within a single
spectrum ~70% of the squared error sits in its worst 10% of pixels -- the
hard pixels are lines, but *which* lines depends on the star. Meanwhile the
standardised loss ``((f-y)/y_std)^2`` is itself a hidden wavelength
weighting: ``1/y_std^2`` spans a factor 214 and over-spends on low-variance
red pixels whose flux error is already the smallest.

``pixel_w``
    Global per-pixel weights ``w_l = clip(r_l / median_l(r), 1, cap)`` where
    ``r_l`` is the train-set RMS residual at pixel ``l`` (flux or standardised
    units, ``--pixel-space``), recomputed every ``--reweight-every`` epochs.
    Loss ``sum_l w_l * mse_l / sum_l w_l``. The literal "weight where the
    lines are" test.
``pixteff_w``
    Same, but the residual profile is computed separately for every Teff
    grid node (30 x n_pixels table) and each row is normalised by its OWN
    median, so a cool star is up-weighted at *its* molecular bands and a hot
    star at *its* lines, with no per-spectrum (epoch_w-style) boost mixed in.
``flux_w``
    Static ``w_l = (y_std_l^2)^p`` (``--flux-power`` p, default 1), i.e. the
    loss becomes plain flux-unit MSE -- the metric we report -- instead of the
    standardised MSE. Tests whether the standardisation bias costs accuracy.

Example
-------
    PYTHONUNBUFFERED=1 BOSS_CACHE_DIR=/export/data/vgiusepp/BOSS_emulator/preprocessed_data \\
        .venv/bin/python script/exp_reweight.py \\
        --generalist trained_model/payne_hpo/trial_0077/payne_1785288314 \\
        --specialist trained_model/exp_specialist/specialist_1785327806 \\
        --epochs 300

autocvd claims a free GPU before JAX initialises (lab policy).
"""
from __future__ import annotations

# GPU selection MUST happen before importing jax (lab policy). See _gpu.py.
from boss_emulator._gpu import claim_free_gpu

claim_free_gpu()

import argparse
import dataclasses
import functools
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from boss_emulator import data as data_mod, training
from boss_emulator.emulator import build_model, load_checkpoint
from boss_emulator.predict import gate_weight
from boss_emulator.training import save_checkpoint

ARMS = ("control", "batch_w", "epoch_w", "scratch", "logteff",
        "pixel_w", "pixteff_w", "flux_w")
PIXEL_MODES = ("pixel_w", "pixteff_w", "flux_w")
TEFF_TRANSFORMS = {
    # name -> (function, new feature name)
    "none": (lambda t: t, "Teff"),
    "log10": (np.log10, "log10_Teff"),
    "inv": (lambda t: 5040.0 / t, "theta_Teff"),
}


# =========================================================================
# Jitted pieces
# =========================================================================

def _standardise(y16, ym, ys):
    return (y16.astype(jnp.float32) - ym) / ys


def _per_row_terms(model, x, y, ys):
    """Per-row standardised MSE and per-row flux RMSE, one forward pass."""
    d = model(x) - y
    mse_i = jnp.mean(d ** 2, axis=1)
    r_i = jnp.sqrt(jnp.mean((d * ys) ** 2, axis=1))
    return mse_i, r_i


@nnx.jit
def step_fixed_w(model, optimizer, x, y16, w, ym, ys):
    """One AdamW step on ``sum(w_i * mse_i) / sum(w_i)`` with given weights."""
    y = _standardise(y16, ym, ys)

    def loss_fn(m):
        mse_i, _ = _per_row_terms(m, x, y, ys)
        return jnp.sum(w * mse_i) / jnp.sum(w)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def step_batch_w(model, optimizer, x, y16, ym, ys, cap):
    """One AdamW step, weights from the batch's own current residuals."""
    y = _standardise(y16, ym, ys)

    def loss_fn(m):
        mse_i, r_i = _per_row_terms(m, x, y, ys)
        r = jax.lax.stop_gradient(r_i)
        w = jnp.clip(r / jnp.median(r), 1.0, cap)
        return jnp.sum(w * mse_i) / jnp.sum(w)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def step_pixel_w(model, optimizer, x, y16, w, ym, ys):
    """One AdamW step on ``sum(w * d^2) / sum(w)`` with per-PIXEL weights.

    ``w`` is either ``(n_pixels,)`` (one profile for every spectrum) or
    ``(batch, n_pixels)`` (a profile per spectrum, e.g. looked up by Teff
    node); it is broadcast against the standardised squared residual.
    """
    y = _standardise(y16, ym, ys)

    def loss_fn(m):
        d2 = (m(x) - y) ** 2
        wb = jnp.broadcast_to(w, d2.shape)
        return jnp.sum(wb * d2) / jnp.sum(wb)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@functools.partial(nnx.jit, static_argnames=("n_bins", "in_flux"))
def _pixel_sse_batch(model, x, y16, bins, ym, ys, n_bins, in_flux):
    """Per-(bin, pixel) sum of squared residuals + per-bin counts for a batch."""
    d = model(x) - _standardise(y16, ym, ys)
    if in_flux:
        d = d * ys
    sse = jax.ops.segment_sum(d ** 2, bins, num_segments=n_bins)
    cnt = jax.ops.segment_sum(jnp.ones(x.shape[0], jnp.float32), bins,
                              num_segments=n_bins)
    return sse, cnt


def pixel_rms_device(model, X, Y16, ym, ys, batch, *, bins=None, n_bins=1,
                     in_flux=True):
    """RMS residual per (Teff-bin, pixel) over device-resident arrays.

    ``bins`` is a device int32 array aligned with ``X`` (all zeros -> one
    global profile). Accumulates in float64 on the host. Returns
    ``(n_bins, n_pixels)``.
    """
    n = X.shape[0]
    P = int(ym.shape[0])
    sse = np.zeros((n_bins, P), np.float64)
    cnt = np.zeros(n_bins, np.float64)
    zeros = None if bins is not None else jnp.zeros(batch, jnp.int32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        b = bins[s:e] if bins is not None else zeros[: e - s]
        sb, cb = _pixel_sse_batch(model, X[s:e], Y16[s:e], b, ym, ys,
                                  n_bins=n_bins, in_flux=in_flux)
        sse += np.asarray(sb, np.float64)
        cnt += np.asarray(cb, np.float64)
    return np.sqrt(sse / np.maximum(cnt, 1.0)[:, None]).astype("float32")


def pixel_weights(rms, cap):
    """``clip(r / median_pixels(r), 1, cap)`` row by row (one row per bin)."""
    med = np.median(rms, axis=1, keepdims=True)
    return np.clip(rms / med, 1.0, cap).astype("float32")


@nnx.jit
def _flux_rmse_batch(model, x, y16, ym, ys):
    pred = model(x) * ys + ym
    return jnp.sqrt(jnp.mean((pred - y16.astype(jnp.float32)) ** 2, axis=1))


@nnx.jit
def _flux_batch(model, x, ym, ys):
    return model(x) * ys + ym


def flux_rmse_device(model, X, Y16, ym, ys, batch):
    """Per-spectrum flux RMSE over device-resident (X, float16 flux) arrays."""
    n = X.shape[0]
    out = np.empty(n, np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        out[s:e] = np.asarray(_flux_rmse_batch(model, X[s:e], Y16[s:e], ym, ys))
    return out


def predict_flux_host(model, Xn, ym, ys, batch):
    """Float32 flux predictions for host-side standardised inputs ``Xn``."""
    out = np.empty((Xn.shape[0], ym.shape[0]), np.float32)
    for s in range(0, Xn.shape[0], batch):
        e = min(s + batch, Xn.shape[0])
        out[s:e] = np.asarray(_flux_batch(model, jnp.asarray(Xn[s:e]), ym, ys))
    return out


# =========================================================================
# Data helpers
# =========================================================================

def stage_f16(Y, idx, chunk=200_000):
    """Gather rows ``idx`` of the float32 flux ``Y`` as float16 on the device."""
    buf = np.empty((idx.size, Y.shape[1]), np.float16)
    for s in range(0, idx.size, chunk):
        buf[s:s + chunk] = Y[idx[s:s + chunk]]
    return jax.device_put(buf)


def transform_teff(ds, kind):
    """Copy of ``ds`` with Teff remapped and input stats recomputed (train split).

    Only ``X``/``x_mean``/``x_std``/``feature_names`` change; ``Y`` is shared.
    The renamed feature is persisted by ``save_checkpoint`` so a reloaded
    checkpoint declares the input it expects.
    """
    if kind == "none":
        return ds
    fn, new_name = TEFF_TRANSFORMS[kind]
    j = ds.feature_names.index("Teff")
    X = ds.X.copy()
    X[:, j] = fn(X[:, j])
    x_mean = X[ds.train_idx].mean(axis=0)
    x_std = X[ds.train_idx].std(axis=0)
    x_std[x_std == 0] = 1.0
    feat = list(ds.feature_names)
    feat[j] = new_name
    return dataclasses.replace(
        ds, X=X, x_mean=x_mean.astype("float32"), x_std=x_std.astype("float32"),
        feature_names=feat)


def per_spectrum_rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean(axis=1)).astype("float32")


def summ(persp):
    return {
        "median": float(np.median(persp)),
        "mean": float(persp.mean()),
        "p90": float(np.percentile(persp, 90)),
        "p99": float(np.percentile(persp, 99)),
        "max": float(persp.max()),
    }


REGIMES = [
    ("ALL", lambda T, C, G: np.ones(T.size, bool)),
    ("Teff<=4000", lambda T, C, G: T <= 4000),
    ("Teff<=3500", lambda T, C, G: T <= 3500),
    ("Teff<=3500 & C>=0.5", lambda T, C, G: (T <= 3500) & (C >= 0.5)),
    ("Teff>4800 (untouched by gate)", lambda T, C, G: T > 4800),
]


def regime_stats(persp, T, C, G):
    out = {}
    for name, f in REGIMES:
        m = f(T, C, G)
        if m.any():
            out[name] = {"n": int(m.sum()), "median": float(np.median(persp[m])),
                         "p99": float(np.percentile(persp[m], 99)),
                         "max": float(persp[m].max())}
    return out


def print_regimes(tag, rs):
    print(f"\n--- {tag}: error vs regime ---", flush=True)
    for name, s in rs.items():
        print(f"  {name:32s} n={s['n']:>7,}  med={s['median']:.4e}  "
              f"p99={s['p99']:.4e}  max={s['max']:.4e}", flush=True)


# =========================================================================
# One arm
# =========================================================================

def run_arm(name, model, ds_used, Xtr, Ytr16, Xva, Yva16, *, mode, epochs, lr,
            weight_decay, batch_size, eval_batch_size, cap, reweight_every,
            warmup_steps, outdir, base_args, log_every, tr_bins=None, n_bins=1,
            pixel_in_flux=True, flux_power=1.0):
    """Train ``model`` in place; write checkpoints/history; return the best model.

    ``tr_bins`` (device int32, aligned with ``Xtr``) and ``n_bins`` are only
    used by ``pixteff_w``; ``pixel_in_flux`` picks the residual units for the
    pixel arms; ``flux_power`` is the exponent of ``flux_w``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    ym = jnp.asarray(ds_used.y_mean, jnp.float32)
    ys = jnp.asarray(ds_used.y_std, jnp.float32)
    n_tr = Xtr.shape[0]
    steps = max(1, n_tr // batch_size)
    total = steps * epochs
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr,
        warmup_steps=min(warmup_steps, total // 20 + 1),
        decay_steps=total, end_value=lr * 0.01)
    opt = nnx.Optimizer(model, optax.adamw(sched, weight_decay=weight_decay),
                        wrt=nnx.Param)
    model.train()
    tr_pos = np.arange(n_tr)
    ones = jnp.ones(batch_size, jnp.float32)
    cap_j = jnp.float32(cap)
    w_all = None
    w_pix = None
    if mode == "flux_w":
        w_np = (np.asarray(ds_used.y_std, np.float64) ** (2.0 * flux_power))
        w_np = (w_np / w_np.mean()).astype("float32")
        w_pix = jax.device_put(w_np)
        print(f"[{name}] flux_w: p={flux_power} weight range "
              f"{w_np.min():.3e}..{w_np.max():.3e}", flush=True)
    pix_bins = tr_bins if mode == "pixteff_w" else None
    pix_nb = n_bins if mode == "pixteff_w" else 1
    best = np.inf
    history = []
    print(f"\n[{name}] mode={mode} epochs={epochs} lr={lr:.2e} wd={weight_decay:.1e} "
          f"n_train={n_tr:,} -> {outdir}", flush=True)
    t_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        winfo = ""
        if mode == "epoch_w" and epoch % reweight_every == 0:
            model.eval()
            r = flux_rmse_device(model, Xtr, Ytr16, ym, ys, eval_batch_size)
            model.train()
            w_np = np.clip(r / np.median(r), 1.0, cap).astype("float32")
            w_all = jax.device_put(w_np)
            winfo = (f"  [reweight: mean_w={w_np.mean():.3f} "
                     f"frac_w>1={np.mean(w_np > 1):.3f} frac_w=cap={np.mean(w_np >= cap):.4f}]")
        if mode in ("pixel_w", "pixteff_w") and epoch % reweight_every == 0:
            model.eval()
            rms = pixel_rms_device(model, Xtr, Ytr16, ym, ys, eval_batch_size,
                                   bins=pix_bins, n_bins=pix_nb, in_flux=pixel_in_flux)
            model.train()
            w_np = pixel_weights(rms, cap)
            w_pix = jax.device_put(w_np[0] if mode == "pixel_w" else w_np)
            np.savez(outdir / f"pixel_weights_ep{epoch:04d}.npz", rms=rms, w=w_np)
            winfo = (f"  [pixel reweight: mean_w={w_np.mean():.3f} "
                     f"frac_w>1={np.mean(w_np > 1):.3f} max_w={w_np.max():.2f}]")
        losses = []
        for b in data_mod.iter_batches(tr_pos, batch_size, seed=10_000 + epoch):
            xb, yb = Xtr[b], Ytr16[b]
            if mode == "batch_w":
                loss = step_batch_w(model, opt, xb, yb, ym, ys, cap_j)
            elif mode == "epoch_w":
                loss = step_fixed_w(model, opt, xb, yb, w_all[b], ym, ys)
            elif mode == "pixteff_w":
                loss = step_pixel_w(model, opt, xb, yb, w_pix[tr_bins[b]], ym, ys)
            elif mode in ("pixel_w", "flux_w"):
                loss = step_pixel_w(model, opt, xb, yb, w_pix, ym, ys)
            else:
                loss = step_fixed_w(model, opt, xb, yb, ones[:b.size], ym, ys)
            losses.append(loss)
        model.eval()
        persp = flux_rmse_device(model, Xva, Yva16, ym, ys, eval_batch_size)
        model.train()
        s = summ(persp)
        obj = s["mean"]
        rec = {"epoch": epoch + 1, "train_loss": float(jnp.mean(jnp.stack(losses))),
               "val_median": s["median"], "val_mean": s["mean"],
               "val_p90": s["p90"], "val_p99": s["p99"], "val_max": s["max"]}
        history.append(rec)
        if obj < best:
            best = obj
            save_checkpoint(outdir / "best.pkl", model, ds_used, base_args)
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
        if epoch % log_every == 0 or epoch == epochs - 1 or winfo:
            print(f"[{name}] epoch {epoch+1:4d}/{epochs}  loss={rec['train_loss']:.4e}  "
                  f"val med={s['median']:.4e} mean={s['mean']:.4e} "
                  f"p90={s['p90']:.4e} p99={s['p99']:.4e}  best_mean={best:.4e}  "
                  f"({time.time()-t0:.1f}s){winfo}", flush=True)
    save_checkpoint(outdir / "final.pkl", model, ds_used, base_args)
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    (outdir / "config.json").write_text(json.dumps({
        "arm": name, "mode": mode, "epochs": epochs, "lr": lr,
        "weight_decay": weight_decay, "batch_size": batch_size, "cap": cap,
        "reweight_every": reweight_every, "pixel_in_flux": pixel_in_flux,
        "flux_power": flux_power, "n_bins": n_bins,
        "selection": "val mean per-spectrum flux RMSE",
        "feature_names": ds_used.feature_names,
        "train_minutes": (time.time() - t_start) / 60}, indent=2))
    plot_history(history, outdir / "history.png", name)
    best_model, _ = load_checkpoint(outdir / "best.pkl")
    best_model.eval()
    return best_model


def plot_history(history, path, name):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in ("val_median", "val_mean", "val_p90", "val_p99"):
        ax.plot(ep, [h[k] for h in history], label=k)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("val flux RMSE")
    ax.set_title(name); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generalist", required=True,
                   help="run dir of the generalist to post-train (holds best.pkl)")
    p.add_argument("--specialist", default=None,
                   help="optional specialist run dir: adds the gated baseline row")
    p.add_argument("--gate-lo", type=float, default=3800.0, dest="gate_lo")
    p.add_argument("--gate-hi", type=float, default=4800.0, dest="gate_hi")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--epochs", type=int, default=300,
                   help="fine-tuning epochs for control/batch_w/epoch_w")
    p.add_argument("--lr", type=float, default=1e-4, help="peak LR for fine-tuning")
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--cap", type=float, default=10.0,
                   help="max per-spectrum weight (relative to the median residual)")
    p.add_argument("--reweight-every", type=int, default=25, dest="reweight_every",
                   help="epoch_w: recompute the train-set weights every N epochs")
    p.add_argument("--pixel-space", default="flux", choices=("flux", "std"),
                   dest="pixel_space",
                   help="pixel_w/pixteff_w: units in which the per-pixel residual "
                        "that defines the weights is measured")
    p.add_argument("--flux-power", type=float, default=1.0, dest="flux_power",
                   help="flux_w: w_l = (y_std_l^2)^p; p=1 is plain flux-unit MSE")
    p.add_argument("--teff-transform", default="log10", choices=list(TEFF_TRANSFORMS),
                   dest="teff_transform")
    p.add_argument("--scratch-epochs", type=int, default=1000, dest="scratch_epochs",
                   help="from-scratch epochs for the scratch/logteff arms")
    p.add_argument("--batch-size", type=int, default=4096, dest="batch_size")
    p.add_argument("--eval-batch-size", type=int, default=8192, dest="eval_batch_size")
    p.add_argument("--max-rows", type=int, default=None, dest="max_rows",
                   help="subsample the grid (smoke tests only)")
    p.add_argument("--log-every", type=int, default=10, dest="log_every")
    p.add_argument("--outdir", default="trained_model/exp_reweight")
    p.add_argument("--plotdir", default="plots/exp_reweight")
    args = p.parse_args()

    training.claim_gpu()
    gen_dir = Path(args.generalist)
    run_dir = Path(args.outdir) / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    plotdir = Path(args.plotdir); plotdir.mkdir(parents=True, exist_ok=True)
    print(f"[rw] writing to {run_dir}", flush=True)

    # --- Data: raw flux + input stats; targets are standardised on-device ---
    data_mod.build_cache(rebuild=False)
    ds = data_mod.load_dataset(
        core_labels=list(data_mod.CORE_LABELS), abundance_elements=[],
        target="norm_flux", val_frac=0.1, test_frac=0.1, seed=0,
        max_rows=args.max_rows, standardize_y=True)
    print(f"[rw] n_train={ds.train_idx.size:,} n_val={ds.val_idx.size:,} "
          f"n_test={ds.test_idx.size:,} n_pixels={ds.n_pixels}", flush=True)
    feat = ds.feature_names
    ci = {k: feat.index(k) for k in ("Teff", "carbon_m", "logg")}

    gen_model, gpay = load_checkpoint(gen_dir / "best.pkl")
    gen_model.eval()
    same_stats = np.allclose(np.asarray(gpay["y_mean"], np.float32),
                             np.asarray(ds.y_mean, np.float32), atol=1e-4)
    if not same_stats:
        msg = "generalist y_mean != dataset y_mean (different split/config)"
        if args.max_rows is None:
            raise AssertionError(msg)
        print(f"[rw] WARNING (smoke test): {msg}", flush=True)

    # checkpoint metadata must describe the generalist's architecture
    base = training.build_parser().parse_args([])
    base.model = "mlp"; base.labels = list(data_mod.CORE_LABELS)
    base.abundances = []; base.val_frac = 0.1; base.test_frac = 0.1
    base.seed = 0; base.max_rows = args.max_rows
    base.hidden = list(gpay["hidden"]); base.dropout = gpay.get("dropout", 0.0)
    mk = gpay.get("model_kwargs") or {}
    base.activation = mk.get("activation", base.activation)
    base.residual = mk.get("residual", base.residual)
    base.layernorm = mk.get("layernorm", base.layernorm)

    # --- Device staging (float16 flux; see module docstring) ----------------
    Xn = ds.normalize_x(ds.X).astype("float32")
    Xtr = jax.device_put(Xn[ds.train_idx]); Xva = jax.device_put(Xn[ds.val_idx])
    t0 = time.time()
    Ytr16 = stage_f16(ds.Y, ds.train_idx); Yva16 = stage_f16(ds.Y, ds.val_idx)
    Ytr16.block_until_ready()
    print(f"[rw] staged f16 targets on device in {time.time()-t0:.0f}s "
          f"({Ytr16.nbytes/2**30:.1f} GB train, {Yva16.nbytes/2**30:.1f} GB val)",
          flush=True)
    ym = jnp.asarray(ds.y_mean, jnp.float32); ys = jnp.asarray(ds.y_std, jnp.float32)

    # Teff grid node of every training row (for pixteff_w's per-node profiles)
    teff_nodes = np.unique(ds.X[:, ci["Teff"]])
    tr_bins = jax.device_put(np.searchsorted(
        teff_nodes, ds.X[ds.train_idx][:, ci["Teff"]]).astype("int32"))
    print(f"[rw] {teff_nodes.size} Teff nodes: {teff_nodes.min():.0f}..{teff_nodes.max():.0f}",
          flush=True)

    # --- Test set (float32 truth on host) -------------------------------------
    idx = ds.test_idx
    T = ds.X[idx][:, ci["Teff"]]; C = ds.X[idx][:, ci["carbon_m"]]
    G = ds.X[idx][:, ci["logg"]]
    true = ds.Y[idx]
    results = {}
    persps = {}
    pixrms = {}     # per-pixel RMS flux residual over the test set, per arm

    def record(tag, flux):
        persp = per_spectrum_rmse(flux, true)
        persps[tag] = persp
        sse = np.zeros(flux.shape[1], np.float64)
        for s in range(0, flux.shape[0], 20_000):
            sse += ((flux[s:s + 20_000] - true[s:s + 20_000]) ** 2).sum(0, dtype=np.float64)
        pixrms[tag] = np.sqrt(sse / flux.shape[0]).astype("float32")
        results[tag] = {**summ(persp), "regimes": regime_stats(persp, T, C, G)}
        print(f"\n[{tag}] test: med={results[tag]['median']:.4e} "
              f"mean={results[tag]['mean']:.4e} p90={results[tag]['p90']:.4e} "
              f"p99={results[tag]['p99']:.4e} max={results[tag]['max']:.4e}", flush=True)
        print_regimes(tag, results[tag]["regimes"])

    gen_flux = predict_flux_host(gen_model, Xn[idx], ym, ys, args.eval_batch_size)
    record("baseline", gen_flux)
    if args.specialist:
        spec_model, _ = load_checkpoint(Path(args.specialist) / "best.pkl")
        spec_model.eval()
        spec_flux = predict_flux_host(spec_model, Xn[idx], ym, ys, args.eval_batch_size)
        w = gate_weight(T, args.gate_lo, args.gate_hi)[:, None]
        record("gated", w * spec_flux + (1.0 - w) * gen_flux)
        del spec_flux, spec_model
    del gen_model

    # --- Arms ---------------------------------------------------------------
    common = dict(batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                  cap=args.cap, reweight_every=args.reweight_every,
                  log_every=args.log_every, tr_bins=tr_bins, n_bins=int(teff_nodes.size),
                  pixel_in_flux=(args.pixel_space == "flux"), flux_power=args.flux_power)
    for arm in args.arms:
        if arm in ("scratch", "logteff"):
            ds_t = transform_teff(ds, args.teff_transform if arm == "logteff" else "none")
            Xn_t = ds_t.normalize_x(ds_t.X).astype("float32")
            Xtr_t = jax.device_put(Xn_t[ds_t.train_idx])
            Xva_t = jax.device_put(Xn_t[ds_t.val_idx])
            model = build_model(
                "mlp", ds_t.n_features, ds_t.n_pixels, seed=0,
                hidden=tuple(base.hidden), activation=base.activation,
                residual=base.residual, layernorm=base.layernorm,
                **({"dropout": base.dropout} if base.dropout else {}))
            # trial 77's own optimiser settings (from its run config)
            gcfg = json.loads((gen_dir / "config.json").read_text())
            lr, wd = float(gcfg["lr"]), float(gcfg["weight_decay"])
            best_model = run_arm(
                arm, model, ds_t, Xtr_t, Ytr16, Xva_t, Yva16, mode="control",
                epochs=args.scratch_epochs, lr=lr, weight_decay=wd,
                warmup_steps=500, outdir=run_dir / arm, base_args=base, **common)
            flux = predict_flux_host(best_model, Xn_t[idx], ym, ys, args.eval_batch_size)
            del Xtr_t, Xva_t, Xn_t
        else:
            model, _ = load_checkpoint(gen_dir / "best.pkl")
            best_model = run_arm(
                arm, model, ds, Xtr, Ytr16, Xva, Yva16, mode=arm,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                warmup_steps=300, outdir=run_dir / arm, base_args=base, **common)
            flux = predict_flux_host(best_model, Xn[idx], ym, ys, args.eval_batch_size)
        record(arm, flux)
        del model, best_model, flux
        # checkpoint the summary after every arm so a crash loses nothing
        (run_dir / "summary.json").write_text(json.dumps(
            {"generalist": str(gen_dir), "specialist": args.specialist,
             "args": vars(args), "results": results}, indent=2, default=str))
        np.savez(run_dir / "test_persp.npz", test_idx=idx, **persps)
        np.savez(run_dir / "test_pixrms.npz", y_std=ds.y_std, **pixrms)

    # --- Report ---------------------------------------------------------------
    print("\n================ RESIDUAL-WEIGHTED POST-TRAINING  (test set) ================")
    print(f"{'':10s} {'median':>11s} {'mean':>11s} {'p90':>11s} {'p99':>11s} {'max':>11s}"
          f"   {'cool med':>10s} {'cool p99':>10s}   (cool = Teff<=4000)")
    for tag, r in results.items():
        cool = r["regimes"].get("Teff<=4000", {})
        print(f"{tag:10s} {r['median']:11.4e} {r['mean']:11.4e} {r['p90']:11.4e} "
              f"{r['p99']:11.4e} {r['max']:11.4e}   {cool.get('median', np.nan):10.4e} "
              f"{cool.get('p99', np.nan):10.4e}")
    base_r = results["baseline"]
    print("\nrelative to baseline (ratio; <1 is better):")
    for tag, r in results.items():
        if tag == "baseline":
            continue
        cool = r["regimes"].get("Teff<=4000", {}); bcool = base_r["regimes"]["Teff<=4000"]
        print(f"  {tag:10s} median x{r['median']/base_r['median']:.3f}  "
              f"p90 x{r['p90']/base_r['p90']:.3f}  p99 x{r['p99']/base_r['p99']:.3f}  "
              f"cool-med x{cool.get('median', np.nan)/bcool['median']:.3f}  "
              f"cool-p99 x{cool.get('p99', np.nan)/bcool['p99']:.3f}")

    plot_summary(persps, results, plotdir / "reweight_summary.png")
    plot_pixel_profiles(pixrms, ds.y_std, plotdir / "pixel_profiles.png")
    print(f"\n[rw] done -> {run_dir}  figures -> {plotdir/'reweight_summary.png'}, "
          f"{plotdir/'pixel_profiles.png'}", flush=True)


def plot_pixel_profiles(pixrms, y_std, path):
    """Per-pixel test RMS residual per arm (top) and its ratio to baseline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   constrained_layout=True)
    base = pixrms["baseline"]
    for i, (tag, r) in enumerate(pixrms.items()):
        ax1.semilogy(r, lw=0.6, label=tag, color=f"C{i}",
                     alpha=1.0 if tag == "baseline" else 0.8)
        if tag != "baseline":
            ax2.plot(r / base, lw=0.6, label=tag, color=f"C{i}", alpha=0.8)
    ax1b = ax1.twinx()
    ax1b.semilogy(y_std, lw=0.4, color="k", alpha=0.25)
    ax1b.set_ylabel("y_std (grey)")
    ax1.set_ylabel("per-pixel RMS flux residual (test)"); ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(1.0, color="k", lw=0.8); ax2.set_ylabel("ratio to baseline")
    ax2.set_xlabel("pixel index"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_summary(persps, results, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    color = {tag: f"C{i}" for i, tag in enumerate(results)}  # same colour in both panels
    for tag, persp in persps.items():
        srt = np.sort(persp)
        tail = 1.0 - np.arange(1, srt.size + 1) / srt.size
        lw = 2.2 if tag == "baseline" else 1.4
        ax1.plot(srt, tail, lw=lw, label=tag, color=color[tag])
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("per-spectrum flux RMSE"); ax1.set_ylabel("fraction of test spectra above")
    ax1.set_title("Test error tail (lower-left is better)"); ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    base = results["baseline"]
    tags = [t for t in results if t != "baseline"]
    metrics = [("median", lambda r: r["median"]), ("p90", lambda r: r["p90"]),
               ("p99", lambda r: r["p99"]),
               ("cool median", lambda r: r["regimes"]["Teff<=4000"]["median"]),
               ("cool p99", lambda r: r["regimes"]["Teff<=4000"]["p99"]),
               ("hot median", lambda r: r["regimes"]["Teff>4800 (untouched by gate)"]["median"])]
    x = np.arange(len(metrics)); wbar = 0.8 / max(1, len(tags))
    for i, t in enumerate(tags):
        vals = [f(results[t]) / f(base) for _, f in metrics]
        ax2.bar(x + (i - (len(tags) - 1) / 2) * wbar, vals, wbar, label=t, color=color[t])
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([m for m, _ in metrics], rotation=20)
    ax2.set_ylabel("ratio to baseline generalist"); ax2.set_title("Relative test error (<1 better)")
    ax2.grid(alpha=0.3, axis="y"); ax2.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
