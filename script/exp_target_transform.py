"""Experiment: does a target reparameterisation shrink the cool-star tail?

Motivation
----------
The winning Payne emulator's error is not spread evenly -- it concentrates in
cool stars (Teff <~ 4000 K), whose deep TiO/CO molecular bands drive the
continuum-normalised flux toward zero. The training target is **per-pixel
standardised**: each pixel's residual is divided by that pixel's across-star
std. Deep-band pixels swing from ~1 (hot stars) to ~0.1 (cool carbon stars),
so they carry a *large* std and are therefore *down-weighted* in the MSE --
the model is allowed to be sloppy exactly where the hard cases live.

A monotone target transform ``t = f(flux)`` whose slope grows as flux -> 0
(log / optical-depth, or the gentler sqrt) makes a given deep-line flux error
map to a much larger error in ``t``-space, re-upweighting those pixels. This
script trains the **same fixed architecture** (the best HPO config so far)
under several transforms and compares them **in flux units** on the held-out
test set, so the numbers are directly comparable to the untransformed baseline.

What it does per transform
--------------------------
* build a transformed target ``t = fwd(flux)``, standardise it on the *train*
  split, and train via :func:`boss_emulator.training.train` (which trains on
  whatever standardised target it is handed -- no pipeline changes needed);
* reload ``best.pkl``, predict the test set, invert the transform back to flux,
  and report per-spectrum flux RMSE (median / p90 / p99 / max), R^2, and the
  cool-star tail (Teff <= 4000);
* save the 10 worst flux reconstructions for each transform.

A ``none`` transform is included as an in-run control (it reproduces the
standard pipeline), so the comparison is like-for-like on one machine/seed.

Example
-------
    PYTHONUNBUFFERED=1 BOSS_CACHE_DIR=/export/data/vgiusepp/BOSS_emulator/preprocessed_data \\
        .venv/bin/python script/exp_target_transform.py \\
        --transforms none sqrt log --epochs 1000
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint

EPS = 1e-3          # floor for log (deep lines never truly reach 0 flux)
MIN_FLUX = 1e-2     # below this true flux the % residual is meaningless


# =========================================================================
# Target transforms: (name -> (forward(flux)->t, inverse(t)->flux))
# forward runs on numpy host arrays; inverse runs on jnp device arrays.
# =========================================================================

TRANSFORMS = {
    "none": (lambda f: f,
             lambda t: t),
    "sqrt": (lambda f: np.sqrt(np.clip(f, 0.0, None)),
             lambda t: jnp.square(t)),
    "log":  (lambda f: np.log(np.clip(f, EPS, None)),
             lambda t: jnp.exp(t)),
}


def _per_pixel_stats(t: np.ndarray, train_idx: np.ndarray):
    """Per-pixel mean/std of the transformed target on the train split."""
    m = t[train_idx].mean(axis=0)
    s = t[train_idx].std(axis=0)
    s[s == 0] = 1.0
    return m.astype("float32"), s.astype("float32")


# =========================================================================
# Flux-space evaluation of a trained transform run
# =========================================================================

def flux_eval(run_dir: Path, ds, Xn, Yflux, test_idx, inverse, batch_size):
    """Predict the test set, invert to flux, return (metrics, persp, preds).

    ``ds.y_mean``/``ds.y_std`` are the *transform-space* stats stored in the
    checkpoint, so ``pred_t = predn * y_std + y_mean`` recovers the transformed
    target and ``inverse(pred_t)`` recovers flux.
    """
    model, payload = load_checkpoint(run_dir / "best.pkl")
    model.eval()          # deterministic: disable dropout for prediction
    y_mean = jnp.asarray(np.asarray(payload["y_mean"], np.float32))
    y_std = jnp.asarray(np.asarray(payload["y_std"], np.float32))
    n_pix = Yflux.shape[1]

    persp = np.empty(test_idx.size, np.float32)
    ss_res = 0.0
    sum_abs = 0.0
    # keep a small pool of predictions only for the worst-case panels
    write = 0
    for batch in data_mod.iter_batches(test_idx, batch_size, shuffle=False):
        predn = model(jnp.asarray(Xn[batch]))
        pred_flux = np.asarray(inverse(predn * y_std + y_mean))
        d = pred_flux - Yflux[batch]
        ss_res += float((d ** 2).sum())
        sum_abs += float(np.abs(d).sum())
        persp[write:write + batch.size] = np.sqrt((d ** 2).mean(axis=1))
        write += batch.size

    ybar = Yflux[test_idx].mean(axis=0)
    ss_tot = float(((Yflux[test_idx] - ybar) ** 2).sum())
    n = test_idx.size
    metrics = {
        "n_test": int(n),
        "r2": 1.0 - ss_res / ss_tot,
        "per_spectrum_rmse_median": float(np.median(persp)),
        "per_spectrum_rmse_mean": float(persp.mean()),
        "per_spectrum_rmse_p90": float(np.percentile(persp, 90)),
        "per_spectrum_rmse_p99": float(np.percentile(persp, 99)),
        "per_spectrum_rmse_max": float(persp.max()),
        "mae_mean": sum_abs / (n * n_pix),
    }
    return metrics, persp, (model, y_mean, y_std)


def tail_stats(ds, test_idx, persp):
    feat = ds.feature_names
    T = ds.X[test_idx][:, feat.index("Teff")]
    C = ds.X[test_idx][:, feat.index("carbon_m")]
    out = {}
    for name, m in [("Teff<=4000", T <= 4000),
                    ("Teff<=4000 & C>=0.5", (T <= 4000) & (C >= 0.5))]:
        if m.any():
            out[name] = {"frac": float(m.mean()),
                         "median": float(np.median(persp[m])),
                         "p99": float(np.percentile(persp[m], 99)),
                         "max": float(persp[m].max())}
    return out


def plot_worst(run_dir_tag, ds, Xn, Yflux, test_idx, persp, handles,
               inverse, path, n=10):
    """The n highest-flux-RMSE test spectra, true vs emulator + resid strip."""
    model, y_mean, y_std = handles
    worst = np.argsort(persp)[::-1][:n]
    pix = np.arange(ds.n_pixels)
    feat = ds.feature_names
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(14, 3.4 * n), sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n}, constrained_layout=True)
    for i, loc in enumerate(worst):
        ax, rax = axes[2 * i], axes[2 * i + 1]
        g = int(test_idx[loc])
        predn = model(jnp.asarray(Xn[g:g + 1]))
        pred = np.asarray(inverse(predn * y_std + y_mean))[0]
        true = Yflux[g]
        lab = "  ".join(f"{feat[k]}={ds.X[g][k]:.2f}" for k in range(6))
        ax.plot(pix, true, lw=0.6, color="k", label="Korg (true)")
        ax.plot(pix, pred, lw=0.6, color="C3", alpha=0.8, label="emulator")
        ax.set_ylabel("norm flux")
        ax.set_title(f"[worst #{i + 1}] RMSE={persp[loc]:.4f}   {lab}",
                     fontsize=9, loc="left")
        ax.legend(loc="lower right", fontsize=8)
        denom = np.where(np.abs(true) < MIN_FLUX, np.nan, true)
        pct = 100.0 * (pred - true) / denom
        rax.axhline(0.0, color="k", lw=0.6)
        rax.plot(pix, pct, lw=0.5, color="C0")
        lim = max(np.nanpercentile(np.abs(pct), 99.5), 0.05)
        rax.set_ylim(-lim, lim)
        rax.set_ylabel("resid [%]", fontsize=8)
        rax.tick_params(labelsize=7)
    axes[-1].set_xlabel("pixel")
    fig.suptitle(f"{run_dir_tag}: 10 worst test reconstructions (flux space)",
                 fontsize=12)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[exp] wrote", path, flush=True)


# =========================================================================
# Main
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transforms", nargs="+", default=["none", "sqrt", "log"],
                   choices=list(TRANSFORMS))
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4096, dest="batch_size")
    p.add_argument("--outdir", default="trained_model/exp_transform")
    p.add_argument("--eval-batch-size", type=int, default=8192,
                   dest="eval_batch_size")
    p.add_argument("--n-worst", type=int, default=10, dest="n_worst")
    p.add_argument("--max-rows", type=int, default=None, dest="max_rows",
                   help="use only N rows (fast end-to-end smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    training.claim_gpu()

    # Fixed architecture = best categorical-width HPO trial (#52), abundance-free.
    base = training.build_parser().parse_args([])
    base.model = "payne"
    base.labels = list(data_mod.CORE_LABELS)
    base.abundances = []                      # 6 stellar params only
    base.val_frac = 0.1
    base.test_frac = 0.1
    base.seed = 0
    base.max_rows = args.max_rows
    base.epochs = args.epochs
    base.batch_size = args.batch_size
    base.ckpt_every = 0
    base.hidden = [2048] * 5
    base.activation = "gelu"
    base.residual = False
    base.layernorm = False
    base.dropout = 0.0031378704241138977
    base.lr = 0.00017976797708267256
    base.weight_decay = 6.749396383571241e-06

    # Load flux ONCE; every transform reuses the same split and inputs.
    ds, Xn, Yn_flux = training.prepare_data(base)
    Yflux = ds.Y                               # physical flux, ground truth
    del Yn_flux                                # rebuilt per transform below

    exp_root = Path(args.outdir)
    exp_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in args.transforms:
        fwd, inverse = TRANSFORMS[name]
        print(f"\n========== transform: {name} ==========", flush=True)
        t = fwd(Yflux).astype("float32")
        t_mean, t_std = _per_pixel_stats(t, ds.train_idx)
        Yn_t = ((t - t_mean) / t_std).astype("float32")
        ds_t = dataclasses.replace(ds, Y=t, y_mean=t_mean, y_std=t_std)

        a = copy.deepcopy(base)
        a.outdir = str(exp_root / name)
        run_dir, _ds, best_obj_tspace = training.train(
            a, prepared=(ds_t, Xn, Yn_t))

        metrics, persp, handles = flux_eval(
            run_dir, ds, Xn, Yflux, ds.test_idx, inverse, args.eval_batch_size)
        metrics["tail"] = tail_stats(ds, ds.test_idx, persp)
        metrics["val_medRMSE_transform_space"] = float(best_obj_tspace)
        metrics["run_dir"] = str(run_dir)
        (run_dir / "flux_eval.json").write_text(json.dumps(metrics, indent=2))
        summary[name] = metrics

        plot_worst(f"transform={name}", ds, Xn, Yflux, ds.test_idx, persp,
                   handles, inverse, exp_root / f"{name}_worst{args.n_worst}.png",
                   n=args.n_worst)

        m = metrics
        print(f"[exp] {name:5s}  fluxMed={m['per_spectrum_rmse_median']:.4e}  "
              f"p99={m['per_spectrum_rmse_p99']:.4e}  "
              f"max={m['per_spectrum_rmse_max']:.4e}  R2={m['r2']:.6f}",
              flush=True)
        del t, Yn_t, ds_t

    # Comparison table.
    (exp_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n================ TARGET-TRANSFORM COMPARISON ================")
    print(f"{'transform':10s} {'flux_med':>11s} {'flux_p90':>11s} "
          f"{'flux_p99':>11s} {'flux_max':>11s} {'R2':>9s} "
          f"{'tail_med':>10s} {'tail_p99':>10s}")
    for name, m in summary.items():
        tl = m["tail"].get("Teff<=4000", {})
        print(f"{name:10s} {m['per_spectrum_rmse_median']:11.4e} "
              f"{m['per_spectrum_rmse_p90']:11.4e} "
              f"{m['per_spectrum_rmse_p99']:11.4e} "
              f"{m['per_spectrum_rmse_max']:11.4e} {m['r2']:9.5f} "
              f"{tl.get('median', float('nan')):10.4e} "
              f"{tl.get('p99', float('nan')):10.4e}")


if __name__ == "__main__":
    main()
