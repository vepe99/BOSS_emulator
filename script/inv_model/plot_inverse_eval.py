"""Evaluation figures + summary for a full-test-set inversion.

Reads the ``<tag>_gradient_full.npz`` written by ``run_gradient_full.py`` (or
any ``save_results`` npz) and produces the inverse-model counterpart of
``eval_mlp.py``:

* ``<tag>_recovery_density.png``  truth vs recovered, 2D log-density per label
* ``<tag>_error_vs_teff.png``     median / p90 |error| per label as a function
                                  of true Teff, split by dwarfs / giants
* ``<tag>_error_maps.png``        median |error| per label over (Teff, logg)
* ``<tag>_chi2.png``              chi2/pixel distribution and chi2 vs Teff
* ``<tag>_summary.json``          per-label MAE / medAE / RMSE / p90 / bias,
                                  overall and per Teff regime

Example::

    .venv/bin/python script/inv_model/plot_inverse_eval.py \\
        plots/combined/inverse_full/combined_gradient_full.npz \\
        --outdir plots/combined/inverse_full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

UNITS = {"Teff": " [K]"}
REGIMES = {"all": None, "Teff<=4000": lambda T: T <= 4000,
           "4000<Teff<=6000": lambda T: (T > 4000) & (T <= 6000),
           "Teff>6000": lambda T: T > 6000}


def label_stats(d: np.ndarray) -> dict:
    a = np.abs(d)
    return {"bias": float(d.mean()), "mae": float(a.mean()),
            "median_abs_error": float(np.median(a)),
            "p90_abs_error": float(np.percentile(a, 90)),
            "p99_abs_error": float(np.percentile(a, 99)),
            "rmse": float(np.sqrt(np.mean(d ** 2))), "n": int(d.size)}


def plot_recovery_density(names, truth, est, title, path):
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.3 * nrow),
                             constrained_layout=True, squeeze=False)
    for j, nm in enumerate(names):
        ax = axes[divmod(j, ncol)]
        t, e = truth[:, j], est[:, j]
        lo, hi = float(t.min()), float(t.max())
        pad = 0.03 * (hi - lo) if hi > lo else 1.0
        lim = [lo - pad, hi + pad]
        h = ax.hist2d(t, e, bins=90, range=[lim, lim], density=True,
                      cmap="viridis", norm=LogNorm())
        ax.plot(lim, lim, color="w", lw=1.0, ls="--", alpha=0.9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        u = UNITS.get(nm, "")
        ax.set_xlabel(f"true {nm}{u}", fontsize=9)
        ax.set_ylabel(f"recovered {nm}{u}", fontsize=9)
        d = e - t
        ax.set_title(f"{nm}:  RMSE={np.sqrt(np.mean(d**2)):.3g}  "
                     f"MAE={np.mean(np.abs(d)):.3g}  bias={np.mean(d):+.2g}",
                     fontsize=9, loc="left")
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.02, label="density")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("[inv-eval] wrote", path, flush=True)


def binned(x, y, edges, q):
    idx = np.digitize(x, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        m = idx == b
        if m.sum() >= 20:
            out[b] = np.percentile(y[m], q)
    return out


def plot_error_vs_teff(names, truth, est, T, G, title, path):
    edges = np.linspace(T.min(), T.max(), 27)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.6 * nrow),
                             constrained_layout=True, squeeze=False)
    groups = [("dwarfs (logg>3.5)", G > 3.5, "#1f77b4"),
              ("giants (logg<=3.5)", G <= 3.5, "#d62728")]
    for j, nm in enumerate(names):
        ax = axes[divmod(j, ncol)]
        a = np.abs(est[:, j] - truth[:, j])
        for lab, m, c in groups:
            ax.plot(mid, binned(T[m], a[m], edges, 50), color=c, lw=2, label=f"{lab} median")
            ax.plot(mid, binned(T[m], a[m], edges, 90), color=c, lw=1.2, ls="--", label=f"{lab} p90")
        ax.set_yscale("log")
        ax.set_xlabel("true Teff [K]", fontsize=9)
        ax.set_ylabel(f"|error| {nm}{UNITS.get(nm, '')}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        if j == 0:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("[inv-eval] wrote", path, flush=True)


def plot_error_maps(names, truth, est, T, G, title, path):
    nb = 24
    te = np.linspace(T.min(), T.max(), nb + 1)
    ge = np.linspace(G.min(), G.max(), nb + 1)
    ti = np.clip(np.digitize(T, te) - 1, 0, nb - 1)
    gi = np.clip(np.digitize(G, ge) - 1, 0, nb - 1)
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.2 * nrow),
                             constrained_layout=True, squeeze=False)
    for j, nm in enumerate(names):
        ax = axes[divmod(j, ncol)]
        a = np.abs(est[:, j] - truth[:, j])
        med = np.full((nb, nb), np.nan)
        for gb in range(nb):
            for tb in range(nb):
                m = (gi == gb) & (ti == tb)
                if m.sum() >= 10:
                    med[gb, tb] = np.median(a[m])
        im = ax.pcolormesh(te, ge, med, cmap="viridis", norm=LogNorm(),
                           shading="flat")
        ax.set_xlabel("true Teff [K]", fontsize=9)
        ax.set_ylabel("true logg", fontsize=9)
        ax.set_title(f"{nm}: median |error|{UNITS.get(nm, '')}", fontsize=9,
                     loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("[inv-eval] wrote", path, flush=True)


def plot_chi2(chi2_pix, T, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    ax = axes[0]
    lo, hi = np.percentile(chi2_pix, [0.1, 99.9])
    # Log x only when the spread is wide; a narrow range on a log axis gets
    # unreadable overlapping minor-tick labels.
    wide = hi / max(lo, 1e-3) > 3
    bins = (np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 80) if wide
            else np.linspace(lo, hi, 80))
    ax.hist(chi2_pix, bins=bins, color="#1f77b4", alpha=0.85)
    if wide:
        ax.set_xscale("log")
    ax.set_yscale("log")
    for q, c in [(50, "k"), (90, "#ff7f0e"), (99, "#d62728")]:
        v = np.percentile(chi2_pix, q)
        ax.axvline(v, color=c, ls="--", lw=1, label=f"p{q}={v:.3g}")
    ax.set_xlabel("chi2 / pixel"); ax.set_ylabel("count"); ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    edges = np.linspace(T.min(), T.max(), 27); mid = 0.5 * (edges[1:] + edges[:-1])
    for q, c, ls in [(50, "#1f77b4", "-"), (90, "#1f77b4", "--"), (99, "#d62728", ":")]:
        ax.plot(mid, binned(T, chi2_pix, edges, q), color=c, ls=ls, lw=1.8, label=f"p{q}")
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("true Teff [K]"); ax.set_ylabel("chi2 / pixel")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("[inv-eval] wrote", path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", type=Path)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--n-pixels", type=int, default=4479, dest="n_pixels")
    args = p.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    names = [str(s) for s in z["fit_names"]]
    truth, est, chi2 = z["truth"], z["est"], z["chi2"]
    full = z["theta_true_full"]
    T = full[:, names.index("Teff")] if "Teff" in names else full[:, 0]
    G = full[:, names.index("logg")] if "logg" in names else full[:, 1]
    chi2_pix = chi2 / args.n_pixels
    tag = args.tag or args.npz.stem
    outdir = args.outdir or args.npz.parent
    outdir.mkdir(parents=True, exist_ok=True)
    n = truth.shape[0]
    title = f"inverse model, {tag}: N={n:,} test spectra"

    summary = {"n": n, "chi2_per_pixel": {f"p{q}": float(np.percentile(chi2_pix, q))
                                          for q in (50, 90, 99)},
               "regimes": {}}
    for rname, fn in REGIMES.items():
        m = np.ones(n, bool) if fn is None else fn(T)
        summary["regimes"][rname] = {
            "n": int(m.sum()),
            "labels": {nm: label_stats(est[m, j] - truth[m, j]) for j, nm in enumerate(names)}}
    (outdir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("[inv-eval] wrote", outdir / f"{tag}_summary.json")

    print(f"{'label':9s} {'bias':>9s} {'MAE':>9s} {'medAE':>9s} {'p90AE':>9s} {'RMSE':>9s}")
    for nm, s in summary["regimes"]["all"]["labels"].items():
        print(f"{nm:9s} {s['bias']:9.4f} {s['mae']:9.4f} {s['median_abs_error']:9.4f} "
              f"{s['p90_abs_error']:9.4f} {s['rmse']:9.4f}")

    plot_recovery_density(names, truth, est, title, outdir / f"{tag}_recovery_density.png")
    plot_error_vs_teff(names, truth, est, T, G, title, outdir / f"{tag}_error_vs_teff.png")
    plot_error_maps(names, truth, est, T, G, title, outdir / f"{tag}_error_maps.png")
    plot_chi2(chi2_pix, T, title, outdir / f"{tag}_chi2.png")


if __name__ == "__main__":
    main()
