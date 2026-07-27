"""Evaluation suite for trained Payne runs.

One entry point for every test-set diagnostic in this project. Point it at one
or more run directories (each containing ``best.pkl`` and, ideally,
``split_idx.npz``); the ~38 GB dataset cache is loaded once and reused across
all of them.

Per run it writes ``reconstruction_stats_full.json`` (into the run dir and the
plots dir) and, unless ``--metrics-only``, five figures prefixed by the run's
tag:

  * ``<tag>_error_distribution.png``    -- per-spectrum RMSE histogram (lin+log)
  * ``<tag>_sample_reconstructions.png``-- best / p25 / median / p90 spectra
  * ``<tag>_worst_reconstructions.png`` -- the 4 highest-RMSE spectra
  * ``<tag>_teff_carbon_map.png``       -- error over the (Teff, carbon_m) plane
  * ``<tag>_corner_error.png``          -- error over the 6-D core-label space

Examples::

    # full suite for the best HPO trial
    UV_CACHE_DIR=... uv run python script/eval_payne.py \
        trained_model/payne_hpo/trial_0031/payne_* --tags trial31

    # just the numbers, for a batch of runs
    UV_CACHE_DIR=... uv run python script/eval_payne.py --metrics-only \
        trained_model/payne_hpo/trial_00{16,25,31}/payne_*

autocvd claims a free GPU before JAX initialises (lab policy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import jax.numpy as jnp

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint

NBIN = 24            # bins per axis in the label-space corner plot
NT, NC = 40, 24      # Teff / carbon_m bins in the 2-D error map
BATCH = 8192

# Candidate cuts reported as a text summary -- the regions a targeted retrain
# would plausibly cover.
REGIONS = [
    ("Teff<=4000", lambda T, C: T <= 4000),
    ("Teff<=4000 & carbon_m>=0.5", lambda T, C: (T <= 4000) & (C >= 0.5)),
    ("Teff<=3500", lambda T, C: T <= 3500),
]


# =========================================================================
# Figures
# =========================================================================

def _recon_panels(ax_specs, title, path, *, pix, persp, predict_row, lab,
                  test_idx):
    fig, axes = plt.subplots(len(ax_specs), 1,
                             figsize=(14, 3.0 * len(ax_specs)),
                             constrained_layout=True)
    for ax, (name, loc) in zip(np.atleast_1d(axes), ax_specs):
        g = int(test_idx[loc])
        pred, true = predict_row(g)
        ax.plot(pix, true, lw=0.6, color="k", label="Korg (true)")
        ax.plot(pix, pred, lw=0.6, color="C3", alpha=0.8, label="emulator")
        ax.set_ylabel("norm flux")
        ax.set_title(f"[{name}] RMSE={persp[loc]:.4f}   {lab(g)}",
                     fontsize=9, loc="left")
        ax.legend(loc="lower right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("pixel")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[eval] wrote", path, flush=True)


def _error_distribution(persp, metrics, tag, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2),
                                 constrained_layout=True)
    a1.hist(persp, bins=120, color="C0", alpha=0.85)
    for q, c in [(50, "k"), (90, "C1"), (99, "C3")]:
        v = np.percentile(persp, q)
        a1.axvline(v, color=c, ls="--", lw=1, label=f"p{q}={v:.4f}")
    a1.set_xlabel("per-spectrum RMSE (norm flux)")
    a1.set_ylabel("count")
    a1.set_title(f"{tag} test-set error distribution")
    a1.legend(fontsize=8)

    a2.hist(persp, bins=np.logspace(np.log10(persp.min() + 1e-6),
                                    np.log10(persp.max()), 120),
            color="C0", alpha=0.85)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("per-spectrum RMSE (log)")
    a2.set_ylabel("count (log)")
    a2.set_title("Error distribution (log-log, tail)")
    fig.suptitle(f"Payne {tag}  n_test={persp.size:,}  "
                 f"median={metrics['per_spectrum_rmse_median']:.4f}  "
                 f"R2={metrics['r2']:.5f}", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[eval] wrote", path, flush=True)


def _teff_carbon_map(T, C, persp, tag, path):
    """Median / p99 / count of per-spectrum RMSE over the (Teff, carbon_m) plane."""
    te = np.linspace(T.min(), T.max(), NT + 1)
    ce = np.linspace(C.min(), C.max(), NC + 1)
    it = np.clip(np.digitize(T, te) - 1, 0, NT - 1)
    ic = np.clip(np.digitize(C, ce) - 1, 0, NC - 1)

    med = np.full((NC, NT), np.nan)
    p99 = np.full((NC, NT), np.nan)
    cnt = np.zeros((NC, NT))
    for a in range(NT):
        for b in range(NC):
            m = (it == a) & (ic == b)
            cnt[b, a] = m.sum()
            if m.any():
                med[b, a] = np.median(persp[m])
                p99[b, a] = np.percentile(persp[m], 99)

    ext = [te[0], te[-1], ce[0], ce[-1]]
    vmin = max(np.nanpercentile(med, 2), 1e-4)
    vmax = np.nanpercentile(p99, 98)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
    for ax, M, ttl, cmap, norm in [
        (axes[0], med, "median RMSE", "viridis", LogNorm(vmin=vmin, vmax=vmax)),
        (axes[1], p99, "p99 RMSE", "inferno", LogNorm(vmin=vmin, vmax=vmax)),
        (axes[2], np.where(cnt > 0, cnt, np.nan), "count", "cividis", None),
    ]:
        im = ax.imshow(M, origin="lower", aspect="auto", extent=ext,
                       cmap=cmap, norm=norm)
        ax.set_xlabel("Teff [K]")
        ax.set_ylabel("carbon_m")
        ax.set_title(f"{tag} test-set {ttl}")
        ax.axvline(4000, color="w", ls="--", lw=1.2)   # proposed retrain cut
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Payne {tag}: reconstruction error over (Teff, carbon_m)",
                 fontsize=13)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("[eval] wrote", path, flush=True)


def _corner_error(labels, persp, tag, path):
    """Where in the 6-D core-label space the error lives.

    diagonal  : median (blue) & p99 (red) RMSE profiled vs each label
    lower tri : 2-D binned MEDIAN RMSE  (typical behaviour)
    upper tri : 2-D binned MAX RMSE     (tail locator)
    """
    core = list(labels)

    def edges(v):
        return np.linspace(v.min(), v.max(), NBIN + 1)

    def stat2d(x, y, ex, ey, fn):
        ix = np.clip(np.digitize(x, ex) - 1, 0, NBIN - 1)
        iy = np.clip(np.digitize(y, ey) - 1, 0, NBIN - 1)
        out = np.full((NBIN, NBIN), np.nan)
        flat = ix * NBIN + iy
        order = np.argsort(flat)
        fs, ps = flat[order], persp[order]
        uniq, start = np.unique(fs, return_index=True)
        for cell, seg in zip(uniq, np.split(ps, start[1:])):
            out[cell // NBIN, cell % NBIN] = fn(seg)
        return out

    E = {name: edges(labels[name]) for name in core}
    n = len(core)
    fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 3.0 * n))
    vmin = max(np.percentile(persp, 1), 1e-4)
    vmax = np.percentile(persp, 99.9)
    im = im2 = None

    for i, yi in enumerate(core):
        for j, xj in enumerate(core):
            ax = axes[i, j]
            if i == j:
                ex = E[xj]
                cen = 0.5 * (ex[:-1] + ex[1:])
                ixb = np.clip(np.digitize(labels[xj], ex) - 1, 0, NBIN - 1)
                med = np.array([np.median(persp[ixb == k]) if np.any(ixb == k)
                                else np.nan for k in range(NBIN)])
                p99 = np.array([np.percentile(persp[ixb == k], 99)
                                if np.any(ixb == k) else np.nan
                                for k in range(NBIN)])
                ax.plot(cen, med, color="C0", lw=1.5, label="median")
                ax.plot(cen, p99, color="C3", lw=1.2, ls="--", label="p99")
                ax.set_yscale("log")
                if i == 0:
                    ax.legend(fontsize=6, loc="upper right")
            elif i > j:
                H = stat2d(labels[xj], labels[yi], E[xj], E[yi], np.median)
                im = ax.imshow(H, origin="lower", aspect="auto",
                               extent=[E[xj][0], E[xj][-1],
                                       E[yi][0], E[yi][-1]],
                               cmap="viridis",
                               norm=LogNorm(vmin=vmin, vmax=vmax))
            else:
                H = stat2d(labels[xj], labels[yi], E[xj], E[yi], np.max)
                im2 = ax.imshow(H, origin="lower", aspect="auto",
                                extent=[E[xj][0], E[xj][-1],
                                        E[yi][0], E[yi][-1]],
                                cmap="inferno",
                                norm=LogNorm(vmin=vmin, vmax=persp.max()))
            if i == n - 1:
                ax.set_xlabel(xj, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(yi, fontsize=9)
            elif j != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Payne ({tag}) test-set RMSE in label space\n"
                 "lower = median RMSE per bin | upper = max RMSE per bin | "
                 "diagonal = median (blue) & p99 (red) vs label", fontsize=12)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    if im is not None:
        fig.colorbar(im, cax=fig.add_axes([0.13, -0.015, 0.32, 0.012]),
                     orientation="horizontal", label="median RMSE")
    if im2 is not None:
        fig.colorbar(im2, cax=fig.add_axes([0.55, -0.015, 0.32, 0.012]),
                     orientation="horizontal", label="max RMSE")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[eval] wrote", path, flush=True)


# =========================================================================
# Driver
# =========================================================================

def evaluate_run(run_dir: Path, tag: str, ds, Xn, Yn, outdir: Path,
                 metrics_only: bool) -> dict:
    sp = run_dir / "split_idx.npz"
    test_idx = np.load(sp)["test_idx"] if sp.exists() else ds.test_idx
    model, payload = load_checkpoint(run_dir / "best.pkl")
    y_mean = np.asarray(payload["y_mean"], np.float32)
    y_std = np.asarray(payload["y_std"], np.float32)

    metrics, persp = training.full_reconstruction_metrics(
        model, Xn, ds.Y, Yn, test_idx, ds.y_std, ds.y_mean, BATCH)
    (run_dir / "reconstruction_stats_full.json").write_text(
        json.dumps({"test_set": metrics}, indent=2))
    print(f"[eval] {tag}: R2={metrics['r2']:.6f} "
          f"std_rmse={metrics['standardized_rmse']:.5f} "
          f"med={metrics['per_spectrum_rmse_median']:.5f} "
          f"mae={metrics['mae_mean']:.5f} "
          f"snr~{metrics['snr_equivalent_median']:.0f}", flush=True)

    feat = ds.feature_names
    T = ds.X[test_idx][:, feat.index("Teff")]
    C = ds.X[test_idx][:, feat.index("carbon_m")]
    for name, cut in REGIONS:
        m = cut(T, C)
        print(f"[eval] {tag} {name:28s} frac={m.mean():6.2%} "
              f"med={np.median(persp[m]):.4f} "
              f"p99={np.percentile(persp[m], 99):.4f} "
              f"max={persp[m].max():.4f}", flush=True)

    if metrics_only:
        return metrics

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{tag}_reconstruction_stats_full.json").write_text(
        json.dumps({"test_set": metrics}, indent=2))
    np.savez(outdir / f"{tag}_error_data.npz", persp=persp,
             X_test=ds.X[test_idx], feature_names=np.array(feat))

    pix = np.arange(ds.n_pixels)
    order = np.argsort(persp)
    n = order.size - 1

    def predict_row(g):
        yn = np.asarray(model(jnp.asarray(Xn[g:g + 1])))[0]
        return yn * y_std + y_mean, ds.Y[g]

    def lab(g):
        v = ds.X[g]
        return "  ".join(f"{feat[i]}={v[i]:.2f}" for i in range(6))

    kw = dict(pix=pix, persp=persp, predict_row=predict_row, lab=lab,
              test_idx=test_idx)
    _recon_panels([("best", order[0]), ("p25", order[n // 4]),
                   ("median", order[n // 2]), ("p90", order[int(0.9 * n)])],
                  f"Payne ({tag}) test-set reconstructions",
                  outdir / f"{tag}_sample_reconstructions.png", **kw)
    _recon_panels([(f"worst #{i + 1}", loc)
                   for i, loc in enumerate(order[::-1][:4])],
                  f"Payne ({tag}) worst test-set reconstructions",
                  outdir / f"{tag}_worst_reconstructions.png", **kw)
    _error_distribution(persp, metrics, tag,
                        outdir / f"{tag}_error_distribution.png")
    _teff_carbon_map(T, C, persp, tag, outdir / f"{tag}_teff_carbon_map.png")
    _corner_error({name: ds.X[test_idx][:, feat.index(name)]
                   for name in data_mod.CORE_LABELS}, persp, tag,
                  outdir / f"{tag}_corner_error.png")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="run directories, each holding best.pkl (+ split_idx.npz)")
    p.add_argument("--tags", nargs="*", default=None,
                   help="figure-name prefix per run (default: run dir name)")
    p.add_argument("--metrics-only", action="store_true",
                   help="write reconstruction_stats_full.json only, no figures")
    p.add_argument("--outdir", type=Path, default=Path("plots/payne_error_analysis"),
                   help="where figures go")
    args = p.parse_args()

    tags = args.tags or [d.parent.name if d.name.startswith("payne_") else d.name
                         for d in args.run_dirs]
    if len(tags) != len(args.run_dirs):
        p.error(f"--tags has {len(tags)} entries for {len(args.run_dirs)} runs")

    training.claim_gpu()

    # Same data config as the HPO worker, so the split matches the trials.
    base = training.build_parser().parse_args([])
    base.model = "payne"
    base.labels = list(data_mod.CORE_LABELS)
    base.abundances = list(data_mod.ABUNDANCE_ELEMENTS)
    base.val_frac = 0.1
    base.test_frac = 0.1
    base.seed = 0
    ds, Xn, Yn = training.prepare_data(base)

    for run_dir, tag in zip(args.run_dirs, tags):
        evaluate_run(run_dir, tag, ds, Xn, Yn, args.outdir, args.metrics_only)


if __name__ == "__main__":
    main()
