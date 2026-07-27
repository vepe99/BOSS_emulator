"""Full evaluation suite for a trained Payne run (parametrized by run dir + tag).

Produces, into plots/payne_error_analysis/ with a ``<tag>_`` prefix:
  * <tag>_error_distribution.png
  * <tag>_sample_reconstructions.png
  * <tag>_worst_reconstructions.png
  * <tag>_corner_error.png
  * <tag>_teff_carbon_map.png
and writes reconstruction_stats_full.json (full scale-relative metrics) into
both the run dir and the plots dir.

    UV_CACHE_DIR=... uv run python script/eval_suite_payne.py RUN_DIR TAG

autocvd claims a free GPU at import (lab policy).
"""
from __future__ import annotations

import json
import sys
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

OUTDIR = Path("plots/payne_error_analysis")
NBIN = 24
NT, NC = 40, 24


def main(run_dir, tag):
    run_dir = Path(run_dir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    training.claim_gpu()

    base = training.build_parser().parse_args([])
    base.model = "payne"
    base.labels = list(data_mod.CORE_LABELS)
    base.abundances = list(data_mod.ABUNDANCE_ELEMENTS)
    base.val_frac = 0.1; base.test_frac = 0.1; base.seed = 0
    ds, Xn, Yn = training.prepare_data(base)

    sp = run_dir / "split_idx.npz"
    test_idx = np.load(sp)["test_idx"] if sp.exists() else ds.test_idx
    model, payload = load_checkpoint(run_dir / "best.pkl")
    y_mean = np.asarray(payload["y_mean"], np.float32)
    y_std = np.asarray(payload["y_std"], np.float32)

    # --- full metrics (+ per-spectrum RMSE) --------------------------------
    metrics, persp = training.full_reconstruction_metrics(
        model, Xn, ds.Y, Yn, test_idx, ds.y_std, ds.y_mean, 8192)
    for d in (run_dir, OUTDIR):
        (d / (f"reconstruction_stats_full.json" if d is run_dir
              else f"{tag}_reconstruction_stats_full.json")).write_text(
            json.dumps({"test_set": metrics}, indent=2))
    print(f"[{tag}] full metrics:", json.dumps(metrics), flush=True)

    feat = ds.feature_names
    pix = np.arange(ds.n_pixels)
    order = np.argsort(persp)
    T = ds.X[test_idx][:, feat.index("Teff")]
    C = ds.X[test_idx][:, feat.index("carbon_m")]

    def predict_row(g):
        yn = np.asarray(model(jnp.asarray(Xn[g:g + 1])))[0]
        return yn * y_std + y_mean, ds.Y[g]

    def lab(g):
        v = ds.X[g]
        return "  ".join(f"{feat[i]}={v[i]:.2f}" for i in range(6))

    def recon_panel(picks, title, fname):
        fig, axes = plt.subplots(len(picks), 1, figsize=(14, 3.0 * len(picks)),
                                 constrained_layout=True)
        for ax, (name, loc) in zip(axes, picks):
            g = int(test_idx[loc]); pred, true = predict_row(g)
            ax.plot(pix, true, lw=0.6, color="k", label="Korg (true)")
            ax.plot(pix, pred, lw=0.6, color="C3", alpha=0.8, label="emulator")
            ax.set_ylabel("norm flux")
            ax.set_title(f"[{name}] RMSE={persp[loc]:.4f}   {lab(g)}",
                         fontsize=9, loc="left")
            ax.legend(loc="lower right", fontsize=8)
        axes[-1].set_xlabel("pixel")
        fig.suptitle(title, fontsize=12)
        fig.savefig(OUTDIR / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[{tag}] wrote", OUTDIR / fname, flush=True)

    n = order.size - 1
    recon_panel([("best", order[0]), ("p25", order[n // 4]),
                 ("median", order[n // 2]), ("p90", order[int(0.9 * n)])],
                f"Payne ({tag}) test-set reconstructions",
                f"{tag}_sample_reconstructions.png")
    recon_panel([(f"worst #{i+1}", loc) for i, loc in enumerate(order[::-1][:4])],
                f"Payne ({tag}) worst test-set reconstructions",
                f"{tag}_worst_reconstructions.png")

    # --- error distribution ------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    a1.hist(persp, bins=120, color="C0", alpha=0.85)
    for q, c in [(50, "k"), (90, "C1"), (99, "C3")]:
        v = np.percentile(persp, q)
        a1.axvline(v, color=c, ls="--", lw=1, label=f"p{q}={v:.4f}")
    a1.set_xlabel("per-spectrum RMSE (norm flux)"); a1.set_ylabel("count")
    a1.set_title(f"{tag} test-set error distribution"); a1.legend(fontsize=8)
    a2.hist(persp, bins=np.logspace(np.log10(persp.min() + 1e-6),
                                    np.log10(persp.max()), 120),
            color="C0", alpha=0.85)
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("per-spectrum RMSE (log)"); a2.set_ylabel("count (log)")
    a2.set_title("Error distribution (log-log, tail)")
    fig.suptitle(f"Payne {tag}  n_test={persp.size:,}  "
                 f"median={metrics['per_spectrum_rmse_median']:.4f}  "
                 f"R2={metrics['r2']:.5f}", fontsize=12)
    fig.savefig(OUTDIR / f"{tag}_error_distribution.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] wrote", OUTDIR / f"{tag}_error_distribution.png", flush=True)

    # --- Teff-carbon_m map -------------------------------------------------
    te = np.linspace(T.min(), T.max(), NT + 1)
    ce = np.linspace(C.min(), C.max(), NC + 1)
    it = np.clip(np.digitize(T, te) - 1, 0, NT - 1)
    ic = np.clip(np.digitize(C, ce) - 1, 0, NC - 1)
    med = np.full((NC, NT), np.nan); p99 = np.full((NC, NT), np.nan)
    cnt = np.zeros((NC, NT))
    for a in range(NT):
        for b in range(NC):
            m = (it == a) & (ic == b); cnt[b, a] = m.sum()
            if m.sum():
                med[b, a] = np.median(persp[m])
                p99[b, a] = np.percentile(persp[m], 99)
    ext = [te[0], te[-1], ce[0], ce[-1]]
    vmin = max(np.nanpercentile(med, 2), 1e-4); vmax = np.nanpercentile(p99, 98)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
    for ax, M, ttl, cmap, norm in [
        (axes[0], med, "median RMSE", "viridis", LogNorm(vmin=vmin, vmax=vmax)),
        (axes[1], p99, "p99 RMSE", "inferno", LogNorm(vmin=vmin, vmax=vmax)),
        (axes[2], np.where(cnt > 0, cnt, np.nan), "count", "cividis", None)]:
        im = ax.imshow(M, origin="lower", aspect="auto", extent=ext, cmap=cmap,
                       norm=norm)
        ax.set_xlabel("Teff [K]"); ax.set_ylabel("carbon_m")
        ax.set_title(f"{tag} test-set {ttl}"); ax.axvline(4000, color="w",
                                                          ls="--", lw=1.2)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Payne {tag}: reconstruction error over (Teff, carbon_m)",
                 fontsize=13)
    fig.savefig(OUTDIR / f"{tag}_teff_carbon_map.png", dpi=125,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] wrote", OUTDIR / f"{tag}_teff_carbon_map.png", flush=True)

    # --- retrain-region summary -------------------------------------------
    for name, mask in [("Teff<=4000", T <= 4000),
                       ("Teff<=4000 & carbon>=0.5", (T <= 4000) & (C >= 0.5)),
                       ("Teff<=3500", T <= 3500)]:
        print(f"[{tag}] {name:28s} frac={mask.mean():6.2%} "
              f"med={np.median(persp[mask]):.4f} "
              f"p99={np.percentile(persp[mask],99):.4f} "
              f"max={persp[mask].max():.4f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: eval_suite_payne.py RUN_DIR TAG")
    main(sys.argv[1], sys.argv[2])
