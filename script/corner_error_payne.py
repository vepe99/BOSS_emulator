"""Corner plot of test-set reconstruction error in label space (best Payne).

Reconstructs the test set with trial #13's best.pkl, then shows WHERE in the
6-D core-label space the error concentrates:
  * diagonal : median & p99 per-spectrum RMSE profiled vs each label
  * lower tri: 2-D binned MEDIAN RMSE heatmap for each label pair
  * upper tri: 2-D binned MAX RMSE heatmap (tail locator)

autocvd claims a free GPU at import (lab policy).
"""
from __future__ import annotations

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

RUN_DIR = Path("trained_model/payne_hpo/trial_0013/payne_1784950098")
OUTDIR = Path("trained_model/payne_hpo/eval_best_trial13")
OUTDIR.mkdir(parents=True, exist_ok=True)
CORE = list(data_mod.CORE_LABELS)          # 6 core stellar labels
NBIN = 24


def main():
    training.claim_gpu()
    base = training.build_parser().parse_args([])
    base.model = "payne"
    base.labels = list(data_mod.CORE_LABELS)
    base.abundances = list(data_mod.ABUNDANCE_ELEMENTS)
    base.val_frac = 0.1; base.test_frac = 0.1; base.seed = 0
    ds, Xn, Yn = training.prepare_data(base)

    test_idx = np.load(RUN_DIR / "split_idx.npz")["test_idx"]
    model, payload = load_checkpoint(RUN_DIR / "best.pkl")
    y_mean = np.asarray(payload["y_mean"], np.float32)
    y_std = np.asarray(payload["y_std"], np.float32)

    persp = np.empty(test_idx.size, np.float32)
    w = 0
    for s in range(0, test_idx.size, 8192):
        b = test_idx[s:s + 8192]
        pred = np.asarray(model(jnp.asarray(Xn[b]))) * y_std + y_mean
        persp[w:w + b.size] = np.sqrt(np.mean((pred - ds.Y[b]) ** 2, axis=1))
        w += b.size

    feat = ds.feature_names
    col = {name: feat.index(name) for name in CORE}
    Xt = ds.X[test_idx]                       # (n_test, n_features)
    labels = {name: Xt[:, col[name]] for name in CORE}
    np.savez(OUTDIR / "corner_error_data.npz",
             persp=persp, X_test=Xt, feature_names=np.array(feat))
    print("[corner] persp median=%.4f p99=%.4f max=%.4f"
          % (np.median(persp), np.percentile(persp, 99), persp.max()))

    # ---- binned statistic helper -----------------------------------------
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
        segs = np.split(ps, start[1:])
        for cell, seg in zip(uniq, segs):
            out[cell // NBIN, cell % NBIN] = fn(seg)
        return out

    E = {name: edges(labels[name]) for name in CORE}
    n = len(CORE)
    fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 3.0 * n))
    vmin = max(np.percentile(persp, 1), 1e-4)
    vmax = np.percentile(persp, 99.9)

    for i, yi in enumerate(CORE):
        for j, xj in enumerate(CORE):
            ax = axes[i, j]
            if i == j:
                # diagonal: median & p99 RMSE vs this label
                ex = E[xj]; cen = 0.5 * (ex[:-1] + ex[1:])
                ixb = np.clip(np.digitize(labels[xj], ex) - 1, 0, NBIN - 1)
                med = np.array([np.median(persp[ixb == k]) if np.any(ixb == k)
                                else np.nan for k in range(NBIN)])
                p99 = np.array([np.percentile(persp[ixb == k], 99)
                                if np.any(ixb == k) else np.nan
                                for k in range(NBIN)])
                ax.plot(cen, med, color="C0", lw=1.5, label="median")
                ax.plot(cen, p99, color="C3", lw=1.2, ls="--", label="p99")
                ax.set_yscale("log")
                ax.tick_params(labelsize=7)
                if i == 0:
                    ax.legend(fontsize=6, loc="upper right")
            elif i > j:
                # lower: median RMSE heatmap
                H = stat2d(labels[xj], labels[yi], E[xj], E[yi], np.median)
                im = ax.imshow(H, origin="lower", aspect="auto",
                               extent=[E[xj][0], E[xj][-1], E[yi][0], E[yi][-1]],
                               cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
            else:
                # upper: MAX RMSE heatmap (tail locator)
                H = stat2d(labels[xj], labels[yi], E[xj], E[yi], np.max)
                im2 = ax.imshow(H, origin="lower", aspect="auto",
                                extent=[E[xj][0], E[xj][-1], E[yi][0], E[yi][-1]],
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

    fig.suptitle("Payne (trial 13) test-set RMSE in label space\n"
                 "lower = median RMSE per bin | upper = max RMSE per bin | "
                 "diagonal = median (blue) & p99 (red) vs label",
                 fontsize=12)
    cax1 = fig.add_axes([0.13, -0.015, 0.32, 0.012])
    fig.colorbar(im, cax=cax1, orientation="horizontal", label="median RMSE")
    cax2 = fig.add_axes([0.55, -0.015, 0.32, 0.012])
    fig.colorbar(im2, cax=cax2, orientation="horizontal", label="max RMSE")
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(OUTDIR / "corner_error.png", dpi=120, bbox_inches="tight")
    print("[corner] wrote", OUTDIR / "corner_error.png")


if __name__ == "__main__":
    main()
