"""Fine-resolution Teff-carbon_m error map for the current-best Payne (trial 16).

Reconstructs the held-out test set with trial #16's best.pkl and maps the
per-spectrum RMSE over the (Teff, carbon_m) plane at high resolution, so the
exact region to target for a retrain can be read off directly.

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

RUN_DIR = Path("trained_model/payne_hpo/trial_0016/payne_1784971665")
OUTDIR = Path("trained_model/payne_hpo/eval_best_trial16")
OUTDIR.mkdir(parents=True, exist_ok=True)
NT, NC = 40, 24                      # Teff / carbon_m bins


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
    T = ds.X[test_idx][:, feat.index("Teff")]
    C = ds.X[test_idx][:, feat.index("carbon_m")]
    np.savez(OUTDIR / "teff_carbon_map_data.npz", persp=persp, Teff=T, carbon_m=C)
    print("[map] trial16 test: median=%.4f p99=%.4f max=%.4f"
          % (np.median(persp), np.percentile(persp, 99), persp.max()))

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
            if m.sum():
                med[b, a] = np.median(persp[m])
                p99[b, a] = np.percentile(persp[m], 99)

    ext = [te[0], te[-1], ce[0], ce[-1]]
    vmin = max(np.nanpercentile(med, 2), 1e-4)
    vmax = np.nanpercentile(p99, 98)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
    for ax, M, title, cmap, norm in [
        (axes[0], med, "median RMSE", "viridis",
         LogNorm(vmin=vmin, vmax=vmax)),
        (axes[1], p99, "p99 RMSE", "inferno",
         LogNorm(vmin=vmin, vmax=vmax)),
        (axes[2], np.where(cnt > 0, cnt, np.nan), "count", "cividis", None)]:
        im = ax.imshow(M, origin="lower", aspect="auto", extent=ext,
                       cmap=cmap, norm=norm)
        ax.set_xlabel("Teff [K]"); ax.set_ylabel("carbon_m")
        ax.set_title(f"trial 16 test-set {title}")
        # mark the proposed retrain region: Teff<=4000 (all carbon)
        ax.axvline(4000, color="w", ls="--", lw=1.2)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].text(3550, ce[-1] - 0.15, "retrain\nregion", color="w",
                 fontsize=9, ha="center", va="top")
    fig.suptitle("Current-best Payne (trial 16): reconstruction error over "
                 "(Teff, carbon_m)", fontsize=13)
    fig.savefig(OUTDIR / "teff_carbon_map.png", dpi=125, bbox_inches="tight")
    print("[map] wrote", OUTDIR / "teff_carbon_map.png")

    # Quantify a few candidate retrain cuts.
    print("\n[map] candidate retrain regions (test-set fraction & error):")
    n = persp.size
    for name, mask in [
        ("Teff<=4000", T <= 4000),
        ("Teff<=4000 & carbon_m>=0.5", (T <= 4000) & (C >= 0.5)),
        ("Teff<=3500", T <= 3500),
    ]:
        frac = mask.mean()
        print(f"  {name:32s} frac={frac:6.2%}  "
              f"median={np.median(persp[mask]):.4f}  "
              f"p99={np.percentile(persp[mask],99):.4f}  "
              f"max={persp[mask].max():.4f}")


if __name__ == "__main__":
    main()
