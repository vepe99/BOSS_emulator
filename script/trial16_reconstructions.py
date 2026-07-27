"""Sample + worst test-set reconstruction curves for current-best Payne (trial 16).

Writes two figures into plots/payne_error_analysis/:
  * trial16_sample_reconstructions.png  (best / p25 / median / p90 spectra)
  * trial16_worst_reconstructions.png   (4 highest-RMSE spectra)

autocvd claims a free GPU at import (lab policy).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint

RUN_DIR = Path("trained_model/payne_hpo/trial_0016/payne_1784971665")
OUTDIR = Path("plots/payne_error_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)


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

    metrics, persp = training.full_reconstruction_metrics(
        model, Xn, ds.Y, Yn, test_idx, ds.y_std, ds.y_mean, 8192)
    import json
    for d in (OUTDIR, RUN_DIR):
        (d / "reconstruction_stats_full.json").write_text(
            json.dumps({"test_set": metrics}, indent=2))
    print("[recon] trial16 full metrics:", json.dumps(metrics))

    feat = ds.feature_names
    pix = np.arange(ds.n_pixels)
    order = np.argsort(persp)

    def predict_row(g):
        yn = np.asarray(model(jnp.asarray(Xn[g:g + 1])))[0]
        return yn * y_std + y_mean, ds.Y[g]

    def lab(g):
        v = ds.X[g]
        return "  ".join(f"{feat[i]}={v[i]:.2f}" for i in range(6))

    def panel(picks, title, fname):
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
        print("[recon] wrote", OUTDIR / fname)

    n = order.size - 1
    panel([("best", order[0]), ("p25", order[n // 4]),
           ("median", order[n // 2]), ("p90", order[int(0.9 * n)])],
          "Payne (trial 16) test-set reconstructions",
          "trial16_sample_reconstructions.png")
    panel([(f"worst #{i+1}", loc) for i, loc in enumerate(order[::-1][:4])],
          "Payne (trial 16) worst test-set reconstructions",
          "trial16_worst_reconstructions.png")


if __name__ == "__main__":
    main()
