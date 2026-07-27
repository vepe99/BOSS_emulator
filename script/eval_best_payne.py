"""Evaluate the best Payne HPO model on the test set and make diagnostic plots.

Loads trial #13's ``best.pkl``, reconstructs the held-out test spectra, and
writes three figures:
  1. sample reconstructions (representative test spectra: true vs predicted)
  2. distribution of per-spectrum reconstruction RMSE (physical flux units)
  3. the worst test-set reconstructions

Importing ``boss_emulator.training`` runs autocvd(num_gpus=1) -> claims a free
GPU before JAX initialises (lab policy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint

RUN_DIR = Path("trained_model/payne_hpo/trial_0013/payne_1784950098")
OUTDIR = Path("trained_model/payne_hpo/eval_best_trial13")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    training.claim_gpu()
    print("[eval] jax devices:", jax.devices())

    # --- data (same config as the HPO worker so the split is identical) ----
    base = training.build_parser().parse_args([])
    base.model = "payne"
    base.labels = list(data_mod.CORE_LABELS)
    base.abundances = list(data_mod.ABUNDANCE_ELEMENTS)
    base.val_frac = 0.1
    base.test_frac = 0.1
    base.seed = 0
    ds, Xn, Yn = training.prepare_data(base)

    # Prefer the split saved with the trial (authoritative); fall back to ds.
    split_p = RUN_DIR / "split_idx.npz"
    if split_p.exists():
        test_idx = np.load(split_p)["test_idx"]
        print(f"[eval] using saved split_idx.npz  n_test={test_idx.size:,}")
    else:
        test_idx = ds.test_idx
        print(f"[eval] using ds.test_idx  n_test={test_idx.size:,}")

    # --- model -------------------------------------------------------------
    model, payload = load_checkpoint(RUN_DIR / "best.pkl")
    y_mean = np.asarray(payload["y_mean"], dtype=np.float32)
    y_std = np.asarray(payload["y_std"], dtype=np.float32)
    n_pix = ds.n_pixels

    # --- predict test set in batches, physical flux units ------------------
    bs = 8192
    persp = np.empty(test_idx.size, dtype=np.float32)   # per-spectrum RMSE
    pred_store = {}                                      # cache preds we plot
    write = 0
    for start in range(0, test_idx.size, bs):
        batch = test_idx[start:start + bs]
        xb = jnp.asarray(Xn[batch])
        yn = np.asarray(model(xb))                      # standardized output
        pred = yn * y_std + y_mean                      # physical flux
        true = ds.Y[batch]                              # physical flux (target)
        rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
        persp[write:write + batch.size] = rmse
        write += batch.size
    print(f"[eval] per-spectrum RMSE: median={np.median(persp):.5f} "
          f"mean={persp.mean():.5f} p90={np.percentile(persp,90):.5f} "
          f"p99={np.percentile(persp,99):.5f} max={persp.max():.5f}")

    # Helper: recompute prediction for a single global row index.
    def predict_row(gidx):
        yn = np.asarray(model(jnp.asarray(Xn[gidx:gidx + 1])))[0]
        return yn * y_std + y_mean, ds.Y[gidx]

    order = np.argsort(persp)                # ascending error (local positions)
    pix = np.arange(n_pix)
    feat = ds.feature_names

    def label_str(gidx):
        vals = ds.X[gidx]
        core = [f"{feat[i]}={vals[i]:.2f}" for i in range(min(6, len(feat)))]
        return "  ".join(core)

    # ---------------------------------------------------------------------
    # Figure 1: representative reconstructions (best / p25 / median / p75)
    # ---------------------------------------------------------------------
    picks = {
        "best": order[0],
        "p25": order[int(0.25 * (order.size - 1))],
        "median": order[int(0.50 * (order.size - 1))],
        "p90": order[int(0.90 * (order.size - 1))],
    }
    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 3.0 * len(picks)),
                             constrained_layout=True)
    for ax, (name, loc) in zip(axes, picks.items()):
        g = int(test_idx[loc])
        pred, true = predict_row(g)
        ax.plot(pix, true, lw=0.6, color="k", label="Korg (true)")
        ax.plot(pix, pred, lw=0.6, color="C3", alpha=0.8, label="emulator")
        ax.set_ylabel("norm flux")
        ax.set_title(f"[{name}] RMSE={persp[loc]:.4f}   {label_str(g)}",
                     fontsize=9, loc="left")
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("pixel")
    fig.suptitle("Payne (trial 13) test-set reconstructions", fontsize=12)
    fig.savefig(OUTDIR / "sample_reconstructions.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figure 2: error distribution
    # ---------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    a1.hist(persp, bins=120, color="C0", alpha=0.85)
    for q, c in [(50, "k"), (90, "C1"), (99, "C3")]:
        v = np.percentile(persp, q)
        a1.axvline(v, color=c, ls="--", lw=1, label=f"p{q}={v:.4f}")
    a1.set_xlabel("per-spectrum RMSE (norm flux)")
    a1.set_ylabel("count")
    a1.set_title("Test-set error distribution")
    a1.legend(fontsize=8)

    a2.hist(persp, bins=np.logspace(np.log10(persp.min() + 1e-6),
                                    np.log10(persp.max()), 120),
            color="C0", alpha=0.85)
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("per-spectrum RMSE (norm flux, log)")
    a2.set_ylabel("count (log)")
    a2.set_title("Error distribution (log-log, tail)")
    fig.suptitle(f"Payne (trial 13)  n_test={test_idx.size:,}", fontsize=12)
    fig.savefig(OUTDIR / "error_distribution.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figure 3: worst reconstructions
    # ---------------------------------------------------------------------
    worst = order[::-1][:4]
    fig, axes = plt.subplots(len(worst), 1, figsize=(14, 3.0 * len(worst)),
                             constrained_layout=True)
    for rank, (ax, loc) in enumerate(zip(axes, worst), 1):
        g = int(test_idx[loc])
        pred, true = predict_row(g)
        ax.plot(pix, true, lw=0.6, color="k", label="Korg (true)")
        ax.plot(pix, pred, lw=0.6, color="C3", alpha=0.8, label="emulator")
        ax.set_ylabel("norm flux")
        ax.set_title(f"[worst #{rank}] RMSE={persp[loc]:.4f}   {label_str(g)}",
                     fontsize=9, loc="left")
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("pixel")
    fig.suptitle("Payne (trial 13) worst test-set reconstructions", fontsize=12)
    fig.savefig(OUTDIR / "worst_reconstructions.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Dump worst-sample summary
    # ---------------------------------------------------------------------
    summary = {
        "n_test": int(test_idx.size),
        "median": float(np.median(persp)),
        "mean": float(persp.mean()),
        "p90": float(np.percentile(persp, 90)),
        "p99": float(np.percentile(persp, 99)),
        "max": float(persp.max()),
        "worst10": [
            {"rank": i + 1, "global_idx": int(test_idx[loc]),
             "rmse": float(persp[loc]),
             "labels": {feat[j]: float(ds.X[int(test_idx[loc])][j])
                        for j in range(len(feat))}}
            for i, loc in enumerate(order[::-1][:10])
        ],
    }
    (OUTDIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote figures + summary to {OUTDIR}")


if __name__ == "__main__":
    main()
