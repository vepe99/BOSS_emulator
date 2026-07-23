"""Diagnose the reconstruction error tail of a trained BOSS emulator.

Model-agnostic: works for any run directory produced by ``training.py``
(payne, deeponet, ...), because it drives the model purely through
``load_checkpoint`` + ``predict`` and reads the split from ``split_idx.npz``.
Reads ONLY the held-out test rows from the cache (via ``split_idx.npz``),
so it never triggers the full in-memory dataset load.

For a given run it will:

  * correlate per-spectrum RMSE with every input label (Spearman), and
    tabulate how the worst-N spectra differ from the test set overall;
  * plot the worst-N true-vs-predicted spectra with labels annotated,
    both full-range and zoomed on the band that dominates their error.

Outputs (written into the run dir):
    tail_worst<N>.json        labels + rmse + cache rows of the worst-N
    tail_worst_full.png       worst-N overlays, full pixel range
    tail_worst_zoom.png       worst-N overlays, zoomed on the worst band
    tail_correlations.json    Spearman corr of RMSE vs each label

Usage (from the project root)::

    .venv/bin/python script/tail_diagnostics.py \
        --run trained_model/payne_1784824855

    # after a deeponet run, same command with its run dir:
    .venv/bin/python script/tail_diagnostics.py \
        --run trained_model/deeponet_1700000000 --n-plot 16

Importing ``boss_emulator.predict`` triggers ``autocvd(num_gpus=1)`` to
grab a free GPU before JAX initialises (set ``JAX_PLATFORMS=cpu`` for a
CPU-only run).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from boss_emulator import data as data_mod
from boss_emulator.emulator import load_checkpoint
from boss_emulator.predict import predict


# =========================================================================
# Load test-split predictions (model-agnostic)
# =========================================================================

def load_test_predictions(run: Path, checkpoint: str, batch_size: int):
    """Return ``(feat, X, Y_true, Y_pred, test_rows)`` for the held-out
    test split of ``run``, reading only those rows from the cache."""
    model, payload = load_checkpoint(run / f"{checkpoint}.pkl")
    feat = payload["feature_names"]
    cfg = payload.get("data_config")
    if cfg is None:
        raise SystemExit(
            f"{run}/{checkpoint}.pkl has no data_config; retrain to embed it.")

    split_path = run / "split_idx.npz"
    if not split_path.exists():
        raise SystemExit(f"{split_path} not found (needed for the test split).")
    test_rows = np.sort(np.load(split_path)["test_idx"])
    print(f"[tail] model={payload['model']}  features={len(feat)}  "
          f"test rows={test_rows.size:,}")

    with h5py.File(data_mod.CACHE_PATH, "r") as f:
        x_cols = [f[name][test_rows] for name in cfg["core_labels"]]
        if cfg["abundance_elements"]:
            ab = f["abundances"][test_rows, :]
            for e in cfg["abundance_elements"]:
                x_cols.append(ab[:, data_mod.ABUNDANCE_ELEMENTS.index(e)])
        X = np.stack(x_cols, axis=1).astype("float32")
        Y_true = f[cfg["target"]][test_rows, :].astype("float32")

    Y_pred = predict(model, payload, X, batch_size=batch_size)
    return feat, X, Y_true, Y_pred, test_rows


# =========================================================================
# Correlation of per-spectrum RMSE with labels
# =========================================================================

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype("float64")
    rb = np.argsort(np.argsort(b)).astype("float64")
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def correlate(feat, X, per_spec, run: Path, n_core: int) -> dict:
    """Print + save Spearman corr of RMSE with each label, and the
    worst-N-vs-overall label table for the core (non-abundance) labels."""
    print("\n[tail] Spearman rank corr( per-spec RMSE , label ):")
    corrs = {name: _spearman(per_spec, X[:, j]) for j, name in enumerate(feat)}
    for name, c in sorted(corrs.items(), key=lambda kv: -abs(kv[1])):
        print(f"        {name:10s} {c:+.3f}")

    tables = {}
    for N in (100, 1000):
        if N > per_spec.size:
            continue
        worst = np.argsort(per_spec)[-N:]
        thr = per_spec[worst].min()
        print(f"\n[tail] worst {N} spectra (RMSE >= {thr:.4e}):")
        print(f"        {'label':10s} {'worst_med':>12s} {'all_med':>12s} "
              f"{'worst_min':>10s} {'worst_max':>10s} "
              f"{'all_min':>10s} {'all_max':>10s}")
        tbl = {}
        for j in range(min(n_core, len(feat))):
            name = feat[j]
            w, a = X[worst, j], X[:, j]
            print(f"        {name:10s} {np.median(w):12.3f} {np.median(a):12.3f} "
                  f"{w.min():10.3f} {w.max():10.3f} {a.min():10.3f} {a.max():10.3f}")
            tbl[name] = dict(worst_median=float(np.median(w)),
                             all_median=float(np.median(a)),
                             worst_min=float(w.min()), worst_max=float(w.max()))
        tables[f"worst{N}"] = tbl

    out = {"spearman": corrs, "worst_vs_all": tables}
    (run / "tail_correlations.json").write_text(json.dumps(out, indent=2))
    print(f"\n[tail] wrote {run/'tail_correlations.json'}")
    return out


# =========================================================================
# Worst-N spectra plots
# =========================================================================

def _worst_band(resid: np.ndarray, worst_rows: np.ndarray, half: int = 45):
    """Locate the contiguous pixel band that dominates the worst spectra:
    centred on the peak per-pixel RMSE of the worst set."""
    pp = np.sqrt((resid[worst_rows] ** 2).mean(axis=0))
    c = int(pp.argmax())
    n_pix = resid.shape[1]
    return max(0, c - half), min(n_pix, c + half), c


def plot_worst(feat, X, Y_true, Y_pred, per_spec, test_rows,
               run: Path, n_plot: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resid = Y_pred - Y_true
    n_pix = Y_true.shape[1]
    pix = np.arange(n_pix)
    li = {n: i for i, n in enumerate(feat)}

    worst = np.argsort(per_spec)[::-1][:n_plot]
    b0, b1, bc = _worst_band(resid, worst)
    # Zoom window: pad the dominant band generously.
    zlo, zhi = max(0, bc - 150), min(n_pix, bc + 150)
    print(f"[tail] dominant band pixels [{b0}, {b1}] (peak {bc}); "
          f"zoom [{zlo}, {zhi}]")

    # Pick a compact, informative label set: always Teff/logg/m_H, plus any
    # carbon label present, plus vmic if present.
    show = [n for n in ("Teff", "logg", "m_H", "alpha_m", "carbon_m", "vmic")
            if n in li]

    def label_str(r, terse=False):
        keys = ("Teff", "logg", "carbon_m") if terse else show
        parts = []
        for k in keys:
            if k not in li:
                continue
            v = X[r, li[k]]
            parts.append(f"{k}={v:.0f}" if k == "Teff" else f"{k}={v:.2f}")
        s = "  ".join(parts)
        return s if terse else f"{s}   RMSE={per_spec[r]:.3f}"

    # ---- full-range overlays --------------------------------------------
    fig, axes = plt.subplots(n_plot, 1, figsize=(15, 1.9 * n_plot), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, worst):
        ax.plot(pix, Y_true[r], lw=0.5, color="#222222", label="true")
        ax.plot(pix, Y_pred[r], lw=0.5, color="#C44E52", alpha=0.85, label="pred")
        ax.axvspan(b0, b1, color="#4C72B0", alpha=0.12)
        ax.set_ylabel("flux", fontsize=8)
        ax.text(0.005, 0.04, label_str(r), transform=ax.transAxes, fontsize=8,
                va="bottom", family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("pixel index")
    axes[0].set_title(f"Worst-{n_plot} test spectra: true vs predicted "
                      f"(blue band = pixels {b0}-{b1})")
    fig.tight_layout()
    fig.savefig(run / "tail_worst_full.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- zoom on the dominant band --------------------------------------
    fig, axes = plt.subplots(n_plot, 1, figsize=(11, 1.6 * n_plot), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, worst):
        ax.plot(pix[zlo:zhi], Y_true[r, zlo:zhi], lw=0.9, color="#222222",
                label="true")
        ax.plot(pix[zlo:zhi], Y_pred[r, zlo:zhi], lw=0.9, color="#C44E52",
                alpha=0.9, label="pred")
        ax.axvspan(b0, b1, color="#4C72B0", alpha=0.12)
        ax.set_ylabel("flux", fontsize=8)
        ax.text(0.005, 0.05, label_str(r, terse=True), transform=ax.transAxes,
                fontsize=8, va="bottom", family="monospace")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("pixel index")
    axes[0].set_title(f"Worst-{n_plot} test spectra zoomed to pixels "
                      f"{zlo}-{zhi}")
    fig.tight_layout()
    fig.savefig(run / "tail_worst_zoom.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- persist the worst-N record -------------------------------------
    rec = {"feature_names": feat, "n_plot": int(n_plot),
           "dominant_band": [int(b0), int(b1)], "band_peak_pixel": int(bc),
           "worst_row_in_test": worst.tolist(),
           "orig_cache_row": test_rows[worst].tolist(),
           "rmse": per_spec[worst].tolist(),
           "labels": X[worst].tolist()}
    (run / f"tail_worst{n_plot}.json").write_text(json.dumps(rec))
    print(f"[tail] wrote {run/'tail_worst_full.png'}, "
          f"{run/'tail_worst_zoom.png'}, {run/f'tail_worst{n_plot}.json'}")


# =========================================================================
# CLI
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose the reconstruction error tail of a trained "
                    "emulator (model-agnostic).")
    p.add_argument("--run", required=True,
                   help="run directory under trained_model/ (any model)")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best")
    p.add_argument("--batch-size", type=int, default=8192, dest="batch_size")
    p.add_argument("--n-plot", type=int, default=12, dest="n_plot",
                   help="number of worst spectra to overlay")
    p.add_argument("--n-core", type=int, default=6, dest="n_core",
                   help="how many leading (core) labels to tabulate")
    p.add_argument("--no-correlate", action="store_true",
                   help="skip the RMSE-vs-label correlation report")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the worst-N spectra plots")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run = Path(args.run)
    if not run.is_dir():
        raise SystemExit(f"--run must be a run directory; got {run}")

    feat, X, Y_true, Y_pred, test_rows = load_test_predictions(
        run, args.checkpoint, args.batch_size)
    per_spec = np.sqrt(((Y_pred - Y_true) ** 2).mean(axis=1))
    print(f"[tail] per-spec RMSE  median={np.median(per_spec):.4e}  "
          f"p90={np.percentile(per_spec, 90):.4e}  "
          f"p99={np.percentile(per_spec, 99):.4e}  max={per_spec.max():.4e}")

    if not args.no_correlate:
        correlate(feat, X, per_spec, run, args.n_core)
    if not args.no_plot:
        plot_worst(feat, X, Y_true, Y_pred, per_spec, test_rows,
                   run, args.n_plot)


if __name__ == "__main__":
    main()
