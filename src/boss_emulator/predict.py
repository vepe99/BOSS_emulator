"""Run a trained BOSS emulator forward: labels -> predicted spectra.

Two modes, given a run directory produced by ``training.py`` (under
``trained_model/``):

1. ``--inputs FILE`` : predict the normalised-flux spectrum for each row of
   an arbitrary set of input labels and write them out.

2. ``--test`` : reconstruct the held-out **test** split (the rows never seen
   during training or validation), predict them, and report reconstruction
   metrics comparing predicted vs. true spectra -- i.e. how well the
   emulator generates spectra on unseen labels.

Optionally pass ``--specialist <run>`` to blend a cool-region specialist onto
the generalist (``--run``) with a smooth Teff gate
(``w*specialist + (1-w)*generalist``, ``w`` ramping from 1 below ``--gate-lo``
to 0 above ``--gate-hi``). This applies in both modes above. It cuts
cool-star (molecular-band) error by ~12-13% while leaving hot stars exactly as
the generalist predicts, and stays continuous in parameter space.

Usage (from the project root)::

    # predict for a user-supplied set of labels
    uv run python -m boss_emulator.predict \
        --run trained_model/payne_1700000000 --inputs my_labels.npy \
        --output predictions.h5 --plot 8

    # evaluate reconstruction metrics on the held-out test set
    uv run python -m boss_emulator.predict \
        --run trained_model/payne_1700000000 --test

Supported ``--inputs`` formats
------------------------------
* ``.npy``  : array of shape (n, n_features), columns already in the model's
  ``feature_names`` order.
* ``.csv``  : text with a header row; columns are matched to the model's
  ``feature_names`` by name (extra columns ignored, order-independent).
* ``.h5``   : either a single dataset ``X`` of shape (n, n_features), or one
  1-D dataset per feature name.
"""

from __future__ import annotations

# --- GPU selection MUST happen before importing jax (see training.py) ----
import os

if not (
    os.environ.get("BOSS_NO_AUTOCVD")
    or os.environ.get("CUDA_VISIBLE_DEVICES")
    or os.environ.get("JAX_PLATFORMS", "").lower() == "cpu"
):
    from autocvd import autocvd

    autocvd(num_gpus=1)

import argparse
import json
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from boss_emulator import data as data_mod
from boss_emulator.emulator import load_checkpoint


def resolve_checkpoint(run: str, which: str) -> Path:
    """Return the checkpoint path from a run dir or an explicit .pkl path."""
    p = Path(run)
    if p.is_file():
        return p
    ckpt = p / f"{which}.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"no checkpoint {ckpt}; pass a run dir containing "
            f"best.pkl/final.pkl or a direct .pkl path")
    return ckpt


def load_inputs(path: Path, feature_names: list[str]) -> np.ndarray:
    """Load inputs and return a (n, n_features) float32 array ordered to
    match ``feature_names``."""
    path = Path(path)
    suffix = path.suffix.lower()
    n_feat = len(feature_names)

    if suffix == ".npy":
        X = np.asarray(np.load(path), dtype="float32")
        if X.ndim != 2 or X.shape[1] != n_feat:
            raise ValueError(
                f".npy must have shape (n, {n_feat}) in feature order "
                f"{feature_names}; got {X.shape}")
        return X

    if suffix == ".csv":
        header = path.read_text().splitlines()[0].lstrip("#").strip()
        cols = [c.strip() for c in header.split(",")]
        raw = np.genfromtxt(path, delimiter=",", skip_header=1)
        raw = np.atleast_2d(raw)
        missing = [f for f in feature_names if f not in cols]
        if missing:
            raise ValueError(
                f"CSV header {cols} is missing required features {missing}")
        idx = [cols.index(f) for f in feature_names]
        return raw[:, idx].astype("float32")

    if suffix in (".h5", ".hdf5"):
        with h5py.File(path, "r") as fh:
            if "X" in fh:
                X = np.asarray(fh["X"][:], dtype="float32")
                if X.ndim != 2 or X.shape[1] != n_feat:
                    raise ValueError(
                        f"dataset 'X' must be (n, {n_feat}); got {X.shape}")
                return X
            missing = [name for name in feature_names if name not in fh]
            if missing:
                raise ValueError(
                    f"h5 file lacks dataset 'X' and per-feature datasets "
                    f"{missing}")
            return np.stack(
                [np.asarray(fh[name][:], dtype="float32")
                 for name in feature_names], axis=1)

    raise ValueError(f"unsupported input format {suffix!r}; "
                     "use .npy, .csv or .h5")


def predict(model, payload, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Predict de-normalised spectra for raw label array ``X``."""
    x_mean, x_std = payload["x_mean"], payload["x_std"]
    y_mean, y_std = payload["y_mean"], payload["y_std"]
    out = np.empty((X.shape[0], payload["n_pixels"]), dtype="float32")
    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        xn = (X[start:stop] - x_mean) / x_std
        yn = np.asarray(model(jnp.asarray(xn)))
        out[start:stop] = yn * y_std + y_mean
    return out


def gate_weight(teff: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smooth specialist weight per row: 1 for Teff<=lo, 0 for Teff>=hi, linear
    in between. A smooth ramp (not a hard switch) keeps the emulator continuous
    in parameter space, which matters for downstream gradient-based fitting."""
    return np.clip((hi - teff) / (hi - lo), 0.0, 1.0).astype("float32")


def predict_gated(gen, gpay, spec, spay, X: np.ndarray, *, lo: float, hi: float,
                  gate_feature: str, batch_size: int = 4096) -> np.ndarray:
    """Blend a frozen generalist with a cool-region specialist by a Teff gate.

    ``output = w*specialist + (1-w)*generalist`` with ``w = gate_weight(Teff)``.
    Each model de-normalises with its own stats and returns flux, so the blend
    in flux space is valid regardless of standardisation; only the feature
    layout and pixel count must agree so the label columns and outputs align.
    """
    if list(spay["feature_names"]) != list(gpay["feature_names"]):
        raise ValueError(
            f"specialist feature_names {spay['feature_names']} differ from "
            f"generalist {gpay['feature_names']}; cannot gate")
    if spay["n_pixels"] != gpay["n_pixels"]:
        raise ValueError("specialist/generalist n_pixels differ; cannot gate")
    if gate_feature not in gpay["feature_names"]:
        raise ValueError(
            f"gate feature {gate_feature!r} not in model features "
            f"{gpay['feature_names']}")
    ti = list(gpay["feature_names"]).index(gate_feature)
    w = gate_weight(X[:, ti], lo, hi)[:, None]
    gen_flux = predict(gen, gpay, X, batch_size)
    spec_flux = predict(spec, spay, X, batch_size)
    return w * spec_flux + (1.0 - w) * gen_flux


# =========================================================================
# Reconstruction metrics (test set)
# =========================================================================

def reconstruction_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute reconstruction metrics between predicted and true spectra.

    All quantities are in the (de-normalised) flux space. Per-spectrum RMSE
    is the RMSE over pixels within one spectrum; per-pixel RMSE is over
    spectra at one pixel.
    """
    resid = pred - true
    abs_resid = np.abs(resid)
    per_spec_rmse = np.sqrt((resid ** 2).mean(axis=1))
    per_pixel_rmse = np.sqrt((resid ** 2).mean(axis=0))

    # Per-pixel (per-wavelength) median absolute error: the median over
    # spectra of the absolute error at each pixel. Reduced across pixels
    # this gives a single, outlier-robust "typical approximation error at a
    # wavelength" figure.
    per_pixel_median_abs_error = np.median(abs_resid, axis=0)

    # Coefficient of determination pooled over all pixels/spectra.
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n_spectra": int(pred.shape[0]),
        "n_pixels": int(pred.shape[1]),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "mae": float(abs_resid.mean()),
        "max_abs_error": float(abs_resid.max()),
        "median_abs_error": float(np.median(abs_resid)),
        # Median (across pixel/wavelengths) of the per-pixel median absolute
        # error -- the typical approximation error at a wavelength.
        "per_pixel_median_abs_error_median": float(
            np.median(per_pixel_median_abs_error)),
        "per_pixel_median_abs_error_p90": float(
            np.percentile(per_pixel_median_abs_error, 90)),
        "per_pixel_median_abs_error_max": float(
            per_pixel_median_abs_error.max()),
        "r2": r2,
        "per_spectrum_rmse_mean": float(per_spec_rmse.mean()),
        "per_spectrum_rmse_median": float(np.median(per_spec_rmse)),
        "per_spectrum_rmse_p90": float(np.percentile(per_spec_rmse, 90)),
        "per_spectrum_rmse_max": float(per_spec_rmse.max()),
        # arrays kept out of the JSON summary; returned for plotting
        "_per_pixel_rmse": per_pixel_rmse,
        "_per_pixel_median_abs_error": per_pixel_median_abs_error,
        "_per_spectrum_rmse": per_spec_rmse,
    }


def plot_test_metrics(pred, true, metrics, outdir: Path, n_show: int = 6):
    """Save diagnostic plots for the test-set reconstruction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pix = np.arange(pred.shape[1])

    # Per-pixel RMSE curve.
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pix, metrics["_per_pixel_rmse"], lw=0.7, color="#C44E52")
    ax.set_xlabel("pixel index")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-pixel reconstruction RMSE (test set)")
    fig.tight_layout()
    fig.savefig(outdir / "test_per_pixel_rmse.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Per-pixel median absolute (approximation) error curve, with the median
    # across pixels drawn as a reference line.
    ppmae = metrics["_per_pixel_median_abs_error"]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pix, ppmae, lw=0.7, color="#55A868")
    ax.axhline(np.median(ppmae), color="#333333", ls="--", lw=1.0,
               label=f"median across pixels = {np.median(ppmae):.3e}")
    ax.set_xlabel("pixel index")
    ax.set_ylabel("median |error|")
    ax.set_title("Per-pixel median approximation error (test set)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "test_per_pixel_median_abs_error.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Per-spectrum RMSE histogram.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(metrics["_per_spectrum_rmse"], bins=80, color="#4C72B0")
    ax.set_xlabel("per-spectrum RMSE")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("Distribution of per-spectrum reconstruction RMSE (test)")
    fig.tight_layout()
    fig.savefig(outdir / "test_rmse_hist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Example true-vs-predicted overlays spanning the RMSE range
    # (best -> worst) so the examples are representative.
    order = np.argsort(metrics["_per_spectrum_rmse"])
    picks = np.linspace(0, order.size - 1, n_show).astype(int)
    rows = order[picks]
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.4 * n_show), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        ax.plot(pix, true[r], lw=0.6, color="#333333", label="true")
        ax.plot(pix, pred[r], lw=0.6, color="#C44E52", alpha=0.8, label="pred")
        rmse = np.sqrt(((pred[r] - true[r]) ** 2).mean())
        ax.set_ylabel("flux")
        ax.text(0.01, 0.9, f"RMSE={rmse:.3e}", transform=ax.transAxes,
                fontsize=8, va="top")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("pixel index")
    axes[0].set_title("Test spectra: true vs predicted (best -> worst)")
    fig.tight_layout()
    fig.savefig(outdir / "test_examples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def evaluate_test(predict_fn, payload, run_dir: Path, batch_size: int,
                  n_plot: int, save_predictions: bool,
                  outdir: Path | None = None) -> dict:
    """Reconstruct the held-out test split, predict it, and report metrics.

    ``predict_fn(X) -> flux`` is the (possibly gated) prediction closure; the
    generalist ``payload`` supplies the ``data_config`` that defines the split.
    ``outdir`` overrides where metrics/plots are written (default: the run dir);
    pass a dedicated folder to keep gated-model artefacts separate from the
    generalist's own.
    """
    cfg = payload.get("data_config")
    if cfg is None:
        raise ValueError(
            "checkpoint has no data_config; retrain to enable --test "
            "(or supply --inputs instead)")

    print(f"[eval] rebuilding dataset to recover the test split: {cfg}")
    ds = data_mod.load_dataset(
        core_labels=cfg["core_labels"],
        abundance_elements=cfg["abundance_elements"],
        target=cfg["target"],
        val_frac=cfg["val_frac"],
        test_frac=cfg["test_frac"],
        seed=cfg["seed"],
        max_rows=cfg["max_rows"],
        standardize_y=cfg["standardize_y"],
    )
    test = ds.test_idx
    print(f"[eval] test set: {test.size:,} spectra")

    X_test = ds.X[test]
    Y_true = ds.Y[test]
    Y_pred = predict_fn(X_test)

    metrics = reconstruction_metrics(Y_pred, Y_true)
    summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
    print("[eval] test reconstruction metrics:")
    for k, v in summary.items():
        print(f"        {k:26s} {v}")

    out = outdir if outdir is not None else (
        run_dir if run_dir.is_dir() else run_dir.parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    plot_test_metrics(Y_pred, Y_true, metrics, out, n_show=max(1, n_plot or 6))
    if save_predictions:
        with h5py.File(out / "test_predictions.h5", "w") as f:
            f.create_dataset("inputs", data=X_test)
            f.create_dataset("pred", data=Y_pred)
            f.create_dataset("true", data=Y_true)
        print("[eval] wrote test_predictions.h5")
    print(f"[eval] wrote test_metrics.json + plots -> {out}")
    return summary


def plot_spectra(spectra: np.ndarray, path: Path, n_show: int = 8) -> None:
    """Overplot a handful of predicted spectra."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_show, spectra.shape[0])
    pix = np.arange(spectra.shape[1])
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_show):
        ax.plot(pix, spectra[i], lw=0.6, alpha=0.8)
    ax.set_xlabel("pixel index")
    ax.set_ylabel("predicted normalised flux")
    ax.set_title(f"{n_show} predicted spectra")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict spectra / evaluate a trained BOSS emulator.")
    p.add_argument("--run", required=True,
                   help="run directory under trained_model/ or a .pkl path")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best",
                   help="which checkpoint to load from a run dir")
    p.add_argument("--specialist", default=None,
                   help="optional cool-region specialist run dir or .pkl; when "
                        "given, output is a smooth Teff-gated blend of the "
                        "generalist (--run) and this specialist")
    p.add_argument("--gate-lo", type=float, default=3800.0, dest="gate_lo",
                   help="Teff below which the specialist fully takes over")
    p.add_argument("--gate-hi", type=float, default=4800.0, dest="gate_hi",
                   help="Teff above which only the generalist is used")
    p.add_argument("--gate-feature", default="Teff", dest="gate_feature",
                   help="label the gate ramps on (default Teff)")
    p.add_argument("--inputs",
                   help="predict mode: input labels file (.npy/.csv/.h5)")
    p.add_argument("--test", action="store_true",
                   help="evaluate reconstruction metrics on the held-out "
                        "test set instead of predicting from --inputs")
    p.add_argument("--output", default=None,
                   help="predict mode: output path (.h5 or .npy); default "
                        "<run>/predicted_spectra.h5")
    p.add_argument("--batch-size", type=int, default=4096, dest="batch_size")
    p.add_argument("--plot", type=int, default=0, metavar="N",
                   help="predict mode: overplot N predicted spectra; "
                        "test mode: number of example overlays")
    p.add_argument("--save-predictions", action="store_true",
                   dest="save_predictions",
                   help="test mode: also write test_predictions.h5")
    p.add_argument("--test-outdir", default=None, dest="test_outdir",
                   help="test mode: write metrics/plots to this folder instead "
                        "of the run dir (use to keep gated-model plots separate)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.test == bool(args.inputs):
        raise SystemExit("choose exactly one of --inputs or --test")

    ckpt = resolve_checkpoint(args.run, args.checkpoint)
    model, payload = load_checkpoint(ckpt)
    print(f"[eval] loaded {ckpt}  model={payload['model']}  "
          f"features={payload['feature_names']}  n_pixels={payload['n_pixels']}")

    # Build the prediction closure: plain generalist, or a gated generalist +
    # specialist blend when --specialist is given.
    if args.specialist:
        spec_ckpt = resolve_checkpoint(args.specialist, args.checkpoint)
        spec_model, spec_payload = load_checkpoint(spec_ckpt)
        print(f"[eval] loaded specialist {spec_ckpt}  -> Teff gate "
              f"[{args.gate_lo:.0f}, {args.gate_hi:.0f}] on {args.gate_feature!r}")

        def predict_fn(X):
            return predict_gated(
                model, payload, spec_model, spec_payload, X,
                lo=args.gate_lo, hi=args.gate_hi,
                gate_feature=args.gate_feature, batch_size=args.batch_size)
    else:
        def predict_fn(X):
            return predict(model, payload, X, batch_size=args.batch_size)

    if args.test:
        evaluate_test(predict_fn, payload, Path(args.run), args.batch_size,
                      args.plot, args.save_predictions,
                      outdir=Path(args.test_outdir) if args.test_outdir else None)
        return

    # Prediction mode.
    X = load_inputs(Path(args.inputs), payload["feature_names"])
    print(f"[eval] {X.shape[0]:,} input rows -> predicting spectra")
    spectra = predict_fn(X)

    out = Path(args.output) if args.output else Path(args.run) / "predicted_spectra.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".npy":
        np.save(out, spectra)
    else:
        with h5py.File(out, "w") as f:
            f.create_dataset("inputs", data=X)
            f.create_dataset("spectra", data=spectra)
            f.attrs["feature_names"] = np.array(
                payload["feature_names"], dtype="S16")
            f.attrs["model"] = payload["model"]
            f.attrs["checkpoint"] = str(ckpt)
            if args.specialist:
                f.attrs["specialist"] = str(spec_ckpt)
                f.attrs["gate_feature"] = args.gate_feature
                f.attrs["gate_lo"] = args.gate_lo
                f.attrs["gate_hi"] = args.gate_hi
    print(f"[eval] wrote {spectra.shape} spectra -> {out}")

    if args.plot:
        png = out.with_suffix(".png")
        plot_spectra(spectra, png, n_show=args.plot)
        print(f"[eval] wrote plot -> {png}")


if __name__ == "__main__":
    main()
