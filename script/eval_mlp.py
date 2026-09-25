"""Evaluation suite for trained MLP emulator runs.

One entry point for every test-set diagnostic in this project. Point it at one
or more run directories (each containing ``best.pkl`` and, ideally,
``split_idx.npz``); the dataset is loaded once and reused across all of them.

Only test rows are ever read, so by default this loads a contiguous test-only
sidecar (``*.testset.*.npz``, ~4 GB) instead of the full 40 GB cache plus its
~38 GB standardised copy. The sidecar is built automatically on first use --
one sequential pass over the main cache -- and reused thereafter. Pass
``--full-data`` to load everything the old way, which is what runs trained
with a different split need.

Per run it writes ``reconstruction_stats_full.json`` (into the run dir and the
plots dir) and, unless ``--metrics-only``, five figures prefixed by the run's
tag:

  * ``<tag>_error_distribution.png``    -- per-spectrum RMSE histogram (lin+log)
  * ``<tag>_sample_reconstructions.png``-- best / p25 / median / p90 spectra
  * ``<tag>_worst_reconstructions.png`` -- the 4 highest-RMSE spectra

Each reconstruction panel carries a percentage-residual strip underneath it.
  * ``<tag>_teff_carbon_map.png``       -- error over the (Teff, carbon_m) plane
  * ``<tag>_corner_error.png``          -- error over the 6-D core-label space

Examples::

    # full suite for the best HPO trial
    BOSS_CACHE_DIR=/path/to/preprocessed_data .venv/bin/python \
        script/eval_mlp.py <run_dir> --tags best

    # just the numbers, for a batch of runs
    .venv/bin/python script/eval_mlp.py --metrics-only <run_a> <run_b> <run_c>

    # different example spectra in the panels (same typicality, new draws)
    .venv/bin/python script/eval_mlp.py --sample-seed 7 <run_dir> --tags best

autocvd claims a free GPU before JAX initialises (lab policy).
"""
from __future__ import annotations

import argparse
import pickle
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
MIN_FLUX = 1e-2      # below this true flux, % residual is meaningless (masked)

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
    """Spectrum panels, each with a percentage-residual strip underneath.

    The residual is ``100 * (pred - true) / true``. In deep absorption lines
    the true flux approaches zero and that ratio blows up while the *absolute*
    error stays small, so pixels with ``|true| < MIN_FLUX`` are masked out
    rather than allowed to dominate the axis; the panel reports how many were
    dropped. The y-range is clipped to a robust percentile for the same reason.
    """
    n = len(ax_specs)
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(14, 3.8 * n), sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n}, constrained_layout=True)
    for i, (name, loc) in enumerate(ax_specs):
        ax, rax = axes[2 * i], axes[2 * i + 1]
        g = int(test_idx[loc])
        pred, true = predict_row(g)
        ax.plot(pix, true, lw=0.6, color="k", label="Korg (true)")
        ax.plot(pix, pred, lw=0.6, color="C3", alpha=0.8, label="emulator")
        ax.set_ylabel("norm flux")
        ax.set_title(f"[{name}] RMSE={persp[loc]:.4f}   {lab(g)}",
                     fontsize=9, loc="left")
        ax.legend(loc="lower right", fontsize=8)

        denom = np.where(np.abs(true) < MIN_FLUX, np.nan, true)
        pct = 100.0 * (pred - true) / denom
        n_masked = int(np.isnan(pct).sum())
        rax.axhline(0.0, color="k", lw=0.6)
        rax.plot(pix, pct, lw=0.5, color="C0")
        lim = max(np.nanpercentile(np.abs(pct), 99.5), 0.05)
        rax.set_ylim(-lim, lim)
        rax.set_ylabel("resid [%]", fontsize=8)
        med = np.nanmedian(np.abs(pct))
        note = f"median |resid| = {med:.3f}%"
        if n_masked:
            note += f"   ({n_masked} px masked, |flux|<{MIN_FLUX})"
        rax.text(0.005, 0.04, note, transform=rax.transAxes, fontsize=7,
                 color="0.3", va="bottom",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                           pad=1))
        rax.tick_params(labelsize=7)
    axes[-1].set_xlabel("pixel")
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
    for q, c in [(50, "k"), (90, "C1"), (99, "C3")]:
        v = np.percentile(persp, q)
        a2.axvline(v, color=c, ls="--", lw=1, label=f"p{q}={v:.4f}")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("per-spectrum RMSE (log)")
    a2.set_ylabel("count (log)")
    a2.set_title("Error distribution (log-log, tail)")
    fig.suptitle(f"MLP {tag}  n_test={persp.size:,}  "
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
    fig.suptitle(f"MLP {tag}: reconstruction error over (Teff, carbon_m)",
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

    fig.suptitle(f"MLP ({tag}) test-set RMSE in label space\n"
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

def _resolve_test_idx(run_dir: Path, ds, tag: str) -> np.ndarray:
    """The run's test rows, as indices into ``ds``.

    A run's ``split_idx.npz`` is numbered in the full-dataset row space. Under
    the test-only fast path ``ds`` holds just the test rows, so those indices
    have to be translated to local positions via ``ds.sub_idx`` (which is
    sorted, so a searchsorted lookup suffices). A run whose split doesn't sit
    inside the loaded test set can't be evaluated this way at all -- that means
    it was trained with a different data config, so say so rather than
    silently scoring the wrong rows.
    """
    sp = run_dir / "split_idx.npz"
    if not sp.exists():
        return ds.test_idx
    saved = np.load(sp)["test_idx"]
    if ds.sub_idx is None:              # full load: already the right space
        return saved
    pos = np.clip(np.searchsorted(ds.sub_idx, saved), 0, ds.sub_idx.size - 1)
    bad = ds.sub_idx[pos] != saved
    if bad.any():
        raise SystemExit(
            f"[eval] {tag}: {int(bad.sum()):,}/{saved.size:,} of this run's "
            f"test rows are not in the loaded test split -- it was trained "
            f"with a different seed/val_frac/test_frac/max_rows. Re-run with "
            f"--full-data, or with data flags matching the run.")
    return pos


def _sample_positions(order, quantiles, rng, band):
    """Positions into the RMSE ordering ``order`` for each target quantile.

    With ``rng is None`` this returns the exact quantile row (the historical,
    fully reproducible behaviour). Otherwise each pick is drawn uniformly from
    the rows whose rank lies within ``+-band`` of the target quantile, so
    repeated runs show *different* spectra of the same typicality.
    """
    n = order.size
    out = []
    for q in quantiles:
        if rng is None:
            out.append(order[min(int(round(q * (n - 1))), n - 1)])
            continue
        lo = int(np.floor(max(q - band, 0.0) * (n - 1)))
        hi = int(np.ceil(min(q + band, 1.0) * (n - 1)))
        out.append(order[rng.integers(lo, hi + 1)])
    return out


def evaluate_run(run_dir: Path, tag: str, ds, Xn, Yn, outdir: Path,
                 metrics_only: bool, rng=None, band: float = 0.02,
                 worst_pool: int = 4, n_worst: int = 4) -> dict:
    test_idx = _resolve_test_idx(run_dir, ds, tag)
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
             X_test=ds.X[test_idx], feature_names=np.array(feat),
             cache_row=(ds.row_idx[test_idx] if ds.row_idx is not None
                        else test_idx))

    pix = np.arange(ds.n_pixels)
    order = np.argsort(persp)

    def predict_row(g):
        yn = np.asarray(model(jnp.asarray(Xn[g:g + 1])))[0]
        return yn * y_std + y_mean, ds.Y[g]

    def lab(g):
        v = ds.X[g]
        return "  ".join(f"{feat[i]}={v[i]:.2f}" for i in range(6))

    kw = dict(pix=pix, persp=persp, predict_row=predict_row, lab=lab,
              test_idx=test_idx)
    qs = [0.0, 0.25, 0.5, 0.9]
    picks = _sample_positions(order, qs, rng, band)
    names = ["best", "p25", "median", "p90"]
    if rng is not None:                 # the label is now approximate
        names = [f"~p{int(100 * q)}" for q in qs]

    tail = order[::-1]
    if rng is None or worst_pool <= n_worst:
        worst = [(f"worst #{i + 1}", loc)
                 for i, loc in enumerate(tail[:n_worst])]
    else:
        pool = min(worst_pool, tail.size)
        sel = np.sort(rng.choice(pool, size=n_worst, replace=False))
        worst = [(f"tail rank {int(r) + 1}", tail[r]) for r in sel]

    # Row provenance for every panel, so a figure can be traced back to
    # specific grid points (and reproduced by re-passing --sample-seed).
    # ``test_idx[loc]`` is a row of ``ds``; under the fast path that is local,
    # so map it back through row_idx to name the row in the h5 cache.
    def cache_row(loc):
        g = int(test_idx[loc])
        return int(ds.row_idx[g]) if ds.row_idx is not None else g

    for name, loc in list(zip(names, picks)) + worst:
        print(f"[eval] {tag} panel {name:14s} row={cache_row(loc):,} "
              f"rmse={persp[loc]:.5f}", flush=True)

    _recon_panels(list(zip(names, picks)),
                  f"MLP ({tag}) test-set reconstructions",
                  outdir / f"{tag}_sample_reconstructions.png", **kw)
    _recon_panels(worst,
                  f"MLP ({tag}) worst test-set reconstructions",
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
    p.add_argument("--outdir", type=Path, default=Path("plots/mlp_error_analysis"),
                   help="where figures go")
    p.add_argument("--abundances", nargs="*", default=None,
                   help="abundance elements used as inputs when the run was "
                        "trained. Default: read from the first run's checkpoint "
                        "(data_config); pass explicitly to override, e.g. empty "
                        "(--abundances) for abundance-free 6-input runs.")
    p.add_argument("--full-data", action="store_true",
                   help="load the whole 40 GB dataset instead of the test-only "
                        "cache (needed for runs whose split differs)")
    p.add_argument("--rebuild-test-cache", action="store_true",
                   help="rewrite the test-only sidecar before evaluating")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="randomise which spectra the reconstruction panels "
                        "show: each panel is drawn from a band around its "
                        "target quantile instead of the exact quantile row. "
                        "Omit for the deterministic exact-quantile picks.")
    p.add_argument("--quantile-band", type=float, default=0.02,
                   help="half-width, in quantile units, of the band each "
                        "randomised panel is drawn from (default 0.02)")
    p.add_argument("--worst-pool", type=int, default=4,
                   help="with --sample-seed, draw the 'worst' panels from "
                        "the top-N highest-RMSE spectra (default 4 = the "
                        "deterministic top 4)")
    p.add_argument("--n-worst", type=int, default=4, dest="n_worst",
                   help="number of highest-RMSE spectra to show in the worst-"
                        "reconstruction figure (default 4)")
    args = p.parse_args()

    tags = args.tags or [
        d.parent.name if d.name.startswith(("mlp_", "payne_")) else d.name
        for d in args.run_dirs]
    if len(tags) != len(args.run_dirs):
        p.error(f"--tags has {len(tags)} entries for {len(args.run_dirs)} runs")

    training.claim_gpu()

    # Same data config as the HPO worker, so the split matches the trials.
    base = training.build_parser().parse_args([])
    base.model = "mlp"
    base.labels = list(data_mod.CORE_LABELS)
    if args.abundances is None:
        # Match the checkpoint: guessing wrong silently misaligns the inputs
        # (a 27-feature dataset fed to a 6-input model crashes in the matmul).
        with open(args.run_dirs[0] / "best.pkl", "rb") as fh:
            cfg = pickle.load(fh).get("data_config") or {}
        base.abundances = list(cfg.get("abundance_elements",
                                       data_mod.ABUNDANCE_ELEMENTS))
    else:
        base.abundances = list(args.abundances)
    base.val_frac = 0.1
    base.test_frac = 0.1
    base.seed = 0

    # Evaluation reads test rows only, so by default load just those (~4 GB
    # from a contiguous sidecar, built on first use) rather than the full
    # ~76 GB of dataset + standardised copy. --full-data restores the old path.
    if args.full_data:
        ds, Xn, Yn = training.prepare_data(base)
    else:
        if args.rebuild_test_cache:
            data_mod.build_test_cache(
                target=base.target, val_frac=base.val_frac,
                test_frac=base.test_frac, seed=base.seed,
                max_rows=base.max_rows, rebuild=True)
        ds, Xn, Yn = training.prepare_test_data(base)

    rng = (np.random.default_rng(args.sample_seed)
           if args.sample_seed is not None else None)

    for run_dir, tag in zip(args.run_dirs, tags):
        evaluate_run(run_dir, tag, ds, Xn, Yn, args.outdir, args.metrics_only,
                     rng=rng, band=args.quantile_band,
                     worst_pool=args.worst_pool, n_worst=args.n_worst)


if __name__ == "__main__":
    main()
