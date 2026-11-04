"""Evaluate amortized vs amortized_cont vs amortized_sc posteriors.

Two held-out test sets, neither seen in training:

* **Korg test** -- the cache's test split (seed 0, 10 %), noised with real
  BOSS noise templates from the BOSS *test* split. All six labels are scored
  against the true labels.
* **BOSS test** -- real spectra of stars in the BOSS test split (grouped by
  ``sdss_id``) whose reference labels lie in the grid box. Only Teff, logg,
  m_H (= [Fe/H]) and alpha_m are scored; carbon_m and vmic have no reference.

For each model and set we report, per label: bias / RMSE / MAE / median |err|
of the posterior median, 68 % / 95 % interval coverage (BOSS: z-scores include
the reference-label error), SBC rank uniformity (Korg) and posterior
contraction (Korg), plus a posterior-predictive chi2/pixel through the
emulator at the posterior median.

v2 additions (work on v1 and v2 runs; the run's ``config.json`` decides how
inputs are built, so v1 runs are evaluated exactly as before):

* BOSS coverage also with a systematic floor added to the reference errors
  (``--ref-floor``, default: the BOSS-CLAM test-stage scatter 275 K / 0.3 /
  0.15 / 0.11 dex quoted in arXiv 2607.22822, App. A);
* breakdown by ``data_type`` and by HR-diagram region;
* repeat-visit consistency: two spectra of the same BOSS test star should give
  compatible posteriors (label-free check, covers all six labels);
* posterior-predictive chi2 with photon sigma and, if the run has a mismatch
  file, with ``sigma^2 + s_m^2``;
* loss curves (train loss, per-epoch val NLL, SC terms) from ``history.json``;
* any ``<name>.weights.h5`` in the run dir is evaluated (``_last`` /
  ``_bestval`` checkpoints only with ``--models``); a diverged model shows as
  NaN instead of crashing.

``sc_sbi/legacy_v1/evaluate.py`` is the frozen original.

Example::

    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/evaluate.py trained_model/sc_sbi/smoke \\
        --n-korg 500 --n-boss 500 --n-samples 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["amortized", "amortized_cont", "amortized_sc"]
_BASE_COLORS = {"amortized": "#D55E00", "amortized_cont": "#7f7f7f", "amortized_sc": "#0072B2"}
_EXTRA_COLORS = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


class _Colors(dict):
    def __missing__(self, k):
        self[k] = _EXTRA_COLORS[(len(self) - len(_BASE_COLORS)) % len(_EXTRA_COLORS)]
        return self[k]


COLORS = _Colors(_BASE_COLORS)
REF_FLOOR_PRESETS = {
    "clam_test_scatter": {"Teff": 275.0, "logg": 0.3, "m_H": 0.15, "alpha_m": 0.11},
    "none": {"Teff": 0.0, "logg": 0.0, "m_H": 0.0, "alpha_m": 0.0},
}
HR_REGIONS = {   # on the BOSS reference labels
    "M dwarf (Teff<4000, logg>=4)": lambda T, g: (T < 4000) & (g >= 4.0),
    "FGK dwarf (4000-6500, logg>=4)": lambda T, g: (T >= 4000) & (T < 6500) & (g >= 4.0),
    "subgiant (3.5<=logg<4)": lambda T, g: (T < 6500) & (g >= 3.5) & (g < 4.0),
    "giant (logg<3.5)": lambda T, g: (T < 6500) & (g < 3.5),
    "hot (Teff>=6500)": lambda T, g: T >= 6500,
}
UNITS = {"Teff": "K"}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def point_metrics(est, ref):
    ok = np.isfinite(ref) & np.isfinite(est)
    d = est[ok] - ref[ok]
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "bias": float(d.mean()), "rmse": float(np.sqrt((d ** 2).mean())),
            "mae": float(np.abs(d).mean()), "median_abs": float(np.median(np.abs(d))),
            "nmad": float(1.4826 * np.median(np.abs(d - np.median(d))))}


def interval_coverage(samples, ref):
    """Fraction of truths inside the central 68 / 95 % posterior intervals."""
    ok = np.isfinite(ref)
    s, r = samples[ok], ref[ok]
    out = {}
    for lev in (68, 95):
        lo, hi = np.percentile(s, [50 - lev / 2, 50 + lev / 2], axis=1)
        out[f"cov{lev}"] = float(np.mean((r >= lo) & (r <= hi)))
    return out


def sbc_ranks(samples, ref):
    return np.mean(samples < ref[:, None], axis=1)


def prior_variance(box, n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    v = {}
    for k, (lo, hi) in box.items():
        x = (10 ** rng.uniform(np.log10(lo), np.log10(hi), n) if k in C.LOG_LABELS
             else rng.uniform(lo, hi, n))
        v[k] = float(x.var())
    return v


def make_chi2(emulator_fn, cont_deg: int = 0):
    """chi2/pixel of the run's forward model at the posterior median: emulator
    [-> LSF of sigma ``lsf``] [-> profiled Chebyshev continuum] (v3)."""
    basis = jnp.asarray(C.cheb_basis(cont_deg)) if cont_deg > 0 else None

    @jax.jit
    def chi2(theta, flux, ivar, lsf):
        mu = emulator_fn(theta)
        if lsf is not None:
            mu = C.apply_lsf(mu, lsf)
        if basis is not None:
            mu = C.profile_continuum(mu, flux, ivar, basis)
        good = jnp.sum(ivar > 0, axis=1)
        return jnp.sum(ivar * (flux - mu) ** 2, axis=1) / jnp.maximum(good, 1)

    def run(theta, flux, ivar, batch=1024, s_m=None, lsf=None):
        if s_m is not None:
            ivar = ivar / (1.0 + ivar * s_m[None] ** 2)
        sl = lambda x, s: None if x is None else jnp.asarray(x[s:s + batch])
        return np.concatenate([np.asarray(chi2(jnp.asarray(theta[s:s + batch]),
                                               jnp.asarray(flux[s:s + batch]),
                                               jnp.asarray(ivar[s:s + batch]), sl(lsf, s)))
                               for s in range(0, theta.shape[0], batch)])
    return run


def z_metrics(med, std, ref, err, floor):
    """Coverage of |z| < 1, 2 with z = (median - ref) / sqrt(std^2 + err^2 + floor^2)."""
    e = np.where(np.isfinite(err) & (err > 0), err, 0.0)
    z = (med - ref) / np.sqrt(std ** 2 + e ** 2 + floor ** 2)
    ok = np.isfinite(z)
    if not ok.any():
        return {"cov68": float("nan"), "cov95": float("nan"), "z_std": float("nan")}
    return {"cov68": float(np.mean(np.abs(z[ok]) < 1)), "cov95": float(np.mean(np.abs(z[ok]) < 2)),
            "z_std": float(np.std(z[ok]))}


def repeat_pairs(boss, box, n_stars, rng):
    """Two in-box test-split spectra of each of up to ``n_stars`` stars."""
    rows = np.flatnonzero((boss["split"] == 2) & C.boss_in_box(boss, box))
    ids = boss["sdss_id"][rows]
    uniq, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    multi = np.flatnonzero(cnt >= 2)
    if multi.size == 0:
        return np.zeros((0, 2), int)
    pick = rng.choice(multi, min(n_stars, multi.size), replace=False)
    pairs = [rng.choice(rows[inv == u], 2, replace=False) for u in pick]
    return np.array(pairs)


def summarize(samples):
    return {"median": np.median(samples, axis=1), "std": samples.std(axis=1)}


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_recovery(ref, med, labels, models, path, title):
    fig, axes = plt.subplots(len(models), len(labels),
                             figsize=(3.1 * len(labels), 3.0 * len(models)), squeeze=False)
    for r, m in enumerate(models):
        for c, k in enumerate(labels):
            ax = axes[r, c]
            x, y = ref[:, c], med[m][:, c]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 2:
                ax.text(0.5, 0.5, "no finite\nestimates", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(k, fontsize=9)
                continue
            lo, hi = np.percentile(np.r_[x[ok], y[ok]], [0.5, 99.5])
            ax.hexbin(x[ok], y[ok], gridsize=45, bins="log", cmap="viridis", mincnt=1,
                      extent=(lo, hi, lo, hi))
            ax.plot([lo, hi], [lo, hi], "r-", lw=0.8)
            pm = point_metrics(y, x)
            ax.set_title(f"{k}: MAE {pm.get('mae', np.nan):.3g}", fontsize=9)
            if r == len(models) - 1:
                ax.set_xlabel(f"reference {k}")
            if c == 0:
                ax.set_ylabel(f"{m}\nposterior median")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sbc(ranks, labels, models, path):
    fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 3.0), squeeze=False)
    grid = np.linspace(0, 1, 101)
    for c, k in enumerate(labels):
        ax = axes[0, c]
        for m in models:
            r = np.sort(ranks[m][:, c])
            ecdf = np.searchsorted(r, grid, side="right") / r.size
            ax.plot(grid, ecdf - grid, color=COLORS[m], label=m)
        n = ranks[models[0]].shape[0]
        ax.fill_between(grid, -1.36 / np.sqrt(n), 1.36 / np.sqrt(n), color="0.85")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(k)
        ax.set_xlabel("fractional rank")
    axes[0, 0].set_ylabel("ECDF − uniform")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Korg test: SBC rank ECDF difference (grey: 95 % KS band)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_err_vs_snr(ref, med, snr, labels, models, path, title):
    edges = np.quantile(snr, np.linspace(0, 1, 9))
    centers = np.sqrt(edges[1:] * edges[:-1])
    fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.0), squeeze=False)
    for c, k in enumerate(labels):
        ax = axes[0, c]
        for m in models:
            e = np.abs(med[m][:, c] - ref[:, c])
            vals = [np.nanmedian(e[(snr >= lo) & (snr < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(centers, vals, "o-", color=COLORS[m], label=m, ms=3)
        ax.set_xscale("log")
        ax.set_xlabel("SNR")
        ax.set_title(f"median |err| {k}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_chi2(chi2, models, path, title):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    chi2 = {m: v[np.isfinite(v)] for m, v in chi2.items()}
    models = [m for m in models if chi2[m].size]
    if not models:
        return
    allv = np.concatenate([chi2[m] for m in models])
    bins = np.logspace(np.log10(max(np.nanpercentile(allv, 0.5), 1e-2)),
                       np.log10(np.nanpercentile(allv, 99.5)), 50)
    for m in models:
        ax.hist(chi2[m], bins=bins, histtype="step", lw=1.5, color=COLORS[m],
                label=f"{m} (median {np.nanmedian(chi2[m]):.2f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^2$/pixel at posterior median")
    ax.legend(fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loss_curves(history, path):
    keys = [m for m in history if history[m]]
    sc_keys = [m for m in keys if any("self-consistency" in k for k in history[m])]
    ncol = 3 if sc_keys else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 3.8), squeeze=False)
    off = 0
    for m in keys:
        h = history[m]
        lab = h.get("labeled/posterior/loss", h.get("loss", []))
        if m != "amortized" and "amortized" in history and history["amortized"]:
            off = len(history["amortized"].get("loss", []))
        else:
            off = 0
        ep = np.arange(1, len(lab) + 1) + off
        axes[0, 0].plot(ep, lab, color=COLORS[m], lw=1.4, label=m)
        if h.get("val_nll"):
            axes[0, 1].plot(np.arange(1, len(h["val_nll"]) + 1) + off, h["val_nll"],
                            color=COLORS[m], lw=1.4, label=m)
        elif "val_nll_end" in h:
            axes[0, 1].plot([off + len(lab)], [h["val_nll_end"]], "o", color=COLORS[m], label=m)
    for m in sc_keys:
        h = history[m]
        k = next((k for k in ("unlabeled/self-consistency/raw_var",
                              "unlabeled/self-consistency/raw") if k in h), None)
        if k:
            v = np.array(h[k], float)
            axes[0, 2].plot(np.arange(1, v.size + 1), v, color=COLORS[m], lw=1.4, label=m)
    axes[0, 0].set_title("labeled loss  −log q(θ|y) (train)")
    axes[0, 1].set_title("Korg val NLL")
    for ax in axes[0, :2]:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if sc_keys:
        axes[0, 2].set_yscale("log")
        axes[0, 2].set_title("SC: raw Var_θ[log p(y)] (phase B)")
        axes[0, 2].set_xlabel("phase-B epoch")
        axes[0, 2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------

def parse_floor(spec: str) -> dict:
    if spec in REF_FLOOR_PRESETS:
        return dict(REF_FLOOR_PRESETS[spec])
    out = dict(REF_FLOOR_PRESETS["none"])
    for kv in spec.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--models", nargs="+", default=None,
                   help="default: every <name>.weights.h5 except *_last / *_bestval")
    p.add_argument("--n-korg", type=int, default=None, help="subset of the Korg test set")
    p.add_argument("--n-boss", type=int, default=None, help="subset of the BOSS test set")
    p.add_argument("--n-repeat-stars", type=int, default=300,
                   help="BOSS test stars with 2 visits for the repeat check (0: skip)")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--ref-floor", default="clam_test_scatter",
                   help="systematic floor on the BOSS reference errors: a preset "
                        f"({', '.join(REF_FLOOR_PRESETS)}) or 'Teff=100,logg=0.1,...'")
    p.add_argument("--no-ppc", action="store_true", help="skip the emulator chi2 check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)
    cfg = C.load_run_config(run_dir)
    ncfg = C.noise_config(cfg.get("noise"))
    mm = C.load_mismatch(ncfg)
    floor = parse_floor(args.ref_floor)
    box = {k: tuple(v) for k, v in cfg["box"].items()}
    adapter = C.shared_adapter({k: tuple(v) for k, v in cfg["padded_box"].items()},
                               cfg.get("log_params"))
    splits = np.load(run_dir / "splits.npz")
    if args.models:
        models = args.models
    else:
        found = sorted(f.name[:-len(".weights.h5")] for f in run_dir.glob("*.weights.h5"))
        found = [m for m in found if not m.endswith(("_last", "_bestval"))]
        models = [m for m in MODELS if m in found] + [m for m in found if m not in MODELS]
    rng = np.random.default_rng(args.seed)
    C.log(f"models: {models}   noise: {ncfg}")

    # ---- Korg test ----------------------------------------------------------
    max_rows = int(splits["max_rows"])
    korg, kidx, _ = C.load_korg(cfg["korg_cache"], max_rows=None if max_rows < 0 else max_rows,
                                seed=0, splits=("test",))
    assert np.array_equal(kidx["test"], splits["korg_test_idx"]), "Korg test split mismatch"
    Lk, Fk = korg["test"]
    if args.n_korg and args.n_korg < Lk.shape[0]:
        sel = np.sort(rng.choice(Lk.shape[0], args.n_korg, replace=False))
        Lk, Fk = Lk[sel], Fk[sel]
    boss = C.load_boss()
    t_rows = np.flatnonzero((boss["split"] == 2) & np.isfinite(boss["snr"]) & (boss["snr"] > 0))
    tmpl = C.NoiseTemplates(boss["ivar"][t_rows], boss["snr"][t_rows])
    # residual mode trains with a bank of *train* residuals; the test set uses template + s_m
    korg_noise = C.NoiseModel(tmpl, ncfg | {"mode": "sm" if ncfg["mode"] == "residual"
                                            else ncfg["mode"]}, mm)
    cond_k = korg_noise.make_batch(Fk, Lk, np.random.default_rng(args.seed + 7))
    snr_k = 10 ** cond_k["log_snr"][:, 0]
    C.log(f"Korg test: {Lk.shape[0]:,} spectra (noise from {t_rows.size:,} BOSS test templates)")

    # ---- BOSS test ----------------------------------------------------------
    rows = np.flatnonzero((boss["split"] == 2) & C.boss_in_box(boss, box))
    if args.n_boss and args.n_boss < rows.size:
        rows = np.sort(rng.choice(rows, args.n_boss, replace=False))
    Lb, Eb = boss["labels"][rows], boss["e_labels"][rows]
    Sb = boss["snr"][rows]
    cond_b = C.spectrum_features(boss["flux"][rows], boss["ivar"][rows], Sb, ncfg, mm)
    dtype_b = boss["data_type"][rows].astype(str)
    C.log(f"BOSS test: {rows.size:,} spectra (test-split stars, in box)")
    bcols = [C.LABELS.index(k) for k in C.BOSS_LABELS]
    regions = {name: f(Lb[:, 0], Lb[:, 1]) for name, f in HR_REGIONS.items()}

    pairs = repeat_pairs(boss, box, args.n_repeat_stars, rng) if args.n_repeat_stars else np.zeros((0, 2), int)
    if len(pairs):
        pr = pairs.reshape(-1)
        cond_r = C.spectrum_features(boss["flux"][pr], boss["ivar"][pr], boss["snr"][pr], ncfg, mm)
        C.log(f"repeat check: {len(pairs)} test stars x 2 spectra")

    example = adapter(C.label_dict(Lk[:2]) | {k: v[:2] for k, v in cond_k.items()})
    pvar = prior_variance(box)
    chi2_fn = None if args.no_ppc else make_chi2(C.make_emulator_fn(cfg["emulator"]), ncfg["cont_deg"])
    names = C.params(ncfg)                      # 6 labels (+ lsf_px for v3, scored separately)
    lsf_true_k = cond_k.get(C.LSF)

    metrics = {"config": {"n_korg": int(Lk.shape[0]), "n_boss": int(rows.size),
                          "n_repeat_stars": int(len(pairs)), "n_samples": args.n_samples,
                          "models": models, "noise": ncfg, "ref_floor": floor}}
    med_k, med_b, ranks_k, chi2_k, chi2_b = {}, {}, {}, {}, {}
    for m in models:
        C.log(f"[{m}] sampling ...")
        post = C.load_posterior(run_dir, m, example)
        if not C.weights_finite(post):
            C.log(f"[{m}] WARNING: non-finite weights (diverged run) -- metrics will be NaN")
        sk = C.sample_posterior(post, adapter, cond_k, args.n_samples, names=names)
        sb = C.sample_posterior(post, adapter, cond_b, args.n_samples, names=names)
        lsf_k = lsf_b = None
        if ncfg["lsf"]:                         # lsf_px: nuisance, kept out of the label metrics
            lsf_k, lsf_b = np.median(sk[..., -1], 1), np.median(sb[..., -1], 1)
            sk, sb = sk[..., :-1], sb[..., :-1]
        Sk, Sbb = summarize(sk), summarize(sb)
        med_k[m], med_b[m] = Sk["median"], Sbb["median"][:, bcols]
        ranks_k[m] = np.stack([sbc_ranks(sk[:, :, i], Lk[:, i]) for i in range(len(C.LABELS))], 1)

        mk = {}
        for i, k in enumerate(C.LABELS):
            mk[k] = point_metrics(Sk["median"][:, i], Lk[:, i]) | interval_coverage(sk[:, :, i], Lk[:, i])
            mk[k]["contraction"] = float(1 - np.mean(Sk["std"][:, i] ** 2) / pvar[k])
            hist = np.histogram(ranks_k[m][:, i], bins=10, range=(0, 1))[0]
            mk[k]["sbc_chi2"] = float(((hist - hist.mean()) ** 2 / hist.mean()).sum())
        mb = {}
        for j, k in enumerate(C.BOSS_LABELS):
            i = C.LABELS.index(k)
            ref, err = Lb[:, j], Eb[:, j]
            med, sd = Sbb["median"][:, i], Sbb["std"][:, i]
            z0 = z_metrics(med, sd, ref, err, 0.0)
            zf = z_metrics(med, sd, ref, err, floor[k])
            mb[k] = point_metrics(med, ref) | {
                "cov68_with_ref_err": z0["cov68"], "cov95_with_ref_err": z0["cov95"],
                "z_std": z0["z_std"],
                "cov68_with_floor": zf["cov68"], "cov95_with_floor": zf["cov95"],
                "z_std_floor": zf["z_std"],
                "median_post_std": float(np.median(sd))}
            mb[k]["by_data_type"] = {
                str(t): point_metrics(med[dtype_b == t], ref[dtype_b == t])
                | {"cov68_with_floor": z_metrics(med[dtype_b == t], sd[dtype_b == t], ref[dtype_b == t],
                                                 err[dtype_b == t], floor[k])["cov68"]}
                for t in np.unique(dtype_b)}
            mb[k]["by_region"] = {
                r: point_metrics(med[msk], ref[msk])
                | {"cov68_with_floor": z_metrics(med[msk], sd[msk], ref[msk], err[msk], floor[k])["cov68"]}
                for r, msk in regions.items()}
        metrics[m] = {"korg": mk, "boss": mb}

        if len(pairs):
            sr = C.sample_posterior(post, adapter, cond_r, args.n_samples)
            md, sd = np.median(sr, 1).reshape(-1, 2, len(C.LABELS)), sr.std(1).reshape(-1, 2, len(C.LABELS))
            rep = {}
            for i, k in enumerate(C.LABELS):
                d = md[:, 0, i] - md[:, 1, i]
                z = d / np.sqrt(sd[:, 0, i] ** 2 + sd[:, 1, i] ** 2)
                ok = np.isfinite(z)
                rep[k] = ({"n": int(ok.sum()), "median_abs_diff": float(np.median(np.abs(d[ok]))),
                           "z_std": float(np.std(z[ok])), "frac_abs_z_lt_1": float(np.mean(np.abs(z[ok]) < 1))}
                          if ok.any() else {"n": 0})
            metrics[m]["repeat"] = rep

        if chi2_fn is not None:
            chi2_k[m] = chi2_fn(Sk["median"], cond_k["flux"], cond_k["ivar"], lsf=lsf_k)
            chi2_b[m] = chi2_fn(Sbb["median"], cond_b["flux"], cond_b["ivar"], lsf=lsf_b)
            metrics[m]["korg_chi2_median"] = float(np.nanmedian(chi2_k[m]))
            metrics[m]["boss_chi2_median"] = float(np.nanmedian(chi2_b[m]))
            if mm is not None:
                metrics[m]["boss_chi2_sm_median"] = float(np.nanmedian(
                    chi2_fn(Sbb["median"], cond_b["flux"], cond_b["ivar"], s_m=mm.s_m, lsf=lsf_b)))
        if lsf_k is not None:
            t = np.asarray(lsf_true_k).ravel()
            metrics[m]["lsf"] = {"korg_mae": float(np.mean(np.abs(lsf_k - t))),
                                 "korg_bias": float(np.mean(lsf_k - t)),
                                 "boss_median": float(np.nanmedian(lsf_b)),
                                 "boss_p16_p84": np.nanpercentile(lsf_b, [16, 84]).tolist()}
            C.log(f"[{m}] lsf_px: Korg MAE {metrics[m]['lsf']['korg_mae']:.3f}  "
                  f"BOSS median {metrics[m]['lsf']['boss_median']:.2f}")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- table --------------------------------------------------------------
    g = lambda d, k: d.get(k, np.nan) if isinstance(d, dict) else np.nan
    lines = ["# SC evaluation: " + run_dir.name, "",
             f"Korg test n={Lk.shape[0]:,}, BOSS test n={rows.size:,}, "
             f"{args.n_samples} posterior samples. Noise: `{ncfg}`. "
             f"BOSS reference floor: {floor}.", ""]
    for setname, labels in (("korg", C.LABELS), ("boss", C.BOSS_LABELS)):
        covs = ["cov68"] if setname == "korg" else ["cov68_with_ref_err", "cov68_with_floor"]
        lines += [f"## {setname.upper()} test", "",
                  "| label | " + " | ".join(f"{m} MAE" for m in models) + " | "
                  + " | ".join(f"{m} {c}" for c in covs for m in models) + " |",
                  "|---|" + "---|" * ((1 + len(covs)) * len(models))]
        for k in labels:
            mae = [g(metrics[m][setname][k], "mae") for m in models]
            cv = [g(metrics[m][setname][k], c) for c in covs for m in models]
            lines.append(f"| {k} | " + " | ".join(f"{v:.4g}" for v in mae) + " | "
                         + " | ".join(f"{v:.2f}" for v in cv) + " |")
        if chi2_fn is not None:
            lines.append("| χ²/pix (median) | " + " | ".join(
                f"{metrics[m][setname + '_chi2_median']:.3g}" for m in models) + " |"
                + " |" * (len(covs) * len(models)))
        lines.append("")
    lines += ["## BOSS test by HR region (MAE / cov68 with floor)", "",
              "| region | n | " + " | ".join(f"{m} {k}" for k in ("logg", "m_H") for m in models) + " |",
              "|---|---|" + "---|" * (2 * len(models))]
    for r, msk in regions.items():
        cells = [f"{g(metrics[m]['boss'][k]['by_region'][r], 'mae'):.3g} / "
                 f"{g(metrics[m]['boss'][k]['by_region'][r], 'cov68_with_floor'):.2f}"
                 for k in ("logg", "m_H") for m in models]
        lines.append(f"| {r} | {int(msk.sum())} | " + " | ".join(cells) + " |")
    lines.append("")
    if len(pairs):
        lines += [f"## Repeat-visit consistency ({len(pairs)} test stars x 2 spectra)", "",
                  "std of z = Δmedian / sqrt(σ1² + σ2²); 1 = posteriors consistent across visits", "",
                  "| label | " + " | ".join(f"{m} z_std" for m in models) + " | "
                  + " | ".join(f"{m} med|Δ|" for m in models) + " |",
                  "|---|" + "---|" * (2 * len(models))]
        for k in C.LABELS:
            lines.append(f"| {k} | " + " | ".join(f"{g(metrics[m]['repeat'][k], 'z_std'):.2f}" for m in models)
                         + " | " + " | ".join(f"{g(metrics[m]['repeat'][k], 'median_abs_diff'):.3g}"
                                              for m in models) + " |")
        lines.append("")
    if "amortized" in models and "amortized_sc" in models:
        lines += ["## Δ MAE (amortized_sc − amortized; negative = SC helps)", ""]
        for setname, labels in (("korg", C.LABELS), ("boss", C.BOSS_LABELS)):
            a = [g(metrics["amortized"][setname][k], "mae") for k in labels]
            b = [g(metrics["amortized_sc"][setname][k], "mae") for k in labels]
            lines.append(f"- {setname}: " + ", ".join(
                f"{k} {y - x:+.3g} ({(y - x) / x:+.0%})" for k, x, y in zip(labels, a, b)))
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- plots --------------------------------------------------------------
    plot_recovery(Lk, med_k, C.LABELS, models, out / "korg_recovery.png", "Korg test (all labels)")
    plot_recovery(Lb, med_b, C.BOSS_LABELS, models, out / "boss_recovery.png",
                  "BOSS test (reference labels)")
    plot_sbc(ranks_k, C.LABELS, models, out / "korg_sbc.png")
    plot_err_vs_snr(Lk, med_k, snr_k, C.LABELS, models, out / "korg_err_vs_snr.png", "Korg test")
    plot_err_vs_snr(Lb, med_b, Sb, C.BOSS_LABELS, models, out / "boss_err_vs_snr.png", "BOSS test")
    if chi2_fn is not None:
        plot_chi2(chi2_k, models, out / "korg_chi2.png", "Korg test")
        plot_chi2(chi2_b, models, out / "boss_chi2.png", "BOSS test")
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        plot_loss_curves(json.loads(hist_path.read_text()), out / "loss_curves.png")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
