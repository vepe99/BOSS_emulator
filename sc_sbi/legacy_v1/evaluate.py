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

Example::

    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/legacy_v1/evaluate.py trained_model/sc_sbi/smoke \\
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
COLORS = {"amortized": "#D55E00", "amortized_cont": "#7f7f7f", "amortized_sc": "#0072B2"}
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


def make_chi2(emulator_fn):
    @jax.jit
    def chi2(theta, flux, ivar):
        mu = emulator_fn(theta)
        good = jnp.sum(ivar > 0, axis=1)
        return jnp.sum(ivar * (flux - mu) ** 2, axis=1) / jnp.maximum(good, 1)

    def run(theta, flux, ivar, batch=1024):
        return np.concatenate([np.asarray(chi2(jnp.asarray(theta[s:s + batch]),
                                               jnp.asarray(flux[s:s + batch]),
                                               jnp.asarray(ivar[s:s + batch])))
                               for s in range(0, theta.shape[0], batch)])
    return run


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


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--models", nargs="+", default=None,
                   help="default: every <model>.weights.h5 found among " + ", ".join(MODELS))
    p.add_argument("--n-korg", type=int, default=None, help="subset of the Korg test set")
    p.add_argument("--n-boss", type=int, default=None, help="subset of the BOSS test set")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--no-ppc", action="store_true", help="skip the emulator chi2 check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)
    cfg = C.load_run_config(run_dir)
    box = {k: tuple(v) for k, v in cfg["box"].items()}
    adapter = C.shared_adapter({k: tuple(v) for k, v in cfg["padded_box"].items()})
    splits = np.load(run_dir / "splits.npz")
    models = args.models or [m for m in MODELS if (run_dir / f"{m}.weights.h5").exists()]
    rng = np.random.default_rng(args.seed)
    C.log(f"models: {models}")

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
    Yk, IVk, LSk = tmpl.apply(Fk, np.random.default_rng(args.seed + 7))
    C.log(f"Korg test: {Lk.shape[0]:,} spectra (noise from {t_rows.size:,} BOSS test templates)")

    # ---- BOSS test ----------------------------------------------------------
    rows = np.flatnonzero((boss["split"] == 2) & C.boss_in_box(boss, box))
    if args.n_boss and args.n_boss < rows.size:
        rows = np.sort(rng.choice(rows, args.n_boss, replace=False))
    Lb, Eb = boss["labels"][rows], boss["e_labels"][rows]
    Yb, IVb, Sb = boss["flux"][rows], boss["ivar"][rows], boss["snr"][rows]
    LSb = np.log10(Sb)[:, None].astype("float32")
    dtype_b = boss["data_type"][rows]
    C.log(f"BOSS test: {rows.size:,} spectra (test-split stars, in box)")
    bcols = [C.LABELS.index(k) for k in C.BOSS_LABELS]

    example = adapter(C.label_dict(Lk[:2]) | {"flux": Yk[:2], "ivar": IVk[:2], "log_snr": LSk[:2]})
    pvar = prior_variance(box)
    chi2_fn = None if args.no_ppc else make_chi2(C.make_emulator_fn(cfg["emulator"]))

    metrics = {"config": {"n_korg": int(Lk.shape[0]), "n_boss": int(rows.size),
                          "n_samples": args.n_samples, "models": models}}
    med_k, med_b, ranks_k, chi2_k, chi2_b = {}, {}, {}, {}, {}
    for m in models:
        C.log(f"[{m}] sampling ...")
        post = C.load_posterior(run_dir, m, example)
        sk = C.sample_posterior(post, adapter, Yk, LSk, args.n_samples)
        sb = C.sample_posterior(post, adapter, Yb, LSb, args.n_samples)
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
            e = np.where(np.isfinite(err) & (err > 0), err, 0.0)
            z = (Sbb["median"][:, i] - ref) / np.sqrt(Sbb["std"][:, i] ** 2 + e ** 2)
            ok = np.isfinite(z)
            mb[k] = point_metrics(Sbb["median"][:, i], ref) | {
                "cov68_with_ref_err": float(np.mean(np.abs(z[ok]) < 1)),
                "cov95_with_ref_err": float(np.mean(np.abs(z[ok]) < 2)),
                "z_std": float(np.std(z[ok])),
                "median_post_std": float(np.median(Sbb["std"][:, i]))}
            mb[k]["by_data_type"] = {
                str(t): point_metrics(Sbb["median"][dtype_b == t, i], ref[dtype_b == t])
                for t in np.unique(dtype_b)}
        metrics[m] = {"korg": mk, "boss": mb}
        if chi2_fn is not None:
            chi2_k[m] = chi2_fn(Sk["median"], Yk, IVk)
            chi2_b[m] = chi2_fn(Sbb["median"], Yb, IVb)
            metrics[m]["korg_chi2_median"] = float(np.median(chi2_k[m]))
            metrics[m]["boss_chi2_median"] = float(np.median(chi2_b[m]))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- table --------------------------------------------------------------
    lines = ["# SC evaluation: " + run_dir.name, "",
             f"Korg test n={Lk.shape[0]:,}, BOSS test n={rows.size:,}, "
             f"{args.n_samples} posterior samples.", ""]
    for setname, labels in (("korg", C.LABELS), ("boss", C.BOSS_LABELS)):
        cov = "cov68" if setname == "korg" else "cov68_with_ref_err"
        lines += [f"## {setname.upper()} test", "",
                  "| label | " + " | ".join(f"{m} MAE" for m in models) + " | "
                  + " | ".join(f"{m} {cov}" for m in models) + " |",
                  "|---|" + "---|" * (2 * len(models))]
        for k in labels:
            mae = [metrics[m][setname][k].get("mae", np.nan) for m in models]
            cv = [metrics[m][setname][k].get(cov, np.nan) for m in models]
            lines.append(f"| {k} | " + " | ".join(f"{v:.4g}" for v in mae) + " | "
                         + " | ".join(f"{v:.2f}" for v in cv) + " |")
        if chi2_fn is not None:
            lines.append("| χ²/pix (median) | " + " | ".join(
                f"{metrics[m][setname + '_chi2_median']:.3g}" for m in models) + " |"
                + " |" * len(models))
        lines.append("")
    if "amortized" in models and "amortized_sc" in models:
        lines += ["## Δ MAE (amortized_sc − amortized; negative = SC helps)", ""]
        for setname, labels in (("korg", C.LABELS), ("boss", C.BOSS_LABELS)):
            d = [metrics["amortized_sc"][setname][k]["mae"] - metrics["amortized"][setname][k]["mae"]
                 for k in labels]
            rel = [x / metrics["amortized"][setname][k]["mae"] for x, k in zip(d, labels)]
            lines.append(f"- {setname}: " + ", ".join(
                f"{k} {x:+.3g} ({r:+.0%})" for k, x, r in zip(labels, d, rel)))
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- plots --------------------------------------------------------------
    plot_recovery(Lk, med_k, C.LABELS, models, out / "korg_recovery.png", "Korg test (all labels)")
    plot_recovery(Lb, med_b, C.BOSS_LABELS, models, out / "boss_recovery.png",
                  "BOSS test (reference labels)")
    plot_sbc(ranks_k, C.LABELS, models, out / "korg_sbc.png")
    plot_err_vs_snr(Lk, med_k, 10 ** LSk[:, 0], C.LABELS, models, out / "korg_err_vs_snr.png",
                    "Korg test")
    plot_err_vs_snr(Lb, med_b, Sb, C.BOSS_LABELS, models, out / "boss_err_vs_snr.png", "BOSS test")
    if chi2_fn is not None:
        plot_chi2(chi2_k, models, out / "korg_chi2.png", "Korg test")
        plot_chi2(chi2_b, models, out / "boss_chi2.png", "BOSS test")
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
