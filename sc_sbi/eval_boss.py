"""Recovery + calibration of a train_boss.py run on its own BOSS test split.

The NPE targets p(CLAM label | spectrum), so the CLAM label of a test star is a draw
from the target and plain SBC / coverage against it is the right calibration test
(no reference-error inflation needed; the z-with-CLAM-errors numbers are just extra).

    .venv/bin/python sc_sbi/eval_boss.py trained_model/sc_sbi/boss_only [--weights boss_bestval boss_last]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                      # KERAS_BACKEND=jax + autocvd GPU pick
import evaluate as E                    # metric + plot helpers
import numpy as np, bayesflow as bf
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--weights", nargs="+", default=["boss_bestval"])
p.add_argument("--n-samples", type=int, default=500)
p.add_argument("--n-repeat-stars", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-plot", type=int, default=1000, help="random test stars shown in the bayesflow plots")
a = p.parse_args()
run = Path(a.run_dir); out = run / "eval"; out.mkdir(exist_ok=True)
cfg = C.load_run_config(run); names = cfg["labels"]; ncfg = C.noise_config(cfg["noise"])
rng = np.random.default_rng(a.seed)

adapter = bf.Adapter()
for k in names:
    adapter = adapter.constrain(k, lower=float(cfg["padded_box"][k][0]), upper=float(cfg["padded_box"][k][1]))

boss = C.load_boss()
rows = np.load(run / "test_posterior.npz")["rows"]          # exact test rows used by train_boss.py
L, Err, snr = boss["labels"][rows], boss["e_labels"][rows], boss["snr"][rows]
cond = C.spectrum_features(boss["flux"][rows], boss["ivar"][rows], snr, ncfg, None)
dtype = boss["data_type"][rows].astype(str)
regions = {r: f(L[:, 0], L[:, 1]) for r, f in E.HR_REGIONS.items()}
pairs = E.repeat_pairs(boss, {k: tuple(v) for k, v in cfg["box"].items()}, a.n_repeat_stars, rng)
cond_r = C.spectrum_features(boss["flux"][pairs.ravel()], boss["ivar"][pairs.ravel()],
                             boss["snr"][pairs.ravel()], ncfg, None)
C.log(f"test {len(rows):,} spectra, repeat pairs {len(pairs)}")

example = adapter({k: v[:2] for k, v in cond.items() if k != "ivar"} | {k: L[:2, i:i + 1] for i, k in enumerate(names)})
levels = np.linspace(0.05, 0.95, 19)
metrics, med, ranks, cal, smp = {}, {}, {}, {}, {}
for w in a.weights:
    post = C.build_posterior(cfg["net"], ncfg["noise_channel"], names)
    post.build_from_data(post.adapter(example)); post.load_weights(run / f"{w}.weights.h5")
    s = C.sample_posterior(post, adapter, cond, a.n_samples, names=names)      # (n, S, 4)
    smp[w] = s; med[w], sd = np.median(s, 1), s.std(1)
    ranks[w] = np.stack([E.sbc_ranks(s[:, :, j], L[:, j]) for j in range(len(names))], 1)
    # empirical coverage of central intervals at each nominal level
    cal[w] = np.array([[np.mean(np.abs(ranks[w][:, j] - 0.5) <= lev / 2) for lev in levels]
                       for j in range(len(names))])
    m = {}
    for j, k in enumerate(names):
        r = L[:, j]
        hist = np.histogram(ranks[w][:, j], bins=20, range=(0, 1))[0]
        ecdf_dev = np.abs(np.sort(ranks[w][:, j]) - (np.arange(len(r)) + 0.5) / len(r)).max()
        z = (med[w][:, j] - r) / sd[:, j]
        m[k] = E.point_metrics(med[w][:, j], r) | E.interval_coverage(s[:, :, j], r) | {
            "median_post_std": float(np.median(sd[:, j])),
            "z_std": float(np.std(z)), "z_median_abs": float(np.median(np.abs(z))),   # 0.674 if Gaussian+calibrated
            "sbc_chi2_20bins": float(((hist - hist.mean()) ** 2 / hist.mean()).sum()),
            "sbc_ks_D": float(ecdf_dev), "sbc_ks_crit95": float(1.36 / np.sqrt(len(r))),
            "calibration_curve": dict(zip([f"{x:.2f}" for x in levels], cal[w][j].round(4).tolist())),
            "cov68_incl_clam_err": E.z_metrics(med[w][:, j], sd[:, j], r, Err[:, j], 0.0)["cov68"],
            "by_data_type": {t: E.point_metrics(med[w][dtype == t, j], r[dtype == t])
                             | {"cov68": float(np.mean(np.abs(ranks[w][dtype == t, j] - .5) <= .34))}
                             for t in np.unique(dtype)},
            "by_region": {g: E.point_metrics(med[w][msk, j], r[msk])
                          | {"cov68": float(np.mean(np.abs(ranks[w][msk, j] - .5) <= .34))}
                          for g, msk in regions.items()}}
    sr = C.sample_posterior(post, adapter, cond_r, a.n_samples, names=names)
    md, sdr = np.median(sr, 1).reshape(-1, 2, len(names)), sr.std(1).reshape(-1, 2, len(names))
    zr = (md[:, 0] - md[:, 1]) / np.sqrt(sdr[:, 0] ** 2 + sdr[:, 1] ** 2)
    m["repeat"] = {k: {"z_std": float(np.std(zr[:, j])), "median_abs_diff": float(np.median(np.abs(md[:, 0, j] - md[:, 1, j])))}
                   for j, k in enumerate(names)}
    metrics[w] = m
    C.log(f"[{w}] " + "  ".join(f"{k}: MAE {m[k]['mae']:.3g} cov68 {m[k]['cov68']:.2f} cov95 {m[k]['cov95']:.2f} "
                                 f"zstd {m[k]['z_std']:.2f}" for k in names))
(out / "metrics.json").write_text(json.dumps({"n_test": len(rows), "n_samples": a.n_samples} | metrics, indent=2))

# ---- summary table ----------------------------------------------------------------
lines = [f"# {run.name}: BOSS-CLAM test ({len(rows):,} spectra, {a.n_samples} draws)", ""]
for w in a.weights:
    m = metrics[w]
    lines += [f"## {w}", "", "| label | MAE | RMSE | bias | NMAD | post σ (med) | cov68 | cov95 | z std | SBC KS D (crit) | repeat z std |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {k} | {m[k]['mae']:.4g} | {m[k]['rmse']:.4g} | {m[k]['bias']:+.3g} | {m[k]['nmad']:.3g} | "
              f"{m[k]['median_post_std']:.3g} | {m[k]['cov68']:.3f} | {m[k]['cov95']:.3f} | {m[k]['z_std']:.2f} | "
              f"{m[k]['sbc_ks_D']:.3f} ({m[k]['sbc_ks_crit95']:.3f}) | {m['repeat'][k]['z_std']:.2f} |" for k in names]
    lines += ["", "| region | n | " + " | ".join(f"{k} MAE / cov68" for k in names) + " |", "|---|---|" + "---|" * len(names)]
    lines += [f"| {g} | {int(msk.sum())} | " + " | ".join(
        f"{m[k]['by_region'][g].get('mae', np.nan):.3g} / {m[k]['by_region'][g]['cov68']:.2f}" for k in names) + " |"
        for g, msk in regions.items()]
    lines.append("")
(out / "summary.md").write_text("\n".join(lines)); print("\n".join(lines))

# ---- plots ----------------------------------------------------------------------------
E.plot_err_vs_snr(L, med, snr, names, a.weights, out / "err_vs_snr.png", f"{run.name}: BOSS test")
E.plot_loss_curves({"boss": json.loads((run / "history.json").read_text())}, out / "loss_curves.png")
P = bf.diagnostics.plots
sub = np.sort(np.random.default_rng(a.seed + 1).choice(len(rows), min(a.n_plot, len(rows)), replace=False))
for w in a.weights:                     # bayesflow diagnostics on a random subset, CLAM labels as targets
    for name, fn, kw in (("recovery", P.recovery, {}),
                         ("sbc_ecdf", P.calibration_ecdf, {"difference": True}),
                         ("sbc_hist", P.calibration_histogram, {"num_bins": 20}),
                         ("coverage", P.coverage, {}),
                         ("z_contraction", P.z_score_contraction, {})):
        fig = fn(smp[w][sub], L[sub], variable_names=names, **kw)
        fig.suptitle(f"{run.name} / {w} ({len(sub)} random test stars)", y=1.02); fig.savefig(out / f"{name}_{w}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    # density recovery on the full test set; Δ = median - CLAM: scatter = std Δ, MAD = median |Δ - median Δ|
    fig, ax = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    for j, k in enumerate(names):
        x, y = L[:, j], med[w][:, j]; d = y - x
        lo, hi = np.percentile(np.r_[x, y], [0.5, 99.5])
        ax[j].hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1, extent=(lo, hi, lo, hi))
        ax[j].plot([lo, hi], [lo, hi], "r-", lw=0.8)
        ax[j].set_title(f"{k}\nscatter {d.std():.3g}   MAD {np.median(np.abs(d - np.median(d))):.3g}   bias {np.median(d):+.3g}", fontsize=11)
        ax[j].set_xlabel(f"CLAM {k}")
    ax[0].set_ylabel("posterior median")
    fig.suptitle(f"{run.name} / {w}: full test set ({len(rows):,} spectra)", y=1.02)
    fig.tight_layout(); fig.savefig(out / f"recovery_density_{w}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
C.log(f"wrote {out}")
