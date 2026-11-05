"""One downloaded spec-lite BOSS star (bank test split, own WDISP): SBI posterior vs NUTS.

Same forward model for both: emulator -> Vedant LSF (WDISP) -> profiled Chebyshev continuum
(cont_deg of the run), Gaussian likelihood with IVAR. Prior for NUTS = the SBI training prior
(uniform in the Korg box, log-uniform in vmic). NUTS starts at the MAP, found by Adam from
the SBI posterior median. Outputs (in <run>/hmc_vs_sbi/<sdss_id>/):
  corner.png (SBI vs NUTS contours, BOSS-CLAM labels), ppc_sbi.png, ppc_nuts.png, summary.json

    uv run -q --extra sbi --with astropy python sc_sbi/hmc_vs_sbi.py trained_model/sc_sbi/v4_vedant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (autocvd claims a free GPU)

import corner  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from numpyro.diagnostics import summary  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--row", type=int, default=None, help="bank row; default: a test-split giant with S/N 40-80")
p.add_argument("--spec", default=None, help="a spec-lite FITS outside the bank (needs <spec>_fit.json from fit_boss_spec.py)")
p.add_argument("--model", default="amortized")
p.add_argument("--n-sbi", type=int, default=4000)
p.add_argument("--chains", type=int, default=4)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--samples", type=int, default=1000)
p.add_argument("--n-ppc", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--init", choices=["map", "random", "sbi"], default="map",
               help="NUTS chain starts: MAP (+jitter), uniform random in the prior box, or SBI posterior draws")
a = p.parse_args()
rng = np.random.default_rng(a.seed)

run_dir = Path(a.run_dir)
cfg = C.load_run_config(run_dir)
ncfg = C.noise_config(cfg.get("noise"))
assert ncfg["lsf"] == "wdisp", "needs a v4 (WDISP) run"
box = {k: tuple(v) for k, v in cfg["box"].items()}
adapter = C.shared_adapter({k: tuple(v) for k, v in cfg["padded_box"].items()}, cfg.get("log_params"))
bank = C.load_bank(ncfg["bank"])

# ---- pick the star
emu = C.make_emulator_fn(cfg["emulator"])
if a.spec:      # a spec-lite file outside the bank: process it like build_wdisp_bank.py
    from astropy.io import fits
    sys.path.insert(0, str(C.PROJECT_ROOT / "script"))
    from korg_to_boss import repair_wdisp
    P, X = C.N_PIX, np.arange(4648, dtype=float)
    with fits.open(a.spec) as h:
        d, m = h[1].data, h[2].data[0]
        assert abs(d["LOGLAM"][0] - C.LOGLAM0) < 1e-6 and "WDISP" in d.columns.names
        sid = int(m["SDSS_ID"])
        dpx = np.log10(1 + float(m["Z"])) / 1e-4
        xs = X[:P] + dpx
        y0 = np.interp(xs, X[:len(d)], np.nan_to_num(d["FLUX"]))
        iv_raw = np.where(np.isfinite(d["IVAR"]), d["IVAR"], 0.0)
        i0 = np.clip(np.floor(xs).astype(int), 0, len(iv_raw) - 2); t = xs - i0
        v0, v1 = iv_raw[i0], iv_raw[i0 + 1]
        with np.errstate(divide="ignore"):
            iv0 = np.where((v0 > 0) & (v1 > 0), 1 / ((1 - t) ** 2 / v0 + t ** 2 / v1), 0.0)
        wd = np.interp(xs, X[:len(d)], repair_wdisp(d["WDISP"])).astype("float32")
    # no boss_clam continuum for this star: normalise by the Cheb-12 continuum that best matches the
    # emulator at the earlier MAP (fit_boss_spec.py, *_fit.json), i.e. onto the Korg normalised scale
    mp = json.loads(Path(str(Path(a.spec).with_suffix("")) + "_fit.json").read_text())["best"]
    th0 = jnp.asarray([[mp[k] for k in C.LABELS]], jnp.float32)
    mu0 = C.vedant_lsf(emu(th0), jnp.asarray(wd)[None], ncfg["lsf_kappa"], ncfg["lsf_rgrid"])
    ok0 = (iv0 > 0) & (C.WAVE[:P] > 3650) & (y0 > 0) & (y0 < 3 * np.median(y0[iv0 > 0]))
    B12 = jnp.asarray(C.cheb_basis(12))
    w0 = jnp.asarray(np.where(ok0, iv0, 0)[None], jnp.float32)
    cont = np.asarray(C.profile_continuum(jnp.ones_like(mu0), jnp.asarray(y0[None], jnp.float32) / mu0, w0 * mu0 ** 2, B12))[0]
    with np.errstate(all="ignore"):
        f, ivn = y0 / cont, iv0 * cont ** 2
    bad = ~np.isfinite(f) | ~np.isfinite(ivn) | (ivn <= 0) | (f <= 0) | (f >= 1.2) | (C.WAVE[:P] < 3650) | (cont <= 0)
    y = np.where(bad, 1.0, f).astype("float32"); iv = np.where(bad, 0.0, ivn).astype("float32")
    snr_ = C.measured_snr(y[None], iv[None]).astype("float32")
    pca = ((((wd - bank["pca_mean"]) @ bank["pca_comp"].T) / bank["pca_std"])[None]).astype("float32")
    ref = np.full(4, np.nan, "float32")
    tag = f"spec-lite {Path(a.spec).name} (not in boss_clam; continuum from the emulator MAP)"
else:
    if a.row is None:
        L, s = bank["labels"], bank["snr"]
        cand = np.flatnonzero((bank["split"] == 2) & C.boss_in_box(bank, box) & (L[:, 1] < 3.5) & (L[:, 0] < 6500)
                              & (s > 40) & (s < 80) & (bank["frac_masked"] < 0.08) & np.isfinite(L).all(1))
        a.row = int(rng.choice(cand))
    r = a.row
    sid = int(bank["sdss_id"][r])
    y, iv = bank["flux"][r].astype("float32"), bank["ivar"][r].astype("float32")
    wd = bank["wdisp"][r].astype("float32")
    ref = bank["labels"][r]
    snr_, pca = bank["snr"][r:r + 1], bank["lsf_pca"][r:r + 1]
    tag = f"spec-lite, bank row {r}, split {int(bank['split'][r])}"
out = run_dir / "hmc_vs_sbi" / str(sid)
out.mkdir(parents=True, exist_ok=True)
C.log(f"{tag}, sdss_id {sid}, S/N {float(snr_[0]):.1f}, BOSS-CLAM "
      + ", ".join(f"{k} {v:.3g}" for k, v in zip(C.BOSS_LABELS, ref)))

# ---- SBI posterior
cond = C.spectrum_features(y[None], iv[None], snr_, ncfg, C.load_mismatch(ncfg))
cond |= dict(wdisp=wd[None], lsf_pca=pca)
ex = adapter(C.label_dict(np.zeros((2, 6), "float32")) | {k: np.repeat(v, 2, 0) for k, v in cond.items()})
post = C.load_posterior(run_dir, a.model, ex)
sbi = C.sample_posterior(post, adapter, cond, a.n_sbi)[0]                 # (n, 6)
C.log("SBI sampled")

# ---- forward model (batched over theta)
B = jnp.asarray(C.cheb_basis(ncfg["cont_deg"]))
Y, W, WD = jnp.asarray(y), jnp.asarray(iv), jnp.asarray(wd)
lo = jnp.array([box[k][0] for k in C.LABELS]); hi = jnp.array([box[k][1] for k in C.LABELS])
vi = C.LABELS.index("vmic")


@jax.jit
def model(th):                                               # (n, 6) -> (n, P)
    n = th.shape[0]
    mu = C.vedant_lsf(emu(th), jnp.broadcast_to(WD, (n, WD.size)), ncfg["lsf_kappa"], ncfg["lsf_rgrid"])
    return C.profile_continuum(mu, jnp.broadcast_to(Y, mu.shape), jnp.broadcast_to(W, mu.shape), B)


chi2 = lambda th: jnp.sum(W * (Y - model(th)) ** 2, -1)
to_x = lambda u: lo + (hi - lo) * jax.nn.sigmoid(u)
to_u = lambda x: jnp.log((x - lo) / (hi - x))

# ---- MAP: Adam from the SBI median
x0 = jnp.asarray(np.median(sbi, 0))[None]
opt = optax.adam(optax.cosine_decay_schedule(0.05, 500, 0.02))


@jax.jit
def fit(u):
    def step(c, _):
        u, s = c
        upd, s = opt.update(jax.grad(lambda u: chi2(to_x(u)).sum())(u), s, u)
        return (optax.apply_updates(u, upd), s), None
    return jax.lax.scan(step, (u, opt.init(u)), None, 500)[0][0]


best = np.asarray(to_x(fit(to_u(x0))))[0]
npx = int((iv > 0).sum())
C.log(f"chi2/pix: SBI median {float(chi2(x0)[0]) / npx:.3f} -> MAP {float(chi2(jnp.asarray(best)[None])[0]) / npx:.3f}")


# ---- NUTS in logit space, SBI training prior
def potential(u):
    s = jax.nn.sigmoid(u); th = lo + (hi - lo) * s
    return 0.5 * chi2(th[None])[0] - jnp.sum(jnp.log(s) + jnp.log1p(-s)) + jnp.log(th[vi])


if a.init == "map":
    x_init = np.repeat(best[None], a.chains, 0)
elif a.init == "sbi":
    x_init = sbi[rng.choice(len(sbi), a.chains, replace=False)]
else:                                                     # prior draws: uniform box, log-uniform vmic
    l_, h_ = np.asarray(lo), np.asarray(hi)
    x_init = l_ + (h_ - l_) * rng.uniform(0.02, 0.98, (a.chains, 6))
    x_init[:, vi] = np.exp(rng.uniform(np.log(l_[vi] * 1.02), np.log(h_[vi] * 0.98), a.chains))
init = jnp.asarray(to_u(jnp.asarray(x_init, jnp.float32)) + 0.01 * rng.standard_normal((a.chains, 6)), jnp.float32)
mcmc = MCMC(NUTS(potential_fn=potential, dense_mass=True, target_accept_prob=0.9),
            num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
            chain_method="vectorized", progress_bar=False)
mcmc.run(jax.random.PRNGKey(a.seed), init_params=init, extra_fields=("diverging",))
th = np.asarray(to_x(mcmc.get_samples(group_by_chain=True)))             # (chains, samples, 6)
ndiv = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
sm = summary(th, group_by_chain=True)["Param:0"]
nuts = th.reshape(-1, 6)
C.log(f"NUTS ({a.init} init): {ndiv} divergences, r_hat max {np.max(sm['r_hat']):.3f}, n_eff min {np.min(sm['n_eff']):.0f}")
chain_med = np.median(th, 1)
chain_chi2 = np.asarray(chi2(jnp.asarray(chain_med))) / npx
C.log("  per chain: start -> median (chi2/pix at median)")
for c in range(a.chains):
    C.log("   " + " ".join(f"{v:8.3g}" for v in x_init[c]) + "  ->  "
          + " ".join(f"{v:8.3g}" for v in chain_med[c]) + f"   ({chain_chi2[c]:.4f})")

# ---- summary
q = lambda x: {"median": float(np.median(x)), "p16": float(np.percentile(x, 16)), "p84": float(np.percentile(x, 84))}
res = {"star": tag, "sdss_id": sid, "snr": float(snr_[0]), "n_px": npx,
       "boss_clam": {k: float(v) for k, v in zip(C.BOSS_LABELS, ref) if np.isfinite(v)}, "map": dict(zip(C.LABELS, map(float, best))),
       "sbi": {k: q(sbi[:, i]) for i, k in enumerate(C.LABELS)},
       "nuts": {k: q(nuts[:, i]) | {"r_hat": float(sm["r_hat"][i]), "n_eff": float(sm["n_eff"][i])}
                for i, k in enumerate(C.LABELS)}, "nuts_divergences": ndiv,
       "init": a.init, "chain_start": x_init.tolist(), "chain_median": chain_med.tolist(),
       "chain_chi2pix": chain_chi2.tolist()}
for k in C.LABELS:
    s_, n_ = res["sbi"][k], res["nuts"][k]
    C.log(f"  {k:9s} SBI {s_['median']:9.3f} [{s_['p16']:.3f}, {s_['p84']:.3f}]   "
          f"NUTS {n_['median']:9.3f} [{n_['p16']:.3f}, {n_['p84']:.3f}]   MAP {res['map'][k]:.3f}"
          + (f"   CLAM {res['boss_clam'][k]:.3f}" if k in res["boss_clam"] else ""))

# ---- corner: both posteriors
names = ["Teff", "logg", "[M/H]", "[α/M]", "[C/M]", "vmic"]
lims = [(min(np.percentile(sbi[:, i], 0.5), np.percentile(nuts[:, i], 0.5)),
         max(np.percentile(sbi[:, i], 99.5), np.percentile(nuts[:, i], 99.5))) for i in range(6)]
lims = [(l - 0.1 * (h - l), h + 0.1 * (h - l)) for l, h in lims]
truths = [float(v) if np.isfinite(v) else None for v in ref] + [None, None]
kw = dict(labels=names, range=lims, bins=35, smooth=1.0, levels=(0.39, 0.86), plot_datapoints=False,
          plot_density=False, fill_contours=False)
fig = corner.corner(sbi, color="#0072B2", hist_kwargs=dict(density=True), **kw)
corner.corner(nuts, color="#D55E00", fig=fig, hist_kwargs=dict(density=True), truths=truths, truth_color="k", **kw)
corner.overplot_lines(fig, list(best), color="#009E73", ls="--", lw=1)
fig.legend(handles=[Line2D([], [], color="#0072B2", label=f"SBI ({a.model}, {len(sbi)} samples)"),
                    Line2D([], [], color="#D55E00", label=f"NUTS, {a.init} starts ({a.chains}×{a.samples}, {ndiv} div., r̂ max {np.max(sm['r_hat']):.2f})"),
                    Line2D([], [], color="#009E73", ls="--", label="MAP (Adam from SBI median)"),
                    Line2D([], [], color="k", label="BOSS-CLAM")],
           loc="upper right", bbox_to_anchor=(0.98, 0.93), fontsize=12, frameon=False)
fig.suptitle(f"sdss_id {sid} ({tag}), S/N {float(snr_[0]):.0f}: 1σ / 2σ contours", y=1.01, fontsize=13)
fig.savefig(out / (f"corner_{a.init}.png" if a.init != "map" else "corner.png"), dpi=90, bbox_inches="tight"); plt.close(fig)

# ---- posterior predictive checks
wave = C.WAVE[:C.N_PIX]
zooms = [("full", wave[0], wave[-1]), ("Ca H&K", 3900, 4000), ("Mg b", 5150, 5200),
         ("Hα", 6530, 6600), ("Ca triplet", 8480, 8680)]


def ppc(samples, label, color, fn):
    d = samples[rng.choice(len(samples), a.n_ppc, replace=False)]
    mu = np.concatenate([np.asarray(model(jnp.asarray(d[i:i + 50]))) for i in range(0, len(d), 50)])
    sig = np.where(iv > 0, 1 / np.sqrt(np.maximum(iv, 1e-30)), np.nan)
    pred = mu + rng.standard_normal(mu.shape) * np.nan_to_num(sig)       # predictive: model + noise
    m50 = np.median(mu, 0)
    b_mu = np.percentile(mu, [2.5, 16, 84, 97.5], 0)
    b_pr = np.percentile(pred, [2.5, 97.5], 0)
    good = iv > 0
    inside = np.mean((y[good] >= b_pr[0][good]) & (y[good] <= b_pr[1][good]))
    c2 = np.sum(iv * (y - m50) ** 2) / npx
    # full spectrum on top (flux + residuals), 4 line zooms below
    fig = plt.figure(figsize=(22, 11), layout="constrained")
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1])
    panels = [(fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), zooms[0])]
    for j in range(4):
        panels.append((fig.add_subplot(gs[2, j]), fig.add_subplot(gs[3, j]), zooms[j + 1]))
    for top, bot, (nm, w0, w1) in panels:
        m = (wave >= w0) & (wave <= w1)
        ym = np.where(good, y, np.nan)
        top.fill_between(wave[m], b_pr[0][m], b_pr[1][m], color=color, alpha=0.15, lw=0, label="95 % predictive (model + noise)")
        top.fill_between(wave[m], b_mu[0][m], b_mu[3][m], color=color, alpha=0.35, lw=0, label="95 % model band")
        top.fill_between(wave[m], b_mu[1][m], b_mu[2][m], color=color, alpha=0.6, lw=0, label="68 % model band")
        top.plot(wave[m], ym[m], "k-", lw=0.6, label="data")
        top.plot(wave[m], m50[m], color=color, lw=0.6, label="median model")
        z = (y - m50) * np.sqrt(iv)
        bot.fill_between(wave[m], ((b_mu[0] - m50) * np.sqrt(iv))[m], ((b_mu[3] - m50) * np.sqrt(iv))[m],
                         color=color, alpha=0.35, lw=0)
        bot.plot(wave[m], np.where(good, z, np.nan)[m], "k-", lw=0.5)
        bot.axhline(0, color=color, lw=0.6)
        for h in (-2, 2):
            bot.axhline(h, color="0.5", lw=0.5, ls=":")
        bot.set_ylim(-5, 5)
        top.set_xlim(w0, w1); bot.set_xlim(w0, w1); top.set_title(nm, fontsize=11)
        lo_, hi_ = np.nanpercentile(ym[m], [0.5, 99.5]); pad = 0.08 * (hi_ - lo_)
        top.set_ylim(lo_ - pad, hi_ + pad)
        bot.set_xlabel("rest wavelength [Å]")
    panels[0][0].set_ylabel("normalised flux"); panels[0][1].set_ylabel("(data − median)/σ")
    panels[1][0].set_ylabel("normalised flux"); panels[1][1].set_ylabel("(data − median)/σ")
    panels[0][0].legend(loc="lower right", fontsize=9, ncol=5)
    fig.suptitle(f"Posterior predictive check, {label}: sdss_id {sid}, {a.n_ppc} draws; median χ²/pix {c2:.3f}, "
                 f"{inside:.1%} of pixels inside the 95 % predictive band", fontsize=13)
    fig.savefig(out / fn, dpi=100); plt.close(fig)
    return {"chi2pix_median_model": float(c2), "frac_in_95_predictive": float(inside)}


res["ppc_sbi"] = ppc(sbi, f"SBI ({a.model})", "#0072B2", "ppc_sbi.png")
res["ppc_nuts"] = ppc(nuts, "NUTS", "#D55E00", f"ppc_nuts_{a.init}.png" if a.init != "map" else "ppc_nuts.png")
C.log(f"PPC SBI {res['ppc_sbi']}   NUTS {res['ppc_nuts']}")
(out / (f"summary_{a.init}.json" if a.init != "map" else "summary.json")).write_text(json.dumps(res, indent=1))
np.savez(out / (f"samples_{a.init}.npz" if a.init != "map" else "samples.npz"), sbi=sbi, nuts=th, map=best, names=C.LABELS)
C.log(f"wrote {out}")
