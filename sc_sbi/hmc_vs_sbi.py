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
p.add_argument("--subdir", default=None, help="write outputs to <star>/<subdir>/")
p.add_argument("--nuts-only", action="store_true", help="corner shows only the NUTS posterior (+ PPC inset)")
p.add_argument("--from-samples", action="store_true", help="skip NUTS, reload samples_<init>.npz from the output dir (replot)")
p.add_argument("--sampler", choices=["nuts", "nss"], default="nuts",
               help="nss = blackjax nested slice sampling on the GPU (needs --with blackjax); implies --init nss")
p.add_argument("--nlive", type=int, default=2000)
p.add_argument("--ref", action="append", default=[], metavar="LABEL:Teff,logg,m_H[:eTeff,elogg,em_H]",
               help="external labels drawn (±1σ) on the --overlay plot, e.g. Astra BOSSNet; repeatable")
p.add_argument("--overlay", default=None, metavar="NPZ:LABEL",
               help="with --nuts-only: also write corner_<init>_vs_<LABEL>.png overlaying another run's samples (its 'nuts' array)")
p.add_argument("--num-delete", type=int, default=500, help="nss: particles replaced in parallel per step")
p.add_argument("--inner-steps", type=int, default=30, help="nss: slice steps per new particle (>= 5 x dim)")
p.add_argument("--clam-row", type=int, default=None,
               help="a boss_clam prepared coadd (C.load_boss row) with the WDISP of its bank visit, as evaluate.py "
                    "--boss-source clam; BOSS-CLAM labels ±e_labels become a reference")
p.add_argument("--vs-sbi", action="store_true",
               help="with --nuts-only: corner + PPC comparison of NUTS vs the SBI posterior (+ references)")
p.add_argument("--out", default=None, help="output dir (overrides hmc_vs_sbi/<sdss_id>/<subdir>)")
p.add_argument("--norm", choices=["emu", "clip"], default="emu",
               help="--spec continuum: Cheb-12 matched to the emulator MAP (emu) or data-only sigma-clipped Cheb-12 (clip)")
p.add_argument("--emu-err", default=None, metavar="Tlo,Thi,mHlo,mHhi",
               help="add the emulator error to sigma: per-pixel RMS(emu - Korg) after this star's LSF, "
                    "over emulator-test Korg spectra in this Teff/m_H range")
a = p.parse_args()
if a.sampler == "nss":
    a.init, a.chains = "nss", 1
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
    ok0 = (iv0 > 0) & (C.WAVE[:P] > 3650) & (y0 > 0) & (y0 < 3 * np.median(y0[iv0 > 0]))
    if a.norm == "emu":
        mp = json.loads(Path(str(Path(a.spec).with_suffix("")) + "_fit.json").read_text())["best"]
        th0 = jnp.asarray([[mp[k] for k in C.LABELS]], jnp.float32)
        mu0 = C.vedant_lsf(emu(th0), jnp.asarray(wd)[None], ncfg["lsf_kappa"], ncfg["lsf_rgrid"])
        B12 = jnp.asarray(C.cheb_basis(12))
        w0 = jnp.asarray(np.where(ok0, iv0, 0)[None], jnp.float32)
        cont = np.asarray(C.profile_continuum(jnp.ones_like(mu0), jnp.asarray(y0[None], jnp.float32) / mu0, w0 * mu0 ** 2, B12))[0]
    else:       # data only: Cheb-12, iteratively drop pixels > 1 sigma below (lines) / > 3 sigma above the fit
        B12, keep = C.cheb_basis(12).astype(float), ok0.copy()
        for _ in range(50):
            w = np.sqrt(iv0 * keep)
            cont = B12 @ np.linalg.lstsq(B12 * w[:, None], y0 * w, rcond=None)[0]
            r_ = (y0 - cont) * np.sqrt(iv0)
            new = ok0 & (r_ > -1) & (r_ < 3)
            if (new == keep).all():
                break
            keep = new
    with np.errstate(all="ignore"):
        f, ivn = y0 / cont, iv0 * cont ** 2
    bad = ~np.isfinite(f) | ~np.isfinite(ivn) | (ivn <= 0) | (f <= 0) | (f >= 1.2) | (C.WAVE[:P] < 3650) | (cont <= 0)
    y = np.where(bad, 1.0, f).astype("float32"); iv = np.where(bad, 0.0, ivn).astype("float32")
    snr_ = C.measured_snr(y[None], iv[None]).astype("float32")
    pca = ((((wd - bank["pca_mean"]) @ bank["pca_comp"].T) / bank["pca_std"])[None]).astype("float32")
    ref = np.full(4, np.nan, "float32")
    tag = f"spec-lite {Path(a.spec).name} (not in boss_clam; continuum " + \
          ("from the emulator MAP)" if a.norm == "emu" else "data-only, clipped Cheb-12)")
elif a.clam_row is not None:  # boss_clam coadd (BOSS-CLAM continuum) + WDISP/PCA of its bank visit
    boss, r = C.load_boss(), a.clam_row
    sid = int(boss["sdss_id"][r])
    bi_ = int(np.flatnonzero(bank["sdss_id"] == sid)[-1])          # = evaluate.py's sdss_id -> bank lookup
    y, iv = boss["flux"][r].astype("float32"), boss["ivar"][r].astype("float32")
    wd = bank["wdisp"][bi_].astype("float32")
    ref, e_ref = boss["labels"][r], np.nan_to_num(boss["e_labels"][r])
    snr_, pca = boss["snr"][r:r + 1], bank["lsf_pca"][bi_:bi_ + 1]
    tag = f"boss_clam coadd row {r}, split {int(boss['split'][r])}"
    del boss
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
out = Path(a.out) if a.out else run_dir / "hmc_vs_sbi" / str(sid) / (a.subdir or "")
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

if a.emu_err:   # sigma^2 = 1/ivar + sigma_emu^2, sigma_emu from the emulator's held-out Korg test spectra
    import h5py
    tlo, thi, mlo, mhi = map(float, a.emu_err.split(","))
    with h5py.File(C.KORG_CACHE, "r") as f:
        Lk = np.stack([f[k][:] for k in C.LABELS], 1)
        te = np.sort(np.load(Path(cfg["emulator"]).parent / "split_idx.npz")["test_idx"])
        te = te[(Lk[te, 0] >= tlo) & (Lk[te, 0] <= thi) & (Lk[te, 2] >= mlo) & (Lk[te, 2] <= mhi)]
        fk = f["norm_flux"][te]
    lsf = jax.jit(lambda fl: C.vedant_lsf(fl, jnp.broadcast_to(jnp.asarray(wd), fl.shape), ncfg["lsf_kappa"], ncfg["lsf_rgrid"]))
    d2 = sum(np.sum((np.asarray(lsf(emu(jnp.asarray(Lk[te[i:i + 256]]))))
                     - np.asarray(lsf(jnp.asarray(fk[i:i + 256])))) ** 2, 0) for i in range(0, te.size, 256))
    s_emu = np.sqrt(d2 / te.size).astype("float32")
    g_ = iv > 0
    C.log(f"emulator error from {te.size} Korg test spectra: sigma_emu median {np.median(s_emu[g_]):.2e}, "
          f"max {s_emu[g_].max():.2e}; photon sigma median {np.median(1 / np.sqrt(iv[g_])):.2e}")
    iv = np.where(g_, 1 / (1 / np.maximum(iv, 1e-30) + s_emu ** 2), 0).astype("float32")
    np.save(out / "sigma_emu.npy", s_emu)

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
sfx = f"_{a.init}" if a.init != "map" else ""
if a.from_samples:
    th = np.load(out / f"samples{sfx}.npz")["nuts"]
    old = json.loads((out / f"summary{sfx}.json").read_text())
    ndiv, x_init = old["nuts_divergences"], np.asarray(old["chain_start"])
    ns_info = old.get("nss")
    a.chains, a.samples = th.shape[:2]
elif a.sampler == "nss":                    # unit cube -> box (log-uniform vmic): same prior as NUTS / SBI
    import blackjax
    import blackjax.ns.utils as nsu
    llo, lhi = jnp.log(lo[vi]), jnp.log(hi[vi])
    cube_x = lambda s_: (lo + (hi - lo) * s_).at[..., vi].set(jnp.exp(llo + (lhi - llo) * s_[..., vi]))
    logprior = lambda s_: jnp.where(jnp.all((s_ >= 0) & (s_ <= 1)), 0.0, -jnp.inf)
    loglike = lambda s_: -0.5 * chi2(cube_x(s_)[None])[0]
    algo = blackjax.nss(logprior, loglike, num_inner_steps=a.inner_steps, num_delete=a.num_delete)
    key = jax.random.PRNGKey(a.seed)
    state = algo.init(jax.random.uniform(key, (a.nlive, 6)))
    step = jax.jit(algo.step)
    dead, it = [], 0
    while float(state.integrator.logZ_live - state.integrator.logZ) > -3:     # live points hold < e^-3 of Z
        key, k = jax.random.split(key)
        state, info = step(k, state); dead.append(info); it += 1
        if it % 20 == 0:
            C.log(f"  NS step {it}: logZ {float(state.integrator.logZ):.2f}, "
                  f"max loglike {float(state.particles.loglikelihood.max()):.2f}")
    fin = nsu.finalise(state, dead)
    key, k1, k2 = jax.random.split(key, 3)
    s_post = nsu.sample(k1, fin, a.samples)
    th = np.asarray(cube_x(s_post.position))[None]                                # (1, samples, 6)
    lw = nsu.log_weights(k2, fin, 100)
    logZ = jax.scipy.special.logsumexp(lw, axis=0)
    ns_info = {"logZ": float(logZ.mean()), "logZ_err": float(logZ.std()), "steps": it,
               "n_dead": int(fin.particles.loglikelihood.size), "ess": float(nsu.ess(k2, fin)),
               "nlive": a.nlive, "num_delete": a.num_delete, "inner_steps": a.inner_steps}
    C.log(f"NSS: {ns_info}")
    ndiv = 0
else:
    mcmc = MCMC(NUTS(potential_fn=potential, dense_mass=True, target_accept_prob=0.9),
                num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(a.seed), init_params=init, extra_fields=("diverging",))
    th = np.asarray(to_x(mcmc.get_samples(group_by_chain=True)))         # (chains, samples, 6)
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
if a.sampler == "nss" and ns_info:
    res["nss"] = ns_info
for k in C.LABELS:
    s_, n_ = res["sbi"][k], res["nuts"][k]
    C.log(f"  {k:9s} SBI {s_['median']:9.3f} [{s_['p16']:.3f}, {s_['p84']:.3f}]   "
          f"NUTS {n_['median']:9.3f} [{n_['p16']:.3f}, {n_['p84']:.3f}]   MAP {res['map'][k]:.3f}"
          + (f"   CLAM {res['boss_clam'][k]:.3f}" if k in res["boss_clam"] else ""))

# ---- posterior predictive checks
wave = C.WAVE[:C.N_PIX]
zooms = [("full", wave[0], wave[-1]), ("Ca H&K", 3900, 4000), ("Mg b", 5150, 5200),
         ("Hα", 6530, 6600), ("Ca triplet", 8480, 8680)]


def ppc_bands(samples):
    d = samples[rng.choice(len(samples), a.n_ppc, replace=False)]
    mu = np.concatenate([np.asarray(model(jnp.asarray(d[i:i + 50]))) for i in range(0, len(d), 50)])
    sig = np.where(iv > 0, 1 / np.sqrt(np.maximum(iv, 1e-30)), np.nan)
    pred = mu + rng.standard_normal(mu.shape) * np.nan_to_num(sig)       # predictive: model + noise
    return np.median(mu, 0), np.percentile(mu, [2.5, 16, 84, 97.5], 0), np.percentile(pred, [2.5, 97.5], 0)


def ppc(samples, label, color, fn):
    m50, b_mu, b_pr = ppc_bands(samples)
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


# ---- corner: both posteriors
names = ["Teff", "logg", "[M/H]", "[α/M]", "[C/M]", "vmic"]
lims = [(min(np.percentile(sbi[:, i], 0.5), np.percentile(nuts[:, i], 0.5)),
         max(np.percentile(sbi[:, i], 99.5), np.percentile(nuts[:, i], 99.5))) for i in range(6)]
lims = [(l - 0.1 * (h - l), h + 0.1 * (h - l)) for l, h in lims]
truths = [float(v) if np.isfinite(v) else None for v in ref] + [None, None]
kw = dict(labels=names, range=lims, bins=35, smooth=1.0, levels=(0.39, 0.86), plot_datapoints=False,
          plot_density=False, fill_contours=False)


def nuts_only_plots(th, sm, ndiv, out, title):
    """Corner of the pooled NUTS chains with a simplified PPC in the empty upper triangle; trace plots."""
    nc = th.shape[0]; cc = plt.cm.viridis(np.linspace(0, 0.9, nc))
    pooled = th.reshape(-1, 6)
    lim = [tuple(np.percentile(pooled[:, i], [0.2, 99.8])) for i in range(6)]
    lim = [(l - 0.1 * (h - l), h + 0.1 * (h - l)) for l, h in lim]
    titles = [f"{nm}  r̂ {sm['r_hat'][i]:.3f}, n_eff {sm['n_eff'][i]:.0f}" for i, nm in enumerate(names)]
    fig = corner.corner(pooled, labels=names, range=lim, bins=40, smooth=1.0, levels=(0.39, 0.86), color="#D55E00",
                        plot_datapoints=False, plot_density=False, hist_kwargs=dict(density=True, lw=1.5))
    ax = np.array(fig.axes).reshape(6, 6)
    for i, nm in enumerate(names):                                   # median and central 68 % interval
        q16, q50, q84 = np.percentile(pooled[:, i], [16, 50, 84]); f = ".0f" if i == 0 else ".2f"
        ax[i, i].set_title(f"{nm} = ${q50:{f}}^{{+{q84 - q50:{f}}}}_{{-{q50 - q16:{f}}}}$", fontsize=11)
    # simplified PPC: full spectrum + residuals
    m50, b_mu, b_pr = ppc_bands(pooled)
    good = iv > 0; ym = np.where(good, y, np.nan); z = np.where(good, (y - m50) * np.sqrt(iv), np.nan)
    inside = np.mean((y[good] >= b_pr[0][good]) & (y[good] <= b_pr[1][good]))
    c2 = np.sum(iv * (y - m50) ** 2) / npx
    P_ = lambda i, j: ax[i, j].get_position()                      # empty upper triangle: rows 0-1, cols 2-5
    x0, x1, yt, y1b = P_(0, 2).x0 + 0.03, P_(0, 5).x1, P_(0, 5).y1, P_(1, 5).y0
    h1 = yt - y1b
    boxes = [("full", wave[0], wave[-1], [x0, y1b + 0.42 * h1, x1 - x0, 0.58 * h1], [x0, y1b + 0.12 * h1, x1 - x0, 0.25 * h1])]
    for nm, w0, w1, rt, rb in boxes:
        top, bot = fig.add_axes(rt), fig.add_axes(rb)
        m = (wave >= w0) & (wave <= w1)
        top.fill_between(wave[m], b_pr[0][m], b_pr[1][m], color="#D55E00", alpha=0.3, lw=0, label="95 % predictive")
        top.plot(wave[m], ym[m], "k-", lw=0.4, label="data")
        top.plot(wave[m], m50[m], color="#D55E00", lw=0.5, label="median model")
        bot.plot(wave[m], z[m], "k-", lw=0.4); bot.axhline(0, color="#D55E00", lw=0.6)
        for hh in (-2, 2):
            bot.axhline(hh, color="0.5", lw=0.5, ls=":")
        lo_, hi_ = np.nanpercentile(ym[m], [0.5, 99.5]); pad = 0.08 * (hi_ - lo_)
        top.set_ylim(lo_ - pad, hi_ + pad); bot.set_ylim(-5, 5)
        top.set_xlim(w0, w1); bot.set_xlim(w0, w1); top.set_xticklabels([])
        top.set_title(nm, fontsize=9, loc="left"); top.tick_params(labelsize=8); bot.tick_params(labelsize=8)
        if nm == "full":
            bot.set_xlabel("rest wavelength [Å]", fontsize=8)
        if nm == "full":
            top.set_ylabel("norm. flux", fontsize=8); bot.set_ylabel("resid/σ", fontsize=8)
            top.legend(loc="lower right", fontsize=8, ncol=3, frameon=False)
            top.set_title(f"posterior predictive: median χ²/pix {c2:.3f}, {inside:.1%} of px in 95 % band",
                          fontsize=10, loc="right")
    extra = (f"{th.shape[1]} resampled draws, ln Z = {res['nss']['logZ']:.1f} ± {res['nss']['logZ_err']:.1f}"
             if "nss" in res else f"{nc}×{th.shape[1]} draws, {ndiv} div.")
    fig.suptitle(f"{title} ({extra})", y=1.01, fontsize=13)
    fig.savefig(out / f"corner_{a.init}.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    if a.sampler == "nss":                                           # resampled iid draws: no chains to trace
        return
    fig, ax = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    step = max(1, th.shape[1] // 2000)
    for i in range(6):
        for c in range(nc):
            ax[i].plot(np.arange(0, th.shape[1], step), th[c, ::step, i], color=cc[c], lw=0.3, alpha=0.7)
        ax[i].set_ylabel(names[i]); ax[i].set_title(titles[i], fontsize=9, loc="right")
    ax[-1].set_xlabel("post-warmup draw")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out / f"trace_{a.init}.png", dpi=90); plt.close(fig)


def compare_plot(sets, fn, title):
    """Overlay corner + PPC of several posteriors; sets = [(samples (n, 6), label, color)]."""
    allp = np.concatenate([s_ for s_, _, _ in sets])
    lim = [tuple(np.percentile(allp[:, i], [0.2, 99.8])) for i in range(6)]
    for _, v, e, _ in refs:                                         # widen to show the reference ±1σ
        for i in range(len(v)):
            lim[i] = (min(lim[i][0], v[i] - 1.5 * e[i]), max(lim[i][1], v[i] + 1.5 * e[i]))
    lim = [(l - 0.1 * (h - l), h + 0.1 * (h - l)) for l, h in lim]
    fig = None
    for s_, _, col in sets:
        fig = corner.corner(s_, fig=fig, labels=names, range=lim, bins=40, smooth=1.0, levels=(0.39, 0.86), color=col,
                            plot_datapoints=False, plot_density=False, hist_kwargs=dict(density=True, lw=1.5, color=col))
    ax = np.array(fig.axes).reshape(6, 6)
    for lab, v, e, col in refs:
        for i in range(len(v)):
            ax[i, i].axvline(v[i], color=col, lw=1.2); ax[i, i].axvspan(v[i] - e[i], v[i] + e[i], color=col, alpha=0.2, lw=0)
            for j in range(i):
                ax[i, j].errorbar(v[j], v[i], xerr=e[j], yerr=e[i], fmt="s", color=col, ms=5, lw=1.2, capsize=2)
    for i, nm in enumerate(names):
        f = ".0f" if i == 0 else ".2f"
        for k, (s_, _, col) in enumerate(sets):
            q16, q50, q84 = np.percentile(s_[:, i], [16, 50, 84])
            ax[i, i].text(0.5, 1.04 + 0.16 * (len(sets) - 1 - k), f"{nm} = ${q50:{f}}^{{+{q84 - q50:{f}}}}_{{-{q50 - q16:{f}}}}$",
                          color=col, transform=ax[i, i].transAxes, ha="center", va="bottom", fontsize=10)
    P_ = lambda i, j: ax[i, j].get_position()
    x0, x1, yt, y1b = P_(0, 2).x0 + 0.03, P_(0, 5).x1, P_(0, 5).y1, P_(1, 5).y0
    h1 = yt - y1b
    top = fig.add_axes([x0, y1b + 0.5 * h1, x1 - x0, 0.5 * h1]); bot = fig.add_axes([x0, y1b + 0.26 * h1, x1 - x0, 0.2 * h1])
    good = iv > 0; ym = np.where(good, y, np.nan)
    top.plot(wave, ym, "k-", lw=0.4, label="data", zorder=0)
    for hh in (-2, 2):
        bot.axhline(hh, color="0.5", lw=0.5, ls=":")
    leg = []
    for k, (s_, lab, col) in enumerate(sets):            # first set filled, the others as dashed band edges
        m50, _, b_pr = ppc_bands(s_)
        inside = np.mean((y[good] >= b_pr[0][good]) & (y[good] <= b_pr[1][good]))
        c2 = np.sum(iv * (y - m50) ** 2) / npx
        if k == 0:
            top.fill_between(wave, b_pr[0], b_pr[1], color=col, alpha=0.3, lw=0)
        else:
            top.plot(wave, b_pr[0], color=col, lw=0.5, ls="--"); top.plot(wave, b_pr[1], color=col, lw=0.5, ls="--")
        top.plot(wave, m50, color=col, lw=0.5)
        bot.plot(wave, np.where(good, (y - m50) * np.sqrt(iv), np.nan), color=col, lw=0.4, alpha=0.6)
        leg.append(Line2D([], [], color=col, lw=4 if k == 0 else 1, ls="-" if k == 0 else "--", alpha=0.6 if k == 0 else 1,
                          label=f"{lab}: 95 % predictive + median (χ²/pix {c2:.3f}, {inside:.1%} in band)"))
    lo_, hi_ = np.nanpercentile(ym, [0.5, 99.5]); pad = 0.08 * (hi_ - lo_)
    top.set_ylim(lo_ - pad, hi_ + pad); bot.set_ylim(-5, 5)
    for A in (top, bot):
        A.set_xlim(wave[0], wave[-1]); A.tick_params(labelsize=8)
    top.set_xticklabels([]); top.set_ylabel("norm. flux", fontsize=8); bot.set_ylabel("resid/σ", fontsize=8)
    bot.set_xlabel("rest wavelength [Å]", fontsize=8)
    top.legend(handles=[Line2D([], [], color="k", lw=0.8, label="data")] + leg, loc="lower right", fontsize=8, frameon=False)
    top.set_title("posterior predictive checks", fontsize=10, loc="left")
    if refs:
        fig.legend(handles=[Line2D([], [], color=col, lw=2, label=lab) for _, lab, col in sets]
                   + [Line2D([], [], color=col, marker="s", lw=1.2, label=f"{lab} (±1σ)") for lab, _, _, col in refs],
                   loc="upper right", bbox_to_anchor=(0.98, 0.6), fontsize=12, frameon=False)
    fig.suptitle(title, y=1.04, fontsize=13)
    fig.savefig(out / fn, dpi=90, bbox_inches="tight"); plt.close(fig)


refs = []
for r_, col in zip(a.ref, ["k", "#CC79A7", "#009E73"]):
    parts = r_.split(":")
    v = [float(t) for t in parts[1].split(",")]
    refs.append((parts[0], v, [float(t) for t in parts[2].split(",")] if len(parts) > 2 else [0.0] * len(v), col))
if a.clam_row is not None:
    refs.insert(0, ("BOSS-CLAM", [float(t) for t in ref], [float(t) for t in e_ref], "k"))

def ref_samples(v, e, n=300, steps=400):
    """External (Teff, logg, m_H) ~ N(v, e); the remaining labels (alpha, C, vmic) profiled by Adam per draw,
    i.e. the best this forward model can do at those labels."""
    k = len(v)
    fixed = jnp.asarray(np.asarray(v) + np.asarray(e) * rng.standard_normal((n, k)), jnp.float32)
    fixed = jnp.clip(fixed, lo[:k] + 1e-3 * (hi[:k] - lo[:k]), hi[:k] - 1e-3 * (hi[:k] - lo[:k]))
    u0 = jnp.asarray(np.repeat(np.asarray(to_u(jnp.asarray(np.median(nuts, 0))))[None, k:], n, 0))
    full = lambda u: jnp.concatenate([fixed, lo[k:] + (hi[k:] - lo[k:]) * jax.nn.sigmoid(u)], 1)
    opt_ = optax.adam(optax.cosine_decay_schedule(0.05, steps, 0.02))

    @jax.jit
    def run(u):
        def st(c, _):
            u, s_ = c
            upd, s_ = opt_.update(jax.grad(lambda u: chi2(full(u)).sum())(u), s_, u)
            return (optax.apply_updates(u, upd), s_), None
        return jax.lax.scan(st, (u, opt_.init(u)), None, steps)[0][0]
    return np.asarray(full(run(u0)))


def ppc_compare(sets, fn, title):
    """PPC of several posteriors on the full spectrum + logg-sensitive lines; sets = [(samples, label, color)]."""
    good = iv > 0; ym = np.where(good, y, np.nan)
    bands = []
    for s_, lab, col in sets:
        m50, _, b_pr = ppc_bands(s_)
        c2 = np.sum(iv * (y - m50) ** 2) / npx
        inside = np.mean((y[good] >= b_pr[0][good]) & (y[good] <= b_pr[1][good]))
        bands.append((m50, b_pr, f"{lab}: χ²/pix {c2:.3f}, {inside:.1%} in 95 % band", col))
        C.log(f"PPC {lab}: chi2/pix {c2:.4f}, in-band {inside:.3f}")
    zz = zooms                                                     # same line zooms as ppc()
    fig = plt.figure(figsize=(22, 11), layout="constrained")
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1])
    panels = [(fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), zz[0])]
    panels += [(fig.add_subplot(gs[2, j]), fig.add_subplot(gs[3, j]), zz[j + 1]) for j in range(4)]
    for top, bot, (nm, w0, w1) in panels:
        m = (wave >= w0) & (wave <= w1)
        top.plot(wave[m], ym[m], "k-", lw=0.6, label="data", zorder=5)
        for ib, (m50, b_pr, lab, col) in enumerate(bands):
            if ib == 0:                                            # first set filled, others dashed band edges
                top.fill_between(wave[m], b_pr[0][m], b_pr[1][m], color=col, alpha=0.2, lw=0)
            else:
                for b_ in b_pr:
                    top.plot(wave[m], b_[m], color=col, lw=0.6, ls="--", alpha=0.8)
            top.plot(wave[m], m50[m], color=col, lw=0.9, label=lab)
            bot.plot(wave[m], np.where(good, (y - m50) * np.sqrt(iv), np.nan)[m], color=col, lw=0.6, alpha=0.8)
        for hh in (-2, 0, 2):
            bot.axhline(hh, color="0.5", lw=0.5, ls=":")
        lo_, hi_ = np.nanpercentile(ym[m], [0.5, 99.5]); pad = 0.08 * (hi_ - lo_)
        top.set_ylim(lo_ - pad, hi_ + pad); bot.set_ylim(-5, 5)
        top.set_xlim(w0, w1); bot.set_xlim(w0, w1); top.set_title(nm, fontsize=11)
        bot.set_xlabel("rest wavelength [Å]")
    for top, bot, _ in panels[:2]:
        top.set_ylabel("normalised flux"); bot.set_ylabel("(data − median)/σ")
    panels[0][0].legend(loc="lower right", fontsize=10)
    fig.suptitle(title + " — bands: 95 % predictive (model + noise)", fontsize=13)
    fig.savefig(out / fn, dpi=100); plt.close(fig)


if a.nuts_only and refs:
    me = "nested sampling" if a.sampler == "nss" else "NUTS"
    sets = [(nuts, me, "#D55E00")] + ([(sbi, f"SBI ({a.model})", "#0072B2")] if a.vs_sbi else [])
    nm6 = ["Teff", "logg", "[M/H]", "[α/M]", "[C/M]", "vmic"]
    for (lab, v, e, _), col in zip(refs, ["#009E73", "#CC79A7", "#56B4E9"]):
        k = len(v)
        rs = ref_samples(v, e)
        C.log(f"{lab}: profiled {nm6[k:]} median " + " ".join(f"{x:.3f}" for x in np.median(rs[:, k:], 0)))
        sets.append((rs, f"{lab} ({', '.join(nm6[:k])} fixed; {', '.join(nm6[k:])} fitted)", col))
    others = (["SBI"] if a.vs_sbi else []) + [r[0] for r in refs]
    ppc_compare(sets, f"ppc_{a.init}_vs_" + "_".join(others) + ".png",
                f"sdss_id {sid}, S/N {float(snr_[0]):.0f}: posterior predictive, {me} vs " + ", ".join(others))

if a.nuts_only and a.vs_sbi:
    ppc_compare([(nuts, "NUTS", "#D55E00"), (sbi, f"SBI ({a.model})", "#0072B2")], f"ppc_{a.init}_vs_SBI.png",
                f"sdss_id {sid}, S/N {float(snr_[0]):.0f}: posterior predictive, NUTS vs SBI")
    compare_plot([(nuts, "NUTS", "#D55E00"), (sbi, f"SBI ({a.model})", "#0072B2")],
                 f"corner_{a.init}_vs_SBI" + "".join(f"_{r[0]}" for r in refs) + ".png",
                 f"sdss_id {sid}, S/N {float(snr_[0]):.0f} ({tag}): NUTS vs SBI, 1σ / 2σ contours")

if a.nuts_only and a.overlay:
    opath, olab = a.overlay.rsplit(":", 1)
    other = np.load(opath)["nuts"].reshape(-1, 6)
    me = "nested sampling" if a.sampler == "nss" else "NUTS"
    compare_plot([(nuts, me, "#D55E00"), (other, olab, "#0072B2")], f"corner_{a.init}_vs_{olab}" + "".join(f"_{r[0]}" for r in refs) + ".png",
                 f"sdss_id {sid}, S/N {float(snr_[0]):.0f}: {me} vs {olab}, 1σ / 2σ contours")

if a.nuts_only:
    nuts_only_plots(th, sm, ndiv, out, f"sdss_id {sid}, S/N {float(snr_[0]):.0f}: "
                    + ("nested slice sampling" if a.sampler == "nss" else f"NUTS from {a.init} starts") + ", 1σ / 2σ contours")
else:
    fig = corner.corner(sbi, color="#0072B2", hist_kwargs=dict(density=True), **kw)
    fig = corner.corner(nuts, color="#D55E00", fig=fig, hist_kwargs=dict(density=True), truths=truths, truth_color="k", **kw)
    corner.overplot_lines(fig, list(best), color="#009E73", ls="--", lw=1)
    fig.legend(handles=[Line2D([], [], color="#0072B2", label=f"SBI ({a.model}, {len(sbi)} samples)"),
                        Line2D([], [], color="#D55E00", label=f"NUTS, {a.init} starts ({a.chains}×{a.samples}, {ndiv} div., r̂ max {np.max(sm['r_hat']):.2f})"),
                        Line2D([], [], color="#009E73", ls="--", label="MAP (Adam from SBI median)"),
                        Line2D([], [], color="k", label="BOSS-CLAM")],
               loc="upper right", bbox_to_anchor=(0.98, 0.93), fontsize=12, frameon=False)
    fig.suptitle(f"sdss_id {sid} ({tag}), S/N {float(snr_[0]):.0f}: 1σ / 2σ contours", y=1.01, fontsize=13)
    fig.savefig(out / (f"corner_{a.init}.png" if a.init != "map" else "corner.png"), dpi=90, bbox_inches="tight"); plt.close(fig)

res["ppc_sbi"] = ppc(sbi, f"SBI ({a.model})", "#0072B2", "ppc_sbi.png")
res["ppc_nuts"] = ppc(nuts, a.sampler.upper(), "#D55E00", f"ppc_nuts_{a.init}.png" if a.init != "map" else "ppc_nuts.png")
C.log(f"PPC SBI {res['ppc_sbi']}   NUTS {res['ppc_nuts']}")
(out / (f"summary_{a.init}.json" if a.init != "map" else "summary.json")).write_text(json.dumps(res, indent=1))
np.savez(out / (f"samples_{a.init}.npz" if a.init != "map" else "samples.npz"), sbi=sbi, nuts=th, map=best, names=C.LABELS)
C.log(f"wrote {out}")
