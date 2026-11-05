"""Posterior predictive check for hmc_boss_spec.py samples on one BOSS spec-lite file.

For N posterior draws: model m_k (same forward model + mask + MAP clip as the fit) and a
replicated spectrum y_rep = m_k + N(0, 1/IVAR). Plots the data over the 68/95 % bands of
m_k (posterior uncertainty of the mean model) and of y_rep (predictive, incl. noise), all
divided by the MAP continuum so they sit on the normalised scale. Reports the chi2
posterior-predictive p-value P(chi2(y_rep) > chi2(y)) and per-pixel band coverage.

    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/ppc_boss_spec.py sc_sbi/vedant_lsf/spec-015018-59164-4551583147.fits
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np, jax, jax.numpy as jnp
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(C.PROJECT_ROOT / "script"))
from korg_to_boss import repair_wdisp

p = argparse.ArgumentParser()
p.add_argument("spec")
p.add_argument("--n-draws", type=int, default=300)
p.add_argument("--r-grid", type=float, default=5000.0)
p.add_argument("--kappa", type=float, default=1.1)
p.add_argument("--cont-deg", type=int, default=12)
p.add_argument("--clip", type=float, default=5.0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
P = C.N_PIX
out = Path(a.spec).with_suffix("")
mp = json.loads(Path(f"{out}_fit.json").read_text())
smp = np.load(f"{out}_nuts.npz")["samples"].reshape(-1, 7)
rng = np.random.default_rng(a.seed)
draws = smp[rng.choice(len(smp), a.n_draws, replace=False)]

# ---- data + forward model (as in fit_boss_spec.py / hmc_boss_spec.py) -------------------
h = fits.open(a.spec); d = h[1].data
y = d["FLUX"][:P].astype("float32"); wave = 10 ** d["LOGLAM"][:P]
ok = np.isfinite(d["IVAR"][:P]) & (d["WDISP"][:P] > 0) & (wave > 3650) & (y > 0) & (y < 3 * np.median(y))
iv = np.where(ok, d["IVAR"][:P], 0).astype("float32")
S = jnp.asarray(C.korg_lsf_sigma(repair_wdisp(d["WDISP"][:P]), kappa=a.kappa, r_grid=a.r_grid))
emu = C.make_emulator_fn()
B = jnp.asarray(C.cheb_basis(a.cont_deg)); Y, X = jnp.asarray(y), jnp.arange(P, dtype=jnp.float32)


@jax.jit
def model(theta, dpx, W):                               # (n, 6), (n,) -> continuum-multiplied (n, P), normalised (n, P)
    mu = C.apply_lsf(emu(theta), jnp.broadcast_to(S, (theta.shape[0], P)))
    mu = jax.vmap(lambda m, s: jnp.interp(X - s, X, m))(mu, dpx)
    return C.profile_continuum(mu, jnp.broadcast_to(Y, mu.shape), jnp.broadcast_to(W, mu.shape), B), mu


names = C.LABELS + ["dpx"]
best = np.array([mp["best"][k] for k in names], "float32")
m0, mu0 = (np.asarray(x)[0] for x in model(jnp.asarray(best[None, :6]), jnp.asarray(best[None, 6]), jnp.asarray(iv)))
z0 = (y - m0) * np.sqrt(iv)
iv = np.where(np.abs(z0) > a.clip, 0, iv).astype("float32"); W = jnp.asarray(iv); good = iv > 0
m0, mu0 = (np.asarray(x)[0] for x in model(jnp.asarray(best[None, :6]), jnp.asarray(best[None, 6]), W))
cont = m0 / mu0                                         # MAP continuum, for display only

M = np.concatenate([np.asarray(model(jnp.asarray(draws[s:s + 50, :6]), jnp.asarray(draws[s:s + 50, 6]), W)[0])
                    for s in range(0, a.n_draws, 50)])
sig = np.where(good, 1 / np.sqrt(np.maximum(iv, 1e-30)), np.nan)
Yrep = M + rng.standard_normal(M.shape) * np.nan_to_num(sig)

# ---- statistics ------------------------------------------------------------------------------
chi_obs = np.nansum(np.where(good, (y - M) ** 2 / sig ** 2, 0), 1)
chi_rep = np.nansum(np.where(good, (Yrep - M) ** 2 / sig ** 2, 0), 1)
ppp = float(np.mean(chi_rep > chi_obs))
qp = np.percentile(Yrep, [2.5, 16, 50, 84, 97.5], 0)
qm = np.percentile(M, [2.5, 16, 50, 84, 97.5], 0)
cov68 = float(np.mean(((y >= qp[1]) & (y <= qp[3]))[good]))
cov95 = float(np.mean(((y >= qp[0]) & (y <= qp[4]))[good]))
zp = np.where(good, (y - qp[2]) / ((qp[3] - qp[1]) / 2), np.nan)
res = {"file": Path(a.spec).name, "n_draws": a.n_draws, "n_px": int(good.sum()),
       "chi2_per_px_obs_median": float(np.median(chi_obs) / good.sum()),
       "chi2_per_px_rep_median": float(np.median(chi_rep) / good.sum()),
       "ppp_chi2": ppp, "coverage_68": cov68, "coverage_95": cov95,
       "model_band_68_over_sigma_median": float(np.nanmedian((qm[3] - qm[1]) / 2 / sig))}
for k, v in res.items():
    C.log(f"  {k}: {v}")
Path(f"{out}_ppc.json").write_text(json.dumps(res, indent=2))

# ---- plot (all divided by the MAP continuum) --------------------------------------------------
n = lambda v: v / cont
yo = np.where(good, y, np.nan)
zooms = [(3850, 4400, "Balmer / Ca H&K"), (4800, 4950, "Hβ"), (5140, 5220, "Mg b"), (6520, 6620, "Hα"), (8480, 8700, "Ca triplet")]
fig = plt.figure(figsize=(17, 12)); gs = fig.add_gridspec(3, len(zooms) + 1, height_ratios=[1.2, 0.6, 1.1])
ax0 = fig.add_subplot(gs[0, :]); axr = fig.add_subplot(gs[1, :], sharex=ax0)


def bands(ax, s, lw):
    ax.fill_between(wave[s], n(qp[0])[s], n(qp[4])[s], color="C0", alpha=0.18, lw=0, step="mid", label="95 % predictive (model + noise)")
    ax.fill_between(wave[s], n(qp[1])[s], n(qp[3])[s], color="C0", alpha=0.35, lw=0, step="mid", label="68 % predictive")
    ax.fill_between(wave[s], n(qm[0])[s], n(qm[4])[s], color="C3", alpha=0.6, lw=0, step="mid", label="95 % posterior model")
    ax.plot(wave[s], n(yo)[s], "k", lw=lw, drawstyle="steps-mid", label="BOSS")


bands(ax0, slice(None), 0.4); ax0.set_ylim(0, 1.2); ax0.set_ylabel("flux / MAP continuum"); ax0.legend(loc="lower right", ncol=4, fontsize=8)
axr.plot(wave, zp, "k", lw=0.4); axr.axhline(0, color="C3", lw=0.6)
for v in (-2, 2):
    axr.axhline(v, color="0.6", lw=0.5, ls="--")
axr.set_ylim(-6, 6); axr.set_ylabel("(data - pred. median) / pred. 68 % half-width"); axr.set_xlabel("observed wavelength [Å]")
for k, (w0, w1, lab) in enumerate(zooms):
    ax = fig.add_subplot(gs[2, k]); s = (wave > w0) & (wave < w1); bands(ax, s, 0.8); ax.set_title(lab, fontsize=9)
ax = fig.add_subplot(gs[2, -1])
ax.hist(chi_rep / good.sum(), 30, histtype="step", color="C0", label="χ²(y_rep)/px")
ax.hist(chi_obs / good.sum(), 30, histtype="step", color="k", label="χ²(y_obs)/px")
ax.set_title(f"χ² PPC  p = {ppp:.2f}", fontsize=9); ax.legend(fontsize=7)
fig.suptitle(f"{res['file']}  posterior predictive check ({a.n_draws} NUTS draws)   "
             f"coverage 68 %: {100 * cov68:.1f}%   95 %: {100 * cov95:.1f}%   p(χ²) = {ppp:.2f}", fontsize=11)
fig.tight_layout(); fig.savefig(f"{out}_ppc.png", dpi=110)
C.log(f"wrote {out}_ppc.json, {out}_ppc.png")
