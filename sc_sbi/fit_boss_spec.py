"""MAP labels for one raw BOSS spec-lite file by gradient descent through the emulator.

Forward model, per observed pixel i (rest-frame Korg grid = BOSS log-lambda grid):
    emulator(theta) [R=5000 grid] -> Gaussian LSF with sigma = korg_lsf_sigma(WDISP)
    (BOSS LSF minus the grid's own R=5000 width) -> shift by the radial velocity
    (free, starts at the pipeline Z) -> times a profiled Chebyshev continuum
    (absorbs flux calibration + SED, since the raw file is not normalised).
Likelihood: Gaussian with the file's IVAR. Adam from many random starts in the
label box (all at once on the GPU), best chi2 wins; errors from the Hessian.

    .venv/bin/python sc_sbi/fit_boss_spec.py sc_sbi/vedant_lsf/spec-015018-59164-4551583147.fits
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                      # autocvd claims a free GPU here
import numpy as np, jax, jax.numpy as jnp, optax
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(C.PROJECT_ROOT / "script"))
from korg_to_boss import repair_wdisp

p = argparse.ArgumentParser()
p.add_argument("spec")
p.add_argument("--r-grid", type=float, default=5000.0, help="resolution lambda/FWHM of the Korg grid")
p.add_argument("--kappa", type=float, default=1.1, help="WDISP calibration factor")
p.add_argument("--cont-deg", type=int, default=12)
p.add_argument("--n-starts", type=int, default=256)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--lr", type=float, default=0.03)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--clip", type=float, default=5.0, help="after the first fit, drop |z| > clip pixels and refit")
a = p.parse_args()
P = C.N_PIX
out = Path(a.spec).with_suffix("")

# ---- data -----------------------------------------------------------------------
h = fits.open(a.spec)
d, info = h[1].data, h[2].data[0]
assert abs(d["LOGLAM"][0] - 3.5523) < 1e-6 and len(d) >= P, "not on the standard BOSS grid"
y = d["FLUX"][:P].astype("float32")
wave = 10 ** d["LOGLAM"][:P]
# blue-edge flux-calibration blow-ups (|flux| ~ 1e7) keep IVAR > 0, so cut them explicitly
ok = np.isfinite(d["IVAR"][:P]) & (d["WDISP"][:P] > 0) & (wave > 3650) & (y > 0) & (y < 3 * np.median(y))
iv = np.where(ok, d["IVAR"][:P], 0).astype("float32")
sig_px = C.korg_lsf_sigma(repair_wdisp(d["WDISP"][:P]), kappa=a.kappa, r_grid=a.r_grid)
dpx0 = np.log10(1 + float(info["Z"])) / 1e-4                         # RV shift [px], 69 km/s each
C.log(f"{Path(a.spec).name}: {info['CLASS'].strip()} {info['SUBCLASS'].strip()}  Z={info['Z']:.5f} "
      f"({dpx0:.2f} px)  SNR {info['SN_MEDIAN_ALL']:.1f}  kernel sigma {sig_px.min():.2f}-{sig_px.max():.2f} px")

# ---- forward model + chi2 ---------------------------------------------------------
emu = C.make_emulator_fn()
box = C.label_box()
lo = jnp.array([box[k][0] for k in C.LABELS]); hi = jnp.array([box[k][1] for k in C.LABELS])
B = jnp.asarray(C.cheb_basis(a.cont_deg))
Y, S, X = jnp.asarray(y), jnp.asarray(sig_px), jnp.arange(P, dtype=jnp.float32)


def model(theta, dpx, W):
    """theta (n, 6) physical, dpx (n,) -> (n, P) model in the observed frame."""
    mu = C.apply_lsf(emu(theta), jnp.broadcast_to(S, (theta.shape[0], P)))
    mu = jax.vmap(lambda m, s: jnp.interp(X - s, X, m))(mu, dpx)        # observed px i <- rest px i - dpx
    return C.profile_continuum(mu, jnp.broadcast_to(Y, mu.shape), jnp.broadcast_to(W, mu.shape), B)


def chi2(theta, dpx, W):
    return jnp.sum(W * (Y - model(theta, dpx, W)) ** 2, -1)


to_phys = lambda u: lo + (hi - lo) * jax.nn.sigmoid(u)               # unconstrained -> inside the box
loss = lambda q, W: jnp.sum(chi2(to_phys(q["u"]), q["dpx"], W))

rng = np.random.default_rng(a.seed)
s0 = rng.uniform(0.02, 0.98, (a.n_starts, 6))
q = {"u": jnp.asarray(np.log(s0 / (1 - s0)), jnp.float32), "dpx": jnp.full(a.n_starts, dpx0, jnp.float32)}
opt = optax.adam(optax.cosine_decay_schedule(a.lr, a.steps, 0.01))


@jax.jit
def step(q, st, W):
    g = jax.grad(loss)(q, W)
    upd, st = opt.update(g, st)
    return optax.apply_updates(q, upd), st


for rnd in range(2):                                    # round 2: refit after sigma-clipping
    W, st = jnp.asarray(iv), opt.init(q)
    for i in range(a.steps):
        q, st = step(q, st, W)
        if i % 500 == 0 or i == a.steps - 1:
            c = np.asarray(chi2(to_phys(q["u"]), q["dpx"], W))
            C.log(f"round {rnd} step {i:5d}  best chi2/px {c.min() / (iv > 0).sum():.3f}  median {np.median(c) / (iv > 0).sum():.3f}")
    b = int(np.argmin(c))
    z = (y - np.asarray(model(to_phys(q["u"][b:b + 1]), q["dpx"][b:b + 1], W))[0]) * np.sqrt(iv)
    C.log(f"clipping {int((np.abs(z) > a.clip).sum())} pixels with |z| > {a.clip}")
    iv = np.where(np.abs(z) > a.clip, 0, iv).astype("float32")
W = jnp.asarray(iv)

# ---- best solution, Hessian errors -------------------------------------------------
th, dp = np.asarray(to_phys(q["u"])), np.asarray(q["dpx"])
c = np.asarray(chi2(jnp.asarray(th), jnp.asarray(dp), W))
order = np.argsort(c); b = order[0]; npx = int((iv > 0).sum())
pbest = jnp.asarray(np.r_[th[b], dp[b]], jnp.float32)
H = jax.hessian(lambda v: chi2(v[None, :6], v[None, 6], W)[0])(pbest)
err = np.sqrt(np.clip(np.diag(np.linalg.pinv(0.5 * np.asarray(H, np.float64))), 0, None))
names = C.LABELS + ["dpx"]
res = {"file": Path(a.spec).name, "chi2_per_px": float(c[b] / npx), "n_px": npx, "kappa": a.kappa, "r_grid": a.r_grid,
       "rv_kms_pipeline": float(dpx0 * 1e-4 * np.log(10) * 299792.458),
       "rv_kms_fit": float(dp[b] * 1e-4 * np.log(10) * 299792.458),
       "best": {k: float(v) for k, v in zip(names, np.asarray(pbest))},
       "err_hessian": {k: float(v) for k, v in zip(names, err)},
       "top10_starts": [{"chi2_per_px": float(c[j] / npx), **{k: float(v) for k, v in zip(C.LABELS, th[j])}} for j in order[:10]]}
for k in names:
    C.log(f"  {k:9s} {res['best'][k]:9.3f} +- {res['err_hessian'][k]:.3f}")
C.log(f"  chi2/px {res['chi2_per_px']:.3f}  RV {res['rv_kms_fit']:.1f} km/s (pipeline {res['rv_kms_pipeline']:.1f})")
Path(f"{out}_fit.json").write_text(json.dumps(res, indent=2))

# ---- plots: observed flux, and continuum-normalised (data / fitted continuum vs emulator scale) ----
m = np.asarray(model(pbest[None, :6], pbest[None, 6], W))[0]
th1 = pbest[None, :6]
mu = C.apply_lsf(emu(th1), S[None])                                 # normalised model, LSF applied
mu = np.asarray(jnp.interp(X - pbest[6], X, mu[0]))                 # observed frame
cont = m / mu                                                       # the fitted Chebyshev continuum
good = iv > 0; z = np.where(good, (y - m) * np.sqrt(iv), np.nan)
zooms = [(3850, 4400, "Balmer / Ca H&K"), (4800, 4950, "Hβ"), (5140, 5220, "Mg b"), (6520, 6620, "Hα"), (8480, 8700, "Ca triplet")]
bb, ee = res["best"], res["err_hessian"]
for suf, yo, mm, ylab in (("", np.where(good, y, np.nan), m, "flux [1e-17 cgs]"),
                          ("_norm", np.where(good, y / cont, np.nan), mu, "normalised flux")):
    fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(3, len(zooms), height_ratios=[1.2, 0.6, 1.1])
    ax0 = fig.add_subplot(gs[0, :]); axr = fig.add_subplot(gs[1, :], sharex=ax0)
    ax0.plot(wave, yo, "k", lw=0.5, label="BOSS" + (" / fitted continuum" if suf else ""))
    ax0.plot(wave, mm, "C3", lw=0.6, label="emulator (LSF, RV)" if suf else "best fit")
    if suf:
        ax0.set_ylim(0, 1.15); ax0.axhline(1, color="0.6", lw=0.5)
    else:
        ax0.plot(wave, cont, "C0", lw=0.8, label="fitted continuum")
        ax0.set_ylim(np.nanpercentile(yo, 0.5) * 0.9, np.nanpercentile(yo, 99.8) * 1.05)
    ax0.legend(loc="lower right"); ax0.set_ylabel(ylab)
    axr.plot(wave, z, "k", lw=0.4); axr.axhline(0, color="C3", lw=0.6); axr.set_ylim(-6, 6); axr.set_ylabel("(data - model)/σ")
    axr.set_xlabel("observed wavelength [Å]")
    for k, (w0, w1, lab) in enumerate(zooms):
        ax = fig.add_subplot(gs[2, k]); s_ = (wave > w0) & (wave < w1)
        ax.plot(wave[s_], yo[s_], "k", lw=0.8); ax.plot(wave[s_], mm[s_], "C3", lw=0.8); ax.set_title(lab, fontsize=9)
    fig.suptitle(f"{res['file']}  ({info['SUBCLASS'].strip()})   χ²/px {res['chi2_per_px']:.2f}\n"
                 + "   ".join(f"{k} {bb[k]:.3g}±{ee[k]:.2g}" for k in C.LABELS)
                 + f"   RV {res['rv_kms_fit']:.0f} km/s   (R_grid {a.r_grid:.0f}, κ {a.kappa})", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}_fit{suf}.png", dpi=110); plt.close(fig)
C.log(f"wrote {out}_fit.json, {out}_fit.png, {out}_fit_norm.png")
