"""NUTS posterior (numpyro) for one BOSS spec-lite file, started at the MAP from fit_boss_spec.py.

Same forward model and likelihood as fit_boss_spec.py (emulator -> BOSS LSF minus the
R=5000 grid width -> RV shift -> profiled Chebyshev continuum, Gaussian with IVAR, the
same pixel mask + the MAP's 5-sigma clip). Prior = the emulator's: uniform in the label
box, log-uniform in vmic, flat in the RV shift. Sampling is in logit space (box -> R).
The continuum is profiled, not marginalised: the posterior is conditional on the best
continuum for each theta (its extra width is small next to the label errors).

    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/hmc_boss_spec.py sc_sbi/vedant_lsf/spec-015018-59164-4551583147.fits
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                      # autocvd claims a free GPU here (unless JAX_PLATFORMS=cpu)
import numpy as np, jax, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
from astropy.io import fits
import corner
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(C.PROJECT_ROOT / "script"))
from korg_to_boss import repair_wdisp

p = argparse.ArgumentParser()
p.add_argument("spec")
p.add_argument("--map-json", default=None, help="default: <spec>_fit.json")
p.add_argument("--r-grid", type=float, default=5000.0)
p.add_argument("--kappa", type=float, default=1.1)
p.add_argument("--cont-deg", type=int, default=12)
p.add_argument("--clip", type=float, default=5.0)
p.add_argument("--chains", type=int, default=4)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--samples", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
P = C.N_PIX
out = Path(a.spec).with_suffix("")
mp = json.loads(Path(a.map_json or f"{out}_fit.json").read_text())
C.log(f"backend {jax.default_backend()}, MAP chi2/px {mp['chi2_per_px']:.3f}")

# ---- data + forward model (as in fit_boss_spec.py) -------------------------------------
d = fits.open(a.spec)[1].data
y = d["FLUX"][:P].astype("float32"); wave = 10 ** d["LOGLAM"][:P]
ok = np.isfinite(d["IVAR"][:P]) & (d["WDISP"][:P] > 0) & (wave > 3650) & (y > 0) & (y < 3 * np.median(y))
iv = np.where(ok, d["IVAR"][:P], 0).astype("float32")
S = jnp.asarray(C.korg_lsf_sigma(repair_wdisp(d["WDISP"][:P]), kappa=a.kappa, r_grid=a.r_grid))
emu = C.make_emulator_fn(); box = C.label_box()
lo = jnp.array([box[k][0] for k in C.LABELS]); hi = jnp.array([box[k][1] for k in C.LABELS])
B = jnp.asarray(C.cheb_basis(a.cont_deg)); Y, X = jnp.asarray(y), jnp.arange(P, dtype=jnp.float32)


def model(theta, dpx, W):                               # theta (6,), dpx scalar -> (P,)
    mu = C.apply_lsf(emu(theta[None]), S[None])[0]
    mu = jnp.interp(X - dpx, X, mu)
    return C.profile_continuum(mu[None], Y[None], W[None], B)[0]


names = C.LABELS + ["dpx"]
best = np.array([mp["best"][k] for k in names], "float32")
z = (y - np.asarray(model(jnp.asarray(best[:6]), best[6], jnp.asarray(iv)))) * np.sqrt(iv)
iv = np.where(np.abs(z) > a.clip, 0, iv).astype("float32"); W = jnp.asarray(iv); npx = int((iv > 0).sum())
C.log(f"{npx} pixels, {int((np.abs(z) > a.clip).sum())} clipped at the MAP")

# ---- posterior in (logit-box, dpx) ------------------------------------------------------
vi = C.LABELS.index("vmic")


def potential(v):                                       # -log posterior
    s = jax.nn.sigmoid(v[:6]); th = lo + (hi - lo) * s
    chi2 = jnp.sum(W * (Y - model(th, v[6], W)) ** 2)
    log_jac = jnp.sum(jnp.log(s) + jnp.log1p(-s))       # d theta / d u
    return 0.5 * chi2 - log_jac + jnp.log(th[vi])       # log-uniform vmic prior


u0 = np.r_[np.log((best[:6] - lo) / (hi - best[:6])), best[6]].astype("float32")
rng = np.random.default_rng(a.seed)
init = jnp.asarray(u0 + 0.01 * rng.standard_normal((a.chains, 7)), jnp.float32)
mcmc = MCMC(NUTS(potential_fn=potential, dense_mass=True, target_accept_prob=0.9),
            num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
            chain_method="vectorized", progress_bar=False)
mcmc.run(jax.random.PRNGKey(a.seed), init_params=init, extra_fields=("diverging",))
v = np.asarray(mcmc.get_samples(group_by_chain=True))        # (chains, samples, 7)
th = np.concatenate([np.asarray(lo + (hi - lo) * jax.nn.sigmoid(v[..., :6])), v[..., 6:]], -1)
ndiv = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

# ---- report -------------------------------------------------------------------------------
sm = summary(th, group_by_chain=True)
flat = th.reshape(-1, 7)
res = {"file": Path(a.spec).name, "backend": jax.default_backend(), "n_px": npx, "divergences": ndiv,
       "chains": a.chains, "warmup": a.warmup, "samples": a.samples, "map": mp["best"],
       "posterior": {k: {"median": float(np.median(flat[:, i])), "p16": float(np.percentile(flat[:, i], 16)),
                         "p84": float(np.percentile(flat[:, i], 84)), "std": float(flat[:, i].std()),
                         "r_hat": float(sm["Param:0"]["r_hat"][i]), "n_eff": float(sm["Param:0"]["n_eff"][i])}
                     for i, k in enumerate(names)},
       "corr": np.corrcoef(flat.T).round(3).tolist()}
rv = lambda x: x * 1e-4 * np.log(10) * 299792.458
for k, r in res["posterior"].items():
    C.log(f"  {k:9s} {r['median']:9.3f} +{r['p84'] - r['median']:.3f} -{r['median'] - r['p16']:.3f}   "
          f"(MAP {mp['best'][k]:.3f}, Hessian ±{mp['err_hessian'][k]:.3f})  r_hat {r['r_hat']:.3f}  n_eff {r['n_eff']:.0f}")
C.log(f"  divergences {ndiv}")
np.savez(f"{out}_nuts.npz", samples=th, names=names)
Path(f"{out}_nuts.json").write_text(json.dumps(res, indent=2))

lab = C.LABELS + ["RV [km/s]"]
fl = flat.copy(); fl[:, 6] = rv(fl[:, 6])
fig = corner.corner(fl, labels=lab, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g",
                    truths=list(best[:6]) + [rv(best[6])], truth_color="C3")
fig.suptitle(f"{res['file']}  NUTS {a.chains}x{a.samples}, {ndiv} divergences (red: MAP)", y=1.01)
fig.savefig(f"{out}_nuts_corner.png", dpi=90, bbox_inches="tight")
C.log(f"wrote {out}_nuts.json, {out}_nuts.npz, {out}_nuts_corner.png")
