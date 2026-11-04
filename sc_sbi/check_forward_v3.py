"""Self-check of the v3 forward model (LSF + Chebyshev continuum).
Run: JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/check_forward_v3.py"""
import sys; sys.path.insert(0, "sc_sbi")
import numpy as np, jax.numpy as jnp, common as C

P, rng = C.N_PIX, np.random.default_rng(0)

# 1. LSF: flat stays flat; a delta becomes the normalised Gaussian of that sigma
flat = jnp.ones((3, P))
assert np.allclose(C.apply_lsf(flat, jnp.array([0.5, 1.25, 4.0])), 1.0, atol=1e-6)
d = np.zeros((1, P), "float32"); d[0, 2000] = 1.0
out = np.asarray(C.apply_lsf(jnp.asarray(d), jnp.array([1.25])))[0, 1990:2011]
g = np.exp(-0.5 * (np.arange(-10, 11) / 1.25) ** 2); g /= g.sum()
assert np.allclose(out, g, atol=1e-6), "LSF kernel"
# per-pixel sigma: constant row == scalar path; a delta sees its own pixel's sigma
sp = np.linspace(0.8, 2.0, P, dtype="float32")[None]
assert np.allclose(C.apply_lsf(jnp.asarray(d), jnp.full((1, P), 1.25)), C.apply_lsf(jnp.asarray(d), jnp.array([1.25])), atol=1e-6)
out = np.asarray(C.apply_lsf(jnp.asarray(d), jnp.asarray(sp)))[0]
assert abs(np.sqrt(((np.arange(P) - 2000) ** 2 * out).sum() / out.sum()) - sp[0, 2000]) < 0.02, "per-pixel LSF"
assert abs(C.korg_lsf_sigma(np.sqrt(0.369 ** 2 - 1 / 12) / 1.1)) < 0.06       # grid already = target -> ~0

# 2. continuum profiling recovers a known multiplicative Chebyshev polynomial
B = C.cheb_basis(6)
f = (1 - 0.5 * rng.random((4, P))).astype("float32")
a = np.c_[np.ones(4), 0.03 * rng.standard_normal((4, 6))].astype("float32")
y = f * (a @ B.T)
W = np.ones_like(y); W[:, :218] = 0                              # masked blue end
fit = np.asarray(C.profile_continuum(jnp.asarray(f), jnp.asarray(y), jnp.asarray(W), jnp.asarray(B)))
assert np.allclose(fit[:, 218:], y[:, 218:], atol=1e-4), "continuum profile"

# 3. GPU batch path == CPU path for the deterministic part (no photon noise)
ncfg = C.noise_config(dict(mode="template", snr_mode="measured", noise_channel=True,
                           lsf=True, cont_deg=6, cont_tilt=0.03))
iv = np.full((50, P), 1e4, "float32"); iv[:, :218] = 0
tm = C.NoiseTemplates(iv, np.full(50, 30.0, "float32"))
nm = C.NoiseModel(tm, ncfg, None)
F = (1 - 0.3 * rng.random((200, P))).astype("float32"); L = rng.random((200, 6)).astype("float32") + 1
ds = C.KorgNoisyDataset(L, F, nm, 8, 1, C.shared_adapter({k: (0.0, 3.0) for k in C.LABELS} | {C.LSF: (0.2, 4.1)}))
ds.dev["sig_t"] = ds.dev["sig_t"] * 0                            # switch photon noise off
i = np.arange(8)
lsf, coefs = C.draw_forward_aug(ncfg, 8, np.random.default_rng(5))
g = C._gpu_noisy_batch(C.jax.random.key(0), i=i, j=i, aug=(lsf, coefs, ds.basis), **ds.dev)
ref = np.asarray(C.forward_aug(jnp.asarray(F[i]), lsf, coefs, jnp.asarray(C.cheb_basis(6))))
ref = np.where(iv[i] > 0, ref, 1.0)
assert np.allclose(np.asarray(g["flux"]), ref, atol=1e-5), "GPU vs CPU forward"
b = ds[0]
assert b[C.LSF].shape == (8, 1) and np.all((np.asarray(b[C.LSF]) > -10) & (np.asarray(b[C.LSF]) < 10))
assert C.params(ncfg)[-1] == C.LSF and len(C.params(ncfg)) == 7
print("v3 forward model checks OK")

# 4. log-Teff parametrisation keeps the *physical* Teff prior uniform:
#    uniform Teff draws -> adapter forward -> u; the SC prior density in u-space,
#    exp(prior_log_prob(physical) + log|dT/du|) (what bayesflow forms), must match
#    the histogram of u.
box = {k: (-1.0, 1.0) for k in C.LABELS} | {"Teff": (2748.0, 8052.0), "vmic": (0.3, 5.0)}
ad = C.shared_adapter(box, log_params=["vmic", "Teff"])
T = np.random.default_rng(1).uniform(2800, 8000, 400_000).astype("float32")[:, None]
u = np.asarray(ad({"Teff": T}, strict=False)["Teff"]).ravel()
h, e = np.histogram(u, bins=60, range=(np.percentile(u, 1), np.percentile(u, 99)), density=True)
c = 0.5 * (e[1:] + e[:-1])
_, ldj = ad({"Teff": c[:, None].astype("float32")}, inverse=True, log_det_jac=True, strict=False)
dens = np.exp(np.asarray(ldj["Teff"]).ravel())                 # uniform prior: constant in T
dens *= h.sum() / dens.sum()
assert np.allclose(h, dens, rtol=0.06), "log-Teff adapter changed the Teff prior"
print("log-Teff parametrisation: physical Teff prior stays uniform  OK")
