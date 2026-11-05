"""Assert-only checks of the v4 forward model (real per-star LSF, Vedant's method).

    JAX_PLATFORMS=cpu uv run --extra sbi --with astropy python sc_sbi/check_forward_v4.py [--bank PATH]

1. JAX ``C.vedant_lsf`` == a literal numpy transcription of ``sc_sbi/LSF_vedant.py``
   (same nx -> float32 round-off only; Vedant's own data-dependent nx -> small); the
   fixed ``VEDANT_NX`` is >= Vedant's nx for every WDISP tested.
2. Physics: flat stays flat, equivalent width conserved, line width = expected.
3. GPU batch (``_gpu_noisy_batch``) == CPU ``NoiseModel.make_batch`` with noise off.
4. ``params`` has 6 entries in wdisp mode; posterior conditions = log_snr + lsf_pca.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np, jax.numpy as jnp

p = argparse.ArgumentParser()
p.add_argument("--bank", default=None, help="test on real WDISP from this bank (default: synthetic)")
a = p.parse_args()
P = C.N_PIX
rng = np.random.default_rng(0)


def vedant_numpy(fm, wdisp, kappa=1.1, R=5000.0, nx=None):
    """sc_sbi/LSF_vedant.py lines 17-44, model grid = BOSS grid (Korg), float64."""
    wave = C.WAVE[:P]
    wm = wave
    sigma_lsf = kappa * np.log(10) * 1e-4 * wdisp * wave
    R_in = R * 2.3548
    sigma_k = np.sqrt(np.clip(sigma_lsf ** 2 + np.gradient(wave) ** 2 / 12 - (wave / R_in) ** 2,
                              (0.1 * sigma_lsf) ** 2, None))
    sig = np.interp(wm, wave, sigma_k)
    cdf = np.cumsum(np.gradient(wm) / sig)
    cdf /= cdf[-1]
    x_per_sigma = np.median(np.gradient(cdf) / (np.gradient(wm) / sig))
    nx_v = int(2 ** np.ceil(np.log2(2 / x_per_sigma)))
    nx = nx or nx_v
    lam = np.interp(np.linspace(0, 1, nx), cdf, wm)
    f = np.fft.irfft(np.fft.rfft(np.interp(lam, wm, fm)) * np.exp(-2 * np.pi ** 2 * x_per_sigma ** 2
                                                                  * np.fft.rfftfreq(nx, 1 / nx) ** 2), n=nx)
    return np.interp(wave, lam, f), nx_v


# ---- WDISP curves + test spectra -------------------------------------------------
if a.bank:
    b = C.load_bank(a.bank)
    W = b["wdisp"][rng.choice(len(b["wdisp"]), 16, replace=False)].astype("float64")
else:
    x = np.linspace(0, 1, P)
    W = np.stack([1.0 + 0.2 * np.sin(2 * np.pi * x * rng.uniform(0.5, 2)) - 0.25 * x + rng.normal(0, 0.03)
                  for _ in range(16)])
lines = np.ones((16, P))
for i in range(16):
    for lc, d in zip(rng.uniform(3700, 10000, 300), rng.uniform(0.05, 0.8, 300)):
        lines[i] *= 1 - d * np.exp(-0.5 * ((C.WAVE[:P] - lc) / (lc / (5000 * 2.3548))) ** 2)

# ---- 1. JAX vs numpy -------------------------------------------------------------
jx = np.asarray(C.vedant_lsf(lines, W))
in_ = slice(10, P - 10)                       # periodic-wrap edges (as in the original)
nx_v = []
for i in range(16):
    ref_same, nv = vedant_numpy(lines[i], W[i], nx=C.VEDANT_NX)
    ref_own, _ = vedant_numpy(lines[i], W[i])
    nx_v.append(nv)
    d_same = np.abs(jx[i] - ref_same)[in_].max()
    assert d_same < 2e-4, f"JAX vs numpy (same nx) {d_same:.2e}"
    if i < 4:   # fixed nx is at least as accurate as Vedant's own nx (vs a 262144-point reference)
        fine, _ = vedant_numpy(lines[i], W[i], nx=262144)
        e_own, e_fix = np.abs(ref_own - fine)[in_].max(), np.abs(ref_same - fine)[in_].max()
        assert e_fix <= e_own and e_fix < 5e-3, (e_fix, e_own)
assert max(nx_v) <= C.VEDANT_NX, f"VEDANT_NX {C.VEDANT_NX} < Vedant's nx {max(nx_v)}"
print(f"1. JAX == numpy LSF_vedant at nx={C.VEDANT_NX} (<2e-4); Vedant's own nx {sorted(set(nx_v))} is "
      f"less accurate vs a 262144-point reference  OK")

# ---- 2. physics --------------------------------------------------------------------
flat = np.asarray(C.vedant_lsf(np.ones((2, P)), W[:2]))
assert np.abs(flat[:, in_] - 1).max() < 1e-4, "flat not flat"
k = 2000; lc = C.WAVE[k]; s0 = lc / (5000 * 2.3548)
one = 1 - 0.5 * np.exp(-0.5 * ((C.WAVE[:P] - lc) / s0) ** 2)
wd = np.full((1, P), 1.0)
out = np.asarray(C.vedant_lsf(one[None], wd))[0]
dl = np.gradient(C.WAVE[:P])
w30 = np.abs(C.WAVE[:P] - lc) < 30                 # moments within +-30 A of the line
ew_in, ew_out = ((1 - one) * dl)[w30].sum(), ((1 - out) * dl)[w30].sum()
assert abs(ew_out / ew_in - 1) < 0.01, f"EW {ew_out / ew_in}"
var = ((1 - out) * (C.WAVE[:P] - lc) ** 2)[w30].sum() / (1 - out)[w30].sum()
s_lsf = 1.1 * C.LN10E4 * 1.0 * lc
expect = s_lsf ** 2 + dl[k] ** 2 / 12             # sigma_k^2 + model sigma^2 (removed in sigma_k)
# Vedant's method assumes a finely sampled model; on the coarse Korg grid the linear
# interpolation onto the stretched grid adds ~3 % width (as-is, by choice)
assert abs(np.sqrt(var / expect) - 1) < 0.05, f"width {np.sqrt(var / expect)}"
print(f"2. flat->flat, EW ratio {ew_out / ew_in:.4f}, width ratio {np.sqrt(var / expect):.4f}  OK")

# ---- 3. GPU batch == CPU make_batch (noise off) -------------------------------------
ncfg = C.noise_config({"mode": "template", "snr_mode": "measured", "noise_channel": True,
                       "lsf": "wdisp", "cont_deg": 6, "cont_tilt": 0.03})
ivar = np.full((16, P), 1e12, "float32")
tm = C.NoiseTemplates(ivar, np.full(16, 100.0, "float32"), W.astype("float16"),
                      rng.normal(size=(16, 10)).astype("float32"))
nm = C.NoiseModel(tm, ncfg, None)
labels = np.tile([[5000, 3.0, -0.5, 0.1, 0.0, 1.0]], (8, 1)).astype("float32")
ds = C.KorgNoisyDataset(labels, lines[:8].astype("float32"), nm, 8, 1, adapter=lambda d, strict=True: d, seed=3)
ds.dev["sig_t"] = ds.dev["sig_t"] * 0
i = np.arange(8)
r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
_, coefs = C.draw_forward_aug(ncfg, 8, r1)     # make_batch order: aug draws, then template rows
j = r1.integers(0, 16, size=8)
g = C._gpu_noisy_batch(C.jax.random.key(0), i=i, j=j, aug=(None, coefs, ds.basis), **ds.dev)
nm.t.sigma[:] = 0
c = nm.make_batch(lines[:8].astype("float32"), labels, r2)
assert np.abs(np.asarray(g["flux"]) - c["flux"])[:, 30:-30].max() < 1e-3, "GPU != CPU batch"   # GPU Korg flux is float16
assert np.allclose(np.asarray(g["lsf_pca"]), c["lsf_pca"]) and np.array_equal(np.asarray(g["wdisp"]), c["wdisp"])
print("3. GPU batch == CPU make_batch (noise off), wdisp/lsf_pca drawn jointly with the template  OK")

# ---- 4. params / adapter -------------------------------------------------------------
assert C.params(ncfg) == C.LABELS, C.params(ncfg)
ad = C.posterior_adapter(True, C.LABELS, lsf_pca=True)
ex = C.label_dict(labels) | {k: np.asarray(v) for k, v in g.items()}
o = ad(ex)
assert o["inference_conditions"].shape == (8, 11) and "wdisp" not in o, {k: v.shape for k, v in o.items()}
print("4. params = 6 labels; conditions = [log_snr | 10 lsf_pca]; wdisp dropped  OK")
print("check_forward_v4: all OK")
