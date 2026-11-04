"""Self-check: GPU batch path == CPU NoiseModel path (deterministic parts exact,
noise statistics match) + timing. Run: .venv/bin/python sc_sbi/check_gpu_batches.py"""
import sys, time; sys.path.insert(0, "sc_sbi")
import numpy as np, jax, common as C

mm = C.Mismatch("trained_model/sc_sbi/mismatch/v1/mismatch.npz")
ncfg = C.noise_config(dict(mode="sm", snr_mode="measured", noise_channel=True,
                           mismatch=mm.path, likelihood_sm=True))
boss = C.load_boss()
rows = np.flatnonzero((boss["split"] == 0) & np.isfinite(boss["snr"]) & (boss["snr"] > 0))
nm = C.NoiseModel(C.NoiseTemplates(boss["ivar"][rows], boss["snr"][rows]), ncfg, mm)
korg, _, box = C.load_korg(splits=("val",))
L, F = korg["val"]
ad = C.shared_adapter(C.padded_box(box))
B = 2048
ds = C.KorgNoisyDataset(L, F, nm, B, 10, ad, seed=0)
assert ds.gpu

# deterministic parts: same rows i, j, zero noise -> GPU features == spectrum_features
i = np.arange(B); j = np.arange(B) % len(nm.t)
zero = jax.numpy.zeros(C.N_PIX)
g = C._gpu_noisy_batch(jax.random.key(0), i=i, j=j,
                       **(ds.dev | {"s_m_noise": zero, "sig_t": ds.dev["sig_t"] * 0}))
iv = nm.t.ivar[j]
ref = C.spectrum_features(np.where(iv > 0, F[i], 1.0), iv, nm.t.snr[j], ncfg, mm)
for k in ref:
    np.testing.assert_allclose(np.asarray(g[k]), ref[k], rtol=1e-4, atol=1e-5, err_msg=k)

# noise statistics: (noisy - clean) / sqrt(sigma^2 + s_m^2) ~ N(0, 1) on good pixels
b_cpu = nm.make_batch(F[i], L[i], np.random.default_rng(1))
g = C._gpu_noisy_batch(jax.random.key(1), i=i, j=j, **ds.dev)
for name, y, ivv in [("cpu", b_cpu["flux"], b_cpu["ivar"]),
                     ("gpu", np.asarray(g["flux"]), np.asarray(g["ivar"]))]:
    sig = np.sqrt(1 / np.maximum(ivv, 1e-30) + mm.s_m[None] ** 2)
    z = ((y - F[i]) / sig)[ivv > 0]
    print(f"{name}: z mean {z.mean():+.4f}  std {z.std():.4f}")
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01

# batches go through the adapter and stay on device
out = ds[0]
print({k: (type(v).__name__, v.shape) for k, v in out.items()})
assert isinstance(out["flux"], jax.Array)
bd = C.BossDataset(boss["flux"][rows[:5000]], boss["ivar"][rows[:5000]], boss["snr"][rows[:5000]],
                   64, 10, ad, ncfg=ncfg, mm=mm)
assert isinstance(bd[0]["flux"], jax.Array)


def t(f, n=20):
    jax.block_until_ready(f()); s = time.perf_counter()
    for _ in range(n):
        r = f()
    jax.block_until_ready(r); return 1e3 * (time.perf_counter() - s) / n


for bs in (256, 2048):
    ds.batch_size = bs
    gpu = t(lambda: ds[0])
    cpu = t(lambda: nm.make_batch(F[i[:bs]], L[i[:bs]], ds.rng), 5)
    print(f"batch {bs}: gpu {gpu:.2f} ms   cpu {cpu:.1f} ms")
print("OK")
