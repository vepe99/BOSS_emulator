"""NPE posterior median -> gradient-descent MAP on boss_clam coadds, with the v4 forward model.

For n boss_clam test stars (each with the WDISP of its spec-lite visit):
  1. sample the amortized posterior, take the median;
  2. from there, Adam on chi2 = sum ivar (y - C(emu(theta) * LSF_vedant(WDISP)))^2,
     C = profiled Chebyshev continuum (as in training / SC), theta box-bounded via logit;
  3. recovery plot NPE median vs MAP vs BOSS-CLAM labels (M dwarfs marked) + MAE table.

    uv run -q --extra sbi --with astropy python sc_sbi/npe_then_map.py trained_model/sc_sbi/v4_vedant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (autocvd claims a free GPU)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--model", default="amortized")
p.add_argument("--n", type=int, default=5000)
p.add_argument("--n-samples", type=int, default=200)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--batch", type=int, default=500)
p.add_argument("--out", default="map_clam")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--cut-mdwarfs", action="store_true", help="drop reference Teff < 4000 & logg >= 4")
p.add_argument("--match-eval-korg", type=int, default=0,
               help="reproduce evaluate.py's star draw: its --n-korg (rng draws the Korg subset first)")
a = p.parse_args()

run_dir = Path(a.run_dir)
out = run_dir / a.out
out.mkdir(exist_ok=True)
cfg = C.load_run_config(run_dir)
ncfg = C.noise_config(cfg.get("noise"))
assert ncfg["lsf"] == "wdisp", "needs a v4 (WDISP) run"
box = {k: tuple(v) for k, v in cfg["box"].items()}
adapter = C.shared_adapter({k: tuple(v) for k, v in cfg["padded_box"].items()}, cfg.get("log_params"))

# ---- data: boss_clam test coadds + WDISP of the star's bank visit (as evaluate.py --boss-source clam)
bank = C.load_bank(ncfg["bank"])
boss = C.load_boss()
pos = {int(v): i for i, v in enumerate(bank["sdss_id"])}
bi = np.array([pos.get(int(v), -1) for v in boss["sdss_id"]])
rows = np.flatnonzero((boss["split"] == 2) & C.boss_in_box(boss, box) & (bi >= 0))
if a.cut_mdwarfs:
    L0 = boss["labels"][rows]
    rows = rows[~((L0[:, 0] < 4000) & (L0[:, 1] >= 4))]
rng = np.random.default_rng(a.seed)
if a.match_eval_korg:
    rng.choice(len(np.load(run_dir / "splits.npz")["korg_test_idx"]), a.match_eval_korg, replace=False)
rows = np.sort(rng.choice(rows, min(a.n, rows.size), replace=False))
flux, ivar = boss["flux"][rows], boss["ivar"][rows]
wdisp = bank["wdisp"][bi[rows]].astype("float32")
ref = boss["labels"][rows]                                   # Teff, logg, m_H, alpha_m
C.log(f"{rows.size:,} boss_clam test stars")

# ---- 1. NPE posterior median
cond = C.spectrum_features(flux, ivar, boss["snr"][rows], ncfg, C.load_mismatch(ncfg))
cond |= dict(wdisp=wdisp, lsf_pca=bank["lsf_pca"][bi[rows]])
ex = adapter(C.label_dict(np.zeros((2, 6), "float32")) | {k: v[:2] for k, v in cond.items()})
post = C.load_posterior(run_dir, a.model, ex)
smp = C.sample_posterior(post, adapter, cond, a.n_samples)
npe = np.median(smp, 1)
C.log("NPE sampled")

# ---- 2. MAP by Adam from the NPE median (logit-bounded box)
lo = jnp.array([box[k][0] for k in C.LABELS]); hi = jnp.array([box[k][1] for k in C.LABELS])
to_u = lambda x: jnp.log((x - lo) / (hi - x))
to_x = lambda u: lo + (hi - lo) * jax.nn.sigmoid(u)
emu = C.make_emulator_fn(cfg["emulator"])
basis = jnp.asarray(C.cheb_basis(ncfg["cont_deg"]))


def chi2(theta, y, w, wd):
    mu = C.vedant_lsf(emu(theta), wd, ncfg["lsf_kappa"], ncfg["lsf_rgrid"])
    mu = C.profile_continuum(mu, y, w, basis)
    return jnp.sum(w * (y - mu) ** 2, 1)


opt = optax.adam(optax.cosine_decay_schedule(a.lr, a.steps, 0.05))


@jax.jit
def fit(x0, y, w, wd):
    loss = lambda u: jnp.sum(chi2(to_x(u), y, w, wd))     # stars independent -> per-star Adam
    def step(c, _):
        u, s = c
        g = jax.grad(loss)(u)
        upd, s = opt.update(g, s, u)
        return (optax.apply_updates(u, upd), s), None
    u0 = to_u(jnp.clip(x0, lo + 1e-4 * (hi - lo), hi - 1e-4 * (hi - lo)))
    (u, _), _ = jax.lax.scan(step, (u0, opt.init(u0)), None, a.steps)
    x = to_x(u)
    return x, chi2(x0, y, w, wd), chi2(x, y, w, wd)


mp, c0, c1 = [], [], []
for s in range(0, rows.size, a.batch):
    idx = np.minimum(np.arange(s, s + a.batch), rows.size - 1)   # pad the last batch: one compile
    k = min(a.batch, rows.size - s)
    r = fit(jnp.asarray(npe[idx]), jnp.asarray(flux[idx]), jnp.asarray(ivar[idx]), jnp.asarray(wdisp[idx]))
    mp.append(np.asarray(r[0])[:k]); c0.append(np.asarray(r[1])[:k]); c1.append(np.asarray(r[2])[:k])
    C.log(f"  MAP {min(s + a.batch, rows.size)}/{rows.size}")
mp, c0, c1 = np.concatenate(mp), np.concatenate(c0), np.concatenate(c1)
npix = np.maximum((ivar > 0).sum(1), 1)
chi_npe, chi_map = c0 / npix, c1 / npix
C.log(f"MAP lowered chi2 for {np.mean(c1 < c0):.1%} of stars; median chi2/pix ratio MAP/NPE {np.median(c1 / c0):.3f}")

# ---- 3. metrics + plot
T, g = ref[:, 0], ref[:, 1]
mdw = (T < 4000) & (g >= 4)
bl = C.BOSS_LABELS
cols = [C.LABELS.index(k) for k in bl]
mae = lambda e, m: {k: float(np.nanmean(np.abs(e[m, j] - ref[m, j]))) for j, k in enumerate(bl)}
res = {}
for name, m in [("all", np.ones_like(mdw)), ("no_mdwarf", ~mdw), ("mdwarf", mdw)]:
    res[name] = {"n": int(m.sum()),
                 "npe": mae(npe[:, cols], m), "map": mae(mp[:, cols], m),
                 "chi2pix_npe": float(np.median(chi_npe[m])), "chi2pix_map": float(np.median(chi_map[m]))}
json.dump(res, open(out / "metrics.json", "w"), indent=1)
np.savez(out / "map_results.npz", rows=rows, sdss_id=boss["sdss_id"][rows], ref=ref, npe=npe, map=mp,
         chi2pix_npe=chi_npe, chi2pix_map=chi_map, npe_std=smp.std(1))

lines = [f"# NPE ({a.model}) -> MAP, {rows.size} boss_clam test coadds, Vedant LSF\n",
         "| subset | n | " + " | ".join(f"{k} NPE | {k} MAP" for k in bl) + " | χ²/pix NPE | χ²/pix MAP |",
         "|---" * (2 * len(bl) + 4) + "|"]
for name, r in res.items():
    lines.append(f"| {name} | {r['n']} | " + " | ".join(f"{r['npe'][k]:.3g} | {r['map'][k]:.3g}" for k in bl)
                 + f" | {r['chi2pix_npe']:.3f} | {r['chi2pix_map']:.3f} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
for i, (est, lab) in enumerate([(npe, "NPE median"), (mp, "MAP (from NPE)")]):
    for j, k in enumerate(bl):
        A = ax[i, j]; e = est[:, cols[j]]
        A.hexbin(ref[~mdw, j], e[~mdw], gridsize=60, mincnt=1, bins="log", cmap="viridis")
        A.scatter(ref[mdw, j], e[mdw], s=2, c="#E69F00", alpha=0.4, label="M dwarfs" if j == 0 else None)
        lim = [np.nanmin(ref[:, j]), np.nanmax(ref[:, j])]
        A.plot(lim, lim, "r-", lw=0.8)
        A.set_title(f"{k}: MAE {res['all'][('npe','map')[i]][k]:.3g} (no M dw {res['no_mdwarf'][('npe','map')[i]][k]:.3g})",
                    fontsize=9)
        A.set_xlabel(f"BOSS-CLAM {k}")
        if j == 0:
            A.set_ylabel(lab); A.legend(loc="upper left", markerscale=4, fontsize=8)
fig.suptitle(f"boss_clam test coadds (n={rows.size}): NPE ({a.model}) vs MAP with Vedant LSF")
fig.tight_layout(); fig.savefig(out / "recovery_npe_vs_map.png", dpi=120)

fig, A = plt.subplots(figsize=(5.5, 4.5))
b = np.logspace(np.log10(0.3), np.log10(30), 60)
A.hist(chi_npe, b, histtype="step", label="NPE median"); A.hist(chi_map, b, histtype="step", label="MAP")
A.set_xscale("log"); A.set_xlabel("χ²/pix"); A.legend(); fig.tight_layout(); fig.savefig(out / "chi2_npe_vs_map.png", dpi=120)
C.log(f"wrote {out}")
