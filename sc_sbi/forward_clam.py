"""Forward-model real BOSS stars at their BOSS-CLAM labels (CPU).

Teff, logg, [M/H], [alpha/M] are fixed to the catalogue; the parameters CLAM does
not provide are drawn from the v3 priors -- [C/M] ~ U(-1.5, 3), vmic ~ logU(0.316, 5),
lsf_px ~ U(0.3, 4) -- and every draw goes through the v3 forward model: emulator ->
Gaussian LSF -> Chebyshev-6 continuum profiled against the star (ivar weights).
Per star: observed spectrum vs median model with 68 / 95 % bands (full + 2 zooms),
residuals in sigma, and chi2/pixel over the prior draws.

    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/forward_clam.py --out plots/sc_sbi/forward_clam
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--out", default="plots/sc_sbi/forward_clam")
p.add_argument("--n-draws", type=int, default=1000)
p.add_argument("--min-snr", type=float, default=50)
p.add_argument("--cont-deg", type=int, default=6)
p.add_argument("--zooms", type=int, nargs=4, default=[1300, 1500, 3000, 3200], metavar=("A0", "A1", "B0", "B1"))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-stars", type=int, default=0, help="random test stars instead of the 6 HR groups; no per-star plots, "
               "aggregate (BOSS - median)/sigma statistics instead")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)

# ---- pick test-split stars across the HR diagram ------------------------------
b = C.load_boss(); L = b["labels"]; box = C.label_box()
ok = (C.boss_in_box(b, box) & (b["split"] == 2) & np.isfinite(L).all(1) & (b["snr"] >= a.min_snr)
      & ((b["ivar"] <= 0).mean(1) <= 0.3))
T, g, mh = L[:, 0], L[:, 1], L[:, 2]
groups = {"metal-poor giant": ok & (g < 2.5) & (mh < -1.2),
          "metal-rich giant": ok & (g < 2.5) & (mh > -0.2),
          "red clump": ok & (g > 2.3) & (g < 2.7) & (T > 4600) & (T < 5000) & (mh > -0.5),
          "subgiant": ok & (g >= 3.5) & (g < 4.0) & (T > 4800),
          "FGK dwarf": ok & (g >= 4.2) & (T > 4800) & (T < 6200),
          "M dwarf": ok & (g >= 4.4) & (T < 3800)}
stars = {k: int(rng.choice(np.flatnonzero(m))) for k, m in groups.items() if m.any()}
if a.n_stars:
    stars = {f"star{i:03d}": int(r) for i, r in enumerate(np.sort(rng.choice(np.flatnonzero(ok), a.n_stars, replace=False)))}
Z, Zb = [], []      # pulls vs median / vs best draw, (n_stars, N_PIX), nan where masked

# ---- forward model --------------------------------------------------------------
emu = C.make_emulator_fn()
basis = jnp.asarray(C.cheb_basis(a.cont_deg))
lo_c, hi_c = box["carbon_m"]; lo_v, hi_v = box["vmic"]


@jax.jit
def model(theta6, lsf, y, w):
    mu = C.apply_lsf(emu(theta6), lsf)
    return C.profile_continuum(mu, jnp.broadcast_to(y, mu.shape), jnp.broadcast_to(w, mu.shape), basis)


summary = {}
for name, r in stars.items():
    y, iv = b["flux"][r], b["ivar"][r]
    n = a.n_draws
    th = np.zeros((n, 6), "float32")
    th[:, :4] = L[r]                                                      # CLAM labels
    th[:, 4] = rng.uniform(lo_c, hi_c, n)                                 # [C/M] prior
    th[:, 5] = 10 ** rng.uniform(np.log10(lo_v), np.log10(hi_v), n)       # vmic prior
    lsf = rng.uniform(0.3, 4.0, n).astype("float32")                      # lsf_px prior
    M = np.concatenate([np.asarray(model(jnp.asarray(th[s:s + 250]), jnp.asarray(lsf[s:s + 250]),
                                         jnp.asarray(y), jnp.asarray(iv)))
                        for s in range(0, n, 250)])
    good = iv > 0
    chi2 = ((iv * (y - M) ** 2)[:, good]).sum(1) / good.sum()
    q = np.percentile(M, [2.5, 16, 50, 84, 97.5], axis=0)
    best = int(np.argmin(chi2))
    summary[name] = {"row": r, "sdss_id": int(b["sdss_id"][r]), "data_type": str(b["data_type"][r]),
                     "snr": float(b["snr"][r]),
                     "clam": dict(zip(C.BOSS_LABELS, map(float, L[r]))),
                     "chi2_per_px_median": float(np.median(chi2)),
                     "chi2_per_px_16_84": np.percentile(chi2, [16, 84]).tolist(),
                     "chi2_per_px_best": float(chi2[best]),
                     "best_draw": {"carbon_m": float(th[best, 4]), "vmic": float(th[best, 5]), "lsf_px": float(lsf[best])}}

    sig = np.where(good, 1 / np.sqrt(np.maximum(iv, 1e-30)), np.nan)
    z, zb = (np.where(good, y, np.nan) - q[2]) / sig, (np.where(good, y, np.nan) - M[best]) / sig
    Z.append(z); Zb.append(zb)
    summary[name] |= {"z_median": float(np.nanmedian(z)), "z_rms": float(np.sqrt(np.nanmean(z ** 2))),
                      "z_mad": float(1.4826 * np.nanmedian(np.abs(z - np.nanmedian(z)))),
                      "frac_abs_z_gt3": float(np.nanmean(np.abs(z) > 3)), "zbest_rms": float(np.sqrt(np.nanmean(zb ** 2))),
                      "abs_flux_err_median": float(np.nanmedian(np.abs(np.where(good, y, np.nan) - q[2])))}
    if a.n_stars:
        C.log(f"{name} row {r}  Teff {L[r, 0]:.0f} logg {L[r, 1]:.2f}  z rms {summary[name]['z_rms']:.2f}  best {summary[name]['zbest_rms']:.2f}")
        continue
    # ---- plot ----------------------------------------------------------------------
    x = np.arange(C.N_PIX); yo = np.where(good, y, np.nan); sig = np.where(good, 1 / np.sqrt(np.maximum(iv, 1e-30)), np.nan)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 0.8])
    ax0 = fig.add_subplot(gs[0, :]); axA = fig.add_subplot(gs[1, 0]); axB = fig.add_subplot(gs[1, 1])
    axr = fig.add_subplot(gs[2, :], sharex=ax0)
    for ax, (x0, x1) in ((ax0, (0, C.N_PIX)), (axA, a.zooms[:2]), (axB, a.zooms[2:])):
        s = slice(x0, x1)
        ax.fill_between(x[s], q[0, s], q[4, s], color="C0", alpha=0.18, lw=0, label="95 % (prior draws)")
        ax.fill_between(x[s], q[1, s], q[3, s], color="C0", alpha=0.35, lw=0, label="68 %")
        ax.plot(x[s], yo[s], color="k", lw=0.6, label="BOSS")
        ax.plot(x[s], q[2, s], color="C3", lw=0.7, label="median model")
        ax.set_xlim(x0, x1)
    ax0.set_ylim(np.nanpercentile(yo, 0.5) - 0.05, 1.15)
    ax0.legend(loc="lower right", fontsize=8, ncol=4)
    axA.set_title(f"zoom px {a.zooms[0]}-{a.zooms[1]}", fontsize=9); axB.set_title(f"zoom px {a.zooms[2]}-{a.zooms[3]}", fontsize=9)
    axr.plot(x, (yo - q[2]) / sig, color="k", lw=0.4)
    axr.axhline(0, color="C3", lw=0.6); axr.set_ylim(-8, 8); axr.set_ylabel("(BOSS - median) / σ"); axr.set_xlabel("pixel")
    c = summary[name]["clam"]
    fig.suptitle(f"{name}  (sdss_id {summary[name]['sdss_id']}, {summary[name]['data_type']}, SNR {summary[name]['snr']:.0f})   "
                 f"CLAM Teff {c['Teff']:.0f}  logg {c['logg']:.2f}  [M/H] {c['m_H']:+.2f}  [α/M] {c['alpha_m']:+.2f}\n"
                 f"[C/M], vmic, lsf_px from the prior ({n} draws)   χ²/px median {np.median(chi2):.2f} "
                 f"[16-84 %: {np.percentile(chi2, 16):.2f}-{np.percentile(chi2, 84):.2f}]   best {chi2[best]:.2f} at "
                 f"[C/M] {th[best, 4]:+.2f}, vmic {th[best, 5]:.2f}, lsf {lsf[best]:.2f}", fontsize=10)
    fig.tight_layout()
    f = out / f"forward_{name.replace(' ', '_')}.png"; fig.savefig(f, dpi=110); plt.close(fig)
    C.log(f"{name:17s} chi2/px median {np.median(chi2):6.2f}  16-84 {np.percentile(chi2, 16):6.2f}-{np.percentile(chi2, 84):6.2f}  "
          f"best {chi2[best]:6.2f}  -> {f}")

if a.n_stars:
    Z, Zb = np.array(Z), np.array(Zb); rows = np.array([v["row"] for v in summary.values()])
    gi = L[rows, 1] < 3.5; zr = np.array([v["z_rms"] for v in summary.values()]); zbr = np.array([v["zbest_rms"] for v in summary.values()])
    agg = {}
    for lab, m in (("all", np.ones(len(rows), bool)), ("giants logg<3.5", gi), ("dwarfs logg>=3.5", ~gi)):
        if not m.any():
            continue
        v, vb = Z[m][np.isfinite(Z[m])], Zb[m][np.isfinite(Zb[m])]
        agg[lab] = {"n_stars": int(m.sum()), "z_pct_2.5_16_50_84_97.5": np.percentile(v, [2.5, 16, 50, 84, 97.5]).tolist(),
                    "z_mad": float(1.4826 * np.median(np.abs(v - np.median(v)))), "frac_abs_z_gt3": float(np.mean(np.abs(v) > 3)),
                    "per_star_z_rms_16_50_84": np.percentile(zr[m], [16, 50, 84]).tolist(),
                    "zbest_pct_16_50_84": np.percentile(vb, [16, 50, 84]).tolist(),
                    "per_star_zbest_rms_16_50_84": np.percentile(zbr[m], [16, 50, 84]).tolist()}
        C.log(f"{lab:17s} n={m.sum():3d}  z 16/50/84 % {agg[lab]['z_pct_2.5_16_50_84_97.5'][1]:+.2f}/{agg[lab]['z_pct_2.5_16_50_84_97.5'][2]:+.2f}/"
              f"{agg[lab]['z_pct_2.5_16_50_84_97.5'][3]:+.2f}  MAD {agg[lab]['z_mad']:.2f}  |z|>3 {100 * agg[lab]['frac_abs_z_gt3']:.1f}%  "
              f"per-star rms median {np.median(zr[m]):.2f} (best draw {np.median(zbr[m]):.2f})")
    summary = {"aggregate": agg, "stars": summary}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    bins = np.linspace(-8, 8, 161); xx = np.linspace(-8, 8, 400)
    for lab, m, c in (("giants", gi, "C0"), ("dwarfs", ~gi, "C1")):
        if m.any():
            ax[0, 0].hist(Z[m][np.isfinite(Z[m])], bins, density=True, histtype="step", color=c, label=f"{lab} vs median (n={m.sum()})")
            ax[0, 0].hist(Zb[m][np.isfinite(Zb[m])], bins, density=True, histtype="step", color=c, ls="--", label=f"{lab} vs best draw")
    ax[0, 0].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k:", label="N(0,1)")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_ylim(1e-4, 1); ax[0, 0].set_xlabel("(BOSS - model) / σ"); ax[0, 0].legend(fontsize=8)
    sc = ax[0, 1].scatter(L[rows, 0], L[rows, 1], c=zr, cmap="viridis", vmin=1, vmax=np.percentile(zr, 95), s=25)
    ax[0, 1].invert_xaxis(); ax[0, 1].invert_yaxis(); ax[0, 1].set_xlabel("CLAM Teff"); ax[0, 1].set_ylabel("CLAM logg")
    fig.colorbar(sc, ax=ax[0, 1], label="per-star rms (BOSS - median)/σ")
    x = np.arange(C.N_PIX)
    for lab, m, c in (("giants", gi, "C0"), ("dwarfs", ~gi, "C1")):
        if m.any():
            ax[1, 0].plot(x, np.nanmedian(np.abs(Z[m]), 0), color=c, lw=0.5, label=f"{lab}: median over stars of |z|")
    ax[1, 0].axhline(0.674, color="k", ls=":", lw=0.8, label="pure noise (0.674)"); ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("pixel"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].hist([zr[gi], zr[~gi]], bins=30, label=["giants", "dwarfs"], color=["C0", "C1"], histtype="step")
    ax[1, 1].hist([zbr[gi], zbr[~gi]], bins=30, color=["C0", "C1"], histtype="step", ls="--")
    ax[1, 1].set_xlabel("per-star rms z (solid: vs median, dashed: vs best draw)"); ax[1, 1].legend(fontsize=8)
    fig.suptitle(f"{len(rows)} random BOSS test stars (SNR>={a.min_snr:.0f}) at CLAM labels, [C/M]/vmic/lsf from prior ({a.n_draws} draws)")
    fig.tight_layout(); fig.savefig(out / "residual_distribution.png", dpi=110); plt.close(fig)
    C.log(f"wrote {out / 'residual_distribution.png'}")
(out / "forward_clam.json").write_text(json.dumps(summary, indent=2))
C.log(f"wrote {out / 'forward_clam.json'}")
