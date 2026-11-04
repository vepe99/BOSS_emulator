"""Posterior corner plot for one BOSS giant (boss_only NPE) + posterior-predictive
forward model in the empty upper-right triangle
(full spectrum + data-minus-model residuals).

The NPE gives (Teff, logg, [M/H], [alpha/M]); the emulator also needs [C/M] and vmic,
which are marginalised by drawing them from the Korg-grid prior ([C/M] uniform, vmic
log-uniform). Gaussian LSF with one width fitted to the star (min chi2 of the median model;
boss_clam has no WDISP), Chebyshev continuum profiled against the star. The plotted model is
the median plus Gaussian noise drawn from the star's ivar (its S/N); residuals are BOSS minus that.

    .venv/bin/python sc_sbi/corner_forward.py [--row R] [--out trained_model/sc_sbi/boss_only/eval/corner_giant.png]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                      # KERAS_BACKEND=jax + autocvd GPU pick
import numpy as np, jax, jax.numpy as jnp, bayesflow as bf, corner
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--run", default="trained_model/sc_sbi/boss_only")
p.add_argument("--weights", default="boss_bestval")
p.add_argument("--emulator", default="trained_model/combined_best/mlp_1790330838/best.pkl")
p.add_argument("--row", type=int, default=None, help="BOSS row; default: random high-SNR test giant")
p.add_argument("--n-samples", type=int, default=2000)
p.add_argument("--min-snr", type=float, default=50)
p.add_argument("--cont-deg", type=int, default=6)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="trained_model/sc_sbi/boss_only/eval/corner_giant.png")
a = p.parse_args()
rng = np.random.default_rng(a.seed)
run = Path(a.run); cfg = C.load_run_config(run); names = cfg["labels"]; ncfg = C.noise_config(cfg["noise"])

# ---- pick a test giant ---------------------------------------------------------------
b = C.load_boss(); L = b["labels"]
test = np.load(run / "test_posterior.npz")["rows"]
if a.row is None:
    ok = (L[test, 1] < 2.5) & (b["snr"][test] >= a.min_snr) & ((b["ivar"][test] <= 0).mean(1) <= 0.1)
    a.row = int(rng.choice(test[ok]))
r = a.row; y, iv = b["flux"][r], b["ivar"][r]
C.log(f"row {r}  sdss_id {b['sdss_id'][r]}  SNR {b['snr'][r]:.0f}  CLAM {dict(zip(names, L[r].round(3)))}")

# ---- posterior --------------------------------------------------------------------------
adapter = bf.Adapter()
for k in names:
    adapter = adapter.constrain(k, lower=float(cfg["padded_box"][k][0]), upper=float(cfg["padded_box"][k][1]))
cond = C.spectrum_features(b["flux"][[r, r]], b["ivar"][[r, r]], b["snr"][[r, r]], ncfg, None)
example = adapter({k: v for k, v in cond.items() if k != "ivar"} | {k: L[[r, r], i:i + 1] for i, k in enumerate(names)})
post = C.build_posterior(cfg["net"], ncfg["noise_channel"], names)
post.build_from_data(post.adapter(example)); post.load_weights(run / f"{a.weights}.weights.h5")
S = C.sample_posterior(post, adapter, {k: v[:1] for k, v in cond.items()}, a.n_samples, names=names)[0]   # (S, 4)

# ---- forward model, marginalising [C/M], vmic over their priors; LSF fitted --------------------
emu = C.make_emulator_fn(a.emulator); basis = jnp.asarray(C.cheb_basis(a.cont_deg)); box = C.label_box()


@jax.jit
def model(theta6, lsf, y, w):
    mu = C.apply_lsf(emu(theta6), lsf)
    return C.profile_continuum(mu, jnp.broadcast_to(y, mu.shape), jnp.broadcast_to(w, mu.shape), basis)


n = len(S); th = np.zeros((n, 6), "float32"); th[:, :4] = S
th[:, 4] = rng.uniform(*box["carbon_m"], n)
th[:, 5] = 10 ** rng.uniform(*np.log10(box["vmic"]), n)
good = iv > 0


def median_model(lsf_px):
    lsf = np.full(n, lsf_px, "float32")
    M = np.concatenate([np.asarray(model(jnp.asarray(th[s:s + 250]), jnp.asarray(lsf[s:s + 250]), jnp.asarray(y), jnp.asarray(iv)))
                        for s in range(0, n, 250)])
    return np.median(M, axis=0)


def chi2(m):
    return (iv * (y - m) ** 2)[good].sum() / good.sum()


# best-fit LSF width (no WDISP in boss_clam): grid over the lsf_px prior box
# ponytail: 1-D grid, 0.1 px step; swap for a bounded scalar minimiser if finer precision matters
grid = np.round(np.arange(0.3, 4.0001, 0.1), 2)
c2 = np.array([chi2(median_model(g)) for g in grid])
lsf_best = float(grid[c2.argmin()]); med = median_model(lsf_best)
C.log(f"best LSF {lsf_best:.1f} px  chi2/px {c2.min():.2f}  (grid {grid[0]}-{grid[-1]})")
# noisy realisation of the median model at this observation's S/N (its ivar)
model_obs = np.where(good, med + rng.standard_normal(med.shape) / np.sqrt(np.maximum(iv, 1e-30)), np.nan)

# ---- corner + forward panels in the upper-right triangle -------------------------------
lab = {"Teff": r"$T_{\rm eff}$", "logg": r"$\log g$", "m_H": "[M/H]", "alpha_m": r"[$\alpha$/M]"}
fig = corner.corner(S, labels=[lab[k] for k in names], truths=L[r], truth_color="C3",
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g", fig=plt.figure(figsize=(13, 13)))
x = np.arange(C.N_PIX); yo = np.where(good, y, np.nan)
# full spectrum + residuals inside the empty upper-right triangle (cols K-2..K-1, rows 0-1), no overlap
K = len(names); ax_ = np.array(fig.axes).reshape(K, K)
x0, x1 = ax_[0, K - 2].get_position().x0 + 0.04, ax_[0, K - 1].get_position().x1
y0, y1 = ax_[1, K - 1].get_position().y0 + 0.05, ax_[0, K - 1].get_position().y1 - 0.02
h = y1 - y0; axF = fig.add_axes([x0, y0 + 0.35 * h, x1 - x0, 0.65 * h])
axR = fig.add_axes([x0, y0, x1 - x0, 0.33 * h], sharex=axF)
axF.plot(x, yo, "k", lw=0.4, label="BOSS"); axF.plot(x, model_obs, "C3", lw=0.4, alpha=0.8, label="median model + noise")
axF.set_xlim(0, C.N_PIX); axF.set_ylim(np.nanpercentile(yo, 0.5) - 0.05, 1.15)
axF.legend(fontsize=7, ncol=2, loc="lower right"); axF.set_ylabel("norm. flux", fontsize=8)
axF.set_title(f"sdss_id {b['sdss_id'][r]}", fontsize=9)
axR.plot(x, yo - model_obs, "k", lw=0.4); axR.axhline(0, color="C3", lw=0.6)
axR.set_ylim(-0.3, 0.3); axR.set_ylabel("BOSS − model", fontsize=8); axR.set_xlabel("pixel", fontsize=8)
plt.setp(axF.get_xticklabels(), visible=False)
for ax in (axF, axR):
    ax.tick_params(labelsize=7)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight"); C.log(f"wrote {out}")
