"""Phase 1: how well can Korg + emulator explain real boss_clam spectra?

For a random subset of BOSS **train-split** stars (labels inside the Korg box)
we fit the six labels by maximum likelihood through the emulator and compare
with the BOSS-CLAM reference labels:

1. ``clam_fixed`` -- Teff, logg, [M/H], [alpha/M] held at the CLAM values,
   only carbon_m and vmic free: chi2 *at* the reference labels.
2. ``free``       -- all six free; best of a CLAM-started and
   ``--n-random-inits`` prior-started optimisation (Adam on a sigmoid-bounded
   parametrisation of the emulator box). If the free fit lands far from CLAM
   (e.g. logg off by ~1 dex), the problem is physics / preprocessing, not the
   posterior network.
3. ``s_m``        -- per-pixel model-mismatch scatter maximising
   ``sum_n [r^2 / (sigma^2 + s^2) + ln(sigma^2 + s^2)]`` (BOSS-CLAM Eq. 6,
   labels fixed), iterated ``--sm-iters`` times with the free fit redone under
   ``sigma^2 + s_m^2``. Also per data_type and per (Teff, dwarf/giant) group.
4. Emulator-alone error on noiseless Korg test spectra, per pixel, to split
   ``s_m`` into emulator error vs Korg-vs-real mismatch.

Outputs in ``<outdir>/<tag>/``:
  ``mismatch.npz``   -- ``s_m`` + residual bank, read by ``train.py --mismatch``
  ``fits.npz``       -- per-star labels / chi2 for every fit
  ``summary.json``   -- label offsets & scatter vs CLAM, chi2 statistics
  ``*.png``          -- Kiel diagrams, fit-vs-CLAM, chi2, s_m spectrum

Example (GPU picked by autocvd)::

    .venv/bin/python sc_sbi/diagnose_mismatch.py --n-stars 4000 --tag v1
    # quick CPU check
    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/diagnose_mismatch.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_LAB = len(C.LABELS)
I_TEFF, I_LOGG = C.LABELS.index("Teff"), C.LABELS.index("logg")
BOSS_IDX = [C.LABELS.index(k) for k in C.BOSS_LABELS]
DT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7f7f7f"]


# --------------------------------------------------------------------------
# label fitting through the emulator
# --------------------------------------------------------------------------

class Fitter:
    """Batched max-likelihood label fits, labels bounded to the emulator box."""

    def __init__(self, emulator_fn, box: dict, steps: int, lr: float, batch: int):
        self.lo = np.array([np.log10(box[k][0]) if k in C.LOG_LABELS else box[k][0]
                            for k in C.LABELS], "float32")
        self.hi = np.array([np.log10(box[k][1]) if k in C.LOG_LABELS else box[k][1]
                            for k in C.LABELS], "float32")
        self.is_log = np.array([k in C.LOG_LABELS for k in C.LABELS])
        self.batch, self.steps = batch, steps
        lo, hi, is_log = jnp.asarray(self.lo), jnp.asarray(self.hi), jnp.asarray(self.is_log)

        def to_theta(u):
            z = lo + (hi - lo) * jax.nn.sigmoid(u)
            # power only on the log columns (10**Teff would overflow -> NaN grads)
            return jnp.where(is_log, 10.0 ** jnp.where(is_log, z, 0.0), z)

        def chi2(theta, y, iv_eff):
            mu = emulator_fn(theta)
            return jnp.sum(iv_eff * (y - mu) ** 2, axis=-1)

        opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.01))

        @jax.jit
        def run(u0, fixed, free, y, iv_eff):
            def loss(u):
                th = jnp.where(free, to_theta(u), fixed)
                return chi2(th, y, iv_eff).sum()

            def step(carry, _):
                u, s = carry
                g = jax.grad(loss)(u)
                upd, s = opt.update(g, s, u)
                return (optax.apply_updates(u, upd), s), None

            (u, _), _ = jax.lax.scan(step, (u0, opt.init(u0)), None, length=steps)
            th = jnp.where(free, to_theta(u), fixed)
            return th, chi2(th, y, iv_eff)

        self._run = run
        self.chi2 = jax.jit(chi2)

    def to_u(self, theta):
        z = np.where(self.is_log, np.log10(np.maximum(theta, 1e-6)), theta)
        f = np.clip((z - self.lo) / (self.hi - self.lo), 1e-4, 1 - 1e-4)
        return np.log(f / (1 - f)).astype("float32")

    def clip(self, theta):
        z = np.where(self.is_log, np.log10(np.maximum(theta, 1e-6)), theta)
        z = np.clip(z, self.lo + 1e-4 * (self.hi - self.lo), self.hi - 1e-4 * (self.hi - self.lo))
        return np.where(self.is_log, 10.0 ** np.where(self.is_log, z, 0.0), z).astype("float32")

    def random_theta(self, n, rng):
        z = rng.uniform(self.lo, self.hi, size=(n, N_LAB))
        return np.where(self.is_log, 10.0 ** np.where(self.is_log, z, 0.0), z).astype("float32")

    def fit(self, theta0, free, y, iv_eff):
        """theta0 (n, 6) start (and value of fixed dims); free (6,) bool."""
        n = y.shape[0]
        th_out = np.empty((n, N_LAB), "float32")
        chi_out = np.empty(n, "float32")
        free_j = jnp.asarray(free)
        for s in range(0, n, self.batch):
            e = min(s + self.batch, n)
            pad = self.batch - (e - s)          # fixed shapes -> one compilation
            sl = lambda a: np.concatenate([a[s:e], np.repeat(a[e - 1:e], pad, 0)]) if pad else a[s:e]
            t0 = sl(theta0)
            th, chi = self._run(jnp.asarray(self.to_u(t0)), jnp.asarray(t0), free_j,
                                jnp.asarray(sl(y)), jnp.asarray(sl(iv_eff)))
            th_out[s:e], chi_out[s:e] = np.asarray(th)[:e - s], np.asarray(chi)[:e - s]
        return th_out, chi_out

    def residuals(self, emulator_fn, theta, y, ivar):
        out = np.empty_like(y)
        for s in range(0, y.shape[0], self.batch):
            e = min(s + self.batch, y.shape[0])
            out[s:e] = y[s:e] - np.asarray(emulator_fn(jnp.asarray(theta[s:e])))
        return np.where(ivar > 0, out, 0.0).astype("float32")


def iv_eff(ivar, s_m):
    if s_m is None:
        return ivar
    return (ivar / (1.0 + ivar * s_m[None] ** 2)).astype("float32")


# --------------------------------------------------------------------------
# per-pixel mismatch scatter
# --------------------------------------------------------------------------

S_GRID = np.concatenate([[0.0], np.logspace(-4, 0, 161)]).astype("float32")


def fit_sm(resid, ivar, min_good=20):
    """argmax_s of the Gaussian likelihood with variance sigma^2 + s^2, per pixel
    (grid search on S_GRID). Pixels with < min_good good stars get 0."""
    good = ivar > 0
    var = np.where(good, 1.0 / np.maximum(ivar, 1e-30), 1.0)
    r2 = resid ** 2
    best = np.full(resid.shape[1], np.inf, "float64")
    s_m = np.zeros(resid.shape[1], "float32")
    for s in S_GRID:
        tot = var + s * s
        nll = np.where(good, r2 / tot + np.log(tot), 0.0).sum(0)
        better = nll < best
        best[better], s_m[better] = nll[better], s
    ngood = good.sum(0)
    s_m[ngood < min_good] = 0.0
    return s_m, ngood


def groups_of(teff, logg, dtype):
    g = {f"data_type={t}": dtype == t for t in np.unique(dtype)}
    for lo, hi in ((0, 4000), (4000, 5000), (5000, 6000), (6000, 1e5)):
        for name, m in (("dwarf", logg >= 3.5), ("giant", logg < 3.5)):
            g[f"Teff {lo:.0f}-{hi:.0f} {name}"] = (teff >= lo) & (teff < hi) & m
    return g


# --------------------------------------------------------------------------
# summaries & plots
# --------------------------------------------------------------------------

def nmad(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def compare(fit, ref):
    ok = np.isfinite(fit) & np.isfinite(ref)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    d = fit[ok] - ref[ok]
    slope, icpt = np.polyfit(ref[ok], fit[ok], 1)
    return {"n": int(ok.sum()), "median_diff": float(np.median(d)), "nmad": nmad(d),
            "mae": float(np.abs(d).mean()), "slope_fit_vs_ref": float(slope),
            "intercept": float(icpt)}


def plot_kiel(ref, fits, path):
    fig, axes = plt.subplots(1, 1 + len(fits), figsize=(4.2 * (1 + len(fits)), 4.0),
                             sharex=True, sharey=True, squeeze=False)
    panels = [("BOSS-CLAM reference", ref)] + list(fits.items())
    for ax, (name, th) in zip(axes[0], panels):
        sc_ = ax.scatter(th[:, I_TEFF], th[:, I_LOGG], c=th[:, BOSS_IDX[2]], s=2,
                         cmap="viridis", vmin=-2, vmax=0.5, rasterized=True)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Teff [K]")
    axes[0, 0].set_ylabel("logg")
    axes[0, 0].invert_xaxis()
    axes[0, 0].invert_yaxis()
    fig.colorbar(sc_, ax=axes[0, -1], label="[M/H]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fit_vs_ref(ref, fit, dtype, path, title):
    types = list(np.unique(dtype))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for c, i in enumerate(BOSS_IDX):
        k = C.LABELS[i]
        x, y = ref[:, i], fit[:, i]
        ok = np.isfinite(x) & np.isfinite(y)
        lo, hi = np.percentile(np.r_[x[ok], y[ok]], [0.5, 99.5])
        ax = axes[0, c]
        ax.hexbin(x[ok], y[ok], gridsize=45, bins="log", cmap="viridis", mincnt=1,
                  extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color="#D55E00", lw=0.8)
        cmp_ = compare(y, x)
        ax.set_title(f"{k}: med Δ {cmp_.get('median_diff', np.nan):+.3g}, "
                     f"NMAD {cmp_.get('nmad', np.nan):.3g}", fontsize=9)
        ax.set_xlabel(f"CLAM {k}")
        ax.set_ylabel(f"emulator fit {k}")
        ax = axes[1, c]
        for t, col in zip(types, DT_COLORS):
            m = ok & (dtype == t)
            ax.scatter(ref[m, I_TEFF], (y - x)[m], s=1.5, alpha=0.4, color=col,
                       label=t, rasterized=True)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("CLAM Teff [K]")
        ax.set_ylabel(f"Δ{k} (fit − CLAM)")
        ax.set_ylim(np.nanpercentile((y - x)[ok], [1, 99]))
    axes[1, 0].legend(fontsize=7, markerscale=5)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_chi2(chis: dict, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    allv = np.concatenate(list(chis.values()))
    bins = np.logspace(np.log10(max(np.nanpercentile(allv, 0.5), 0.1)),
                       np.log10(np.nanpercentile(allv, 99.5)), 60)
    for (name, v), col in zip(chis.items(), DT_COLORS):
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=col,
                label=f"{name} (median {np.nanmedian(v):.2f})")
    ax.axvline(1, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^2$ / good pixel")
    ax.set_ylabel("stars")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sm(s_m, emu_rms, sig_med, resid_med, frac_masked, path):
    pix = np.arange(C.N_PIX)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.3, 0.8]})
    ax = axes[0]
    ax.plot(pix, sig_med, color="#7f7f7f", lw=0.7, label="median photon σ (BOSS)")
    ax.plot(pix, s_m, color="#0072B2", lw=0.8, label="s_m (mismatch scatter)")
    if emu_rms is not None:
        ax.plot(pix, emu_rms, color="#D55E00", lw=0.8, label="emulator RMS vs Korg")
    ax.set_yscale("log")
    ax.set_ylabel("flux units")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Per-pixel scatter budget (continuum-normalised flux)")
    axes[1].plot(pix, resid_med, color="#0072B2", lw=0.6)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("median residual\n(BOSS − emu)")
    lim = np.nanpercentile(np.abs(resid_med), 99.5)
    axes[1].set_ylim(-1.2 * lim, 1.2 * lim)
    axes[2].plot(pix, frac_masked, color="#7f7f7f", lw=0.6)
    axes[2].set_ylabel("masked\nfraction")
    axes[2].set_xlabel("pixel index")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sm_groups(sm_groups: dict, s_m, path):
    pix = np.arange(C.N_PIX)
    keys = [k for k in sm_groups if k.startswith("Teff")]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(pix, s_m, color="k", lw=1.0, label="all")
    for k, col in zip(keys, DT_COLORS * 2):
        ax.plot(pix, sm_groups[k], lw=0.6, color=col, label=k,
                ls="-" if "dwarf" in k else "--")
    ax.set_yscale("log")
    ax.set_xlabel("pixel index")
    ax.set_ylabel("s_m")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emulator", default=str(C.EMULATOR_CKPT))
    p.add_argument("--korg-cache", default=str(C.KORG_CACHE))
    p.add_argument("--n-stars", type=int, default=4000)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--n-random-inits", type=int, default=2)
    p.add_argument("--sm-iters", type=int, default=2)
    p.add_argument("--n-korg-test", type=int, default=5000,
                   help="noiseless Korg test spectra for the emulator-alone RMS (0: skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(C.PROJECT_ROOT / "trained_model/sc_sbi/mismatch"))
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true", help="200 stars, 30 steps, 1 s_m iteration")
    args = p.parse_args()
    if args.smoke:
        args.n_stars, args.steps, args.sm_iters = 200, 30, 1
        args.batch, args.n_korg_test = 100, 300
        args.tag = args.tag or "smoke"
    out = Path(args.outdir) / (args.tag or time.strftime("run_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    C.log(f"out {out}  devices={jax.devices()}")

    # ---- stars --------------------------------------------------------------
    box = C.label_box(args.korg_cache)
    boss = C.load_boss()
    L = boss["labels"]
    ok = (boss["split"] == 0) & C.boss_in_box(boss, box) & np.isfinite(L).all(1)
    rows = np.flatnonzero(ok)
    rows = np.sort(rng.choice(rows, min(args.n_stars, rows.size), replace=False))
    y, ivar = boss["flux"][rows], boss["ivar"][rows]
    ref4, snr, dtype = L[rows], boss["snr"][rows], boss["data_type"][rows].astype(str)
    C.log(f"{rows.size:,} BOSS train stars (in box, all 4 reference labels finite)")

    emu = C.make_emulator_fn(args.emulator)
    emu_j = jax.jit(emu)
    fitter = Fitter(emu, box, args.steps, args.lr, args.batch)
    ngood = np.maximum((ivar > 0).sum(1), 1)

    # reference in 6-label space: carbon_m = 0, vmic = geometric box centre
    ref6 = np.empty((rows.size, N_LAB), "float32")
    ref6[:, BOSS_IDX] = ref4
    ref6[:, C.LABELS.index("carbon_m")] = 0.0
    lv = np.log10(box["vmic"])
    ref6[:, C.LABELS.index("vmic")] = 10 ** lv.mean()
    ref6 = fitter.clip(ref6)

    # ---- 1. chi2 at the CLAM labels ----------------------------------------
    free_cv = np.array([k in ("carbon_m", "vmic") for k in C.LABELS])
    t0 = time.perf_counter()
    th_clam, chi_clam = fitter.fit(ref6, free_cv, y, ivar)
    C.log(f"clam_fixed fit: median chi2/pix {np.median(chi_clam / ngood):.2f} "
          f"({time.perf_counter() - t0:.0f} s)")

    # ---- 2. free fit, multi-start -------------------------------------------
    def free_fit(s_m, starts):
        ive = iv_eff(ivar, s_m)
        best_th, best_chi, which = None, None, None
        for name, th0 in starts.items():
            th, chi = fitter.fit(th0, np.ones(N_LAB, bool), y, ive)
            if best_th is None:
                best_th, best_chi = th, chi
                which = np.full(rows.size, name, dtype=object)
            else:
                b = chi < best_chi
                best_th[b], best_chi[b], which[b] = th[b], chi[b], name
            C.log(f"  start {name:8s}: median chi2/pix {np.median(chi / ngood):.2f}")
        return best_th, best_chi, which

    starts = {"clam": th_clam} | {f"random{r}": fitter.random_theta(rows.size, rng)
                                  for r in range(args.n_random_inits)}
    t0 = time.perf_counter()
    th_free, chi_free, which = free_fit(None, starts)
    C.log(f"free fit (photon noise): median chi2/pix {np.median(chi_free / ngood):.2f} "
          f"({time.perf_counter() - t0:.0f} s); winning start: "
          + ", ".join(f"{k} {np.mean(which == k):.0%}" for k in starts))

    # ---- 3. s_m, iterated ----------------------------------------------------
    s_m = None
    th_cur = th_free
    sm_hist = []
    for it in range(args.sm_iters):
        resid = fitter.residuals(emu_j, th_cur, y, ivar)
        s_m, npix_good = fit_sm(resid, ivar)
        sm_hist.append(float(np.median(s_m)))
        C.log(f"s_m iter {it}: median {np.median(s_m):.4f}, p90 {np.percentile(s_m, 90):.4f}")
        th_cur, chi_cur, which_sm = free_fit(s_m, {"prev": th_cur, "clam": th_clam})
    th_sm = th_cur
    resid = fitter.residuals(emu_j, th_sm, y, ivar)
    s_m, npix_good = fit_sm(resid, ivar)
    chi_sm = np.asarray(fitter.chi2(jnp.asarray(th_sm), jnp.asarray(y),
                                    jnp.asarray(iv_eff(ivar, s_m))))
    chi_clam_sm = np.asarray(fitter.chi2(jnp.asarray(th_clam), jnp.asarray(y),
                                         jnp.asarray(iv_eff(ivar, s_m))))
    groups = groups_of(ref4[:, 0], ref4[:, 1], dtype)
    sm_groups = {}
    for g, m in groups.items():
        if m.sum() >= 100:
            sm_groups[g] = fit_sm(resid[m], ivar[m])[0]

    # ---- 4. emulator alone on Korg ------------------------------------------
    emu_rms = None
    if args.n_korg_test:
        korg, _, _ = C.load_korg(args.korg_cache, seed=0, splits=("test",))
        Lk, Fk = korg["test"]
        sel = np.sort(rng.choice(Lk.shape[0], min(args.n_korg_test, Lk.shape[0]), replace=False))
        Lk, Fk = Lk[sel], Fk[sel]
        d = np.concatenate([np.asarray(emu_j(jnp.asarray(Lk[s:s + 1024]))) - Fk[s:s + 1024]
                            for s in range(0, Lk.shape[0], 1024)])
        emu_rms = np.sqrt((d ** 2).mean(0)).astype("float32")
        C.log(f"emulator RMS vs Korg test: median over pixels {np.median(emu_rms):.4f}")

    # ---- save -----------------------------------------------------------------
    np.savez(out / "mismatch.npz", s_m=s_m, n_good_stars=npix_good,
             emu_rms=emu_rms if emu_rms is not None else np.zeros(0),
             bank_rows=rows, bank_resid=resid,
             bank_teff=th_sm[:, I_TEFF], bank_logg=th_sm[:, I_LOGG],
             group_names=np.array(list(sm_groups)),
             s_m_groups=np.stack(list(sm_groups.values())) if sm_groups else np.zeros(0))
    np.savez(out / "fits.npz", rows=rows, ref=ref4, snr=snr, data_type=dtype,
             theta_clam_fixed=th_clam, chi2_clam_fixed=chi_clam,
             theta_free=th_free, chi2_free=chi_free, start_won=which.astype(str),
             theta_free_sm=th_sm, chi2_free_sm=chi_sm, chi2_clam_sm=chi_clam_sm,
             n_good_pix=ngood)

    sig = np.where(ivar > 0, 1 / np.sqrt(np.maximum(ivar, 1e-30)), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        sig_pix_med = np.nanmedian(sig, 0)
    summary = {"config": vars(args), "n_stars": int(rows.size),
               "chi2_per_pix_median": {
                   "clam_fixed": float(np.median(chi_clam / ngood)),
                   "free": float(np.median(chi_free / ngood)),
                   "clam_fixed_with_sm": float(np.median(chi_clam_sm / ngood)),
                   "free_with_sm": float(np.median(chi_sm / ngood))},
               "start_won_free": {k: float(np.mean(which == k)) for k in starts},
               "s_m": {"median": float(np.median(s_m)),
                       "p90": float(np.percentile(s_m, 90)),
                       "median_photon_sigma": float(np.median(
                           np.where(ivar > 0, 1 / np.sqrt(np.maximum(ivar, 1e-30)), np.nan)[ivar > 0])),
                       "frac_pix_sm_gt_photon": float(np.mean(s_m > sig_pix_med)),
                       "iter_medians": sm_hist},
               "labels": {}}
    for i, k in zip(BOSS_IDX, C.BOSS_LABELS):
        j = C.BOSS_LABELS.index(k)
        summary["labels"][k] = {
            "free_vs_clam": compare(th_free[:, i], ref4[:, j]),
            "free_sm_vs_clam": compare(th_sm[:, i], ref4[:, j]),
            "by_data_type": {t: compare(th_sm[dtype == t, i], ref4[dtype == t, j])
                             for t in np.unique(dtype)}}
    dl = th_sm[:, I_LOGG] - ref4[:, 1]
    summary["frac_abs_dlogg_gt_0.5"] = float(np.mean(np.abs(dl) > 0.5))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- plots ------------------------------------------------------------------
    plot_kiel(ref6, {"emulator fit (photon σ)": th_free, "emulator fit (σ² + s_m²)": th_sm},
              out / "kiel.png")
    plot_fit_vs_ref(ref6, th_sm, dtype, out / "fit_vs_clam.png",
                    "Free emulator fit (σ² + s_m²) vs BOSS-CLAM labels")
    plot_chi2({"CLAM labels, photon σ": chi_clam / ngood,
               "free fit, photon σ": chi_free / ngood,
               "CLAM labels, σ² + s_m²": chi_clam_sm / ngood,
               "free fit, σ² + s_m²": chi_sm / ngood}, out / "chi2.png")
    with np.errstate(all="ignore"):
        resid_med = np.nanmedian(np.where(ivar > 0, resid, np.nan), 0)
    plot_sm(s_m, emu_rms, sig_pix_med, resid_med, (ivar <= 0).mean(0), out / "sm_spectrum.png")
    if sm_groups:
        plot_sm_groups(sm_groups, s_m, out / "sm_groups.png")
    print(json.dumps(summary["chi2_per_pix_median"], indent=2))
    print(json.dumps({k: v["free_sm_vs_clam"] for k, v in summary["labels"].items()}, indent=2))
    C.log(f"wrote {out}")


if __name__ == "__main__":
    main()
