"""Build the v4 spec-lite bank: one DR20 BOSS visit per boss_clam star with its real LSF.

For every downloaded spec-lite file (``Ilija_download.py``; no manifest needed, so a
partial download works):

* rest frame: shift by log10(1+Z)/1e-4 px (rest px i <- observed px i + dpx), keep the
  4479 Korg/emulator pixels; WDISP repaired (``korg_to_boss.repair_wdisp``) and shifted alike;
* normalise with that star's boss_clam continuum (same rest-frame grid, what
  ``_prepare_boss`` divides by) times the median ratio to its boss_clam normalised coadd; ivar * (cont * s)^2; the
  ``_prepare_boss`` mask (non-finite, ivar<=0, f<=0, f>=1.2) plus lambda < 3650 A;
* labels / e_labels / split / data_type from ``cache/boss_clam_prepared.npz`` (same
  star-level split as v1-v3);
* PCA of WDISP (fit on split 0): prints reconstruction error vs number of components and
  keeps the smallest K whose 99th-pct kernel-sigma error is within --pca-tol (x) of the
  plateau at --pca-max (the rest is pixel jitter, not LSF shape).

Output (``load_bank``): flux, ivar (float32), wdisp (float16), lsf_pca (standardised
coefficients), pca_mean/pca_comp/pca_std, labels, e_labels, snr (measured), sdss_id,
boss_row, split, data_type, telescope, z, frac_masked.

    JAX_PLATFORMS=cpu uv run --extra sbi --with astropy python sc_sbi/build_wdisp_bank.py
"""
import argparse, glob, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
import numpy as np
from astropy.io import fits
from korg_to_boss import repair_wdisp
import common as C

p = argparse.ArgumentParser()
p.add_argument("--spectra", default="/export/scratch/vgiusepp/sdss_sas/dr20/spectro/boss/redux/v6_2_1/spectra/daily/lite")
p.add_argument("--out", default=str(C.BANK_DEFAULT))
p.add_argument("--max-files", type=int, default=0, help="random subset (smoke)")
p.add_argument("--pca-max", type=int, default=20)
p.add_argument("--pca-tol", type=float, default=1.15,
               help="keep the smallest K whose 99th-pct sigma_k error is <= tol x the error at --pca-max")
p.add_argument("--workers", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
P = C.N_PIX
X = np.arange(4648, dtype=float)


def shift(v, dpx):
    return np.interp(X[:P] + dpx, X[:len(v)], v)


def shift_ivar(iv, dpx):
    """ivar of the linearly interpolated flux: var = (1-t)^2 var0 + t^2 var1 (independent
    pixels); any masked neighbour -> masked."""
    x = X[:P] + dpx
    i0 = np.clip(np.floor(x).astype(int), 0, len(iv) - 2)
    t = x - i0
    v0, v1 = iv[i0], iv[i0 + 1]
    with np.errstate(divide="ignore"):
        var = (1 - t) ** 2 / v0 + t ** 2 / v1
    return np.where((v0 > 0) & (v1 > 0), 1.0 / var, 0.0)


def read(f):
    try:
        with fits.open(f) as h:
            d, m = h[1].data, h[2].data[0]
            if abs(d["LOGLAM"][0] - C.LOGLAM0) > 1e-6 or len(d) < P or "WDISP" not in d.columns.names:
                return None
            dpx = np.log10(1 + float(m["Z"])) / 1e-4
            iv = np.where(np.isfinite(d["IVAR"]), d["IVAR"], 0.0)
            return (int(m["SDSS_ID"]), float(m["Z"]), m["OBS"].strip(),
                    shift(np.nan_to_num(d["FLUX"]), dpx).astype("float32"),
                    shift_ivar(iv, dpx).astype("float32"),
                    shift(repair_wdisp(d["WDISP"]), dpx).astype("float32"))
    except Exception as e:
        print(f"skip {f}: {e}", flush=True)
        return None


files = sorted(glob.glob(f"{a.spectra}/*/*/*/spec-*.fits"))
if a.max_files and len(files) > a.max_files:
    files = sorted(np.random.default_rng(a.seed).choice(files, a.max_files, replace=False))
C.log(f"reading {len(files):,} spec-lite files")
with Pool(a.workers) as pool:
    res = [r for r in pool.imap(read, files, chunksize=64) if r is not None]
sid = np.array([r[0] for r in res]); z = np.array([r[1] for r in res]); obs = np.array([r[2] for r in res])
flux = np.stack([r[3] for r in res]); ivar = np.stack([r[4] for r in res]); wdisp = np.stack([r[5] for r in res])
del res

# ---- match to boss_clam rows (labels, split) + continuum ------------------------------
boss = C.load_boss()
first = {}
for r, s in enumerate(boss["sdss_id"]):
    first.setdefault(int(s), r)
_, uniq = np.unique(sid, return_index=True)                  # one file per star
keep = np.array([int(s) in first for s in sid]) & np.isin(np.arange(sid.size), uniq)
sid, z, obs, flux, ivar, wdisp = sid[keep], z[keep], obs[keep], flux[keep], ivar[keep], wdisp[keep]
row = np.array([first[int(s)] for s in sid])
C.log(f"{sid.size:,} stars matched to boss_clam; loading continua from {C.BOSS_NPZ} (one pass, a few min)")
cont = np.load(C.BOSS_NPZ)["lux_data"]["continuum"][row, :P].astype("float32")

with np.errstate(all="ignore"):
    r = flux / cont
    # scale so the visit matches the star's boss_clam normalised coadd (median absorption ~ the
    # same), i.e. flux / (cont * s) is on the boss_clam scale
    cf, civ = boss["flux"][row], boss["ivar"][row]
    good0 = (ivar > 0) & np.isfinite(r) & (cont > 0) & (civ > 0)
    s = np.array([np.median(rr[g] / c[g]) if g.sum() > 100 else np.nan for rr, c, g in zip(r, cf, good0)])
    del cf, civ
    f = r / s[:, None]
    iv = ivar * (cont * s[:, None]) ** 2
bad = (~np.isfinite(f) | ~np.isfinite(iv) | (iv <= 0) | (f <= 0) | (f >= 1.2) | (cont <= 0)
       | (C.WAVE[None, :P] < 3650) | ~np.isfinite(s)[:, None])
flux = np.where(bad, 1.0, f).astype("float32")
ivar = np.where(bad, 0.0, iv).astype("float32")
del r, f, iv, cont
frac_masked = bad.mean(1).astype("float32")
snr = C.measured_snr(flux, ivar)
ok = np.isfinite(s) & (frac_masked < 0.9)
C.log(f"normalised; dropped {(~ok).sum()} stars with no usable pixels; "
      f"median masked {np.median(frac_masked[ok]):.1%}, measured S/N 5/50/95 % "
      f"{np.percentile(snr[ok], [5, 50, 95]).round(1)}")

split = boss["split"][row]

# ---- PCA of WDISP: choose K -----------------------------------------------------------------
s_k = lambda w: np.sqrt(np.clip((1.1 * w) ** 2 - 0.369 ** 2 + 1 / 12, 0.05 ** 2, None))
tr = np.flatnonzero(ok & (split == 0)); te = np.flatnonzero(ok & (split != 0))
rng = np.random.default_rng(a.seed)
fit = rng.choice(tr, min(20000, tr.size), replace=False)
mu = wdisp[fit].mean(0)
_, S, Vt = np.linalg.svd(wdisp[fit] - mu, full_matrices=False)
ev = np.cumsum(S ** 2) / (S ** 2).sum()
tst = wdisp[rng.choice(te, min(5000, te.size), replace=False)] if te.size else wdisp[fit[:500]]
C.log(f"WDISP PCA ({fit.size} train curves; test on {len(tst)} non-train; "
      f"APO {np.sum(obs[ok] == 'APO')}, LCO {np.sum(obs[ok] == 'LCO')})")
print(" K  cum.var   RMS[px]   99%|err|[px]   99% sigma_k rel.err", flush=True)
rels = []
for k in range(1, a.pca_max + 1):
    rec = mu + (tst - mu) @ Vt[:k].T @ Vt[:k]
    e = rec - tst
    rels.append(np.percentile(np.abs(s_k(rec) / s_k(tst) - 1), 99))
    print(f"{k:2d}  {ev[k - 1]:.5f}  {np.sqrt((e ** 2).mean()):.5f}   {np.percentile(np.abs(e), 99):.5f}       {rels[-1]:.4f}", flush=True)
K = 1 + int(np.argmax(np.array(rels) <= a.pca_tol * rels[-1]))   # beyond K: pixel jitter, not LSF shape
C.log(f"PCA: K = {K} (99th-pct sigma_k error {rels[K - 1]:.2%}, plateau {rels[-1]:.2%})")
coef = (wdisp - mu) @ Vt[:K].T
std = coef[fit].std(0)
lsf_pca = (coef / std).astype("float32")

# ---- save ----------------------------------------------------------------------------------
sel = np.flatnonzero(ok)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(".tmp.npz")
np.savez(tmp, flux=flux[sel], ivar=ivar[sel], wdisp=wdisp[sel].astype("float16"), lsf_pca=lsf_pca[sel],
         pca_mean=mu.astype("float32"), pca_comp=Vt[:K].astype("float32"), pca_std=std.astype("float32"),
         labels=boss["labels"][row[sel]], e_labels=boss["e_labels"][row[sel]], snr=snr[sel],
         sdss_id=sid[sel], boss_row=row[sel], split=split[sel], data_type=boss["data_type"][row[sel]],
         telescope=obs[sel], z=z[sel], frac_masked=frac_masked[sel], label_names=boss["label_names"])
tmp.rename(out)
C.log(f"wrote {out}: {sel.size:,} stars (split 0/1/2: {[(split[sel] == k).sum() for k in range(3)]}), K={K}")
