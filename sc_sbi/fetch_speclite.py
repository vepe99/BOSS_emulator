"""Download BOSS spec-lite files (they carry WDISP + WRESL) for a random subset of the
S/N-matched selection made by fetch_mwmvisit.py (mwmvisit_selection.npz): per star, the
highest-S/N visit classified STAR in spAll-lite (FIELD, MJD, CATALOGID -> URL). Resumable.

    .venv/bin/python sc_sbi/fetch_speclite.py --n 5000
"""
import argparse, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from astropy.io import fits

R = Path("/export/data/vgiusepp/sdss_dr19")
p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=5000)
p.add_argument("--selection", default=str(R / "mwmvisit_selection.npz"))
p.add_argument("--spall", default=str(R / "spAll-lite-v6_1_3.fits.gz"))
p.add_argument("--out", default=str(R / "speclite"))
p.add_argument("--workers", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
URL = "https://dr19.sdss.org/sas/dr19/spectro/boss/redux/v6_1_3/spectra/lite/{f:06d}/{m}/spec-{f:06d}-{m}-{c}.fits"

sel = np.load(a.selection)
ids = np.random.default_rng(a.seed).permutation(sel["sdss_id"])[:a.n]
s = fits.open(a.spall)[1].data
sid, snr = np.asarray(s["SDSS_ID"]), np.asarray(s["SN_MEDIAN_ALL"], float)
m = np.isin(sid, ids) & (np.char.strip(np.asarray(s["CLASS"]).astype(str)) == "STAR")   # pipeline class STAR only
rows = np.flatnonzero(m)[np.lexsort((-snr[m], sid[m]))]                 # per star: highest S/N first
rows = rows[np.r_[True, sid[rows][1:] != sid[rows][:-1]]]
jobs = [(int(s["FIELD"][r]), int(s["MJD"][r]), int(s["CATALOGID"][r])) for r in rows]
print(f"{len(ids)} stars requested, {len(jobs)} found in spAll; S/N of chosen visits 5/50/95 %: "
      f"{np.percentile(snr[rows], [5, 50, 95]).round(1)}", flush=True)
np.savez(out.parent / f"speclite_jobs_{a.n}.npz", sdss_id=sid[rows], field=[j[0] for j in jobs],
         mjd=[j[1] for j in jobs], catalogid=[j[2] for j in jobs], snr_visit=snr[rows])


def fetch(j):
    f, mj, c = j; dst = out / f"spec-{f:06d}-{mj}-{c}.fits"
    if dst.exists():
        return 0
    tmp = dst.with_suffix(".part")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(URL.format(f=f, m=mj, c=c), tmp)
            tmp.rename(dst); return dst.stat().st_size
        except Exception as e:
            err = e
            if getattr(e, "code", None) == 404:
                break
            time.sleep(2 ** attempt)
    print(f"FAILED {j}: {err}", file=sys.stderr, flush=True); return -1


t0, done, nb, nf = time.time(), 0, 0, 0
with ThreadPoolExecutor(a.workers) as ex:
    for b in ex.map(fetch, jobs):
        done += 1; nb += max(b, 0); nf += b < 0
        if done % 500 == 0 or done == len(jobs):
            r = done / (time.time() - t0)
            print(f"{done}/{len(jobs)}  {r:.1f} files/s  {nb / 1e9:.2f} GB  failed {nf}  ETA {(len(jobs) - done) / r / 60:.0f} min", flush=True)
