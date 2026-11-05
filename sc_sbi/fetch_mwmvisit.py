"""Pick a BOSSNet (DR19 Astra 0.6.0) subset whose S/N follows boss_clam, and download
its mwmVisit files (per-visit rest-frame flux/ivar + `wresl` = the BOSS LSF FWHM [A]).

Selection: every boss_clam star that is in the BOSSNet summary (they also carry
BOSS-CLAM labels), then BOSSNet stars with flag_bad == 0 drawn per S/N bin so the
total S/N histogram matches boss_clam's. Download over HTTPS, a few parallel
connections, resumable (existing files are skipped).

    .venv/bin/python sc_sbi/fetch_mwmvisit.py --n 90000            # select + download
    .venv/bin/python sc_sbi/fetch_mwmvisit.py --n 300 --test       # timing test
"""
import argparse, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from astropy.io import fits

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=90000)
p.add_argument("--summary", default="/export/data/vgiusepp/sdss_dr19/astraAllStarBossNet-0.6.0.fits.gz")
p.add_argument("--boss-clam", default=str(Path(__file__).resolve().parent / "cache/boss_clam_prepared.npz"))
p.add_argument("--out", default="/export/data/vgiusepp/sdss_dr19/mwmVisit")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--test", action="store_true", help="download only the first --n of a random order, report speed")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
URL = "https://dr19.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/visit/{:02d}/{:02d}/mwmVisit-0.6.0-{}.fits"

# ---- selection --------------------------------------------------------------------------
d = fits.open(a.summary)[1].data
sid, snr, bad = np.asarray(d["sdss_id"]), np.asarray(d["snr"]), np.asarray(d["flag_bad"])
z = np.load(a.boss_clam)
ref_snr = z["snr"][np.isfinite(z["snr"])]
rng = np.random.default_rng(a.seed)
edges = np.r_[0, np.percentile(ref_snr, np.linspace(2, 98, 49)), np.inf]          # 50 equal-count bins of boss_clam S/N
target = np.round(np.histogram(ref_snr, edges)[0] / len(ref_snr) * a.n).astype(int)
forced = np.isin(sid, z["sdss_id"])                                                 # boss_clam stars present in BOSSNet
pool = ~forced & ~bad & np.isfinite(snr)
b_all = np.clip(np.digitize(snr, edges) - 1, 0, len(target) - 1)
pick = [np.flatnonzero(forced)]
for k in range(len(target)):
    need = target[k] - int((forced & (b_all == k)).sum())
    cand = np.flatnonzero(pool & (b_all == k))
    if need > 0:
        pick.append(rng.choice(cand, min(need, len(cand)), replace=False))
sel = np.unique(np.concatenate(pick))
print(f"selected {len(sel)} stars ({forced.sum()} boss_clam overlap); S/N 5/25/50/75/95 %: "
      f"{np.percentile(snr[sel], [5, 25, 50, 75, 95]).round(1)}  vs boss_clam {np.percentile(ref_snr, [5, 25, 50, 75, 95]).round(1)}")
np.savez(out.parent / "mwmvisit_selection.npz", sdss_id=sid[sel], snr=snr[sel], in_boss_clam=forced[sel],
         teff=d["teff"][sel], logg=d["logg"][sel], fe_h=d["fe_h"][sel], flag_bad=bad[sel],
         n_apogee_visits=d["n_apogee_visits"][sel])

# ---- download ---------------------------------------------------------------------------
ids = sid[sel]
if a.test:
    ids = rng.permutation(ids)[:a.n]


def fetch(i):
    i = int(i); f = out / f"mwmVisit-0.6.0-{i}.fits"
    if f.exists():
        return 0
    tmp = f.with_suffix(".part")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(URL.format(i // 100 % 100, i % 100, i), tmp)
            tmp.rename(f); return f.stat().st_size
        except Exception as e:
            err = e
            if getattr(e, "code", None) == 404:
                break
            time.sleep(2 ** attempt)
    print(f"FAILED {i}: {err}", file=sys.stderr, flush=True); return -1


t0, done, nbytes, nfail = time.time(), 0, 0, 0
with ThreadPoolExecutor(a.workers) as ex:
    for s in ex.map(fetch, ids):
        done += 1; nbytes += max(s, 0); nfail += s < 0
        if done % 2000 == 0 or done == len(ids):
            dt = time.time() - t0
            print(f"{done}/{len(ids)}  {done / dt:.1f} files/s  {nbytes / 1e9:.2f} GB  failed {nfail}  "
                  f"ETA {(len(ids) - done) / max(done / dt, 1e-9) / 60:.0f} min", flush=True)
print(f"done in {(time.time() - t0) / 60:.1f} min, {nbytes / 1e9:.2f} GB, {nfail} failed")
