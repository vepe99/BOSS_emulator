"""BOSS spec files (with WDISP) for boss_clam stars, by sdss_id -- Ilija's sdss_access recipe.

Ilija's snippet reads spAll from the Utah CHPC disk and opens one row's specFull file.
Here the same sdss_access paths point at a local mirror ($SAS_BASE_DIR, default
/export/data/vgiusepp/sdss_sas) and missing files are fetched from the public DR20 SAS
(p.url). Rows are chosen by SDSS_ID (not row number): per star the highest-S/N visit
with CLASS == STAR. spAll-lite (2.6 GB) is read once and cached as a slim npz.

    .venv/bin/python sc_sbi/Ilija_download.py --n 20              # 20 random boss_clam stars
    .venv/bin/python sc_sbi/Ilija_download.py --ids 57587370 ...  # specific sdss_ids
"""
import argparse, io, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--ids", type=int, nargs="*", help="sdss_ids (default: --n random boss_clam stars)")
p.add_argument("--n", type=int, default=20, help="0 = all boss_clam stars")
p.add_argument("--kind", default="specLite", choices=["specLite", "specFull"])
p.add_argument("--run2d", default="v6_2_1")
p.add_argument("--release", default="dr20")
p.add_argument("--sas", default="/export/data/vgiusepp/sdss_sas")
p.add_argument("--boss-clam", default=str(Path(__file__).resolve().parent / "cache/boss_clam_prepared.npz"))
p.add_argument("--workers", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.environ.setdefault("SAS_BASE_DIR", a.sas)

from astropy.io import fits
import sdss_access.path

sdss_path = sdss_access.path.Path(release=a.release, preserve_envvars=True)


def get(path, url):
    """Local mirror path + file bytes; download from the public SAS if missing. One write
    per file: /export/data is NFS, where urlretrieve's 8 kB writes are ~3x slower."""
    path = Path(path)
    if path.exists():
        return path, path.read_bytes()
    for attempt in range(4):
        try:
            b = urllib.request.urlopen(url, timeout=300).read(); break
        except Exception as e:
            if getattr(e, "code", None) == 404 or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(b); tmp.rename(path)
    return path, b


# ---- spAll (slim cache of the columns we need) -----------------------------------------------
spall_path = sdss_path.full("spAll-lite", run2d=a.run2d)              # sdss_access resolves the .gz once it exists
slim = Path(spall_path.removesuffix(".gz")).with_suffix(".slim.npz")
if not slim.exists():
    get(spall_path, sdss_path.url("spAll-lite", run2d=a.run2d))
    print(f"reading {spall_path} (one-time)", flush=True)
    d = fits.open(spall_path)[1].data
    np.savez(slim, **{k: np.asarray(d[k]) for k in ["SDSS_ID", "RUN2D", "FIELD", "MJD", "CATALOGID", "CLASS", "SN_MEDIAN_ALL"]})
spall = dict(np.load(slim))
cls = np.char.strip(spall["CLASS"].astype(str))

# ---- choose stars and, per star, the best STAR visit ----------------------------------------
if a.ids:
    ids = np.asarray(a.ids)
else:
    bc = np.unique(np.load(a.boss_clam)["sdss_id"])
    ids = bc if a.n == 0 else np.random.default_rng(a.seed).choice(bc, a.n, replace=False)
m = np.isin(spall["SDSS_ID"], ids) & (cls == "STAR")
rows = np.flatnonzero(m)[np.lexsort((-spall["SN_MEDIAN_ALL"][m], spall["SDSS_ID"][m]))]
rows = rows[np.r_[True, spall["SDSS_ID"][rows][1:] != spall["SDSS_ID"][rows][:-1]]]
print(f"{len(ids)} sdss_ids requested, {len(rows)} with a STAR visit in spAll {a.run2d}", flush=True)


def fetch(idx):
    kw = dict(run2d=str(spall["RUN2D"][idx]).strip(), mjd=int(spall["MJD"][idx]),
              fieldid="%06d" % spall["FIELD"][idx], isplate="", catalogid=int(spall["CATALOGID"][idx]))
    try:
        path, b = get(sdss_path.full(a.kind, **kw), sdss_path.url(a.kind, **kw))
    except Exception as e:
        return idx, None, str(e)
    with fits.open(io.BytesIO(b)) as h:
        d, obs = h[1].data, h[2].data["OBS"][0].strip()
        if "WDISP" not in d.columns.names:
            return idx, path, f"NO WDISP: {d.columns.names}"
        on_grid = abs(d["LOGLAM"][0] - 3.5523) < 1e-6 and len(d) == 4648        # = Korg/emulator grid
        return idx, path, f"WDISP {'grid-ok' if on_grid else 'OFF-GRID'} {obs} {(d['WDISP'] > 0).mean():.3f}"


t0, res = time.time(), []
with ThreadPoolExecutor(a.workers) as ex:
    for k, (idx, path, msg) in enumerate(ex.map(fetch, rows), 1):
        res.append((idx, path, msg))
        if len(rows) <= 50 or not msg.startswith("WDISP grid-ok"):
            print(f"sdss_id {spall['SDSS_ID'][idx]}  S/N {spall['SN_MEDIAN_ALL'][idx]:5.1f}  {msg}  {path}", flush=True)
        if k % 2000 == 0:
            r = k / (time.time() - t0)
            print(f"{k}/{len(rows)}  {r:.1f} files/s  ETA {(len(rows) - k) / r / 60:.0f} min", flush=True)
ok = sum(m.startswith("WDISP grid-ok") for _, _, m in res)
print(f"{ok}/{len(res)} files with WDISP on the Korg grid, in {(time.time() - t0) / 60:.1f} min")
out = Path(a.sas) / f"boss_clam_{a.kind}_{a.run2d}.npz"
np.savez(out, sdss_id=spall["SDSS_ID"][rows], path=np.array([str(p) for _, p, _ in res]),
         status=np.array([m for _, _, m in res]), snr=spall["SN_MEDIAN_ALL"][rows],
         field=spall["FIELD"][rows], mjd=spall["MJD"][rows], catalogid=spall["CATALOGID"][rows])
print(f"wrote {out}")
