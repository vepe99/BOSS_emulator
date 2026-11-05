# Downloaded BOSS spectra with WDISP (for the LSF forward model)

## Main set: boss_clam stars, DR20 spec-lite (use this one)

| what | where |
|---|---|
| spectra | `/export/scratch/vgiusepp/sdss_sas/dr20/spectro/boss/redux/v6_2_1/spectra/daily/lite/<FFF>XXX/<FIELD>/<MJD>/spec-<FIELD>-<MJD>-<CATALOGID>.fits` |
| manifest (one row per star) | `/export/scratch/vgiusepp/sdss_sas/boss_clam_specLite_v6_2_1.npz` — `sdss_id, path, status, snr, field, mjd, catalogid` |
| spAll columns we use (slim cache) | `/export/scratch/vgiusepp/sdss_sas/dr20/spectro/boss/redux/v6_2_1/summary/daily/spAll-lite-v6_2_1.slim.npz` |
| download log | `/export/scratch/vgiusepp/sdss_sas/boss_clam_download.log` |
| script | `sc_sbi/Ilija_download.py` (resumable: re-run the same command) |

- **Stars:** the 88,259 unique `sdss_id` in `boss_clam/boss_clam_training_data.npz`; 87,938 have a visit with `CLASS == 'STAR'` in DR20 `v6_2_1` spAll-lite.
- **One file per star:** the highest-`SN_MEDIAN_ALL` visit classified `STAR` (a single visit, so lower S/N than the multi-visit coadds behind the boss_clam / BOSSNet labels).
- **Each file:** HDU1 has `FLUX, LOGLAM, IVAR, AND_MASK, OR_MASK, WDISP, SKY, MODEL, WRESL`; HDU2 has spAll metadata (`CLASS`, `SUBCLASS`, `Z`, `SN_MEDIAN_ALL`, `OBS`, XCSAO params, Gaia ids, ...).
- **Grid:** `LOGLAM[0] = 3.5523`, step 1e-4 dex, 4648 px = the Korg/emulator grid (emulator keeps px 0..4478). `status` in the manifest is `WDISP grid-ok <APO|LCO> <frac WDISP>0>` when checked OK.
- **Frame:** observed frame (not RV-corrected) — fit the shift (`fit_boss_spec.py` does).
- **WDISP:** Gaussian LSF sigma in 1e-4 dex pixels; sigma[A] = ln(10)·1e-4·WDISP·λ. Vedant's κ = 1.1 was calibrated on APO; LCO WDISP is ~10 % smaller. A few % of pixels have WDISP = 0 (edges) → `korg_to_boss.repair_wdisp`. Blue edge (λ < 3650 Å) can have garbage flux with IVAR > 0 — mask it.
- **Telescopes:** both APO and LCO (column `OBS` in HDU2).
- Note: `/export/scratch` is scratch space — check the purge policy before relying on it long-term.

Read one star:
```python
import numpy as np
from astropy.io import fits
m = np.load("/export/scratch/vgiusepp/sdss_sas/boss_clam_specLite_v6_2_1.npz")
i = np.flatnonzero(m["sdss_id"] == 57587370)[0]
d = fits.open(m["path"][i])[1].data          # FLUX, LOGLAM, IVAR, WDISP, ...
```

## Other downloads (smaller / earlier, DR19)

| what | where |
|---|---|
| BOSSNet summary (DR19 Astra 0.6.0; BOSSNet + XCSAO + Gaia-XP labels, `n_apogee_visits`) | `/export/data/vgiusepp/sdss_dr19/astraAllStarBossNet-0.6.0.fits.gz` |
| DR19 spAll-lite | `/export/data/vgiusepp/sdss_dr19/spAll-lite-v6_1_3.fits.gz` |
| 4,946 DR19 spec-lite (S/N matched to boss_clam, BOSSNet stars, CLASS STAR) | `/export/data/vgiusepp/sdss_dr19/speclite/` + list `speclite_jobs_5000.npz` (script `sc_sbi/fetch_speclite.py`) |
| ~4.8k mwmVisit files (per-visit `wresl`, no `WDISP`; download stopped) | `/export/data/vgiusepp/sdss_dr19/mwmVisit/` (script `sc_sbi/fetch_mwmvisit.py`) |
| test star (F3/F5V halo star, sdss_id 57587370) + fit / NUTS / PPC outputs | `sc_sbi/vedant_lsf/` |
| ~800 duplicate DR20 files + 2.6 GB spAll from the slow first attempt (can be deleted) | `/export/data/vgiusepp/sdss_sas/` |

Single-star pipeline: `sc_sbi/fit_boss_spec.py` (MAP) → `sc_sbi/hmc_boss_spec.py` (NUTS) → `sc_sbi/ppc_boss_spec.py` (posterior predictive check).
