"""Cut a label-region sub-cache out of a preprocessed cache.

Diagnostic helper: writes a new ``korg_grid_train.h5`` holding only the rows
that satisfy a label filter, with the same datasets/attrs as the full cache
(plus ``source_row`` = row index in the parent cache), so the ordinary
training / evaluation scripts run on it unchanged via ``BOSS_CACHE_DIR``.

Example::

    .venv/bin/python script/make_region_cache.py \\
        --src preprocessed_data_combined_v2 --dst preprocessed_data_region_coolgiants \\
        --teff-max 4000 --logg-max 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", required=True, help="parent cache dir")
p.add_argument("--dst", required=True, help="output cache dir")
p.add_argument("--teff-min", type=float, default=-np.inf, dest="teff_min")
p.add_argument("--teff-max", type=float, default=np.inf, dest="teff_max")
p.add_argument("--logg-min", type=float, default=-np.inf, dest="logg_min")
p.add_argument("--logg-max", type=float, default=np.inf, dest="logg_max")
a = p.parse_args()

src = Path(a.src) / "korg_grid_train.h5"
dst_dir = Path(a.dst); dst_dir.mkdir(parents=True, exist_ok=True)
dst = dst_dir / "korg_grid_train.h5"

with h5py.File(src, "r") as f:
    T = f["Teff"][:]; G = f["logg"][:]
    m = (T >= a.teff_min) & (T <= a.teff_max) & (G >= a.logg_min) & (G <= a.logg_max)
    rows = np.flatnonzero(m)
    print(f"[region] {rows.size:,} of {T.size:,} rows  "
          f"Teff in [{a.teff_min:g},{a.teff_max:g}]  logg in [{a.logg_min:g},{a.logg_max:g}]")
    with h5py.File(dst, "w") as g:
        for k in f.keys():
            d = f[k]
            if d.ndim == 1:
                g.create_dataset(k, data=d[:][rows])
            else:
                out = g.create_dataset(k, shape=(rows.size, d.shape[1]), dtype=d.dtype,
                                       chunks=(min(4096, rows.size), d.shape[1]))
                for i in range(0, rows.size, 20000):
                    r = rows[i:i + 20000]
                    out[i:i + r.size] = d[r[0]:r[-1] + 1][r - r[0]]
        g.create_dataset("source_row", data=rows.astype("int64"))
        for k, v in f.attrs.items():
            g.attrs[k] = v
        g.attrs["n_converged"] = rows.size
        g.attrs["region"] = (f"Teff[{a.teff_min:g},{a.teff_max:g}] "
                             f"logg[{a.logg_min:g},{a.logg_max:g}] of {src}")
print(f"[region] wrote {dst} ({dst.stat().st_size/1e9:.2f} GB)")
