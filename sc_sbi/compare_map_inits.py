"""Density recovery plot: NPE median vs MAP warm-started (NPE median) vs MAP from random prior starts,
against BOSS-CLAM, from two npe_then_map.py runs (--init npe / --init random) on the same stars.

    python sc_sbi/compare_map_inits.py trained_model/sc_sbi/v4_vedant map_fulltest_npe map_fulltest_random
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

run, dw, dr = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
W, R = np.load(run / dw / "map_results.npz"), np.load(run / dr / "map_results.npz")
assert np.array_equal(W["rows"], R["rows"]), "runs on different stars"
ref = W["ref"]
mdw = (ref[:, 0] < 4000) & (ref[:, 1] >= 4)
names = ["Teff", "logg", "[M/H]", "[α/M]"]
ests = [("NPE median", W["npe"], W["chi2pix_npe"]), ("MAP, warm start (NPE median)", W["map"], W["chi2pix_map"]),
        ("MAP, random start", R["map"], R["chi2pix_map"])]
lims = [(3000, 7500), (-0.5, 5.5), (-3.5, 0.8), (-0.3, 0.7)]

fig, ax = plt.subplots(3, 4, figsize=(18, 13))
lines = []
for i, (lab, est, chi) in enumerate(ests):
    for j in range(4):
        A, x, y = ax[i, j], ref[:, j], est[:, j]
        ok = np.isfinite(x)
        A.hist2d(x[ok], y[ok], bins=80, range=[lims[j], lims[j]], norm=LogNorm(), cmap="viridis", cmin=1)
        A.plot(lims[j], lims[j], "r-", lw=0.8)
        mae = np.nanmean(np.abs(y - x)); mae_n = np.nanmean(np.abs(y - x)[~mdw])
        bias_n = np.nanmedian((y - x)[~mdw])
        A.set_title(f"{names[j]}: MAE {mae:.3g} (no M dw {mae_n:.3g}, bias {bias_n:+.3g})", fontsize=9)
        A.set_xlabel(f"BOSS-CLAM {names[j]}")
        if j == 0:
            A.set_ylabel(lab)
    lines.append(f"{lab}: median χ²/pix {np.median(chi):.3f} (no M dw {np.median(chi[~mdw]):.3f})")
fig.suptitle(f"boss_clam test coadds (n={ref.shape[0]:,}, {mdw.sum():,} M dwarfs), emulator + Vedant LSF + ivar\n"
             + "   |   ".join(lines), fontsize=11)
fig.tight_layout(); fig.savefig(run / dr / "density_npe_vs_map_warm_vs_random.png", dpi=110)
print("\n".join(lines))
ch_w, ch_r = W["chi2pix_map"], R["chi2pix_map"]
print(f"random start reaches lower chi2 than warm start for {np.mean(ch_r < ch_w - 1e-3):.1%} of stars, "
      f"higher for {np.mean(ch_r > ch_w + 1e-3):.1%}")
print("wrote", run / dr / "density_npe_vs_map_warm_vs_random.png")
