"""Distribution of SBI recovery errors on the eval_clam_noMdwarf test set + 3 stars to inspect.

Per-star NPE (amortized) medians come from npe_then_map.py's map_results.npz (same 5,000 boss_clam
test coadds, no M dwarfs, as evaluate.py --boss-source clam --cut-mdwarfs). Error per star:
    e = sqrt(mean_k ((SBI median_k - CLAM_k) / s_k)^2),  k = Teff, logg, m_H, alpha_m,
    s = BOSS-CLAM test scatter (275 K, 0.3, 0.15, 0.11).
Picks the star closest to the 5th / 50th / 95th percentile of e (best 10 %, median, worst 10 %).

    python sc_sbi/pick_recovery_stars.py trained_model/sc_sbi/v4_vedant
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Teff", "logg", "[M/H]", "[α/M]"]
SCALE = np.array([275.0, 0.3, 0.15, 0.11])          # evaluate.py clam_test_scatter
PCTS = [5, 50, 95]

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--results", default="map_clam_noMdwarf/map_results.npz")
p.add_argument("--out", default="recovery_picks")
a = p.parse_args()
run = Path(a.run_dir); out = run / a.out; out.mkdir(exist_ok=True)

z = np.load(run / a.results)
rows, ref, npe = z["rows"], z["ref"].astype(float), z["npe"][:, :4].astype(float)
print("MAE (check vs eval_clam_noMdwarf amortized):", dict(zip(NAMES, np.round(np.nanmean(np.abs(npe - ref), 0), 4))))
ok = np.isfinite(ref).all(1)                         # 9 stars lack CLAM [M/H], [α/M]
rows, ref, npe, sid = rows[ok], ref[ok], npe[ok], z["sdss_id"][ok]
snr = np.load(ROOT / "sc_sbi/cache/boss_clam_prepared.npz", mmap_mode="r")["snr"][rows]
d = (npe - ref) / SCALE
e = np.sqrt(np.mean(d ** 2, 1))

q = np.percentile(e, PCTS)
picks = [int(np.argmin(np.abs(e - v))) for v in q]
info = []
for pc, i in zip(PCTS, picks):
    info.append(dict(percentile=pc, boss_row=int(rows[i]), sdss_id=int(sid[i]), e=float(e[i]),
                     snr=float(snr[i]), clam=dict(zip(NAMES, ref[i].round(3).tolist())),
                     sbi=dict(zip(NAMES, npe[i].round(3).tolist()))))
    print(f"p{pc:02d}: row {rows[i]}, sdss_id {z['sdss_id'][i]}, e {e[i]:.2f}, S/N {snr[i]:.0f}, "
          f"CLAM {np.round(ref[i], 2)}, SBI {np.round(npe[i], 2)}")
for lo_, hi_, nm in [(0, 10, "best 10 %"), (45, 55, "middle 10 %"), (90, 100, "worst 10 %")]:
    m = (e >= np.percentile(e, lo_)) & (e <= np.percentile(e, hi_))
    print(f"{nm}: median |Δ|/s " + ", ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, np.median(np.abs(d[m]), 0)))
          + f"; median S/N {np.median(snr[m]):.0f}, median CLAM logg {np.median(ref[m, 1]):.2f}")
(out / "picks.json").write_text(json.dumps(info, indent=1))

cols = ["#009E73", "#E69F00", "#D55E00"]
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
A = ax[0, 0]
A.hist(e, np.logspace(np.log10(e.min()), np.log10(e.max()), 60), color="0.6")
for pc, v, i, c in zip(PCTS, q, picks, cols):
    A.axvline(v, color=c, lw=1.5, ls="--", label=f"p{pc}: sdss_id {info[PCTS.index(pc)]['sdss_id']}, e={e[i]:.2f}")
A.legend(fontsize=9, loc="upper right")
A.set_xscale("log"); A.set_xlabel("e = RMS of |SBI median − CLAM| / scatter (Teff, logg, [M/H], [α/M])")
A.set_ylabel("stars"); A.set_title(f"error distribution, {e.size} test stars (median e {np.median(e):.2f})")

A = ax[0, 1]
hb = A.hexbin(ref[:, 0], ref[:, 1], C=e, reduce_C_function=np.median, gridsize=45, mincnt=3,
              cmap="viridis", vmax=np.percentile(e, 95))
fig.colorbar(hb, ax=A, label="median e")
A.invert_xaxis(); A.invert_yaxis(); A.set_xlabel("CLAM Teff"); A.set_ylabel("CLAM logg"); A.set_title("Kiel diagram")

for A, x, xl in [(ax[1, 0], ref[:, 2], "CLAM [M/H]"), (ax[1, 1], snr, "S/N")]:
    A.scatter(x, e, s=2, c="0.6", alpha=0.4, rasterized=True)
    A.set_yscale("log"); A.set_xlabel(xl); A.set_ylabel("e")
ax[1, 1].set_xscale("log")
for A, xs, ys in [(ax[0, 1], ref[:, 0], ref[:, 1]), (ax[1, 0], ref[:, 2], e), (ax[1, 1], snr, e)]:
    for pc, i, c in zip(PCTS, picks, cols):
        A.scatter(xs[i], ys[i], marker="*", s=300, c=c, edgecolor="k", zorder=5, label=f"p{pc}")
ax[1, 1].legend()
fig.suptitle("SBI (amortized) vs BOSS-CLAM on the eval_clam_noMdwarf test set")
fig.tight_layout(); fig.savefig(out / "recovery_errors.png", dpi=110)
print("wrote", out)
