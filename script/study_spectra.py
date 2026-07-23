"""Study how the simulated spectra in the Korg grid are distributed.

Data file: /export/data/vgiusepp/BOSS_emulator/data/korg_grid.h5  (~243 GB)

Contents (per grid point, N = 2,316,600):

  Stellar labels (same as korg_grid_params.h5):
    Teff, logg, m_H, alpha_m, carbon_m, vmic
  Chemistry:
    abundances  float64  (N, 92)      ~1.7 GB
  Spectra (the objects studied here):
    abs_flux    float64  (N, 4648)    ~86 GB   absolute flux
    continuum   float64  (N, 4648)    ~86 GB   continuum
    norm_flux   float64  (N, 4648)    ~86 GB   continuum-normalised flux
  Bookkeeping:
    converged_flag  int64  (N,)       1 = converged, 0 = failed

IMPORTANT FINDINGS about the frequency / wavelength grid
--------------------------------------------------------
* This file is SELF-CONTAINED: it carries `converged_flag` and all six
  stellar labels, so the separate korg_grid_params.h5 is not required.
* There is NO wavelength (or frequency) array stored anywhere -- not in
  this HDF5 file and not in the raw `spectrum_lowres_*.dat` files (whose
  three columns are exactly abs_flux, norm_flux, continuum).
* Every spectrum has the SAME length, 4648 pixels, so all spectra live on
  one common, fixed sampling grid -- regular in *pixel index*. Whether the
  underlying wavelength spacing is linear or log-linear (constant R, as is
  standard for BOSS) cannot be determined from these files; that mapping
  lives in the external BOSS wavelength solution / Korg configuration.
* Consequently the plots below use the pixel index as the x-axis.

VOLUME WARNING
--------------
Each spectral array is ~86 GB; the full file is ~243 GB. Nothing spectral
can be loaded in full. All checks here read a random SAMPLE of rows via
h5py fancy indexing (sorted indices), which touches only those rows on disk.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("/export/data/vgiusepp/BOSS_emulator/data/korg_grid.h5")

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "spectra_exploration"

# The three spectral representations stored per grid point.
SPECTRA = ["norm_flux", "abs_flux", "continuum"]
LABELS = ["Teff", "logg", "m_H", "alpha_m", "carbon_m", "vmic"]

N_PIX = 4648  # pixels per spectrum (fixed grid)

# Fixed seed so repeated runs sample the same rows (reproducible plots).
RNG_SEED = 0


def report_volume(path=DATA_PATH):
    """Summarise the file contents and warn about the volume, so it is
    clear that only sampled reads are feasible.
    """
    file_gb = path.stat().st_size / 1e9
    print(f"File: {path.name}")
    print(f"On-disk size: {file_gb:,.1f} GB")

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        n = f["converged_flag"].shape[0]
        print(f"Grid points (N): {n:,}\n")
        print("Datasets:")
        for key in keys:
            dset = f[key]
            gb = dset.size * dset.dtype.itemsize / 1e9
            print(f"  {key:16s} shape={str(dset.shape):20s} "
                  f"{str(dset.dtype):8s} {gb:8.2f} GB")

        cf = f["converged_flag"][:]

    has_flag = "converged_flag" in keys
    has_labels = all(name in keys for name in LABELS)
    print(f"\nconverged_flag present here? {has_flag}")
    print(f"all six stellar labels present here? {has_labels}")
    print("  -> this file is self-contained; korg_grid_params.h5 not needed.")

    print(f"\nConvergence: {int((cf == 1).sum()):,} converged | "
          f"{int((cf == 0).sum()):,} failed")

    wl_keys = [k for k in keys
               if any(s in k.lower() for s in
                      ("wave", "lam", "freq", "lambda", "wl", "disp"))]
    print(f"\nWavelength/frequency dataset present? {bool(wl_keys)} "
          f"{wl_keys}")
    print(f"  -> spectra are on a fixed {N_PIX}-pixel grid (pixel-index "
          "axis); actual wavelengths are NOT stored in this file.")

    one_spec_gb = n * N_PIX * 8 / 1e9
    print(f"\nEach spectral array is ~{one_spec_gb:.0f} GB -> cannot be "
          "loaded in full; sampling rows instead.")


def sample_indices(n_sample, path=DATA_PATH, converged_only=True):
    """Return sorted random row indices (converged if requested).

    h5py fancy indexing requires increasing indices, so we sort them; this
    also makes the on-disk reads far more efficient.
    """
    rng = np.random.default_rng(RNG_SEED)
    with h5py.File(path, "r") as f:
        cf = f["converged_flag"][:]
    pool = np.flatnonzero(cf == 1) if converged_only else np.arange(cf.size)
    n_sample = min(n_sample, pool.size)
    idx = rng.choice(pool, size=n_sample, replace=False)
    idx.sort()
    return idx


def load_sample(idx, path=DATA_PATH):
    """Read the sampled rows of the spectra and labels into memory.

    Returns a dict with each spectral array (n_sample, N_PIX) and the
    stellar labels for the same rows.
    """
    out = {}
    with h5py.File(path, "r") as f:
        for name in SPECTRA:
            out[name] = f[name][idx, :]
        for name in LABELS:
            out[name] = f[name][idx]
    return out


def _clean(arr):
    """Replace non-finite values with NaN so nan-aware stats ignore them."""
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def plot_sample_spectra(data, n_show=12, outdir=PLOTS_DIR):
    """Overplot a handful of individual normalised spectra to show the
    qualitative shape and the common pixel grid.
    """
    nf = _clean(data["norm_flux"])
    n_show = min(n_show, nf.shape[0])
    pix = np.arange(nf.shape[1])
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_show):
        ax.plot(pix, nf[i], lw=0.5, alpha=0.7)
    ax.set_xlabel("pixel index")
    ax.set_ylabel("normalised flux")
    ax.set_title(f"{n_show} sample normalised spectra (converged)")
    fig.tight_layout()
    out = outdir / "sample_norm_spectra.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_envelope(data, outdir=PLOTS_DIR):
    """Per-pixel median and percentile envelope of the normalised flux
    across the sample, i.e. how the distribution of spectra looks pixel
    by pixel.
    """
    nf = _clean(data["norm_flux"])
    pix = np.arange(nf.shape[1])
    p01, p16, p50, p84, p99 = np.nanpercentile(nf, [1, 16, 50, 84, 99], axis=0)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(pix, p01, p99, color="#4C72B0", alpha=0.2,
                    label="1-99 pct")
    ax.fill_between(pix, p16, p84, color="#4C72B0", alpha=0.4,
                    label="16-84 pct")
    ax.plot(pix, p50, color="#C44E52", lw=0.8, label="median")
    ax.set_xlabel("pixel index")
    ax.set_ylabel("normalised flux")
    ax.set_title(f"Normalised-flux distribution per pixel "
                 f"(N_sample={nf.shape[0]:,})")
    ax.legend()
    fig.tight_layout()
    out = outdir / "norm_flux_envelope.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_pixel_stats(data, outdir=PLOTS_DIR):
    """Per-pixel mean and standard deviation of the normalised flux, plus
    the fraction of non-finite (NaN/inf) values per pixel -- reveals where
    spectral variance concentrates and any bad-pixel structure.
    """
    raw = np.asarray(data["norm_flux"], dtype=float)
    bad_frac = np.mean(~np.isfinite(raw), axis=0)
    nf = _clean(raw)
    pix = np.arange(nf.shape[1])
    mean = np.nanmean(nf, axis=0)
    std = np.nanstd(nf, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(pix, mean, color="#4C72B0", lw=0.7)
    axes[0].set_ylabel("mean norm flux")
    axes[1].plot(pix, std, color="#55A868", lw=0.7)
    axes[1].set_ylabel("std norm flux")
    axes[2].plot(pix, bad_frac, color="#C44E52", lw=0.7)
    axes[2].set_ylabel("non-finite frac")
    axes[2].set_xlabel("pixel index")
    axes[0].set_title("Per-pixel statistics of normalised flux")
    fig.tight_layout()
    out = outdir / "pixel_statistics.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_value_distributions(data, outdir=PLOTS_DIR):
    """Histograms of the pixel values for each spectral representation,
    pooled over the sample -- shows the overall dynamic range and where
    the bulk of flux values sit.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, SPECTRA):
        vals = _clean(data[name]).ravel()
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=200, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("pixel count")
    fig.suptitle("Pooled pixel-value distributions (sample)")
    fig.tight_layout()
    out = outdir / "value_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_median_by_teff(data, n_bins=6, outdir=PLOTS_DIR):
    """Median normalised spectrum in a few Teff bins, to show how the
    spectral morphology changes with effective temperature.
    """
    nf = _clean(data["norm_flux"])
    teff = np.asarray(data["Teff"], dtype=float)
    pix = np.arange(nf.shape[1])
    edges = np.linspace(teff.min(), teff.max(), n_bins + 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap("viridis")
    for b in range(n_bins):
        sel = (teff >= edges[b]) & (teff < edges[b + 1])
        if sel.sum() < 5:
            continue
        med = np.nanmedian(nf[sel], axis=0)
        ax.plot(pix, med, lw=0.8, color=cmap(b / (n_bins - 1)),
                label=f"{edges[b]:.0f}-{edges[b+1]:.0f} K "
                      f"(n={int(sel.sum())})")
    ax.set_xlabel("pixel index")
    ax.set_ylabel("median normalised flux")
    ax.set_title("Median normalised spectrum by Teff bin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = outdir / "median_by_teff.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def analyze_norm_flux_full(path=DATA_PATH, chunk=200_000, outdir=PLOTS_DIR):
    """Full-dataset pass over `norm_flux` (all 2.32M rows) to characterise
    statistics, bad pixels, and the trailing zero-imputation.

    The machine has ~1 TB RAM, so the 86 GB array *could* be loaded whole;
    instead we stream it in row chunks (memory-safe, identical results) and
    accumulate exact per-pixel and per-row summaries.

    Questions answered
    ------------------
    * Overall statistics of the normalised flux (per-pixel mean/std/min/max,
      global percentiles), computed over CONVERGED rows only.
    * Bad pixels: per-pixel fraction of non-finite values and of exact zeros.
    * Trailing zero imputation: for every row we record the trailing run of
      exact zeros, so we can tell whether the same trailing pixels are zeroed
      in *all* simulations (consistent) or only in some.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        cf = f["converged_flag"][:]
        dset = f["norm_flux"]
        N, P = dset.shape
        conv = cf == 1
        n_conv = int(conv.sum())
        print(f"\nFull pass over norm_flux: N={N:,} rows x {P} pix "
              f"({n_conv:,} converged)")

        # per-pixel accumulators (over converged rows only)
        n_finite = np.zeros(P, dtype=np.int64)
        psum = np.zeros(P)
        psumsq = np.zeros(P)
        n_zero = np.zeros(P, dtype=np.int64)         # exact zeros
        n_nonfinite = np.zeros(P, dtype=np.int64)    # NaN/inf
        pmin = np.full(P, np.inf)
        pmax = np.full(P, -np.inf)

        # per-row summaries (all rows; split later with `conv`)
        trailing_zeros = np.zeros(N, dtype=np.int32)  # length of trailing 0-run
        n_zero_row = np.zeros(N, dtype=np.int32)      # total exact zeros in row
        n_nonfinite_row = np.zeros(N, dtype=np.int32)
        all_sentinel = np.zeros(N, dtype=bool)        # entire row == -1

        for start in range(0, N, chunk):
            stop = min(start + chunk, N)
            block = dset[start:stop, :]               # float64 view of chunk
            cblock = conv[start:stop]

            finite = np.isfinite(block)
            iszero = block == 0.0

            # ---- per-row (all rows) ----
            n_zero_row[start:stop] = iszero.sum(axis=1)
            n_nonfinite_row[start:stop] = (~finite).sum(axis=1)
            all_sentinel[start:stop] = np.all(block == -1.0, axis=1)
            nz = block != 0.0
            any_nz = nz.any(axis=1)
            # index of last non-zero pixel (-1 if the whole row is zero)
            last_nz = np.where(
                any_nz, (P - 1) - np.argmax(nz[:, ::-1], axis=1), -1)
            trailing_zeros[start:stop] = np.where(any_nz, (P - 1) - last_nz, P)

            # ---- per-pixel (converged rows only) ----
            if cblock.any():
                b = block[cblock]
                bf = np.isfinite(b)
                bsafe = np.where(bf, b, 0.0)
                n_finite += bf.sum(axis=0)
                psum += bsafe.sum(axis=0)
                psumsq += (bsafe * bsafe).sum(axis=0)
                n_zero += (b == 0.0).sum(axis=0)
                n_nonfinite += (~bf).sum(axis=0)
                bmin = np.where(bf, b, np.inf).min(axis=0)
                bmax = np.where(bf, b, -np.inf).max(axis=0)
                pmin = np.minimum(pmin, bmin)
                pmax = np.maximum(pmax, bmax)

            print(f"  processed rows {stop:,}/{N:,}", end="\r", flush=True)
    print()

    # finalise per-pixel stats
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = psum / n_finite
        var = psumsq / n_finite - mean * mean
        std = np.sqrt(np.clip(var, 0, None))
    frac_nonfinite = n_nonfinite / max(n_conv, 1)
    frac_zero = n_zero / max(n_conv, 1)

    # ---- report ----
    print("\n=== norm_flux full-dataset report (converged rows) ===")
    print(f"converged rows analysed: {n_conv:,}")
    print(f"per-pixel mean:   min={np.nanmin(mean):.4g}  "
          f"max={np.nanmax(mean):.4g}")
    print(f"per-pixel std:    min={np.nanmin(std):.4g}  "
          f"max={np.nanmax(std):.4g}")
    print(f"global finite min/max: {np.nanmin(pmin):.4g} / "
          f"{np.nanmax(pmax):.4g}")

    # bad pixels: never-finite, or always-zero across all converged rows
    never_finite = np.flatnonzero(n_finite == 0)
    always_zero = np.flatnonzero(n_zero == n_conv)
    print(f"\nBad pixels (over converged rows):")
    print(f"  pixels non-finite in >0% of rows: "
          f"{int((frac_nonfinite > 0).sum())}")
    print(f"  pixels non-finite in >50% of rows: "
          f"{int((frac_nonfinite > 0.5).sum())}")
    print(f"  pixels never finite anywhere:     {never_finite.size} "
          f"{never_finite[:10].tolist()}")
    print(f"  pixels zero in EVERY converged row: {always_zero.size}")
    if always_zero.size:
        print(f"    range: [{always_zero.min()}, {always_zero.max()}] "
              f"contiguous={np.array_equal(always_zero, np.arange(always_zero.min(), always_zero.max()+1))}")

    # trailing-zero imputation consistency (converged rows only)
    tz = trailing_zeros[conv]
    uniq, counts = np.unique(tz, return_counts=True)
    order = np.argsort(counts)[::-1]
    print(f"\nTrailing zero-run length (converged rows):")
    print(f"  min={tz.min()}  max={tz.max()}  "
          f"median={int(np.median(tz))}  mean={tz.mean():.2f}")
    print(f"  most common trailing-zero lengths (length: n_rows):")
    for k in order[:8]:
        print(f"    {int(uniq[k]):5d} : {int(counts[k]):,}")
    consistent = uniq.size == 1
    print(f"  => trailing zero imputation identical for all converged rows? "
          f"{consistent}")

    # sentinel / failed rows
    print(f"\nSentinel (all -1) rows: {int(all_sentinel.sum()):,} "
          f"(of these, converged_flag==1: {int((all_sentinel & conv).sum()):,})")

    # ---- plots ----
    pix = np.arange(P)
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    axes[0].plot(pix, mean, color="#4C72B0", lw=0.7)
    axes[0].set_ylabel("mean")
    axes[1].plot(pix, std, color="#55A868", lw=0.7)
    axes[1].set_ylabel("std")
    axes[2].plot(pix, frac_zero, color="#C44E52", lw=0.7)
    axes[2].set_ylabel("frac exactly 0")
    axes[3].plot(pix, frac_nonfinite, color="#8172B3", lw=0.7)
    axes[3].set_ylabel("frac non-finite")
    axes[3].set_xlabel("pixel index")
    axes[0].set_title(f"norm_flux full-dataset per-pixel stats "
                      f"({n_conv:,} converged rows)")
    fig.tight_layout()
    out = outdir / "norm_flux_full_pixel_stats.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(uniq, counts, width=0.9, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xlabel("trailing zero-run length (pixels)")
    ax.set_ylabel("number of converged rows")
    ax.set_title("Consistency of trailing zero-imputation across simulations")
    fig.tight_layout()
    out = outdir / "trailing_zero_histogram.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def make_plots(n_sample=int(1e4)):
    """Load a random sample of converged spectra and write all plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSampling {n_sample:,} converged rows ...")
    idx = sample_indices(n_sample, converged_only=True)
    print(f"  reading {idx.size:,} spectra from disk (this touches only "
          "those rows) ...")
    data = load_sample(idx)
    print(f"  loaded sample: norm_flux shape {data['norm_flux'].shape}")

    print(f"\nWriting plots to {PLOTS_DIR} ...")
    plot_sample_spectra(data)
    plot_envelope(data)
    plot_pixel_stats(data)
    plot_value_distributions(data)
    plot_median_by_teff(data)


def main():
    report_volume()
    make_plots()


if __name__ == "__main__":
    main()
