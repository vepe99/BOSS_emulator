"""Load the BOSS/Korg grid prior parameters.

The HDF5 file stores, per grid point (N = 2,316,600):

  Scalar "prior" parameters (stellar labels):
    Teff      float32   effective temperature [K]      (2800 - 8000)
    logg      float32   surface gravity                (-0.5 - 5.5)
    m_H       float32   metallicity [M/H]              (-2.5 - 1.0)
    alpha_m   float32   [alpha/M]                      (-1.0 - 1.0)
    carbon_m  float32   [C/M]                          (-1.5 - 1.0)
    vmic      float64   microturbulence [km/s]         (0.316 - 5.0)

  Bookkeeping:
    converged_flag  int64   1 = Korg synthesis converged, 0 = failed

  Large derived array:
    abundances  float64  shape (N, 92)  ~1.7 GB on disk
      One column per element, ordered by atomic number Z = 1..92
      (H through U). Each value is A(X), the log abundance on the
      standard scale where A(H) = 12. It is streamed in row-chunks (never
      held whole) to build per-element histograms — see
      `load_abundance_hist` and the `plot_abundance_*` functions.

Total file size ~1.79 GB, almost entirely the `abundances` array.
The six prior parameters together are only ~55 MB in memory, so they
load comfortably for a corner plot.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # headless: write files, no display needed
import matplotlib.pyplot as plt
import numpy as np

# Path to the grid parameter file.
DATA_PATH = Path(
    "/export/data/vgiusepp/BOSS_emulator/data/data.sdss5.org/sas/sdsswork/mwm/"
    "sandbox/boss_korg_grid/BOSS-Grid/korg_grid_params.h5"
)

# Where to write the exploration plots (relative to the project root,
# i.e. the parent of this script's directory).
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "dataset_exploration"

# Sub-directory for the converged-only versions of the plots.
PLOTS_DIR_CONVERGED = PLOTS_DIR / "converged_only"

# The scalar prior parameters we want to visualise in a corner plot.
PRIOR_PARAMS = ["Teff", "logg", "m_H", "alpha_m", "carbon_m", "vmic"]

# Human-readable axis labels.
PARAM_LABELS = {
    "Teff": r"$T_{\rm eff}$ [K]",
    "logg": r"$\log g$",
    "m_H": "[M/H]",
    "alpha_m": r"[$\alpha$/M]",
    "carbon_m": "[C/M]",
    "vmic": r"$v_{\rm mic}$ [km/s]",
}

# The `abundances` array is (N, 92): one column per element, ordered by
# atomic number Z = 1..92 (H through U). Each value is A(X), the log
# abundance on the standard scale where A(H) = 12. Columns 0 (H) and 1
# (He) are held fixed; the rest vary across the grid with [M/H], alpha
# and carbon.
ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
]

# A handful of astrophysically interesting elements to show as detailed
# 1D marginals (in addition to the all-element overview).
KEY_ELEMENTS = ["C", "N", "O", "Mg", "Si", "Ca", "Ti", "Fe", "Ni", "Ba"]

# Common value grid for the per-element abundance histograms. The grid
# spans the observed A(X) range with a little headroom on each side.
ABUND_VALUE_RANGE = (-8.0, 12.5)


def load_prior_params(path=DATA_PATH, converged_only=True):
    """Load the scalar prior parameters into a dict of 1D numpy arrays.

    Parameters
    ----------
    path : Path or str
        Location of the HDF5 grid file.
    converged_only : bool
        If True, keep only grid points with converged_flag == 1.

    Returns
    -------
    params : dict[str, np.ndarray]
        Maps each name in PRIOR_PARAMS to its 1D array.
    converged_flag : np.ndarray
        The (possibly filtered) convergence flag.
    """
    with h5py.File(path, "r") as f:
        params = {name: f[name][:] for name in PRIOR_PARAMS}
        converged_flag = f["converged_flag"][:]

    if converged_only:
        mask = converged_flag == 1
        params = {name: arr[mask] for name, arr in params.items()}
        converged_flag = converged_flag[mask]

    return params, converged_flag


def load_abundance_hist(path=DATA_PATH, converged_only=True,
                        value_bins=205, chunk=200_000):
    """Stream the (N, 92) abundances and accumulate, per element, a 1D
    histogram over a common value grid.

    The full array is ~1.7 GB, so rather than hold it in memory we read
    it in row-chunks and bin each chunk into a shared (92, value_bins)
    count array. Running min/max/sum/count per element are tracked too.

    Parameters
    ----------
    path : Path or str
        Location of the HDF5 grid file.
    converged_only : bool
        If True, only accumulate grid points with converged_flag == 1.
    value_bins : int
        Number of bins spanning ABUND_VALUE_RANGE.
    chunk : int
        Number of rows read per iteration.

    Returns
    -------
    edges : np.ndarray
        Shared bin edges, shape (value_bins + 1,).
    hist : np.ndarray
        Per-element counts, shape (92, value_bins).
    stats : dict[str, np.ndarray]
        "min", "max", "mean" per element, each shape (92,).
    n_points : int
        Number of grid points accumulated.
    """
    edges = np.linspace(*ABUND_VALUE_RANGE, value_bins + 1)

    with h5py.File(path, "r") as f:
        dset = f["abundances"]
        n_total, n_elem = dset.shape
        flag = f["converged_flag"][:] if converged_only else None

        hist = np.zeros((n_elem, value_bins), dtype=np.int64)
        emin = np.full(n_elem, np.inf)
        emax = np.full(n_elem, -np.inf)
        esum = np.zeros(n_elem)
        n_points = 0

        for start in range(0, n_total, chunk):
            stop = min(start + chunk, n_total)
            block = dset[start:stop]  # (rows, 92)
            if flag is not None:
                block = block[flag[start:stop] == 1]
            if block.size == 0:
                continue

            n_points += block.shape[0]
            emin = np.minimum(emin, block.min(axis=0))
            emax = np.maximum(emax, block.max(axis=0))
            esum += block.sum(axis=0)
            # Bin every element's column onto the shared value grid.
            for e in range(n_elem):
                hist[e] += np.histogram(block[:, e], bins=edges)[0]

    stats = {"min": emin, "max": emax, "mean": esum / max(n_points, 1)}
    return edges, hist, stats, n_points


def _hist_percentiles(counts, centers, qs=(0.16, 0.5, 0.84)):
    """Approximate percentiles of a distribution from its binned counts,
    interpolating the cumulative fraction at the bin centres.
    """
    total = counts.sum()
    if total == 0:
        return [np.nan] * len(qs)
    cumfrac = np.cumsum(counts) / total
    return [float(np.interp(q, cumfrac, centers)) for q in qs]


def report_volume(path=DATA_PATH):
    """Print a summary of the file contents and the in-memory footprint
    of the prior parameters, so we know a scatter/corner plot is feasible.
    """
    file_mb = path.stat().st_size / 1e6
    print(f"File: {path.name}")
    print(f"On-disk size: {file_mb:,.1f} MB")

    with h5py.File(path, "r") as f:
        n = f[PRIOR_PARAMS[0]].shape[0]
        print(f"Grid points (N): {n:,}")
        print("\nDatasets:")
        for key in f.keys():
            dset = f[key]
            mb = dset.size * dset.dtype.itemsize / 1e6
            print(f"  {key:16s} shape={str(dset.shape):20s} "
                  f"{str(dset.dtype):8s} {mb:8,.1f} MB")

    # In-memory footprint of just the prior parameters (what a corner
    # plot needs). Each column is N float32/64 -> at most N*8 bytes.
    prior_mb = n * 8 * len(PRIOR_PARAMS) / 1e6
    print(f"\nPrior params in memory (upper bound): ~{prior_mb:,.0f} MB "
          f"({len(PRIOR_PARAMS)} columns x {n:,} rows)")

    # A per-point scatter of 2.3M markers is renderable but slow/heavy;
    # a corner plot with 2D histograms (hist2d / hexbin) is the better
    # choice at this volume.
    print("\nScatter-plot feasibility:")
    print(f"  {n:,} points per panel. Memory is fine, but a raw "
          "point-per-marker scatter is slow to render.")
    print("  -> prefer 2D-histogram / hexbin corner panels at this N.")


def _label(name):
    return PARAM_LABELS.get(name, name)


def plot_corner(params, bins=80, outdir=PLOTS_DIR):
    """Corner plot of the prior parameters.

    Diagonal: 1D marginal histogram of each parameter.
    Lower triangle: 2D density (hist2d, log colour scale) — the right
    tool at N ~ 2.3M points, where a per-point scatter would be an
    unreadable black blob.
    """
    names = list(params.keys())
    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))

    for i, yname in enumerate(names):
        for j, xname in enumerate(names):
            ax = axes[i, j]
            if j > i:  # upper triangle unused
                ax.axis("off")
                continue

            if i == j:  # diagonal -> 1D marginal
                ax.hist(params[xname], bins=bins, color="#4C72B0")
            else:       # lower triangle -> 2D density
                ax.hist2d(
                    params[xname], params[yname],
                    bins=bins, cmap="viridis",
                    norm=matplotlib.colors.LogNorm(),
                )

            # Only label the outer edges to keep the grid clean.
            if i == k - 1:
                ax.set_xlabel(_label(xname))
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(_label(yname))
            elif j != 0:
                ax.set_yticklabels([])

    fig.suptitle("BOSS/Korg grid — prior parameter corner plot "
                 f"(converged points, N={params[names[0]].size:,})", y=0.995)
    fig.tight_layout()
    out = outdir / "corner_prior_params.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_marginals(params_all, mask, bins=80, outdir=PLOTS_DIR):
    """1D marginal histograms of each parameter, overlaying converged vs
    failed points, so we can see whether failures cluster in parameter
    space rather than being uniformly spread.
    """
    names = list(params_all.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.ravel(), names):
        arr = params_all[name]
        edges = np.histogram_bin_edges(arr, bins=bins)
        ax.hist(arr[mask], bins=edges, color="#4C72B0",
                alpha=0.7, label="converged")
        ax.hist(arr[~mask], bins=edges, color="#C44E52",
                alpha=0.7, label="failed")
        ax.set_xlabel(_label(name))
        ax.set_ylabel("count")
        ax.set_yscale("log")
    axes.ravel()[0].legend()
    fig.suptitle("Prior parameter marginals: converged vs failed")
    fig.tight_layout()
    out = outdir / "marginals_converged_vs_failed.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_marginals_single(params, bins=80, outdir=PLOTS_DIR):
    """1D marginal histograms of each parameter for a single sample
    (used for the converged-only set, where there is nothing to overlay).
    """
    names = list(params.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.ravel(), names):
        ax.hist(params[name], bins=bins, color="#4C72B0")
        ax.set_xlabel(_label(name))
        ax.set_ylabel("count")
        ax.set_yscale("log")
    fig.suptitle("Prior parameter marginals (converged points)")
    fig.tight_layout()
    out = outdir / "marginals.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_kiel(params, outdir=PLOTS_DIR):
    """Kiel diagram (Teff vs logg), the standard stellar-parameter view,
    coloured by mean [M/H] in each 2D bin. Axes are inverted so hot/
    high-gravity stars sit top-left, as is conventional.
    """
    teff, logg, mH = params["Teff"], params["logg"], params["m_H"]
    fig, ax = plt.subplots(figsize=(8, 7))
    # Mean metallicity per bin via weighted / unweighted hist2d ratio.
    counts, xe, ye = np.histogram2d(teff, logg, bins=120)
    wsum, _, _ = np.histogram2d(teff, logg, bins=[xe, ye], weights=mH)
    with np.errstate(invalid="ignore"):
        mean_mH = np.where(counts > 0, wsum / counts, np.nan)
    pcm = ax.pcolormesh(xe, ye, mean_mH.T, cmap="plasma", shading="auto")
    fig.colorbar(pcm, ax=ax, label="mean [M/H]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(_label("Teff"))
    ax.set_ylabel(_label("logg"))
    ax.set_title("Kiel diagram — grid coverage coloured by mean [M/H]")
    fig.tight_layout()
    out = outdir / "kiel_diagram.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_failure_rate(params_all, mask, bins=80, outdir=PLOTS_DIR):
    """Convergence-failure rate across the Teff-logg plane, to reveal
    where Korg synthesis struggles (typically cool / low-gravity edges).
    """
    teff, logg = params_all["Teff"], params_all["logg"]
    total, xe, ye = np.histogram2d(teff, logg, bins=bins)
    failed, _, _ = np.histogram2d(teff[~mask], logg[~mask], bins=[xe, ye])
    with np.errstate(invalid="ignore"):
        rate = np.where(total > 0, failed / total, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    pcm = ax.pcolormesh(xe, ye, rate.T, cmap="inferno",
                        shading="auto", vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax, label="failure fraction")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(_label("Teff"))
    ax.set_ylabel(_label("logg"))
    ax.set_title("Non-convergence rate across Teff-logg")
    fig.tight_layout()
    out = outdir / "failure_rate_teff_logg.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_abundance_hist2d(edges, hist, n_points, outdir=PLOTS_DIR):
    """All 92 elements at once: a 2D density of A(X) vs atomic number.

    Each vertical strip is one element's marginal abundance distribution
    over the grid, on a shared value axis and a log colour scale. Reveals
    at a glance which elements are fixed (H, He), which span a wide range,
    and where the bulk of each element's abundances sits.
    """
    n_elem = hist.shape[0]
    xedges = np.arange(0.5, n_elem + 1.5)  # one column per element
    fig, ax = plt.subplots(figsize=(20, 7))
    pcm = ax.pcolormesh(
        xedges, edges, hist.T, cmap="viridis",
        norm=matplotlib.colors.LogNorm(),
    )
    fig.colorbar(pcm, ax=ax, label="count")
    ax.set_xticks(np.arange(1, n_elem + 1))
    ax.set_xticklabels(ELEMENT_SYMBOLS, rotation=90, fontsize=6)
    ax.set_xlim(0.5, n_elem + 0.5)
    ax.set_xlabel("element (by atomic number Z)")
    ax.set_ylabel(r"$A(X) = \log_{10}(N_X/N_{\rm H}) + 12$")
    ax.set_title(f"Abundance distribution across all elements (N={n_points:,})")
    fig.tight_layout()
    out = outdir / "abundances_hist2d.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_abundance_summary(edges, hist, stats, outdir=PLOTS_DIR):
    """Per-element spread as a percentile band vs atomic number.

    For every element we draw the min-max range (whiskers), the 16-84th
    percentile band, and the median, derived from the accumulated
    histogram. Compact way to see the width of each element's grid
    distribution and how it varies along the periodic table.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_elem = hist.shape[0]
    Z = np.arange(1, n_elem + 1)

    p16 = np.empty(n_elem)
    p50 = np.empty(n_elem)
    p84 = np.empty(n_elem)
    for e in range(n_elem):
        p16[e], p50[e], p84[e] = _hist_percentiles(hist[e], centers)

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.vlines(Z, stats["min"], stats["max"], color="#B0B0B0",
              lw=1, label="min-max")
    ax.fill_between(Z, p16, p84, color="#4C72B0", alpha=0.4,
                    step="mid", label="16-84th pct")
    ax.plot(Z, p50, ".", color="#C44E52", ms=5, label="median")
    ax.set_xticks(Z)
    ax.set_xticklabels(ELEMENT_SYMBOLS, rotation=90, fontsize=6)
    ax.set_xlim(0.5, n_elem + 0.5)
    ax.set_xlabel("element (by atomic number Z)")
    ax.set_ylabel(r"$A(X)$")
    ax.set_title("Per-element abundance spread across the grid")
    ax.legend()
    fig.tight_layout()
    out = outdir / "abundances_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_abundance_marginals(edges, hist, outdir=PLOTS_DIR):
    """1D marginal histograms for a selection of key elements, drawn
    directly from the accumulated per-element counts.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, sym in zip(axes.ravel(), KEY_ELEMENTS):
        e = ELEMENT_SYMBOLS.index(sym)
        ax.bar(centers, hist[e], width=width, color="#4C72B0", align="center")
        ax.set_xlabel(f"A({sym})")
        ax.set_ylabel("count")
        ax.set_yscale("log")
        # Zoom onto the populated part of the value axis.
        nz = np.nonzero(hist[e])[0]
        if nz.size:
            lo = centers[nz[0]] - 2 * width
            hi = centers[nz[-1]] + 2 * width
            ax.set_xlim(lo, hi)
    fig.suptitle("Abundance marginals for selected elements")
    fig.tight_layout()
    out = outdir / "abundances_marginals.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def make_plots():
    """Load the grid and write all exploration plots to PLOTS_DIR."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load everything once (all points + flag), then derive the mask.
    params_all, flag = load_prior_params(converged_only=False)
    mask = flag == 1
    params_conv = {name: arr[mask] for name, arr in params_all.items()}

    print(f"\nWriting plots to {PLOTS_DIR} ...")
    plot_corner(params_conv)
    plot_marginals(params_all, mask)
    plot_kiel(params_conv)
    plot_failure_rate(params_all, mask)

    print("  streaming abundances (converged points) ...")
    edges, hist, stats, n_points = load_abundance_hist(converged_only=True)
    plot_abundance_hist2d(edges, hist, n_points)
    plot_abundance_summary(edges, hist, stats)
    plot_abundance_marginals(edges, hist)


def make_plots_converged():
    """Write the exploration plots using only converged grid points.

    The failure-rate map and the converged-vs-failed marginal overlay are
    dropped here (there are no failed points to compare against); in their
    place we write a plain single-sample marginals figure. Corner and Kiel
    are the same plots as in `make_plots`, restricted to converged points.
    """
    PLOTS_DIR_CONVERGED.mkdir(parents=True, exist_ok=True)

    params_conv, _ = load_prior_params(converged_only=True)

    print(f"\nWriting converged-only plots to {PLOTS_DIR_CONVERGED} ...")
    plot_corner(params_conv, outdir=PLOTS_DIR_CONVERGED)
    plot_marginals_single(params_conv, outdir=PLOTS_DIR_CONVERGED)
    plot_kiel(params_conv, outdir=PLOTS_DIR_CONVERGED)

    print("  streaming abundances (converged points) ...")
    edges, hist, stats, n_points = load_abundance_hist(converged_only=True)
    plot_abundance_hist2d(edges, hist, n_points, outdir=PLOTS_DIR_CONVERGED)
    plot_abundance_summary(edges, hist, stats, outdir=PLOTS_DIR_CONVERGED)
    plot_abundance_marginals(edges, hist, outdir=PLOTS_DIR_CONVERGED)


def main():
    report_volume()

    print("\nLoading prior parameters ...")
    params, converged_flag = load_prior_params()

    print(f"Loaded {len(params)} parameters, "
          f"{converged_flag.size:,} grid points.")
    print(f"Converged: {(converged_flag == 1).sum():,} | "
          f"failed: {(converged_flag == 0).sum():,}")

    print("\nPer-parameter ranges:")
    for name, arr in params.items():
        print(f"  {name:10s} min={arr.min():10.4g}  max={arr.max():10.4g}")

    print("\nStreaming abundances for per-element summary ...")
    edges, hist, stats, n_points = load_abundance_hist(converged_only=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    print(f"Accumulated {n_points:,} converged points over "
          f"{hist.shape[0]} elements.")
    print(f"{'elem':>5s} {'min':>8s} {'p16':>8s} {'median':>8s} "
          f"{'p84':>8s} {'max':>8s}")
    for e, sym in enumerate(ELEMENT_SYMBOLS):
        p16, p50, p84 = _hist_percentiles(hist[e], centers)
        print(f"{sym:>5s} {stats['min'][e]:8.3f} {p16:8.3f} {p50:8.3f} "
              f"{p84:8.3f} {stats['max'][e]:8.3f}")

    make_plots()
    make_plots_converged()


if __name__ == "__main__":
    main()
