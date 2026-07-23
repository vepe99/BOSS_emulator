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

  Large derived array (NOT loaded by default):
    abundances  float64  shape (N, 92)  ~1.7 GB on disk

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

    make_plots()
    make_plots_converged()


if __name__ == "__main__":
    main()
