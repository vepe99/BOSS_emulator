import os
import numpy as np
import warnings
from tqdm import tqdm, trange
import logging


def load_data(file_path, convert_alpha=False):
    """Load and preprocess the training data."""
    data = np.load(file_path)['lux_data']

    # Normalize flux
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        norm_flux = data['flux'] / data['continuum']

    # Compute inverse variance for normalized flux
    norm_ivar = data['continuum']**2 * data['ivar']

    # Handle non-finite values
    bad_pixels = (
        ~np.isfinite(norm_flux)
    |   ~np.isfinite(norm_ivar)
    |   (norm_ivar <= 0)
    |   (norm_flux <= 0)
    |   (norm_flux >= 1.2)
    )

    norm_flux = np.where(bad_pixels, 1.0, norm_flux)
    norm_ivar = np.where(bad_pixels, 0.0, norm_ivar)

    # Compute absorption: A = 1 - flux, clipped to be non-negative
    absorption = np.clip(1.0 - norm_flux, 0.0, np.inf)

    if convert_alpha:
        # Extract stellar labels: teff, logg, m_h, alpha_h
        labels = np.column_stack([
            data['teff'],
            data['logg'],
            data['fe_h'],
            data['raw_alpha_m_atm'] + data['fe_h']  # [alpha/H] = [alpha/M] + [M/H]
        ])
    else:
        # Extract stellar labels: teff, logg, m_h, alpha_h
        labels = np.column_stack([
            data['teff'],
            data['logg'],
            data['fe_h'],
            data['raw_alpha_m_atm']
        ])
    
    meta_data = np.column_stack([
            data['bp_mag'],
            data['rp_mag'],
            data['sdss_id'],
            data['telescope'],
            data['snr'],
            data['data_type']
    ])

    return absorption, norm_flux, norm_ivar, labels, meta_data

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_NAMES = [r"$T_\mathrm{eff}$ [K]", r"$\log g$", r"[Fe/H]", r"[$\alpha$/M]"]
META_NAMES = ["bp_mag", "rp_mag", "sdss_id", "telescope", "snr", "data_type"]


def _teff_ticks(ax):
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([3000, 4000, 6000, 10000, 20000, 40000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:g}k"))


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("saved %s", path)


def plot_counts(meta, out_dir):
    """Number of spectra per data_type, split by telescope."""
    dtype, tel = meta[:, 5], meta[:, 3]
    types, tels = np.unique(dtype), np.unique(tel)
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(types))
    for t in tels:
        n = np.array([np.sum((dtype == ty) & (tel == t)) for ty in types])
        ax.bar(types, n, bottom=bottom, label=t)
        bottom += n
    for i, b in enumerate(bottom):
        ax.text(i, b, f"{int(b)}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("N spectra")
    ax.set_title(f"Training set composition (N = {len(dtype)})")
    ax.legend(title="telescope")
    _save(fig, out_dir, "counts_per_data_type.png")


def _names(alpha_h):
    return LABEL_NAMES[:3] + ([r"[$\alpha$/H]"] if alpha_h else [LABEL_NAMES[3]])


def plot_label_histograms(labels, meta, out_dir, alpha_h=False):
    """1D label distributions: all data and split by data_type."""
    dtype = meta[:, 5]
    types = np.unique(dtype)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, ax in enumerate(axes.ravel()):
        x = labels[:, i]
        finite = np.isfinite(x)
        if i == 0:  # Teff spans ~3e3 - 5e4 K: log bins
            bins = np.logspace(np.log10(x[finite].min()), np.log10(x[finite].max()), 60)
            ax.set_xscale("log")
            _teff_ticks(ax)
        else:
            bins = np.linspace(x[finite].min(), x[finite].max(), 60)
        ax.hist(x[finite], bins=bins, color="0.8", label="all")
        for ty in types:
            m = finite & (dtype == ty)
            ax.hist(x[m], bins=bins, histtype="step", lw=1.5, label=ty)
        ax.set_xlabel(_names(alpha_h)[i])
        ax.set_yscale("log")
        ax.set_ylabel("N")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Label distributions")
    _save(fig, out_dir, "label_histograms" + ("_alpha_h" if alpha_h else "") + ".png")


def plot_label_corner(labels, out_dir, alpha_h=False):
    """Pairwise 2D density of the four labels (all data)."""
    finite = np.all(np.isfinite(labels), axis=1)
    L = labels[finite].copy()
    L[:, 0] = np.log10(L[:, 0])
    names = [r"$\log_{10} T_\mathrm{eff}$"] + _names(alpha_h)[1:]
    n = L.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(L[:, i], bins=60, color="C0")
                ax.set_yticks([])
            else:
                ax.hexbin(L[:, j], L[:, i], gridsize=60, bins="log", cmap="viridis", mincnt=1)
            if i == n - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            elif j > 0:
                ax.set_yticklabels([])
    fig.suptitle("Label correlations (log density)")
    _save(fig, out_dir, "label_corner" + ("_alpha_h" if alpha_h else "") + ".png")


def plot_kiel(labels, meta, out_dir):
    """Kiel diagram coloured by [Fe/H], one panel per data_type."""
    dtype = meta[:, 5]
    types = np.unique(dtype)
    fig, axes = plt.subplots(1, len(types) + 1, figsize=(4 * (len(types) + 1), 4.5),
                             sharex=True, sharey=True)
    groups = [("all", np.ones(len(dtype), bool))] + [(t, dtype == t) for t in types]
    for ax, (name, m) in zip(axes, groups):
        nofe = m & ~np.isfinite(labels[:, 2])
        ax.scatter(labels[nofe, 0], labels[nofe, 1], s=2, color="0.5", rasterized=True,
                   label=f"no [Fe/H] ({nofe.sum()})")
        if nofe.any():
            ax.legend(fontsize=7, loc="upper left", markerscale=4)
        sc = ax.scatter(labels[m, 0], labels[m, 1], c=labels[m, 2], s=1,
                        cmap="plasma", vmin=-2.5, vmax=0.5, rasterized=True)
        ax.set_title(f"{name} (N={m.sum()})")
        ax.set_xlabel(LABEL_NAMES[0])
    axes[0].set_xscale("log")
    _teff_ticks(axes[0])
    axes[0].invert_xaxis()
    axes[0].invert_yaxis()
    axes[0].set_ylabel(LABEL_NAMES[1])
    fig.colorbar(sc, ax=axes, label="[Fe/H]", fraction=0.02)
    _save(fig, out_dir, "kiel_by_data_type.png")


def plot_snr_mag(meta, out_dir):
    """SNR and magnitude distributions per data_type."""
    dtype = meta[:, 5]
    snr = meta[:, 4].astype(float)
    rp = meta[:, 1].astype(float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    sbins = np.logspace(np.log10(np.nanmin(snr)), np.log10(np.nanmax(snr)), 50)
    rbins = np.linspace(np.nanmin(rp), np.nanmax(rp), 50)
    for ty in np.unique(dtype):
        m = dtype == ty
        a1.hist(snr[m], bins=sbins, histtype="step", lw=1.5, label=ty)
        a2.hist(rp[m], bins=rbins, histtype="step", lw=1.5, label=ty)
    a1.set_xscale("log"); a1.set_xlabel("SNR"); a1.set_ylabel("N")
    a2.set_xlabel("RP mag")
    a1.legend(fontsize=8)
    _save(fig, out_dir, "snr_mag_by_data_type.png")


def plot_example_spectra(norm_flux, norm_ivar, labels, meta, out_dir, n_per_type=4, seed=0):
    """A few random spectra from each data_type (pixel index: no wavelength grid is stored)."""
    rng = np.random.default_rng(seed)
    dtype = meta[:, 5]
    pix = np.arange(norm_flux.shape[1])
    for ty in np.unique(dtype):
        idx = np.flatnonzero(dtype == ty)
        pick = rng.choice(idx, size=min(n_per_type, len(idx)), replace=False)
        fig, axes = plt.subplots(len(pick), 1, figsize=(14, 2.4 * len(pick)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, pick):
            good = norm_ivar[k] > 0
            f = np.where(good, norm_flux[k], np.nan)
            ax.plot(pix, f, lw=0.5, color="k")
            ax.plot(pix[~good], np.ones((~good).sum()), "|", color="r", ms=4, alpha=0.5)
            t, g, fe, a = labels[k]
            ax.set_title(
                f"sdss_id={meta[k, 2]}  {meta[k, 3]}  SNR={float(meta[k, 4]):.0f}  "
                f"Teff={t:.0f}  logg={g:.2f}  [Fe/H]={fe:.2f}  [a/M]={a:.2f}",
                fontsize=9, loc="left")
            ax.set_ylim(0, 1.25)
            ax.set_ylabel("norm. flux")
        axes[-1].set_xlabel("pixel index (red ticks = masked pixels)")
        fig.suptitle(f"Example spectra: {ty}")
        _save(fig, out_dir, f"spectra_{ty}.png")


def plot_teff_sequence(norm_flux, norm_ivar, labels, out_dir, n=6):
    """Spectra across Teff (log-spaced), offset vertically, at near-solar [Fe/H]."""
    teff = labels[:, 0]
    targets = np.logspace(np.log10(3500), np.log10(30000), n)
    near_solar = np.abs(labels[:, 2]) < 0.3
    fig, ax = plt.subplots(figsize=(14, 1.2 * n + 2))
    for i, T in enumerate(targets):
        cand = np.flatnonzero(near_solar) if near_solar.any() else np.arange(len(teff))
        k = cand[np.argmin(np.abs(np.log(teff[cand] / T)))]
        f = np.where(norm_ivar[k] > 0, norm_flux[k], np.nan)
        ax.plot(f + 0.8 * i, lw=0.5)
        ax.text(norm_flux.shape[1] + 20, 1 + 0.8 * i, f"{teff[k]:.0f} K, logg={labels[k, 1]:.1f}",
                va="center", fontsize=8)
    ax.set_xlabel("pixel index")
    ax.set_ylabel("norm. flux + offset")
    ax.set_title("Spectral sequence in Teff (|[Fe/H]| < 0.3)")
    _save(fig, out_dir, "spectra_teff_sequence.png")


def plot_median_spectra(norm_flux, norm_ivar, meta, out_dir):
    """Median normalised spectrum and masked-pixel fraction per data_type."""
    dtype = meta[:, 5]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1]))
    for ty in np.unique(dtype):
        m = dtype == ty
        f = np.where(norm_ivar[m] > 0, norm_flux[m], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a1.plot(np.nanmedian(f, axis=0), lw=0.6, label=ty)
        a2.plot((norm_ivar[m] <= 0).mean(axis=0), lw=0.6, label=ty)
    a1.set_ylabel("median norm. flux")
    a1.legend(fontsize=8)
    a2.set_ylabel("masked fraction")
    a2.set_xlabel("pixel index")
    fig.suptitle("Median spectrum and masking per data_type")
    _save(fig, out_dir, "median_spectra_by_data_type.png")


def plot_alpha_conventions(labels, meta, out_dir):
    """[alpha/M] vs [Fe/H] and [alpha/H] vs [Fe/H] side by side, per data_type."""
    dtype = meta[:, 5]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ty in np.unique(dtype):
        m = (dtype == ty) & np.isfinite(labels[:, 3])
        if not m.any():
            continue
        axes[0].scatter(labels[m, 2], labels[m, 3], s=1, alpha=0.3, label=ty, rasterized=True)
        axes[1].scatter(labels[m, 2], labels[m, 3] + labels[m, 2], s=1, alpha=0.3, rasterized=True)
    axes[0].set_ylabel(r"[$\alpha$/M]  (convert_alpha=False)")
    axes[1].set_ylabel(r"[$\alpha$/H] = [$\alpha$/M] + [Fe/H]  (convert_alpha=True)")
    for ax in axes:
        ax.set_xlabel("[Fe/H]")
    axes[0].legend(markerscale=10, fontsize=8)
    fig.suptitle("Alpha label convention")
    _save(fig, out_dir, "alpha_conventions.png")


def plot_snr_vs_source(labels, meta, out_dir):
    """SNR per data_type (box) and SNR vs each label coloured by data_type."""
    dtype = meta[:, 5]
    snr = meta[:, 4].astype(float)
    types = np.unique(dtype)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    axes[0].boxplot([snr[dtype == t] for t in types], whis=(5, 95), showfliers=False)
    axes[0].set_xticks(range(1, len(types) + 1), types, rotation=30)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SNR")
    axes[0].set_title("SNR per data_type (5-95%)")
    for i, ax in enumerate(axes[1:]):
        for k, t in enumerate(types):
            m = dtype == t
            # running median in label bins
            x = labels[m, i]
            ok = np.isfinite(x)
            if ok.sum() < 50:
                continue
            edges = np.quantile(x[ok], np.linspace(0, 1, 16))
            idx = np.clip(np.digitize(x[ok], edges) - 1, 0, 14)
            med = [np.median(snr[m][ok][idx == j]) if np.any(idx == j) else np.nan for j in range(15)]
            ax.scatter(x, snr[m], s=0.5, alpha=0.1, color=f"C{k}", rasterized=True)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), med, color=f"C{k}", lw=2, label=t)
        ax.set_xlabel(LABEL_NAMES[i])
        if i == 0:
            ax.set_xscale("log")
            _teff_ticks(ax)
    axes[1].legend(fontsize=8)
    fig.suptitle("SNR vs label source (lines: running median)")
    _save(fig, out_dir, "snr_vs_data_type.png")


def plot_cross_source(labels, meta, norm_flux, norm_ivar, out_dir):
    """Stars (same sdss_id) present in two data_types: label offsets between sources."""
    dtype, ids = meta[:, 5], meta[:, 2]
    _, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    pairs = {}
    for g in np.flatnonzero(cnt > 1):
        idx = np.flatnonzero(inv == g)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ia, ib = idx[a], idx[b]
                if dtype[ia] == dtype[ib]:
                    continue
                if dtype[ia] > dtype[ib]:
                    ia, ib = ib, ia
                pairs.setdefault((dtype[ia], dtype[ib]), []).append((ia, ib))
    pairs = {k: np.array(v) for k, v in pairs.items() if len(v) >= 10}
    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), 4, figsize=(16, 3.8 * len(pairs)), squeeze=False)
    for r, ((ta, tb), v) in enumerate(pairs.items()):
        same_flux = np.mean([np.allclose(norm_flux[i], norm_flux[j]) for i, j in v])
        for i in range(4):
            ax = axes[r, i]
            x, y = labels[v[:, 0], i], labels[v[:, 1], i]
            d = y - x
            ax.scatter(x, d, s=3, alpha=0.5)
            ax.axhline(0, color="k", lw=0.5)
            ax.axhline(np.nanmedian(d), color="r", lw=1)
            ax.set_xlabel(f"{LABEL_NAMES[i]} ({ta})")
            ax.set_title(f"median {np.nanmedian(d):+.3g}, std {np.nanstd(d):.3g}", fontsize=9)
        axes[r, 0].set_ylabel(f"{tb} - {ta}\n(N={len(v)}, identical flux {same_flux:.0%})")
    fig.suptitle("Same star, different label source")
    _save(fig, out_dir, "cross_source_label_offsets.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "boss_clam_training_data.npz")
    out_dir = os.path.join(here, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # convert_alpha=False -> labels are [alpha/M], as stored
    absorption, norm_flux, norm_ivar, labels, meta = load_data(data_path, convert_alpha=False)
    logging.info("loaded %d spectra x %d pixels", *norm_flux.shape)
    for ty, n in zip(*np.unique(meta[:, 5], return_counts=True)):
        logging.info("  %-12s %6d", ty, n)

    plot_counts(meta, out_dir)
    plot_label_histograms(labels, meta, out_dir)
    plot_label_corner(labels, out_dir)
    # same plots with convert_alpha=True ([alpha/H])
    labels_ah = load_data(data_path, convert_alpha=True)[3]
    plot_label_histograms(labels_ah, meta, out_dir, alpha_h=True)
    plot_label_corner(labels_ah, out_dir, alpha_h=True)
    plot_alpha_conventions(labels, meta, out_dir)
    plot_snr_vs_source(labels, meta, out_dir)
    plot_cross_source(labels, meta, norm_flux, norm_ivar, out_dir)
    plot_kiel(labels, meta, out_dir)
    plot_snr_mag(meta, out_dir)
    plot_example_spectra(norm_flux, norm_ivar, labels, meta, out_dir)
    plot_teff_sequence(norm_flux, norm_ivar, labels, out_dir)
    plot_median_spectra(norm_flux, norm_ivar, meta, out_dir)
