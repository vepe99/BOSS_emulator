"""Raw (0.01 A) Korg spectrum -> BOSS pixels with the real BOSS LSF.

BOSS pixel i records  int_{pixel i} (f * G_i)(lam) dlam / width,  G_i = Gaussian with
sigma_i = kappa * ln10 * 1e-4 * WDISP_i * lam_i [A].  Swapping the integrals, the weight
of model point lam_j is the Gaussian integrated over the pixel: a difference of erfs.
So the whole operation is one sparse matrix M (n_pix, n_model): flux_boss = M @ flux.
Build M once per WDISP (one star, or one representative LSF) and apply it to any
number of Korg spectra on the same model grid.
"""
import numpy as np
from scipy import sparse
from scipy.special import erf


def repair_wdisp(wdisp):
    """Interpolate over NaN / >2x-off-median WDISP pixels (reduction glitches)."""
    w = np.asarray(wdisp, float).copy()
    bad = ~np.isfinite(w) | (np.abs(np.log(np.clip(w, 1e-6, None) / np.nanmedian(w))) > np.log(2))
    w[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), w[~bad])
    return w


def lsf_matrix(wl, loglam, wdisp, kappa=1.1, nsig=5.0):
    """wl: model wavelengths [A], sorted, finely sampled (<< sigma). loglam, wdisp: BOSS
    grid and LSF [1e-4 dex px] (same convention: both vacuum). Returns CSR (n_pix, n_wl),
    rows sum to 1, so flux_boss = M @ flux is the pixel-averaged, LSF-smoothed flux."""
    wave = 10 ** loglam
    sig = kappa * np.log(10) * 1e-4 * repair_wdisp(wdisp) * wave
    dlog = np.gradient(loglam)
    lo, hi = 10 ** (loglam - dlog / 2), 10 ** (loglam + dlog / 2)   # pixel edges
    i0 = np.searchsorted(wl, lo - nsig * sig)
    i1 = np.searchsorted(wl, hi + nsig * sig)
    dw = np.gradient(wl)
    rows, cols, vals = [], [], []
    for i in range(len(wave)):
        j = np.arange(i0[i], i1[i])
        s = np.sqrt(2) * sig[i]
        w = (erf((hi[i] - wl[j]) / s) - erf((lo[i] - wl[j]) / s)) * dw[j]
        rows.append(np.full(len(j), i)); cols.append(j); vals.append(w / w.sum())
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(wave), len(wl)))


if __name__ == "__main__":
    # self-check: flat stays flat; an isolated narrow line comes out with
    # variance sigma_lsf^2 + pixel^2/12 and conserved equivalent width
    loglam = 3.5523 + 1e-4 * np.arange(4648)
    wdisp = np.full(4648, 1.2)
    wl = np.arange(3500.0, 10500.0, 0.01)
    M = lsf_matrix(wl, loglam, wdisp)
    assert np.allclose(M @ np.ones_like(wl), 1.0)

    lc, s0 = 6000.0, 0.02
    f = 1 - 0.5 * np.exp(-0.5 * ((wl - lc) / s0) ** 2)
    fb, wave = M @ f, 10 ** loglam
    a = 1 - fb
    dpix = wave * np.log(10) * 1e-4
    ew_in = 0.5 * s0 * np.sqrt(2 * np.pi)
    assert abs((a * dpix).sum() / ew_in - 1) < 0.01, "EW"
    var = (a * (wave - lc) ** 2).sum() / a.sum()
    k = np.argmin(abs(wave - lc))
    expect = (1.1 * np.log(10) * 1e-4 * 1.2 * lc) ** 2 + dpix[k] ** 2 / 12 + s0 ** 2
    assert abs(np.sqrt(var / expect) - 1) < 0.02, "LSF width"
    print("korg_to_boss checks OK", M.shape, M.nnz)
