"""Shared pieces for the self-consistency (SC) SBI experiment.

Labeled data   : Korg spectra from the BOSS-Grid-Combined cache (random,
                 continuous labels; the same cache + split the forward emulator
                 ``combined_best`` was trained on).
Unlabeled data : real BOSS spectra (``boss_clam_training_data.npz``).

The posterior ``q(theta | spectrum, SNR)`` is a bayesflow ``BasicEstimator``
(CouplingFlow + an MLP summary network on the spectrum, SNR as a direct
condition). Training noise is *real BOSS noise*: each simulated spectrum gets
the per-pixel ivar, bad-pixel mask and SNR of a random BOSS train-split
spectrum, so the SNR condition follows the BOSS distribution exactly.

The SC loss (``bayesflow.experimental.self_consistency``) needs analytic
``log p(theta)`` and ``log p(y | theta)``:

* prior      -- uniform in the label box, log-uniform in vmic (how the
                Combined grid was sampled);
* likelihood -- Gaussian, ``-1/2 sum ivar (y - emulator(theta))^2`` using the
                spectrum's own ivar (masked pixels have ivar = 0).

Additive constants cancel in the SC loss (it is a variance over theta at fixed
y), so neither density is normalised.

v2 additions (all opt-in through ``noise_cfg``; an empty config reproduces v1,
and ``sc_sbi/legacy_v1/`` is a frozen copy of the v1 code):

* ``NoiseModel`` -- template noise plus a per-pixel model-mismatch scatter
  ``s_m`` (as in BOSS-CLAM, arXiv 2607.22822, Eq. 6), or *residual
  augmentation* with real residuals ``y_BOSS - emu(theta_fit)`` from
  ``diagnose_mismatch.py``;
* ``snr_mode="measured"``: ``log_snr = log10 mean_good(flux * sqrt(ivar))``
  computed from the arrays themselves (boss_clam ``snr`` is this quantity up
  to ~3 %);
* ``noise_channel``: the summary network also sees the per-pixel effective
  sigma ``sqrt(1/ivar + s_m^2)`` (masked pixels -> 1);
* ``make_likelihood(..., s_m=...)``: Gaussian with ``sigma^2 + s_m^2``;
* ``StabilizedSC``: scale-free SC loss variants (log-variance, normalised
  variance) with symmetric trimming of the log-evidence samples.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

# boss_emulator.training imports autocvd, which claims a free GPU when one
# exists (lab policy); JAX_PLATFORMS=cpu / CUDA_VISIBLE_DEVICES skip it.
from boss_emulator import training as _training  # noqa: F401  (autocvd side effect)
from boss_emulator import data as data_mod
from boss_emulator.emulator import load_checkpoint

import jax
import jax.numpy as jnp
from flax import nnx
import keras
import bayesflow as bf
import bayesflow.experimental.self_consistency as sc

# --- paths ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KORG_CACHE = Path("/export/scratch/lbranca/BOSS_emulator/preprocessed_data_combined_v2/korg_grid_train.h5")
BOSS_NPZ = PROJECT_ROOT / "boss_clam/boss_clam_training_data.npz"
BOSS_PREPARED = PROJECT_ROOT / "sc_sbi/cache/boss_clam_prepared.npz"
EMULATOR_CKPT = PROJECT_ROOT / "trained_model/combined_best/mlp_1790330838/best.pkl"

# --- labels --------------------------------------------------------------

LABELS = list(data_mod.CORE_LABELS)        # Teff, logg, m_H, alpha_m, carbon_m, vmic
LOG_LABELS = ["vmic"]                      # sampled log-uniformly in the grid
N_PIX = data_mod.N_PIX_KEEP                # 4479
# labels that also exist in the BOSS reference catalogue (vmic/carbon do not)
BOSS_LABELS = ["Teff", "logg", "m_H", "alpha_m"]
LSF = "lsf_px"                             # v3: Gaussian LSF sigma [pixels], 7th posterior parameter
BOSS_FIELDS = {"Teff": "teff", "logg": "logg", "m_H": "fe_h", "alpha_m": "raw_alpha_m_atm"}
BOSS_ERR_FIELDS = {"Teff": "e_teff", "logg": "e_logg", "m_H": "e_fe_h",
                   "alpha_m": "raw_e_alpha_m_atm"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================================
# Korg (labeled)
# =========================================================================

def load_korg(cache_path=KORG_CACHE, max_rows=None, seed=0, val_frac=0.1,
              test_frac=0.1, splits=("train", "val", "test")):
    """Labels (N, 6) and flux (N, 4479), float32, with the cache's standard split.

    Uses ``boss_emulator.data.load_dataset`` with the defaults the emulator
    was trained with (seed 0, 10 % val, 10 % test), so the Korg test rows are
    unseen by both the emulator and the posterior. ``max_rows`` subsamples
    (and therefore re-splits) -- only for quick tests.
    Returns ``{split: (labels, flux)}`` plus the raw index arrays.
    """
    ds = data_mod.load_dataset(
        core_labels=LABELS, cache_path=Path(cache_path), val_frac=val_frac,
        test_frac=test_frac, seed=seed, max_rows=max_rows, standardize_y=False)
    idx = {"train": ds.train_idx, "val": ds.val_idx, "test": ds.test_idx}
    out = {s: (ds.X[idx[s]].astype("float32"), ds.Y[idx[s]].astype("float32"))
           for s in splits}
    box = {k: (float(ds.X[:, i].min()), float(ds.X[:, i].max()))
           for i, k in enumerate(LABELS)}
    return out, idx, box


# =========================================================================
# BOSS (unlabeled / realistic test)
# =========================================================================

def _prepare_boss(npz=BOSS_NPZ, out=BOSS_PREPARED, seed=0,
                  fracs=(0.7, 0.1, 0.2)) -> None:
    """One-time: decompress the 10 GB structured npz into compact float32
    arrays (normalised flux/ivar, trimmed to 4479 pixels) and a star-level
    train/val/test split (grouped by ``sdss_id`` so a star never sits in two
    splits). Normalisation follows ``boss_clam/data_exploration.py``."""
    log(f"preparing BOSS cache from {npz} (one-time, a few minutes)")
    d = np.load(npz)["lux_data"]
    n = d.shape[0]
    flux = np.empty((n, N_PIX), "float32")
    ivar = np.empty((n, N_PIX), "float32")
    for s in range(0, n, 5000):
        e = min(s + 5000, n)
        cont = d["continuum"][s:e, :N_PIX]
        with np.errstate(all="ignore"):
            f = d["flux"][s:e, :N_PIX] / cont
            iv = cont ** 2 * d["ivar"][s:e, :N_PIX]
        bad = (~np.isfinite(f) | ~np.isfinite(iv) | (iv <= 0) | (f <= 0) | (f >= 1.2))
        flux[s:e] = np.where(bad, 1.0, f)
        ivar[s:e] = np.where(bad, 0.0, iv)
    labels = np.stack([d[BOSS_FIELDS[k]] for k in BOSS_LABELS], 1).astype("float32")
    e_labels = np.stack([d[BOSS_ERR_FIELDS[k]] for k in BOSS_LABELS], 1).astype("float32")

    ids = d["sdss_id"]
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    n_tr = int(round(fracs[0] * uniq.size))
    n_va = int(round(fracs[1] * uniq.size))
    split_of_star = np.empty(uniq.size, "int8")
    split_of_star[perm[:n_tr]] = 0
    split_of_star[perm[n_tr:n_tr + n_va]] = 1
    split_of_star[perm[n_tr + n_va:]] = 2
    split = split_of_star[np.searchsorted(uniq, ids)]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, flux=flux, ivar=ivar, labels=labels, e_labels=e_labels,
             snr=d["snr"].astype("float32"), sdss_id=ids,
             data_type=d["data_type"], telescope=d["telescope"], split=split,
             label_names=np.array(BOSS_LABELS))
    tmp.rename(out)
    log(f"wrote {out}  ({n} spectra; train/val/test stars "
        f"{n_tr}/{n_va}/{uniq.size - n_tr - n_va})")


def load_boss(path=BOSS_PREPARED) -> dict:
    """Prepared BOSS arrays (built on first use). ``split``: 0 train, 1 val, 2 test."""
    path = Path(path)
    if not path.exists():
        _prepare_boss(out=path)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def label_box(cache=KORG_CACHE) -> dict:
    """Label box of the *full* Korg cache (the emulator's domain)."""
    import h5py
    with h5py.File(cache, "r") as f:
        return {k: (float(f[k][:].min()), float(f[k][:].max())) for k in LABELS}


def boss_in_box(boss: dict, box: dict) -> np.ndarray:
    """Rows usable with this prior: Teff/logg finite, finite labels inside the box,
    positive finite SNR. Rows with NaN [Fe/H] / [alpha/M] are kept (they are
    only skipped when scoring those labels)."""
    L = boss["labels"]
    ok = np.isfinite(L[:, 0]) & np.isfinite(L[:, 1])
    for i, k in enumerate(BOSS_LABELS):
        lo, hi = box[k]
        x = L[:, i]
        ok &= ~np.isfinite(x) | ((x >= lo) & (x <= hi))
    ok &= np.isfinite(boss["snr"]) & (boss["snr"] > 0)
    return ok


# =========================================================================
# Noise templates + datasets
# =========================================================================

class NoiseTemplates:
    """Real BOSS (ivar, mask, SNR) triples used to noise simulated spectra (v1)."""

    def __init__(self, ivar: np.ndarray, snr: np.ndarray):
        self.ivar = ivar
        self.snr = snr
        self.sigma = np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 1e-30)), 0.0).astype("float32")

    def __len__(self):
        return self.snr.size

    def apply(self, flux: np.ndarray, rng: np.random.Generator):
        j = rng.integers(0, len(self), size=flux.shape[0])
        iv = self.ivar[j]
        noisy = flux + rng.standard_normal(flux.shape, dtype="float32") * self.sigma[j]
        noisy = np.where(iv > 0, noisy, 1.0).astype("float32")
        return noisy, iv, np.log10(self.snr[j])[:, None].astype("float32")

    def make_batch(self, flux, labels, rng):
        y, iv, lsnr = self.apply(flux, rng)
        return {"flux": y, "ivar": iv, "log_snr": lsnr}


# ---- v2 noise configuration ---------------------------------------------

NOISE_DEFAULTS = {
    "mode": "template",        # template | sm | residual
    "snr_mode": "catalog",     # catalog | measured
    "noise_channel": False,    # feed per-pixel sigma_eff to the summary net
    "mismatch": None,          # path to diagnose_mismatch.py's mismatch.npz
    "sm_scale": 1.0,           # multiply s_m (sensitivity tests)
    "sm_mask_thresh": None,    # mask pixels with s_m > thresh everywhere
    "likelihood_sm": False,    # SC likelihood uses sigma^2 + s_m^2
    "resid_frac": 0.5,         # residual mode: fraction of rows using a real residual
    # --- v3 forward model (L. Branca's BOSS inversion recipe) ---
    "lsf": False,              # Gaussian LSF conv(emu, lsf_px); lsf_px inferred (7th parameter)
    "lsf_range": [0.3, 4.0],   # lsf_px prior box [pixels]
    "cont_deg": 0,             # SC likelihood: profile a Chebyshev continuum of this degree
    "cont_tilt": 0.0,          # simulations: random multiplicative Chebyshev continuum, coef std
}


def params(ncfg: dict | None) -> list:
    """Posterior parameters: the 6 emulator labels (+ lsf_px with the v3 LSF)."""
    return LABELS + ([LSF] if noise_config(ncfg)["lsf"] else [])


# ---- v3 forward model ------------------------------------------------------

K_HALF = 10                                 # LSF half-window [px] (>= 2.5 sigma_max)


def apply_lsf(flux, sigma):
    """Per-spectrum Gaussian convolution in pixel space (BOSS is uniform in
    log-lambda, so a velocity kernel is a constant-pixel kernel). ``flux``
    (n, P), ``sigma`` in pixels: (n,) one width per spectrum, or (n, P) one width
    per output pixel (e.g. from WDISP, see ``korg_lsf_sigma``). Same kernel as
    script/inv_model/run_observed.py."""
    offs = jnp.arange(-K_HALF, K_HALF + 1, dtype=jnp.float32)
    sig = jnp.maximum(jnp.asarray(sigma, jnp.float32), 0.05)
    sig = sig[:, None] if sig.ndim == 1 else sig                # (n, 1) or (n, P)
    w = jnp.exp(-0.5 * (offs / sig[..., None]) ** 2)            # (n, 1|P, 2K+1)
    w = w / jnp.sum(w, axis=-1, keepdims=True)
    pad = jnp.pad(flux, ((0, 0), (K_HALF, K_HALF)), mode="edge")
    out = jnp.zeros_like(flux)
    for j in range(2 * K_HALF + 1):
        out = out + w[..., j] * pad[:, j:j + flux.shape[1]]
    return out


def korg_lsf_sigma(wdisp, kappa=1.1, r_grid=5000.0, point_sampled=True):
    """Extra Gaussian sigma [px] taking the Korg grid (R=lambda/FWHM=``r_grid``,
    1e-4 dex pixels) to the BOSS LSF given by ``wdisp`` [px] (..., P):
    sigma_k^2 = (kappa*WDISP)^2 - sigma_grid^2 (+ 1/12 px^2 pixel integration if the
    grid was point-sampled onto the pixels rather than flux-rebinned)."""
    s_grid = 1.0 / (r_grid * 2.3548 * np.log(10) * 1e-4)        # 0.369 px at R=5000
    s2 = (kappa * np.asarray(wdisp, np.float64)) ** 2 - s_grid ** 2 + (1 / 12 if point_sampled else 0.0)
    return np.sqrt(np.clip(s2, 0.05 ** 2, None)).astype(np.float32)


def cheb_basis(deg: int, npx: int = None) -> np.ndarray:
    """(npx, deg+1) Chebyshev polynomials on x in [-1, 1] across the pixels."""
    x = np.linspace(-1.0, 1.0, npx or N_PIX)
    return np.polynomial.chebyshev.chebvander(x, deg).astype(np.float32)


def profile_continuum(f, y, W, B):
    """Model ``f`` times the best multiplicative continuum ``B a`` for data ``y``
    with weights ``W``: ``a`` minimises ``sum W (y - f B a)^2`` in closed form."""
    K = B.shape[1]
    BB = (B[:, :, None] * B[:, None, :]).reshape(B.shape[0], K * K)
    M = ((W * f * f) @ BB).reshape(-1, K, K) + 1e-6 * jnp.eye(K)[None]
    a = jnp.linalg.solve(M, ((W * f * y) @ B)[..., None])[..., 0]
    return f * (a @ B.T)


def draw_forward_aug(ncfg: dict, n: int, rng: np.random.Generator):
    """Per-spectrum v3 simulator nuisances: ``lsf`` (n,) ~ U(lsf_range) or None,
    ``coefs`` (n, deg+1) of a multiplicative Chebyshev continuum
    ``1 + sum_j a_j T_j`` with ``a_j ~ N(0, cont_tilt / (j + 1))`` or None."""
    c = noise_config(ncfg)
    lsf = (rng.uniform(*c["lsf_range"], size=n).astype("float32") if c["lsf"] else None)
    coefs = None
    if c["cont_tilt"] > 0:
        deg = max(int(c["cont_deg"]), 1)
        coefs = (rng.standard_normal((n, deg + 1)) * c["cont_tilt"]
                 / np.arange(1, deg + 2)).astype("float32")
    return lsf, coefs


def forward_aug(flux, lsf, coefs, basis):
    """Korg flux -> what BOSS would record before noise: LSF, then continuum tilt."""
    if lsf is not None:
        flux = apply_lsf(flux, lsf)
    if coefs is not None:
        flux = flux * (1.0 + coefs @ basis.T)
    return flux


def noise_config(cfg: dict | None) -> dict:
    """Fill a (possibly missing / v1) noise config with defaults."""
    return NOISE_DEFAULTS | (cfg or {})


class Mismatch:
    """Per-pixel model-mismatch scatter + residual bank from diagnose_mismatch.py."""

    def __init__(self, path, scale: float = 1.0, mask_thresh: float | None = None):
        z = np.load(path)
        self.path = str(path)
        self.s_m = (scale * z["s_m"]).astype("float32")
        self.mask = (np.zeros(N_PIX, bool) if mask_thresh is None
                     else self.s_m > float(mask_thresh))
        self.s_m = np.where(self.mask, 0.0, self.s_m).astype("float32")
        self.bank_rows = z["bank_rows"] if "bank_rows" in z.files else None
        self.bank_resid = z["bank_resid"].astype("float32") if "bank_resid" in z.files else None
        self.bank_teff = z["bank_teff"] if "bank_teff" in z.files else None
        self.bank_logg = z["bank_logg"] if "bank_logg" in z.files else None


def load_mismatch(ncfg: dict) -> Mismatch | None:
    ncfg = noise_config(ncfg)
    if ncfg["mismatch"] is None:
        if ncfg["mode"] != "template" or ncfg["likelihood_sm"] or ncfg["sm_mask_thresh"]:
            raise ValueError("noise mode / likelihood_sm / sm_mask_thresh need --mismatch")
        return None
    return Mismatch(ncfg["mismatch"], ncfg["sm_scale"], ncfg["sm_mask_thresh"])


def measured_snr(flux: np.ndarray, ivar: np.ndarray) -> np.ndarray:
    """Mean per-pixel S/N over good pixels, ``mean(flux * sqrt(ivar))``."""
    good = ivar > 0
    s = np.where(good, flux * np.sqrt(np.maximum(ivar, 0.0)), 0.0).sum(1)
    return (s / np.maximum(good.sum(1), 1)).astype("float32")


def spectrum_features(flux, ivar, snr_catalog, ncfg: dict, mm: Mismatch | None) -> dict:
    """Network / likelihood inputs for observed (or already-noised) spectra.

    Applies the pixel mask, computes ``log_snr`` (catalog or measured) and,
    if ``noise_channel``, ``sigma_feat = clip(sqrt(1/ivar + s_m^2), 0, 1)``
    (masked -> 1). ``ivar`` stays the *photon* ivar; the likelihood adds
    ``s_m`` itself."""
    ncfg = noise_config(ncfg)
    if mm is not None and mm.mask.any():
        ivar = np.where(mm.mask[None], 0.0, ivar).astype("float32")
        flux = np.where(mm.mask[None], 1.0, flux).astype("float32")
    if ncfg["snr_mode"] == "measured":
        snr = measured_snr(flux, ivar)
    else:
        snr = np.asarray(snr_catalog, "float32").reshape(-1)
    out = {"flux": flux.astype("float32", copy=False), "ivar": ivar.astype("float32", copy=False),
           "log_snr": np.log10(np.maximum(snr, 1e-3))[:, None].astype("float32")}
    if ncfg["noise_channel"]:
        s2 = 0.0 if mm is None else mm.s_m[None] ** 2
        with np.errstate(divide="ignore"):
            sig = np.sqrt(np.where(ivar > 0, 1.0 / np.maximum(ivar, 1e-30), 1.0) + s2)
        out["sigma_feat"] = np.where(ivar > 0, np.minimum(sig, 1.0), 1.0).astype("float32")
    return out


class NoiseModel:
    """v2 noise for simulated spectra.

    * ``template`` : v1 (real BOSS ivar/mask template + Gaussian photon noise);
    * ``sm``       : template + extra Gaussian ``N(0, s_m)`` per pixel;
    * ``residual`` : with probability ``resid_frac`` add a *real residual*
      ``y_BOSS - emu(theta_fit)`` from the bank (photon + systematic noise of
      a star in the same (Teff, logg) bin, with that star's ivar/mask);
      otherwise (or if the bin is empty) fall back to ``sm``.
    """

    TEFF_STEP, LOGG_STEP = 250.0, 0.5

    def __init__(self, templates: NoiseTemplates, ncfg: dict, mm: Mismatch | None,
                 bank_ivar: np.ndarray | None = None, bank_snr: np.ndarray | None = None):
        self.t, self.cfg, self.mm = templates, noise_config(ncfg), mm
        self.mode = self.cfg["mode"]
        if self.mode == "residual":
            assert mm is not None and mm.bank_resid is not None and bank_ivar is not None
            self.bank_ivar, self.bank_snr = bank_ivar, bank_snr
            bid = self._bin(mm.bank_teff, mm.bank_logg)
            self.members = {b: np.flatnonzero(bid == b) for b in np.unique(bid)}

    def __len__(self):
        return len(self.t)

    def _bin(self, teff, logg):
        return (np.floor(np.asarray(teff) / self.TEFF_STEP).astype(int) * 1000
                + np.floor((np.asarray(logg) + 5) / self.LOGG_STEP).astype(int))

    def make_batch(self, flux, labels, rng):
        n = flux.shape[0]
        lsf, coefs = draw_forward_aug(self.cfg, n, rng)
        if lsf is not None or coefs is not None:
            basis = jnp.asarray(cheb_basis(coefs.shape[1] - 1)) if coefs is not None else None
            flux = np.asarray(forward_aug(jnp.asarray(flux), lsf, coefs, basis))
        j = rng.integers(0, len(self.t), size=n)
        iv = self.t.ivar[j].copy()
        snr = self.t.snr[j].copy()
        eps = rng.standard_normal(flux.shape, dtype="float32") * self.t.sigma[j]
        if self.mode in ("sm", "residual") and self.mm is not None:
            eps += rng.standard_normal(flux.shape, dtype="float32") * self.mm.s_m[None]
        noisy = flux + eps
        if self.mode == "residual":
            bins = self._bin(labels[:, 0], labels[:, 1])
            use = rng.random(n) < self.cfg["resid_frac"]
            for r in np.flatnonzero(use):
                m = self.members.get(bins[r])
                if m is None:
                    continue
                b = m[rng.integers(0, m.size)]
                noisy[r] = flux[r] + self.mm.bank_resid[b]
                iv[r] = self.bank_ivar[b]
                snr[r] = self.bank_snr[b]
        noisy = np.where(iv > 0, noisy, 1.0).astype("float32")
        out = spectrum_features(noisy, iv, snr, self.cfg, self.mm)
        if lsf is not None:
            out[LSF] = lsf[:, None]
        return out


def label_dict(labels: np.ndarray) -> dict:
    return {k: labels[:, i:i + 1].astype("float32") for i, k in enumerate(LABELS)}


@jax.jit
def _gpu_noisy_batch(key, flux_all, ivar_t, sig_t, snr_t, i, j, s_m_noise, s_m_feat, mask,
                     measured, aug=(None, None, None)):
    """GPU twin of ``NoiseModel.make_batch`` (template / sm) + ``spectrum_features``.
    One Gaussian draw with sigma = sqrt(sigma_photon^2 + s_m^2) == the two
    independent draws of the CPU path. ``measured`` is a traced bool; sigma_feat
    is always computed (the caller drops it without ``noise_channel``)."""
    f, iv, snr = flux_all[i], ivar_t[j], snr_t[j]
    f = forward_aug(f, *aug)                    # v3: LSF + continuum tilt (no-op otherwise)
    sig = jnp.sqrt(sig_t[j] ** 2 + s_m_noise[None] ** 2)
    noisy = f + jax.random.normal(key, f.shape, f.dtype) * sig
    noisy = jnp.where(iv > 0, noisy, 1.0)
    iv = jnp.where(mask[None], 0.0, iv)
    noisy = jnp.where(mask[None], 1.0, noisy)
    good = iv > 0
    snr_m = (jnp.where(good, noisy * jnp.sqrt(jnp.maximum(iv, 0.0)), 0.0).sum(1)
             / jnp.maximum(good.sum(1), 1))
    snr = jnp.where(measured, snr_m, snr)
    out = {"flux": noisy, "ivar": iv, "log_snr": jnp.log10(jnp.maximum(snr, 1e-3))[:, None]}
    sf = jnp.sqrt(jnp.where(good, 1.0 / jnp.maximum(iv, 1e-30), 1.0) + s_m_feat[None] ** 2)
    out["sigma_feat"] = jnp.where(good, jnp.minimum(sf, 1.0), 1.0)
    return out


class KorgNoisyDataset(keras.utils.PyDataset):
    """Labeled batches: Korg spectra + fresh real-BOSS noise every draw.
    ``drop_labels=True`` yields unlabeled batches (SC sanity check on Korg).

    With a template/sm ``NoiseModel`` the flux, templates and noise live on
    the GPU (no per-step host->device copy, no CPU RNG); the residual mode
    keeps the CPU path."""

    def __init__(self, labels, flux, templates, batch_size: int,
                 num_batches: int, adapter, seed: int = 0, drop_labels: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.labels, self.flux, self.templates = labels, flux, templates
        self.batch_size, self._num_batches, self.adapter = batch_size, num_batches, adapter
        self.drop_labels = drop_labels
        self.rng = np.random.default_rng(seed)
        self.gpu = isinstance(templates, NoiseModel) and templates.mode != "residual"
        if self.gpu:
            t, mm, cfg = templates.t, templates.mm, templates.cfg
            zero = np.zeros(N_PIX, "float32")
            s_m = zero if mm is None else mm.s_m
            self.dev = dict(
                flux_all=jax.device_put(flux), ivar_t=jax.device_put(t.ivar),
                sig_t=jax.device_put(t.sigma), snr_t=jax.device_put(t.snr.astype("float32")),
                s_m_noise=jnp.asarray(s_m if templates.mode == "sm" else zero),
                s_m_feat=jnp.asarray(s_m),
                mask=jnp.asarray(np.zeros(N_PIX, bool) if mm is None else mm.mask),
                measured=jnp.asarray(cfg["snr_mode"] == "measured"))
            self.noise_channel = bool(cfg["noise_channel"])
            self.key, self.calls = jax.random.key(seed), 0
            self.basis = (jnp.asarray(cheb_basis(max(int(cfg["cont_deg"]), 1)))
                          if cfg["cont_tilt"] > 0 else None)

    def _gpu_batch(self, i):
        j = self.rng.integers(0, len(self.templates.t), size=self.batch_size)
        lsf, coefs = draw_forward_aug(self.templates.cfg, self.batch_size, self.rng)
        self.calls += 1
        out = _gpu_noisy_batch(jax.random.fold_in(self.key, self.calls), i=i, j=j,
                               aug=(lsf, coefs, self.basis if coefs is not None else None),
                               **self.dev)
        if not self.noise_channel:
            out.pop("sigma_feat")
        if lsf is not None:
            out[LSF] = jnp.asarray(lsf)[:, None]
        return out

    def __getitem__(self, item):
        i = self.rng.integers(0, self.labels.shape[0], size=self.batch_size)
        batch = (self._gpu_batch(i) if self.gpu
                 else self.templates.make_batch(self.flux[i], self.labels[i], self.rng))
        if self.drop_labels:
            batch.pop(LSF, None)
            return self.adapter(batch, strict=False)
        return self.adapter(label_dict(self.labels[i]) | batch)

    @property
    def num_batches(self):
        return self._num_batches

    def __len__(self):
        return self._num_batches


class BossDataset(keras.utils.PyDataset):
    """Unlabeled batches: real BOSS spectra (flux, ivar, log SNR[, sigma_feat]).
    The features are held on the GPU; each batch is a device-side gather."""

    def __init__(self, flux, ivar, snr, batch_size: int, num_batches: int,
                 adapter, seed: int = 1, ncfg: dict | None = None,
                 mm: Mismatch | None = None, **kwargs):
        super().__init__(**kwargs)
        feats = spectrum_features(flux, ivar, snr, ncfg, mm)
        self.feats = {k: jax.device_put(v) for k, v in feats.items()}
        self.n = feats["flux"].shape[0]
        self.batch_size, self._num_batches, self.adapter = batch_size, num_batches, adapter
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, item):
        i = jnp.asarray(self.rng.integers(0, self.n, size=self.batch_size))
        return self.adapter({k: v[i] for k, v in self.feats.items()}, strict=False)

    @property
    def num_batches(self):
        return self._num_batches

    def __len__(self):
        return self._num_batches


# =========================================================================
# Adapters, prior, likelihood
# =========================================================================

def padded_box(box: dict, pad: float = 0.01) -> dict:
    """Box padded by ``pad`` x range (vmic in log10 space) so grid points at
    the edges stay finite under the bounded (sigmoid) constraint."""
    out = {}
    for k, (lo, hi) in box.items():
        if k in LOG_LABELS:
            lo, hi = np.log10(lo), np.log10(hi)
            w = hi - lo
            out[k] = (float(10 ** (lo - pad * w)), float(10 ** (hi + pad * w)))
        else:
            w = hi - lo
            out[k] = (lo - pad * w, hi + pad * w)
    return out


def shared_adapter(pbox: dict, log_params=None) -> bf.Adapter:
    """Maps each label to an unbounded space: ``log`` first for ``log_params``
    (default ``LOG_LABELS`` = vmic), then a bounded logit on the padded box.

    This is only the network's *parametrisation*. The prior is set by the
    simulations (the Korg grid: uniform Teff, log-uniform vmic) and, in SC, by
    ``prior_log_prob`` evaluated in physical space with bayesflow adding the
    adapter's log-Jacobian -- so e.g. ``log_params=["vmic", "Teff"]`` (run
    v3_logTeff) keeps the Teff prior uniform. Do not add -log(Teff) to the prior."""
    logs = LOG_LABELS if log_params is None else list(log_params)
    a = bf.Adapter()
    for k in LABELS + ([LSF] if LSF in pbox else []):
        lo, hi = pbox[k]
        if k in logs:
            a = a.log(k).constrain(k, lower=float(np.log(lo)), upper=float(np.log(hi)))
        else:
            a = a.constrain(k, lower=float(lo), upper=float(hi))
    return a


def posterior_adapter(noise_channel: bool = False, names=None) -> bf.Adapter:
    a = bf.Adapter().drop("ivar").concatenate(list(names or LABELS), into="inference_variables")
    if noise_channel:   # summary input = [flux | sigma_feat], shape (n, 2 * N_PIX)
        a = a.concatenate(["flux", "sigma_feat"], into="summary_variables")
    else:
        a = a.rename("flux", "summary_variables")
    return a.rename("log_snr", "inference_conditions")


def prior_log_prob(vmic, **kwargs):
    """Uniform in the box for 5 labels (and lsf_px), log-uniform in vmic, in
    *physical* units -- the Korg grid's sampling -- whatever ``shared_adapter``'s
    ``log_params`` (unnormalised; samples never leave the box)."""
    return -jnp.log(jnp.squeeze(vmic, -1))


def make_emulator_fn(ckpt=EMULATOR_CKPT):
    """Differentiable ``theta (n, 6) -> flux (n, 4479)`` from a checkpoint."""
    model, payload = load_checkpoint(ckpt)
    names = list(payload.get("feature_names") or LABELS)
    assert names == LABELS, f"emulator inputs {names} != {LABELS}"
    graphdef, state = nnx.split(model)
    x_mean = jnp.asarray(payload["x_mean"], jnp.float32)
    x_std = jnp.asarray(payload["x_std"], jnp.float32)
    y_mean = jnp.asarray(payload["y_mean"], jnp.float32)
    y_std = jnp.asarray(payload["y_std"], jnp.float32)

    def flux(theta):
        yn = nnx.merge(graphdef, state)((theta - x_mean) / x_std)
        return yn * y_std + y_mean

    return flux


def make_likelihood(emulator_fn, sigma_floor: float = 0.0, s_m: np.ndarray | None = None,
                    pixel_mask: np.ndarray | None = None, lsf: bool = False, cont_deg: int = 0):
    """``log p(y | theta) = -1/2 sum ivar_eff (y - emu(theta))^2`` (unnormalised).

    ``ivar_eff = ivar / (1 + ivar * (floor^2 + s_m^2))``: an optional scalar
    systematic floor and/or a per-pixel model-mismatch scatter ``s_m``
    (BOSS-CLAM style). ``pixel_mask`` (True = drop) zeroes pixels. Since
    ``s_m`` is fixed, the omitted ``log(sigma^2 + s_m^2)`` term is a
    constant in theta and cancels in the SC variance.

    v3: ``lsf`` convolves the emulator flux with a Gaussian of sigma
    ``theta["lsf_px"]``; ``cont_deg > 0`` multiplies it by the best-fit
    Chebyshev continuum (profiled per spectrum and draw, weights ``ivar_eff``)."""
    extra = np.full(N_PIX, float(sigma_floor) ** 2, "float32")
    if s_m is not None:
        extra = extra + np.asarray(s_m, "float32") ** 2
    keep = np.ones(N_PIX, "float32") if pixel_mask is None else (~pixel_mask).astype("float32")
    extra_j, keep_j = jnp.asarray(extra), jnp.asarray(keep)
    plain = not np.any(extra) and pixel_mask is None
    basis = jnp.asarray(cheb_basis(cont_deg)) if cont_deg > 0 else None

    def likelihood_log_prob(flux, ivar, **theta):
        th = jnp.concatenate([jnp.reshape(theta[k], (-1, 1)) for k in LABELS], axis=1)
        mu = emulator_fn(th.astype(jnp.float32))
        iv = ivar if plain else keep_j * ivar / (1.0 + ivar * extra_j)
        if lsf:
            mu = apply_lsf(mu, theta[LSF])
        if basis is not None:
            mu = profile_continuum(mu, flux, iv, basis)
        return -0.5 * jnp.sum(iv * (flux - mu) ** 2, axis=-1)

    return likelihood_log_prob


class WeightedSC(sc.SelfConsistencyLoss):
    """SC loss scaled by ``weight`` (its raw scale grows with chi2). (v1)"""

    def __init__(self, num_samples: int, weight: float = 1.0, **kwargs):
        super().__init__(num_samples=num_samples, **kwargs)
        self.weight = float(weight)

    def __call__(self, data, stage):
        m = super().__call__(data, stage)
        return {"loss": self.weight * m["loss"], "raw": m["loss"]}


class StabilizedSC(sc.SelfConsistencyLoss):
    """Self-consistency loss with scale-free variants (v2).

    For each observation the log-evidence estimates
    ``l_s = log p(theta_s) + log p(y|theta_s) - log q(theta_s|y)`` over the
    ``num_samples`` posterior draws give ``V = Var_s(l_s)``; then

    * ``var``     : ``mean(V)``                      (v1 / bayesflow default)
    * ``logvar``  : ``mean(log(V + eps))``           gradient = grad V / (V + eps):
      a batch with 100x larger chi2 scatter takes the same-size step;
    * ``normvar`` : ``mean(V) / stop_grad(mean(V) + eps)``.

    ``trim`` drops the ``round(trim * S)`` lowest *and* highest ``l_s`` per
    observation before the variance (robust to a few catastrophic draws,
    e.g. samples where the emulator extrapolates). ``weight`` multiplies the
    loss; ``raw_var`` / ``lml_std`` are logged for monitoring.
    """

    def __init__(self, num_samples: int, weight: float = 1.0, mode: str = "logvar",
                 trim: float = 0.0, eps: float = 1.0, **kwargs):
        super().__init__(num_samples=num_samples, **kwargs)
        assert mode in ("var", "logvar", "normvar"), mode
        self.weight, self.mode, self.eps = float(weight), mode, float(eps)
        self.n_trim = int(round(trim * num_samples))
        assert 2 * self.n_trim < num_samples - 1, "trim too large"

    def __call__(self, data, stage):
        samples = self._sample_proposal(data)
        data = self._reshape_data(data, samples)
        self._determine_keys(samples, data)
        lml = self._log_marginal_likelihood(data, stage)          # (batch, S)
        # drop observations with any non-finite draw (rare; v2_giants diverged on one)
        ok = keras.ops.all(keras.ops.isfinite(lml), axis=1)
        lml = keras.ops.where(ok[:, None], lml, 0.0)
        if self.n_trim:
            lml = keras.ops.sort(lml, axis=1)[:, self.n_trim:self.num_samples - self.n_trim]
        v = keras.ops.var(lml, axis=1)
        okf = keras.ops.cast(ok, v.dtype)
        mean = lambda x: keras.ops.sum(okf * x) / keras.ops.maximum(keras.ops.sum(okf), 1.0)
        if self.mode == "var":
            loss = mean(v)
        elif self.mode == "logvar":
            loss = mean(keras.ops.log(v + self.eps))
        else:
            loss = mean(v) / keras.ops.stop_gradient(mean(v) + self.eps)
        sg = keras.ops.stop_gradient
        return {"loss": self.weight * loss, "raw_var": sg(mean(v)),
                "lml_std": sg(mean(keras.ops.sqrt(v))), "frac_nonfinite": sg(1.0 - keras.ops.mean(okf))}


class FiniteAdamW(keras.optimizers.AdamW):
    """AdamW that skips a step (keeps weights *and* optimizer state) when any
    gradient is non-finite, so one bad SC batch cannot poison the run."""

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in grads]))
        new_tv, new_opt = super().stateless_apply(optimizer_variables, grads, trainable_variables)
        keep = lambda new, old: [jnp.where(ok, a, b) for a, b in zip(new, old)]
        return keep(new_tv, trainable_variables), keep(new_opt, optimizer_variables)


# =========================================================================
# Posterior network
# =========================================================================

class SpectrumMLP(bf.networks.SummaryNetwork):
    """Dense summary of a 1D spectrum: absorption (1 - flux) -> summary_dim."""

    def __init__(self, hidden=(1024, 512, 256), summary_dim=64,
                 activation="gelu", dropout=0.0, noise_channel=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden, self.summary_dim = tuple(hidden), summary_dim
        self.activation, self.dropout = activation, dropout
        # v2: input is [flux | sigma_feat]; only the flux half becomes 1 - flux
        self.noise_channel = bool(noise_channel)
        self.net = keras.Sequential(
            [layer for h in hidden for layer in
             ([keras.layers.Dense(h, activation=activation)]
              + ([keras.layers.Dropout(dropout)] if dropout else []))]
            + [keras.layers.Dense(summary_dim)])

    def call(self, x, training=False, **kwargs):
        if self.noise_channel:
            x = keras.ops.concatenate([1.0 - x[..., :N_PIX], x[..., N_PIX:]], axis=-1)
        else:
            x = 1.0 - x
        return self.net(x, training=training)

    def get_config(self):
        return {"hidden": self.hidden, "summary_dim": self.summary_dim,
                "activation": self.activation, "dropout": self.dropout,
                "noise_channel": self.noise_channel}


class SpectrumTST(bf.networks.SummaryNetwork):
    """Flat [flux | sigma_feat] (B, 2*N_PIX) -> (B, N_PIX, 3) channels
    [1 - flux, sigma_feat, good = sigma_feat < 1] -> TimeSeriesTransformer.
    The mask travels as the `good` channel (TST's mask= is applied before its
    strided-conv downsampling, so it would not mean what we want)."""

    def __init__(self, summary_dim=64, **tst_kwargs):
        super().__init__()
        self.summary_dim, self.tst_kwargs = summary_dim, tst_kwargs
        self.tst = bf.networks.TimeSeriesTransformer(summary_dim=summary_dim, **tst_kwargs)

    def call(self, x, training=False, **kwargs):
        flux, sig = x[..., :N_PIX], x[..., N_PIX:]
        good = keras.ops.cast(sig < 1.0, x.dtype)
        return self.tst(keras.ops.stack([1.0 - flux, sig, good], axis=-1), training=training)

    def get_config(self):
        return {"summary_dim": self.summary_dim, **self.tst_kwargs}


DEFAULT_NET = dict(hidden=[1024, 512, 256], summary_dim=64, dropout=0.0,
                   coupling_depth=6, coupling_widths=[256, 256], summary="mlp")
# TST kwargs; override per run with --net '{"summary": "tst", "tst": {"downsample": 32}}'
DEFAULT_TST = dict(embed_dims=[128, 128, 128], num_heads=[4, 4, 4], downsample=16,
                   time_embedding="time2vec", dropout=0.05)


def summary_net(cfg: dict, noise_channel: bool):
    if cfg["summary"] == "mlp":
        return SpectrumMLP(hidden=cfg["hidden"], summary_dim=cfg["summary_dim"],
                           dropout=cfg["dropout"], noise_channel=noise_channel)
    if cfg["summary"] == "tst":
        assert noise_channel, "tst summary needs the sigma_feat channel (noise_channel=True)"
        kw = DEFAULT_TST | cfg.get("tst", {})
        return SpectrumTST(summary_dim=cfg["summary_dim"],
                           **{k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()})
    raise ValueError(f"unknown summary network {cfg['summary']!r}")


def build_posterior(net_cfg: dict, noise_channel: bool = False, names=None) -> bf.BasicEstimator:
    cfg = DEFAULT_NET | (net_cfg or {})
    return bf.BasicEstimator(
        inference_network=bf.networks.CouplingFlow(
            depth=cfg["coupling_depth"],
            subnet_kwargs={"widths": tuple(cfg["coupling_widths"])}),
        summary_network=summary_net(cfg, noise_channel),
        adapter=posterior_adapter(noise_channel, names),
    )


def build_estimator(posterior, adapter, likelihood_fn):
    return sc.SemiSupervisedEstimator(
        estimators={"prior": prior_log_prob, "likelihood": likelihood_fn,
                    "posterior": posterior},
        adapter=adapter)


def save_posterior(posterior, run_dir: Path, name: str) -> Path:
    p = Path(run_dir) / f"{name}.weights.h5"
    posterior.save_weights(p)
    return p


def load_run_config(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_posterior(run_dir: Path, name: str, example_batch: dict):
    """Rebuild the posterior from ``config.json`` and load ``<name>.weights.h5``.
    ``example_batch`` is a shared-adapted batch used to build the layers."""
    cfg = load_run_config(run_dir)
    ncfg = noise_config(cfg.get("noise"))
    posterior = build_posterior(cfg["net"], ncfg["noise_channel"], params(ncfg))
    posterior.build_from_data(posterior.adapter(example_batch))
    posterior.load_weights(Path(run_dir) / f"{name}.weights.h5")
    return posterior


def sample_posterior(posterior, adapter, cond: dict, n_samples: int,
                     batch: int = 256, names=None) -> np.ndarray:
    """Posterior samples in physical label space, shape (n, n_samples, len(names)),
    ``names`` defaulting to the 6 labels (pass ``params(ncfg)`` to get lsf_px too).
    ``cond``: dict of condition arrays (``flux``, ``log_snr`` [, ``sigma_feat``];
    ``ivar`` and a true ``lsf_px`` are ignored)."""
    names = list(names or LABELS)
    cond = {k: v for k, v in cond.items() if k not in ("ivar", LSF)}
    n = cond["flux"].shape[0]
    out = np.empty((n, n_samples, len(names)), "float32")
    for s in range(0, n, batch):
        e = min(s + batch, n)
        smp = posterior.sample(num_samples=n_samples, conditions={k: v[s:e] for k, v in cond.items()})
        smp = adapter(smp, inverse=True, strict=False)
        out[s:e] = np.concatenate([np.asarray(smp[k]).reshape(e - s, n_samples, 1)
                                   for k in names], axis=-1)
    return out


def weights_finite(model) -> bool:
    return all(np.isfinite(np.asarray(w)).all() for w in model.weights)
