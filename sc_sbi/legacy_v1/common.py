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

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # legacy copy lives one level deeper
KORG_CACHE = PROJECT_ROOT / "data_jarvis/preprocessed_data_combined_v2/korg_grid_train.h5"
BOSS_NPZ = PROJECT_ROOT / "boss_clam/boss_clam_training_data.npz"
BOSS_PREPARED = PROJECT_ROOT / "sc_sbi/cache/boss_clam_prepared.npz"
EMULATOR_CKPT = PROJECT_ROOT / "trained_model/combined_best/mlp_1790330838/best.pkl"

# --- labels --------------------------------------------------------------

LABELS = list(data_mod.CORE_LABELS)        # Teff, logg, m_H, alpha_m, carbon_m, vmic
LOG_LABELS = ["vmic"]                      # sampled log-uniformly in the grid
N_PIX = data_mod.N_PIX_KEEP                # 4479
# labels that also exist in the BOSS reference catalogue (vmic/carbon do not)
BOSS_LABELS = ["Teff", "logg", "m_H", "alpha_m"]
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
    """Real BOSS (ivar, mask, SNR) triples used to noise simulated spectra."""

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


def label_dict(labels: np.ndarray) -> dict:
    return {k: labels[:, i:i + 1].astype("float32") for i, k in enumerate(LABELS)}


class KorgNoisyDataset(keras.utils.PyDataset):
    """Labeled batches: Korg spectra + fresh real-BOSS noise every draw."""

    def __init__(self, labels, flux, templates: NoiseTemplates, batch_size: int,
                 num_batches: int, adapter, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.labels, self.flux, self.templates = labels, flux, templates
        self.batch_size, self._num_batches, self.adapter = batch_size, num_batches, adapter
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, item):
        i = self.rng.integers(0, self.labels.shape[0], size=self.batch_size)
        y, iv, lsnr = self.templates.apply(self.flux[i], self.rng)
        batch = label_dict(self.labels[i]) | {"flux": y, "ivar": iv, "log_snr": lsnr}
        return self.adapter(batch)

    @property
    def num_batches(self):
        return self._num_batches

    def __len__(self):
        return self._num_batches


class BossDataset(keras.utils.PyDataset):
    """Unlabeled batches: real BOSS spectra (flux, ivar, log SNR)."""

    def __init__(self, flux, ivar, snr, batch_size: int, num_batches: int,
                 adapter, seed: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.flux, self.ivar = flux, ivar
        self.log_snr = np.log10(snr)[:, None].astype("float32")
        self.batch_size, self._num_batches, self.adapter = batch_size, num_batches, adapter
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, item):
        i = self.rng.integers(0, self.flux.shape[0], size=self.batch_size)
        return self.adapter({"flux": self.flux[i], "ivar": self.ivar[i],
                             "log_snr": self.log_snr[i]}, strict=False)

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


def shared_adapter(pbox: dict) -> bf.Adapter:
    """Maps each label to an unbounded space (vmic: log first)."""
    a = bf.Adapter()
    for k in LABELS:
        lo, hi = pbox[k]
        if k in LOG_LABELS:
            a = a.log(k).constrain(k, lower=float(np.log(lo)), upper=float(np.log(hi)))
        else:
            a = a.constrain(k, lower=float(lo), upper=float(hi))
    return a


def posterior_adapter() -> bf.Adapter:
    return (bf.Adapter()
            .drop("ivar")
            .concatenate(LABELS, into="inference_variables")
            .rename("flux", "summary_variables")
            .rename("log_snr", "inference_conditions"))


def prior_log_prob(vmic, **kwargs):
    """Uniform in the box for 5 labels, log-uniform in vmic (unnormalised;
    samples never leave the box because of the bounded constraint)."""
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


def make_likelihood(emulator_fn, sigma_floor: float = 0.0):
    """``log p(y | theta) = -1/2 sum ivar_eff (y - emu(theta))^2`` (unnormalised).

    ``ivar_eff = ivar / (1 + ivar * floor^2)`` adds an optional systematic
    floor for emulator / Korg-vs-real mismatch."""
    floor2 = float(sigma_floor) ** 2

    def likelihood_log_prob(flux, ivar, **theta):
        th = jnp.concatenate([jnp.reshape(theta[k], (-1, 1)) for k in LABELS], axis=1)
        mu = emulator_fn(th.astype(jnp.float32))
        iv = ivar / (1.0 + ivar * floor2) if floor2 > 0 else ivar
        return -0.5 * jnp.sum(iv * (flux - mu) ** 2, axis=-1)

    return likelihood_log_prob


class WeightedSC(sc.SelfConsistencyLoss):
    """SC loss scaled by ``weight`` (its raw scale grows with chi2)."""

    def __init__(self, num_samples: int, weight: float = 1.0, **kwargs):
        super().__init__(num_samples=num_samples, **kwargs)
        self.weight = float(weight)

    def __call__(self, data, stage):
        m = super().__call__(data, stage)
        return {"loss": self.weight * m["loss"], "raw": m["loss"]}


# =========================================================================
# Posterior network
# =========================================================================

class SpectrumMLP(bf.networks.SummaryNetwork):
    """Dense summary of a 1D spectrum: absorption (1 - flux) -> summary_dim."""

    def __init__(self, hidden=(1024, 512, 256), summary_dim=64,
                 activation="gelu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden, self.summary_dim = tuple(hidden), summary_dim
        self.activation, self.dropout = activation, dropout
        self.net = keras.Sequential(
            [layer for h in hidden for layer in
             ([keras.layers.Dense(h, activation=activation)]
              + ([keras.layers.Dropout(dropout)] if dropout else []))]
            + [keras.layers.Dense(summary_dim)])

    def call(self, x, training=False, **kwargs):
        return self.net(1.0 - x, training=training)

    def get_config(self):
        return {"hidden": self.hidden, "summary_dim": self.summary_dim,
                "activation": self.activation, "dropout": self.dropout}


DEFAULT_NET = dict(hidden=[1024, 512, 256], summary_dim=64, dropout=0.0,
                   coupling_depth=6, coupling_widths=[256, 256])


def build_posterior(net_cfg: dict) -> bf.BasicEstimator:
    cfg = DEFAULT_NET | (net_cfg or {})
    return bf.BasicEstimator(
        inference_network=bf.networks.CouplingFlow(
            depth=cfg["coupling_depth"],
            subnet_kwargs={"widths": tuple(cfg["coupling_widths"])}),
        summary_network=SpectrumMLP(hidden=cfg["hidden"], summary_dim=cfg["summary_dim"],
                                    dropout=cfg["dropout"]),
        adapter=posterior_adapter(),
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
    posterior = build_posterior(cfg["net"])
    posterior.build_from_data(posterior.adapter(example_batch))
    posterior.load_weights(Path(run_dir) / f"{name}.weights.h5")
    return posterior


def sample_posterior(posterior, adapter, flux, log_snr, n_samples: int,
                     batch: int = 256) -> np.ndarray:
    """Posterior samples in physical label space, shape (n, n_samples, 6)."""
    out = np.empty((flux.shape[0], n_samples, len(LABELS)), "float32")
    for s in range(0, flux.shape[0], batch):
        e = min(s + batch, flux.shape[0])
        cond = {"flux": flux[s:e], "log_snr": log_snr[s:e]}
        smp = posterior.sample(num_samples=n_samples, conditions=cond)
        smp = adapter(smp, inverse=True, strict=False)
        out[s:e] = np.concatenate([np.asarray(smp[k]).reshape(e - s, n_samples, 1)
                                   for k in LABELS], axis=-1)
    return out
