"""Data preprocessing and loading for the BOSS/Korg spectral emulator.

The raw grid lives in a single ~260 GB HDF5 file:

    /export/data/vgiusepp/BOSS_emulator/data/korg_grid.h5

with, per grid point (N = 2,316,600):

    Teff, logg, m_H, alpha_m, carbon_m, vmic   stellar labels
    abundances       (N, 92)   A(X) per element, Z = 1..92 (H..U)
    norm_flux        (N, 4648) continuum-normalised flux (the target)
    abs_flux, continuum ...    other spectral representations
    converged_flag   (N,)      1 = Korg synthesis converged

Two facts, verified against the file, drive the preprocessing here
(see ``script/study_spectra.py`` / ``study_params.py``):

* 62,108 rows failed to converge and are dropped.
* The final 169 pixels (index 4479..4647) are exactly zero in *every*
  converged row -- a zero-imputation artefact -- so we keep pixels
  0..4478 (``N_PIX_KEEP`` = 4479).
* The abundances are a *deterministic* function of (m_H, alpha_m,
  carbon_m): within any group sharing those three the abundance spread
  is ~1e-15 and corr(A(Fe), m_H) = 1.0. H and He are held fixed across
  the grid; the cache stores only the curated ``ABUNDANCE_ELEMENTS``,
  each mapped from its symbol to its column in the raw 92-element array
  via ``ELEMENT_SYMBOLS_Z``.

Building the 40 GB float32 cache is a one-time cost. Thereafter
``load_dataset`` just opens the cache and assembles whatever subset of
labels the user asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# --- paths ---------------------------------------------------------------

RAW_PATH = Path("/export/data/vgiusepp/BOSS_emulator/data/korg_grid.h5")

# Project root = parent of the ``src`` directory that holds this package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "preprocessed_data"
CACHE_PATH = CACHE_DIR / "korg_grid_train.h5"

# --- grid constants (verified against the raw file) ----------------------

CORE_LABELS = ["Teff", "logg", "m_H", "alpha_m", "carbon_m", "vmic"]

N_PIX_RAW = 4648      # pixels stored per spectrum
N_PIX_TRIM = 169      # trailing pixels imputed to zero in every row
N_PIX_KEEP = N_PIX_RAW - N_PIX_TRIM  # 4479 usable pixels

# Full element table by atomic number Z = 1..92 (H..U). This is the column
# order of the raw (N, 92) ``abundances`` array, used to map an element
# symbol to its column index in the raw file.
ELEMENT_SYMBOLS_Z = [
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

# Curated elements we care about for the emulator. H and He are held fixed
# across the grid (no variation), so they are dropped from the selectable
# abundance inputs. Only ``ABUNDANCE_ELEMENTS`` are extracted into the cache.
CURATED_ELEMENTS = [
    "H", "He", "Li", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
]
FIXED_ELEMENTS = ["H", "He"]
ABUNDANCE_ELEMENTS = [e for e in CURATED_ELEMENTS if e not in FIXED_ELEMENTS]

# Column index of each cached abundance element within the raw (N, 92)
# array, so ``abundances[:, ABUNDANCE_COLS]`` selects them in cache order.
ABUNDANCE_COLS = [ELEMENT_SYMBOLS_Z.index(e) for e in ABUNDANCE_ELEMENTS]

# Backwards-compatible alias (the cached abundance columns, in cache order).
VARYING_ELEMENTS = ABUNDANCE_ELEMENTS


# =========================================================================
# Cache construction
# =========================================================================

def build_cache(
    raw_path: Path = RAW_PATH,
    cache_path: Path = CACHE_PATH,
    chunk: int = 100_000,
    rebuild: bool = False,
) -> Path:
    """Stream the raw grid, drop failed rows and trailing zero pixels, and
    write a compact float32 cache.

    Only converged rows are kept; ``norm_flux`` is trimmed to the first
    ``N_PIX_KEEP`` pixels; all arrays are stored as float32. The cache is
    written once and reused -- if it already exists this is a no-op unless
    ``rebuild`` is True.

    Returns the path to the cache file.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        print(f"[data] cache already present: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".h5.tmp")

    with h5py.File(raw_path, "r") as f:
        cf = f["converged_flag"][:]
        conv = cf == 1
        n_conv = int(conv.sum())
        N = cf.size
        print(f"[data] raw N={N:,}  converged={n_conv:,}  "
              f"trimming {N_PIX_TRIM} trailing pixels -> {N_PIX_KEEP} kept")

        with h5py.File(tmp_path, "w") as g:
            # Target spectrum: chunked for efficient row-batch reads later.
            flux = g.create_dataset(
                "norm_flux", shape=(n_conv, N_PIX_KEEP), dtype="float32",
                chunks=(min(4096, n_conv), N_PIX_KEEP),
            )
            core = {
                name: g.create_dataset(name, shape=(n_conv,), dtype="float32")
                for name in CORE_LABELS
            }
            abund = g.create_dataset(
                "abundances", shape=(n_conv, len(ABUNDANCE_ELEMENTS)),
                dtype="float32",
            )

            src_flux = f["norm_flux"]
            src_abund = f["abundances"]
            src_core = {name: f[name] for name in CORE_LABELS}

            write = 0
            for start in range(0, N, chunk):
                stop = min(start + chunk, N)
                m = conv[start:stop]
                k = int(m.sum())
                if k == 0:
                    continue
                sl = slice(write, write + k)
                # norm_flux: read the chunk, mask, trim trailing zeros.
                block = src_flux[start:stop, :N_PIX_KEEP]
                flux[sl] = block[m].astype("float32")
                # abundances: read the full 92-col chunk, mask, then select
                # the curated element columns (numpy fancy indexing; h5py
                # would require sorted column indices).
                ab_block = src_abund[start:stop, :][m]
                abund[sl] = ab_block[:, ABUNDANCE_COLS].astype("float32")
                for name in CORE_LABELS:
                    core[name][sl] = src_core[name][start:stop][m].astype("float32")
                write += k
                print(f"\r[data] building cache: {write:,}/{n_conv:,} rows",
                      end="", flush=True)
            print()
            assert write == n_conv, (write, n_conv)

            # Bookkeeping metadata.
            g.attrs["n_converged"] = n_conv
            g.attrs["n_pixels"] = N_PIX_KEEP
            g.attrs["n_pixels_raw"] = N_PIX_RAW
            g.attrs["n_pixels_trimmed"] = N_PIX_TRIM
            g.attrs["core_labels"] = np.array(CORE_LABELS, dtype="S16")
            g.attrs["abundance_elements"] = np.array(ABUNDANCE_ELEMENTS, dtype="S4")
            g.attrs["abundance_cols"] = np.array(ABUNDANCE_COLS, dtype="int32")
            g.attrs["source"] = str(raw_path)

    tmp_path.replace(cache_path)
    gb = cache_path.stat().st_size / 1e9
    print(f"[data] wrote cache {cache_path} ({gb:.1f} GB)")
    return cache_path


# =========================================================================
# Dataset loading
# =========================================================================

@dataclass
class Dataset:
    """An assembled, ready-to-train dataset.

    ``X`` are the selected input labels, ``Y`` the target spectra; both are
    float32 numpy arrays. Normalisation statistics (computed on the training
    split only) are stored so predictions can be de-normalised and so a
    saved model can reproduce the exact preprocessing.
    """

    X: np.ndarray                # (n, n_features)
    Y: np.ndarray                # (n, n_pixels)
    feature_names: list[str]
    n_pixels: int
    # standardisation stats (train split)
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    # split masks (indices into X/Y)
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


def _resolve_abundances(elements: list[str]) -> list[int]:
    """Map element symbols (e.g. 'Fe') to their column index in the cached
    ``abundances`` array (which holds only ``ABUNDANCE_ELEMENTS``)."""
    cols = []
    for sym in elements:
        if sym not in ABUNDANCE_ELEMENTS:
            raise ValueError(
                f"element {sym!r} is not a cached abundance; "
                f"choose from {ABUNDANCE_ELEMENTS}")
        cols.append(ABUNDANCE_ELEMENTS.index(sym))
    return cols


def load_dataset(
    core_labels: list[str] | None = None,
    abundance_elements: list[str] | None = None,
    target: str = "norm_flux",
    cache_path: Path = CACHE_PATH,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    max_rows: int | None = None,
    standardize_y: bool = True,
    in_memory: bool = True,
) -> Dataset:
    """Assemble a :class:`Dataset` from the cache.

    Parameters
    ----------
    core_labels : list of str, optional
        Which of ``CORE_LABELS`` to use as inputs. Defaults to all six.
    abundance_elements : list of str, optional
        Element symbols whose A(X) abundance to append as extra inputs
        (e.g. ``["Fe", "C"]``). Defaults to none. Note these are
        deterministic functions of (m_H, alpha_m, carbon_m).
    target : str
        Cached target dataset name (currently only ``"norm_flux"``).
    val_frac : float
        Fraction of rows held out for validation (loss monitoring).
    test_frac : float
        Fraction of rows held out for the test set (final reconstruction
        metrics; never seen during training or model selection).
    max_rows : int, optional
        If set, use only this many (randomly chosen) rows -- handy for
        quick smoke tests without touching the full 40 GB target.
    standardize_y : bool
        If True, per-pixel standardise the target; otherwise leave the
        target as-is (identity stats).
    in_memory : bool
        If True, read the full target into RAM (fits the ~1 TB machine).
        If False, keep it as an h5 dataset handle referenced by index.
    """
    core_labels = list(core_labels) if core_labels is not None else list(CORE_LABELS)
    abundance_elements = list(abundance_elements) if abundance_elements else []
    for name in core_labels:
        if name not in CORE_LABELS:
            raise ValueError(f"unknown core label {name!r}; choose from {CORE_LABELS}")
    abund_cols = _resolve_abundances(abundance_elements)

    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"cache {cache_path} not found -- run build_cache() first")

    rng = np.random.default_rng(seed)

    with h5py.File(cache_path, "r") as f:
        n_total = int(f.attrs["n_converged"])
        n_pixels = int(f.attrs["n_pixels"])

        # Row subset (sorted for efficient h5 fancy indexing).
        if max_rows is not None and max_rows < n_total:
            rows = np.sort(rng.choice(n_total, size=max_rows, replace=False))
        else:
            rows = np.arange(n_total)

        # Inputs: core labels then chosen abundances.
        x_cols = [f[name][rows] for name in core_labels]
        if abund_cols:
            ab = f["abundances"][rows, :]
            for c in abund_cols:
                x_cols.append(ab[:, c])
        X = np.stack(x_cols, axis=1).astype("float32")

        # Target.
        if in_memory:
            Y = f[target][rows, :].astype("float32")
        else:
            # Materialise the subset to a fresh in-file order is expensive;
            # for the streaming path we just load anyway (kept for symmetry).
            Y = f[target][rows, :].astype("float32")

    feature_names = list(core_labels) + [f"A({s})" for s in abundance_elements]
    n = X.shape[0]

    # Train / validation / test split. Val monitors the loss during
    # training; test is held out for final reconstruction metrics.
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    if n_val + n_test >= n:
        raise ValueError(
            f"val_frac + test_frac too large ({val_frac} + {test_frac}) "
            f"for n={n}")
    test_idx = np.sort(perm[:n_test])
    val_idx = np.sort(perm[n_test:n_test + n_val])
    train_idx = np.sort(perm[n_test + n_val:])

    # Standardisation stats from the training split only.
    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)
    x_std[x_std == 0] = 1.0
    if standardize_y:
        y_mean = Y[train_idx].mean(axis=0)
        y_std = Y[train_idx].std(axis=0)
        y_std[y_std == 0] = 1.0
    else:
        y_mean = np.zeros(n_pixels, dtype="float32")
        y_std = np.ones(n_pixels, dtype="float32")

    return Dataset(
        X=X, Y=Y, feature_names=feature_names, n_pixels=n_pixels,
        x_mean=x_mean.astype("float32"), x_std=x_std.astype("float32"),
        y_mean=y_mean.astype("float32"), y_std=y_std.astype("float32"),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )


def iter_batches(
    idx: np.ndarray,
    batch_size: int,
    seed: int = 0,
    shuffle: bool = True,
):
    """Yield arrays of row indices (into a Dataset's X/Y) for one epoch."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(idx) if shuffle else idx
    for start in range(0, order.size, batch_size):
        yield order[start:start + batch_size]
