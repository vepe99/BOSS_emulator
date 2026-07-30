# BOSS emulator

A fast, differentiable neural emulator for BOSS stellar spectra. It maps a
handful of stellar **labels → a continuum-normalised spectrum** (4479 pixels),
reproducing what the [Korg](https://github.com/ajwheeler/Korg.jl)
radiative-transfer code synthesises, but in milliseconds instead of
CPU-minutes. Because the network is a JAX model it is differentiable in its
inputs, so it can also be run *backwards* — recovering labels from an observed
spectrum by gradient-based optimisation or MCMC (see
[`script/inv_model/`](script/inv_model/README.md)).

---

## What it learns

The training set is a Korg grid of **2,254,492 converged spectra** on a regular
factorial grid of six core stellar labels:

| label | meaning | grid |
|---|---|---|
| `Teff` | effective temperature | 2800–8000 K (30 steps) |
| `logg` | surface gravity | −0.5 … 5.5 (13) |
| `m_H` | metallicity [M/H] | −2.5 … 1.0 (15) |
| `alpha_m` | alpha enhancement | −1.0 … 1.0 (9) |
| `carbon_m` | carbon enhancement | −1.5 … 1.0 (11) |
| `vmic` | microturbulence | 4 values, 0.3–5.0 |

Each spectrum is 4479 pixels (trimmed from 4648). The grid also carries 21
individual element abundances, but in this grid they are an **exact affine
function** of `(m_H, alpha_m, carbon_m)`, so they add no degrees of freedom —
the emulator is trained **abundance-free on the six core labels** with no loss
of accuracy, which also removes a nasty input degeneracy for inverse modelling.

## The model & the headline result

The emulator is a plain **MLP** ([`src/boss_emulator/emulator.py`](src/boss_emulator/emulator.py)):
a dense network mapping labels → all pixels in one shot, with optional residual
connections and pre-activation LayerNorm.

Three fancier architectures were evaluated and dropped — a DeepONet (learned
basis × coefficients), a Perceiver-style cross-attention decoder, and a
coordinate/INR network. For this data-rich, low-input-dimension regression the
plain MLP won **decisively** (~38× lower error than the DeepONet): the fancier
decoders impose inductive biases that only bottleneck capacity here.

After an Optuna hyper-parameter search the best model is an **8×1281 SiLU MLP**:

| metric | value (held-out test set, 225k spectra) |
|---|---|
| R² | 0.99996 |
| per-spectrum median RMSE | 7.2 × 10⁻⁴ (norm. flux) |
| standardised RMSE | 6.6 × 10⁻³ |

Emulated vs. Korg spectra spanning the error range (best → p90), with the
percentage residual under each panel:

![Forward reconstructions](docs/forward_reconstructions.png)

### Cool-star specialist + gate

Residual error concentrates in cool, carbon-rich, high-microturbulence stars
(deep molecular bands). A **specialist** MLP trained only on the cool region is
blended onto the generalist by a **smooth gate on `Teff`**
(`w·specialist + (1−w)·generalist`), cutting cool-region error ~12–13% at
**zero cost to hot stars** and keeping the emulator continuous in parameter
space. The remaining extreme outliers are an *under-resolved-grid* effect (the
spectrum varies faster than the grid samples it at a molecular-band cliff), not
a model limitation — they would only shrink with denser Korg sampling there.

---

## Install

Requires Python ≥ 3.11 and [JAX](https://docs.jax.dev/en/latest/installation.html)
(install the CUDA build for GPU, the CPU build otherwise).

```bash
pip install -e .          # or: uv sync
```

**GPU selection.** On import the training/predict modules call
[`autocvd`](https://pypi.org/project/autocvd/) to pick a free GPU automatically.
Override it by setting `CUDA_VISIBLE_DEVICES=<id>` to pin a device, or
`JAX_PLATFORMS=cpu` to run on CPU (both also skip the auto-selection).

**Data.** The preprocessed cache (`korg_grid_train.h5`) is built on first use
from the raw Korg grid. Point at a prebuilt cache without copying via an
environment variable:

```bash
export BOSS_CACHE_DIR=/path/to/preprocessed_data     # dir holding the .h5 + sidecars
# or BOSS_CACHE_PATH=/path/to/korg_grid_train.h5     # the file itself
```

> **Data access.** The underlying BOSS spectra and the Korg synthesis grid are
> **not public**. They are available to members of the SDSS collaboration via
> the SDSS data access pages ([www.sdss.org](https://www.sdss.org)); a valid
> SDSS collaboration login is required to download them.

## Quickstart

Load a trained checkpoint and predict a spectrum from stellar labels:

```python
import numpy as np
from boss_emulator.emulator import load_checkpoint, predict_spectra

model, payload = load_checkpoint("trained_model/mlp_run/best.pkl")

# one row of labels, in payload["feature_names"] order
#   (Teff [K], logg, m_H, alpha_m, carbon_m, vmic)
labels = np.array([[5200.0, 4.4, -0.3, 0.1, 0.0, 1.0]], dtype="float32")

flux = predict_spectra(model, payload, labels)   # (1, n_pixels) normalised flux
```

The gated two-model emulator (generalist + cool-star specialist) is exposed
through the CLI (`predict.py --specialist ...`); see the Workflow section.

---

## Repository layout

```
src/boss_emulator/
  data.py          grid → cached, standardised, split dataset (+ test-only fast path)
  emulator.py      MLP / build_model / load_checkpoint / predict_spectra
  training.py      device-resident training loop, AdamW warmup-cosine, checkpointing, CLI
  predict.py       forward inference; optional generalist+specialist Teff gate
script/
  tune_mlp.py                       Optuna hyper-parameter search (multi-worker)
  eval_mlp.py                       full test-set evaluation suite (metrics + figures)
  exp_specialist_gate.py            train the cool-star specialist for the gate
  exp_target_transform.py           (experiment) target reparameterisation — negative result
  inv_model/                        inverse modelling: labels from spectra (own README)
```

Trained models land in `trained_model/`, figures in `plots/` — both git-ignored.

---

## Workflow

### Train

```bash
python -m boss_emulator.training --model mlp \
    --labels Teff logg m_H alpha_m carbon_m vmic \
    --hidden 1281 1281 1281 1281 1281 1281 1281 1281 --activation silu \
    --epochs 1000 --batch-size 4096 --outdir trained_model/mlp
```

Abundance-free is the default (`--abundances` empty); add elements to train the
27-input variant. Data is staged resident on the GPU, so epochs are compute-
bound, not host-bound.

### Hyper-parameter search

```bash
BOSS_CACHE_DIR=/path/to/preprocessed_data \
    python script/tune_mlp.py --study-name mlp_hpo --n-trials 100 \
    --epochs 1000 --outdir trained_model/mlp_hpo
```

A shared `JournalStorage` file lets several workers (one per GPU) cooperate on
one study; the objective is per-spectrum median reconstruction RMSE with
`MedianPruner` early-stopping.

### Evaluate

```bash
python script/eval_mlp.py <run_dir> --n-worst 10 --outdir plots/mlp_eval
```

Writes R²/RMSE/SNR metrics plus error-distribution, sample/worst
reconstructions, and Teff–carbon / 6-D label-space error maps.

### Predict (forward)

```bash
# single model
python -m boss_emulator.predict --run <run_dir> --inputs labels.csv --output spectra.h5

# generalist + cool-star specialist, smooth Teff gate
python -m boss_emulator.predict --run <generalist> --specialist <specialist> \
    --inputs labels.csv --output spectra.h5
# reconstruction metrics on the held-out test set:
python -m boss_emulator.predict --run <generalist> --specialist <specialist> \
    --test --test-outdir plots/gated
```

Inputs accept `.npy` / `.csv` (matched by column name) / `.h5`.

### Invert (labels from spectra)

Gradient descent and classical least-squares, sharing one likelihood and
results format, both able to run through the gated forward model, optionally
warm-started by an amortized inverse net. See
[`script/inv_model/README.md`](script/inv_model/README.md).

```bash
python script/inv_model/run_gradient.py --run <generalist> --specialist <specialist> \
    --obs korg --n-targets 512
python script/inv_model/run_gradient_full.py --run <generalist> --specialist <specialist>
```

On the full held-out set (true Korg spectra, SNR 100) the gated emulator
recovers **Teff to ~20 K, [M/H] to ~0.03 dex** at chi²/pixel ≈ 1; `carbon_m`
and `vmic` are the weakly-constrained directions. Recovered vs. true, as a 2D
log-density over all ~225k test spectra:

![Inverse recovery](docs/inverse_recovery.png)
