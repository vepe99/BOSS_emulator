# Inverse modelling with the trained emulator

The emulator maps `labels -> spectrum`. These scripts run it backwards: given an
observed spectrum, recover the stellar labels. Because the emulator is a
differentiable JAX model, the same chi-square likelihood can be attacked a few
ways, all sharing one likelihood and one results format so they compare
directly.

| script | technique | gives |
|---|---|---|
| `run_gradient.py` | projected Adam on chi-square + batched Levenberg-Marquardt polish (all spectra & restarts at once) | MAP estimate + Gauss-Newton covariance |
| `run_gradient_full.py` | the same, chunked over the **entire** test set | MAP estimate for every test spectrum |
| `run_curve_fit.py` | `scipy` bounded least-squares with a JAX Jacobian, one spectrum at a time | point estimate + linearised covariance |
| `experimental/run_nuts.py` | NumPyro NUTS — **does not converge yet** (see `experimental/`) | (posterior, once it works) |

`common.py` holds everything they share: the emulator wrapped as a jitted,
differentiable forward map, target selection + noise, the chi-square, the MAP
optimiser, and the reporting/plotting/serialisation.

The models are **abundance-free** — the six core labels (`Teff, logg, m_H,
alpha_m, carbon_m, vmic`) are the emulator inputs, so the free parameters are
exactly those labels. `--fit-labels` selects a subset; the rest are held at
truth.

## Two choices that decide what a run measures

**The forward model** (`--specialist`). By default the forward map is a single
generalist emulator (`--run`). Passing `--specialist <run>` makes it the
deployed **gated blend** — `w*specialist + (1-w)*generalist` with `w` ramping
smoothly across `--gate-lo`..`--gate-hi` in Teff — which is more accurate for
cool stars and stays continuous in parameter space (good for the optimiser).

**What counts as the data** (`--obs`). `--obs korg` (default) inverts the *true*
Korg spectrum, so emulator error enters as model misspecification (the realistic
setting). `--obs emulator` inverts the emulator's own prediction at the true
labels — no misspecification, a pure identifiability/optimiser test. The gap
between them is the share of the label error due to emulator inaccuracy. Noise
is a flat `sigma = 1/snr` per pixel (`--snr`, default 100); `--noiseless` keeps
sigma in the likelihood but does not perturb the data.

## Amortized warm-start (recommended)

From a box-centre start the optimiser gets stuck on a minority of spectra (a
heavy label-error tail). An **amortized inverse net** — a supervised
spectrum→labels regressor — gives a near-truth starting point that removes the
tail. Train one, then pass it as the initial guess:

```bash
# train the warm-start net (add --snr 10 to noise-augment for robustness)
BOSS_CACHE_DIR=/path/to/preprocessed_data \
    .venv/bin/python script/inv_model/train_inverse_net.py \
    --epochs 300 --out trained_model/inverse_net/inv.pkl

# gradient-descent inversion, gated forward model, warm-started
.venv/bin/python script/inv_model/run_gradient.py \
    --run <generalist_run> --specialist <specialist_run> \
    --init amortized --amortized-net trained_model/inverse_net/inv.pkl \
    --n-restarts 8 --n-targets 4096

# invert the entire held-out test set (chunked)
.venv/bin/python script/inv_model/run_gradient_full.py \
    --run <generalist_run> --specialist <specialist_run> \
    --init amortized --amortized-net trained_model/inverse_net/inv.pkl \
    --chunk-size 8192
```

## Evaluating a full-test-set inversion

`plot_inverse_eval.py` turns the `<tag>_gradient_full.npz` written by
`run_gradient_full.py` into the inverse-model counterpart of `eval_mlp.py`:
truth-vs-recovered 2D densities, |error| vs Teff for dwarfs and giants, error
maps over (Teff, logg), the χ²/pixel distribution, and a JSON summary per
label and per Teff regime.

```bash
.venv/bin/python script/inv_model/plot_inverse_eval.py \
    plots/<dir>/<tag>_gradient_full.npz --outdir plots/<dir> --tag <tag>
```

## Output

Each run writes into `--outdir` (default `plots/inverse_modeling/`):

* `<tag>_<method>.npz` — `truth`, `est`, `err`, `chi2`, `rows` (provenance into
  the h5 cache), the prior box, plus method extras (`cov`, ...).
* `<tag>_<method>.json` — per-label bias/RMSE/MAE, RMSE as a fraction of the
  prior range, pull statistics, timings, config.
* `<tag>_<method>_recovery.png` — truth vs recovered per label, with residuals.
* `<tag>_<method>_spectra.png` — observed vs best-fit spectra across the chi²
  range.

## Reading the output

A good spectral fit does not imply good labels: `chi2/pixel ~ 1` with a large
label RMSE means a degenerate ridge the data cannot resolve (raise
`--n-restarts`). At BOSS resolution `carbon_m` and `vmic` are the weakly
constrained directions. `pull_std` (spread of `(est-truth)/err`) and
`coverage_1sigma` check the error bars; the Gauss-Newton bars are known to be
optimistic on this posterior, which is what a working NUTS would quantify.
