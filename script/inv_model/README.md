# Inverse modelling with the trained emulator

The emulator maps `labels -> spectrum`. These scripts run it backwards: given
an observed spectrum, recover the labels. Three techniques, one shared
likelihood, one shared results format so they can be compared directly.

| script | technique | scope | gives |
|---|---|---|---|
| `run_curve_fit.py` | `scipy.optimize.curve_fit` / bounded `least_squares`, JAX Jacobian | one spectrum at a time, serial (~1-2 s each) | point estimate + linearised covariance |
| `run_gradient.py` | projected Adam on the chi-square + batched Levenberg-Marquardt polish | all spectra *and* all restarts in one jitted step (~ms each) | MAP estimate + Gauss-Newton covariance |
| `run_nuts.py` | NumPyro NUTS | one MCMC per spectrum (minutes each) | full posterior, credible intervals, r_hat/ESS, coverage |

`common.py` holds everything they share: the emulator wrapped as a jitted
differentiable forward map, target selection and noise, the chi-square, the MAP
optimiser, and the reporting/plotting/serialisation.

## The two choices that decide what a run measures

**Which labels are free** (`--fit-labels`). The emulator takes 27 inputs: the 6
core labels plus 21 element abundances that are largely redundant with
`m_H`/`alpha_m`/`carbon_m`. Fitting all 27 from a single spectrum is badly
degenerate, so by default only the core 6 float and the abundances are held at
their true values. That separates "can we invert the emulator" from "is the
parametrisation identifiable at all".

**What counts as the data** (`--obs`).

* `--obs korg` (default) fits the *true* Korg spectrum, so emulator error
  enters as model misspecification. This is the realistic setting.
* `--obs emulator` fits the emulator's own prediction at the true labels. No
  misspecification, so this is a pure identifiability + optimiser/sampler test.

The gap between the two is the share of the label error attributable to
emulator inaccuracy. Add `--noiseless` to drop observational noise as well:
`--obs emulator --noiseless` is the well-specified control in which a correct
method must return the truth almost exactly.

Noise is a flat `sigma = 1/snr` per pixel (`--snr`, default 100). Continuum-
normalised flux is O(1), so SNR 100 is a 1% per-pixel error -- roughly where
the Payne runs' median reconstruction RMSE (~1e-2) starts to compete with
photon noise.

## Usage

```bash
RUN=trained_model/payne_hpo/trial_0016/payne_1784971665

# fast, large-N: MAP labels for a few thousand test spectra
UV_CACHE_DIR=/export/data/vgiusepp/.uv_cache uv run \
    python script/inv_model/run_gradient.py --run $RUN --n-targets 4096

# classical least squares on a subsample, for cross-checking
UV_CACHE_DIR=... uv run python script/inv_model/run_curve_fit.py \
    --run $RUN --n-targets 64

# posterior audit of a handful of spectra
UV_CACHE_DIR=... uv run python script/inv_model/run_nuts.py \
    --run $RUN --n-targets 16 --chains 4
```

### NUTS runtime

Each draw costs up to `2**max_tree_depth` emulator forward+backward passes over
all 4479 pixels, and the posterior is a needle inside a label box spanning
5000 K in Teff -- so NUTS routinely saturates the tree depth and runs minutes
per spectrum. Levers, in order of effect: `--max-tree-depth 7/8` (halves the
cost per level), `--n-targets`, `--warmup`/`--samples`. Chains are vectorised on
one device, so `--chains 4` is much cheaper than 4x.

Two defaults exist because of that geometry. `--init map` warm-starts every
chain from a batched Adam MAP fit: from a prior draw the chains essentially
never find the mode (`r_hat` in the thousands, ESS of 1). And a dense mass
matrix is adapted by default -- with a handful of free labels it is nearly free,
and the Teff/logg and m_H/alpha_m/carbon_m directions are too correlated for a
diagonal metric (`--diag-mass` to compare).

**Status: NUTS does not yet converge on this posterior.** The scaffolding runs
end to end (sampling, r_hat/ESS, coverage, corner plots, serialisation) but a
smoke run at `--obs korg --snr 100`, 200 warmup + 200 draws, 2 chains, depth 8
returns `r_hat = 666` and `ESS = 1` with zero divergences: the chains freeze at
their starting point and every marginal collapses to a spike. Warmup length and
`--target-accept` are the first things to push; `--x64` is there because the
potential energy is chi2/2 over 4479 pixels at `1/sigma^2 ~ 1e4`, and float32
rounding in gradients that stiff is a plausible cause of the step size never
adapting. Do not read the coverage numbers from a run whose `r_hat` is not ~1.

Note that `--obs emulator --noiseless` is the *slowest* setting for NUTS even
though it is the easiest for the optimisers: with no noise the posterior
collapses to near-zero width and the sampler saturates the tree depth on every
draw. Give it a low `--max-tree-depth` and few targets.

Every run writes, into `--outdir` (default `plots/inverse_modeling/`):

* `<tag>_<method>.npz` -- `truth`, `est`, `err`, `chi2`, `rows` (provenance
  into the h5 cache), the prior box, plus method extras (`cov`, `samples`,
  `post_*`).
* `<tag>_<method>.json` -- per-label bias/RMSE/MAE, RMSE as a fraction of the
  prior range, pull statistics and coverage, timings, config.
* `<tag>_<method>_recovery.png` -- truth vs recovered per label, with residuals.
* `<tag>_<method>_spectra.png` -- observed vs best-fit spectra spanning the
  chi2 range, residuals against the noise band.
* `<tag>_nuts_corner_NN.png` -- posterior corner plots with the truth marked.

## Reading the output

A good spectral fit does not imply good labels. `chi2/pixel ~ 1` with a large
label RMSE means the likelihood has a degenerate ridge (or a second basin) that
the data cannot resolve -- raise `--n-restarts`, and look at the NUTS corner
plots to see whether the ridge passes through the truth (degeneracy) or misses
it (bias).

That is not hypothetical here. In the well-specified control
(`--obs emulator --noiseless`, 64 targets, 8 random restarts, LM polish) the
median fit reaches `chi2/pixel = 3e-5` -- essentially exact -- yet the label
errors are heavy-tailed: Teff MAE 133 K but RMSE 363 K, m_H MAE 0.67 dex but
RMSE 1.1 dex. Most spectra invert cleanly and a minority land on a distant
label vector that reproduces the spectrum just as well. Since there is no noise
and no misspecification in that run, the cause is the emulator's own
label->spectrum map, not the fitting: worth chasing with the corner plots before
reading anything into the `--obs korg` numbers.

`pull_std` (the spread of `(est - truth)/err`) and `coverage_1sigma` are the
honesty check on the error bars. `pull_std >> 1` with `--obs korg` and
`pull_std ~ 1` with `--obs emulator --noiseless` means the error bars are fine
but the emulator is biased; large in both cases means the linearised covariance
itself is wrong, which the NUTS credible intervals will show directly.
