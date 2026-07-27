# Inverse modelling with the trained emulator

The emulator maps `labels -> spectrum`. These scripts run it backwards: given
an observed spectrum, recover the labels. Three techniques, one shared
likelihood, one shared results format so they can be compared directly.

| script | technique | scope | gives |
|---|---|---|---|
| `run_curve_fit.py` | `scipy.optimize.curve_fit` / bounded `least_squares`, JAX Jacobian | one spectrum at a time, serial (~1-2 s each) | point estimate + linearised covariance |
| `run_gradient.py` | projected Adam on the chi-square + batched Levenberg-Marquardt polish | all spectra *and* all restarts in one jitted step (~ms each) | MAP estimate + Gauss-Newton covariance |
| `run_nuts.py` | NumPyro NUTS | one MCMC per spectrum (minutes each) | full posterior, credible intervals, r_hat/ESS, coverage |
| `diagnose_manifold.py` | directional derivatives of the emulator | one pass over N spectra (seconds) | how much label sensitivity the redundant `A(X)` channels absorb |

`common.py` holds everything they share: the emulator wrapped as a jitted
differentiable forward map, target selection and noise, the chi-square, the MAP
optimiser, and the reporting/plotting/serialisation.

## The two choices that decide what a run measures

**How the abundances are handled** (`--abund-mode`). The emulator takes 27
inputs, but they are not 27 degrees of freedom. In the Korg grid

```
A(E) = A_sun(E) + m_H + alpha_m·[E ∈ {O,Mg,Si,S,Ca,Ti}] + carbon_m·[E == C]
```

holds *exactly* — `common.abundance_slaving` re-verifies it against the loaded
split on every run and refuses to proceed if it ever stops holding. So the
inputs lie on a 6-dimensional manifold, and what to do with the 21 redundant
channels is the single most consequential choice in these scripts:

* `slaved` (**default**) — recompute the abundances from the free core labels
  through the relation above. Six free parameters, which is the grid's true
  dimensionality; every query stays on the manifold the emulator was trained
  on; nothing leaks.
* `fixed` — hold them at the truth. This *leaks the answer*: `A(Fe)` known
  means `m_H` known. Measured on 256 test spectra, only 6% of the emulator's
  metallicity sensitivity is left on the `m_H` input once `A(X)` is pinned
  (`|∂flux/∂m_H|` = 5.3e-3 per dex vs 8.9e-2 on-manifold, against a noise floor
  of 1e-2). For `alpha_m` and `carbon_m` the direct derivative is 3-8× *larger*
  than the on-manifold one and partially cancels it — the network distributed
  the redundant signal across the duplicate channels arbitrarily, as a network
  trained on a degenerate parametrisation will.
* `free` — fit all 27 independently. Off-manifold extrapolation: it fits
  spectra (`chi2/pixel = 6e-3` in the noiseless control) and recovers nothing,
  every label landing at prior width. Kept so that claim stays checkable.

The three, on the identical noiseless control (64 targets, 8 restarts, LM
polish), as `rmse` / prior range:

| label | `free` | `fixed` | `slaved` |
|---|---|---|---|
| Teff | 0.214 | 0.0672 | **0.0041** (22 K) |
| logg | 0.230 | 0.0651 | **0.0265** |
| m_H | 0.408 | 0.307 | **0.0226** (0.082 dex) |
| alpha_m | 0.237 | 0.0806 | **0.0177** |
| carbon_m | 0.294 | 0.123 | **0.0908** |
| vmic | 0.303 | 0.228 | **0.131** |

`--fit-labels` still selects which *core* labels float.

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

# what the redundant abundance channels are worth: same control, three modes
for m in slaved fixed free; do
    UV_CACHE_DIR=... uv run python script/inv_model/run_gradient.py \
        --run $RUN --tag ctl_$m --abund-mode $m \
        --obs emulator --noiseless --n-targets 64 --n-restarts 8
done
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

That is not hypothetical here, and it had a mundane cause worth recording. The
first version of these scripts held the abundances fixed at truth, and in the
well-specified control the median fit reached `chi2/pixel = 3e-5` -- essentially
exact -- while the label errors were heavy-tailed: Teff MAE 133 K but RMSE
363 K, m_H MAE 0.67 dex but RMSE 1.1 dex. That looked like a degeneracy in the
emulator's own label->spectrum map, since the run had neither noise nor
misspecification.

It was not. It was the off-manifold parametrisation described above: with
`A(X)` pinned, the free `m_H` input carries almost none of the metallicity
signal, so the chi-square is nearly flat along it and a minority of spectra
drift far away at no cost. Switching to `--abund-mode slaved` cuts m_H RMSE
from 1.116 to 0.082 dex and Teff from 363 to 22 K with everything else
identical. **Diagnose the parametrisation before blaming the emulator.**

`pull_std` (the spread of `(est - truth)/err`) and `coverage_1sigma` are the
honesty check on the error bars. `pull_std >> 1` with `--obs korg` and
`pull_std ~ 1` with `--obs emulator --noiseless` means the error bars are fine
but the emulator is biased; large in both cases means the linearised covariance
itself is wrong, which the NUTS credible intervals will show directly.
