"""Shared machinery for inverse modelling with a trained emulator.

The forward problem this project solves is ``labels -> spectrum``. Inverse
modelling asks the opposite: given an observed spectrum, recover the stellar
labels that produced it. The trained emulator makes that cheap -- it is fast,
and (being a JAX network) differentiable in its inputs -- so the same
likelihood can be attacked three ways:

  * ``run_curve_fit.py`` -- ``scipy.optimize.curve_fit``, one spectrum at a
    time, analytic Jacobian from JAX (classic least squares + covariance).
  * ``run_gradient.py``  -- projected Adam on the chi-square, all spectra and
    all restarts optimised simultaneously on the GPU (MAP point estimate).
  * ``run_nuts.py``      -- NumPyro NUTS, the full posterior with credible
    intervals, r_hat/ESS diagnostics and coverage.

All three share the pieces defined here so their numbers are comparable:

* :func:`load_problem` -- emulator + test split + label bounds + a jitted,
  differentiable forward map.
* :func:`draw_targets` -- pick test spectra, optionally add Gaussian noise;
  these are the "observations" to be inverted.
* :func:`recovery_report`, :func:`plot_recovery`, :func:`plot_spectrum_fits`,
  :func:`save_results` -- one results format (``<tag>_<method>.npz`` plus a
  JSON summary) for every method.

Two modelling choices deserve to be stated up front, because they decide what
a run actually measures:

**How the abundance inputs are handled** (``--abund-mode``). The emulator takes
27 inputs, but they are not 27 degrees of freedom: in the Korg grid

    A(E) = A_sun(E) + m_H + alpha_m * [E is alpha] + carbon_m * [E == C]

*exactly* (verified per run by :func:`abundance_slaving`), so the inputs lie on
a manifold of dimension 6. Three ways to respect that, none of them neutral:

* ``slaved`` (default) -- the 21 abundances are recomputed from the free core
  labels through the relation above. Six free parameters, every query on the
  training manifold, and no information smuggled in.
* ``fixed`` -- abundances pinned at their true values. This *leaks* the answer:
  with ``A(Fe)`` known, ``m_H`` is determined, and only ~6% of the emulator's
  metallicity sensitivity is left on the ``m_H`` input itself (for ``alpha_m``
  and ``carbon_m`` the direct derivative is 3-8x *larger* than the on-manifold
  one, of the opposite sign -- the network split the redundant signal across
  duplicate channels arbitrarily). The fit then crawls along directions the
  grid never varied.
* ``free`` -- all 27 fitted independently. Pure extrapolation off the manifold;
  it fits spectra well and recovers nothing (label RMSE at prior width). Kept
  only so that claim stays checkable.

``--fit-labels`` still chooses which *core* labels float.

**What counts as the data.** ``--obs korg`` (default) inverts the *true* Korg
spectrum, so emulator error enters as model misspecification -- the realistic
setting. ``--obs emulator`` inverts the emulator's own prediction at the true
labels, which removes misspecification entirely and therefore measures pure
identifiability + optimiser/sampler quality. Run both: the gap between them
is the part of the label error that comes from emulator inaccuracy.

autocvd claims a free GPU at import time (lab policy), via
``boss_emulator.training``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boss_emulator import data as data_mod
from boss_emulator import training           # imports autocvd -> claims a GPU
from boss_emulator.emulator import load_checkpoint

import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Labels free by default: the core physical parameters. The 21 abundance
# inputs are nearly degenerate with m_H/alpha_m/carbon_m and are held fixed
# unless the caller asks otherwise.
DEFAULT_FIT_LABELS = list(data_mod.CORE_LABELS)

BATCH = 8192          # rows per forward pass when scoring the whole test set


# =========================================================================
# Problem definition
# =========================================================================

@dataclass
class Problem:
    """A trained emulator packaged as a differentiable forward model.

    Attributes
    ----------
    names : list of str
        All ``n_features`` emulator inputs, in model order.
    fit_names : list of str
        The subset being inferred (the "free" parameters), in model order.
    fit_idx : ndarray of int
        Column indices of ``fit_names`` within ``names``.
    lo, hi : ndarray, shape (k,)
        Physical box bounds for the free parameters (from the loaded split).
    flux : callable
        ``flux(theta_free, theta_full) -> (b, n_pixels)`` physical flux.
        ``theta_free`` is ``(b, k)`` physical values for the free parameters;
        ``theta_full`` is ``(b, n_features)`` supplying the fixed ones. Jitted
        and differentiable in ``theta_free``.
    n_pixels : int
    """

    names: list[str]
    fit_names: list[str]
    fit_idx: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    flux: callable
    n_pixels: int
    abund_mode: str = "slaved"

    @property
    def k(self) -> int:
        return len(self.fit_names)

    # --- scaled coordinates ------------------------------------------------
    # Optimisers and samplers see s in [0, 1]^k rather than raw physical
    # units, so a 4000 K temperature and a 0.3 dex abundance are equally
    # conditioned. Bounds become the unit box.

    def to_scaled(self, theta_free):
        return (theta_free - self.lo) / (self.hi - self.lo)

    def to_phys(self, s):
        return self.lo + (self.hi - self.lo) * s


def _resolve_test_idx(run_dir: Path, ds, tag: str) -> np.ndarray:
    """The run's test rows as indices into ``ds`` (see eval_payne.py).

    ``split_idx.npz`` is numbered in the full-dataset row space; under the
    test-only fast path ``ds`` holds just the test rows, so saved indices are
    translated to local positions through the sorted ``ds.sub_idx``.
    """
    sp = run_dir / "split_idx.npz"
    if not sp.exists():
        return ds.test_idx
    saved = np.load(sp)["test_idx"]
    if ds.sub_idx is None:
        return saved
    pos = np.clip(np.searchsorted(ds.sub_idx, saved), 0, ds.sub_idx.size - 1)
    bad = ds.sub_idx[pos] != saved
    if bad.any():
        raise SystemExit(
            f"[inv] {tag}: {int(bad.sum()):,}/{saved.size:,} of this run's "
            f"test rows are not in the loaded test split -- it was trained "
            f"with a different seed/val_frac/test_frac/max_rows. Re-run with "
            f"--full-data.")
    return pos


# Elements whose abundance is raised by ``alpha_m`` in the Korg grid.
ALPHA_ELEMENTS = ["O", "Mg", "Si", "S", "Ca", "Ti"]

# The three core labels that drive every abundance input.
DRIVER_LABELS = ["m_H", "alpha_m", "carbon_m"]


def abundance_slaving(names: list[str], X: np.ndarray, tol: float = 1e-4):
    """Express every ``A(X)`` input as an exact affine function of the
    ``(m_H, alpha_m, carbon_m)`` triple, and verify that against ``X``.

    The grid was built with

        A(E) = A_sun(E) + m_H + alpha_m * [E is alpha] + carbon_m * [E == C]

    so the emulator's ``n_features`` inputs really live on a manifold of
    dimension ``len(CORE_LABELS)``. The solar zero point is recovered from the
    data rather than hard-coded, which makes every run re-check the relation
    it is about to rely on.

    Returns ``(mask, const, coef, worst)``: ``mask[j]`` marks the abundance
    features, ``const[j] + coef[j] @ [m_H, alpha_m, carbon_m]`` reproduces
    them, and ``worst`` is the largest residual seen (raises if > ``tol``).
    """
    drv = [names.index(c) for c in DRIVER_LABELS]
    mask = np.zeros(len(names), dtype=bool)
    const = np.zeros(len(names), dtype=np.float64)
    coef = np.zeros((len(names), 3), dtype=np.float64)
    worst = 0.0
    for j, nm in enumerate(names):
        if not (nm.startswith("A(") and nm.endswith(")")):
            continue
        e = nm[2:-1]
        c = np.array([1.0,
                      1.0 if e in ALPHA_ELEMENTS else 0.0,
                      1.0 if e == "C" else 0.0])
        resid = X[:, j].astype(np.float64) - X[:, drv].astype(np.float64) @ c
        c0 = float(np.median(resid))
        worst = max(worst, float(np.abs(resid - c0).max()))
        mask[j], const[j], coef[j] = True, c0, c
    if worst > tol:
        raise SystemExit(
            f"[inv] abundances are not an exact affine function of "
            f"{DRIVER_LABELS} in this split (max residual {worst:.3g} > "
            f"{tol:g}) -- --abund-mode slaved is not valid here.")
    return mask, const, coef, worst


def _make_flux(model, payload, fit_idx: np.ndarray, slaving=None):
    """Jitted, differentiable ``(theta_free, theta_full) -> physical flux``.

    With ``slaving`` (the tuple from :func:`abundance_slaving`) the abundance
    inputs are *recomputed* from the free core labels after the free block is
    written, so the query always lies on the grid's 6-D manifold and no
    information leaks in through the redundant channels. Without it the
    abundances keep whatever ``theta_full`` supplied.
    """
    graphdef, state = nnx.split(model)
    x_mean = jnp.asarray(payload["x_mean"], jnp.float32)
    x_std = jnp.asarray(payload["x_std"], jnp.float32)
    y_mean = jnp.asarray(payload["y_mean"], jnp.float32)
    y_std = jnp.asarray(payload["y_std"], jnp.float32)
    cols = jnp.asarray(fit_idx)

    if slaving is None:
        apply_slaving = None
    else:
        mask, const, coef, _ = slaving
        names = list(payload["feature_names"])
        drv = jnp.asarray([names.index(c) for c in DRIVER_LABELS])
        m = jnp.asarray(mask)
        cst = jnp.asarray(const, jnp.float32)
        cf = jnp.asarray(coef, jnp.float32)

        def apply_slaving(theta):
            slaved = cst + theta[:, drv] @ cf.T      # (b, n_features)
            return jnp.where(m, slaved, theta)

    @jax.jit
    def flux(theta_free, theta_full):
        theta = theta_full.at[:, cols].set(theta_free)
        if apply_slaving is not None:
            theta = apply_slaving(theta)
        yn = nnx.merge(graphdef, state)((theta - x_mean) / x_std)
        return yn * y_std + y_mean

    return flux


def _make_gated_flux(gen_model, gen_payload, spec_model, spec_payload,
                     fit_idx: np.ndarray, *, gate_lo, gate_hi, gate_feature):
    """Differentiable gated forward map: a frozen generalist blended with a
    cool-region specialist by a smooth gate on ``gate_feature`` (Teff).

        flux = w * specialist + (1 - w) * generalist,
        w = clip((gate_hi - Teff) / (gate_hi - gate_lo), 0, 1)

    Each sub-model de-normalises with its own stats and returns physical flux,
    so the blend is standardisation-agnostic; the two must share feature layout
    and pixel count. Differentiable in ``theta_free`` -- the gate ramp is
    piecewise-linear in Teff, so gradients flow everywhere except the clipped
    tails (fine for projected Adam / LM). The gate follows the *current* Teff
    estimate, so the forward model is exactly the deployed gated emulator.
    """
    gdef_g, st_g = nnx.split(gen_model)
    gdef_s, st_s = nnx.split(spec_model)
    names = list(gen_payload["feature_names"])
    ti = names.index(gate_feature)
    cols = jnp.asarray(fit_idx)

    def _stats(p):
        return (jnp.asarray(p["x_mean"], jnp.float32),
                jnp.asarray(p["x_std"], jnp.float32),
                jnp.asarray(p["y_mean"], jnp.float32),
                jnp.asarray(p["y_std"], jnp.float32))
    xmg, xsg, ymg, ysg = _stats(gen_payload)
    xms, xss, yms, yss = _stats(spec_payload)
    lo_g, hi_g = float(gate_lo), float(gate_hi)

    @jax.jit
    def flux(theta_free, theta_full):
        theta = theta_full.at[:, cols].set(theta_free)
        w = jnp.clip((hi_g - theta[:, ti]) / (hi_g - lo_g), 0.0, 1.0)[:, None]
        gen = nnx.merge(gdef_g, st_g)((theta - xmg) / xsg) * ysg + ymg
        spec = nnx.merge(gdef_s, st_s)((theta - xms) / xss) * yss + yms
        return w * spec + (1.0 - w) * gen

    return flux


def load_problem(args) -> tuple[Problem, "data_mod.Dataset", np.ndarray]:
    """Load emulator + test split and assemble the inverse problem.

    Returns ``(problem, ds, test_idx)``. The dataset is the same object
    ``eval_payne.py`` builds, so label/flux arrays and the split match the
    forward-model evaluation exactly.
    """
    training.claim_gpu()

    run_dir = Path(args.run)
    model, payload = load_checkpoint(run_dir / "best.pkl")
    names = list(payload["feature_names"])

    # Build the dataset to MATCH the checkpoint, so ds.X columns line up with
    # the emulator's inputs. The config is read from the checkpoint itself, so
    # this works for both the 27-input runs and the abundance-free (6-input)
    # ones without guessing -- guessing wrong silently misaligns fit_idx.
    cfg = payload.get("data_config") or {}
    base = training.build_parser().parse_args([])
    base.model = payload.get("model", "payne")
    base.labels = list(cfg.get("core_labels", data_mod.CORE_LABELS))
    base.abundances = list(cfg.get("abundance_elements",
                                   data_mod.ABUNDANCE_ELEMENTS))
    base.target = cfg.get("target", "norm_flux")
    base.val_frac = cfg.get("val_frac", 0.1)
    base.test_frac = cfg.get("test_frac", 0.1)
    base.seed = cfg.get("seed", 0)
    base.max_rows = cfg.get("max_rows", None)

    if args.full_data:
        ds, _, _ = training.prepare_data(base)
    else:
        ds, _, _ = training.prepare_test_data(base)

    test_idx = _resolve_test_idx(run_dir, ds, args.tag)
    print(f"[inv] {args.tag}: emulator={payload['model']} "
          f"n_features={len(names)} n_pixels={payload['n_pixels']} "
          f"n_test={test_idx.size:,}", flush=True)

    # Optional cool-region specialist -> smooth Teff-gated forward map.
    spec_model = spec_payload = None
    if getattr(args, "specialist", None):
        spec_dir = Path(args.specialist)
        spec_ckpt = spec_dir if spec_dir.is_file() else spec_dir / "best.pkl"
        spec_model, spec_payload = load_checkpoint(spec_ckpt)
        if list(spec_payload["feature_names"]) != names:
            raise SystemExit(
                f"[inv] specialist features {spec_payload['feature_names']} "
                f"!= generalist {names}; cannot gate")
        if spec_payload["n_pixels"] != payload["n_pixels"]:
            raise SystemExit("[inv] specialist/generalist n_pixels differ")
        if args.gate_feature not in names:
            raise SystemExit(f"[inv] --gate-feature {args.gate_feature!r} not "
                             f"in emulator inputs {names}")
        print(f"[inv] gated: specialist={spec_ckpt}  Teff gate "
              f"[{args.gate_lo:.0f}, {args.gate_hi:.0f}] on "
              f"{args.gate_feature!r}", flush=True)

    fit_names = list(args.fit_labels)
    unknown = [n for n in fit_names if n not in names]
    if unknown:
        raise SystemExit(f"[inv] --fit-labels: unknown {unknown}; "
                         f"choose from {names}")

    # Abundance handling. Abundance-free models (no A(X) inputs) have no
    # redundant channels: the free core labels ARE the inputs, so slaving is a
    # no-op and the mode is irrelevant.
    has_abund = any(n.startswith("A(") for n in names)
    mode = args.abund_mode
    if not has_abund:
        mode = "none"
    else:
        listed_abund = [n for n in fit_names if n.startswith("A(")]
        if mode != "free" and listed_abund:
            raise SystemExit(
                f"[inv] --abund-mode {mode} contradicts --fit-labels "
                f"{listed_abund}; use --abund-mode free to fit abundances "
                f"independently.")
        if mode == "free":
            fit_names += [n for n in names
                          if n.startswith("A(") and n not in fit_names]

    slaving = None
    if mode == "slaved":
        missing = [c for c in DRIVER_LABELS if c not in fit_names]
        if missing:
            print(f"[inv] note: {missing} are not free, so their slaved "
                  f"abundances stay at the truth.", flush=True)
        slaving = abundance_slaving(names, ds.X[test_idx])

    fit_idx = np.array([names.index(n) for n in fit_names], dtype=np.int32)

    # Prior/search box: the label range actually spanned by the loaded split,
    # padded a little so a truth sitting on the grid edge is interior.
    Xt = ds.X[test_idx]
    lo = Xt[:, fit_idx].min(axis=0).astype(np.float64)
    hi = Xt[:, fit_idx].max(axis=0).astype(np.float64)
    pad = args.bound_pad * (hi - lo)
    lo, hi = lo - pad, hi + pad
    for n, a, b in zip(fit_names, lo, hi):
        print(f"[inv]   free {n:10s} in [{a:.4g}, {b:.4g}]", flush=True)

    if slaving is not None:
        print(f"[inv] abundances slaved to {DRIVER_LABELS} "
              f"(relation verified, max residual {slaving[3]:.2g})", flush=True)
    else:
        print(f"[inv] abundances: {mode}", flush=True)

    if spec_model is not None:
        flux = _make_gated_flux(model, payload, spec_model, spec_payload,
                                fit_idx, gate_lo=args.gate_lo,
                                gate_hi=args.gate_hi,
                                gate_feature=args.gate_feature)
    else:
        flux = _make_flux(model, payload, fit_idx, slaving)

    problem = Problem(
        names=names, fit_names=fit_names, fit_idx=fit_idx, lo=lo, hi=hi,
        flux=flux, n_pixels=int(payload["n_pixels"]), abund_mode=mode)
    return problem, ds, test_idx


# =========================================================================
# Targets ("observations" to invert)
# =========================================================================

@dataclass
class Targets:
    theta_true: np.ndarray     # (n, n_features) physical, all inputs
    y_obs: np.ndarray          # (n, n_pixels) the data being fitted
    y_clean: np.ndarray        # (n, n_pixels) noise-free version of y_obs
    sigma: np.ndarray          # (n_pixels,) likelihood sigma
    rows: np.ndarray           # provenance: row in the h5 cache
    obs_kind: str              # 'korg' | 'emulator'

    @property
    def n(self) -> int:
        return self.theta_true.shape[0]

    def theta_true_free(self, problem: Problem) -> np.ndarray:
        return self.theta_true[:, problem.fit_idx]


def draw_targets(args, problem: Problem, ds, test_idx: np.ndarray) -> Targets:
    """Pick test spectra and build the observations to be inverted.

    The likelihood sigma is a flat ``1 / snr`` per pixel -- continuum-
    normalised flux is O(1), so ``--snr 100`` is a 1% per-pixel error, the
    scale at which emulator error (median per-spectrum RMSE ~ 1e-2 for the
    Payne runs) starts to compete with photon noise. ``--noiseless`` keeps
    that sigma in the likelihood but does not perturb the data, which
    separates "does noise limit the labels" from "does bias limit them".
    """
    rng = np.random.default_rng(args.target_seed)
    n = min(args.n_targets, test_idx.size)
    sel = np.sort(rng.choice(test_idx.size, size=n, replace=False))
    loc = test_idx[sel]

    theta_true = np.ascontiguousarray(ds.X[loc], dtype=np.float64)
    if args.obs == "korg":
        y_clean = np.ascontiguousarray(ds.Y[loc], dtype=np.float64)
    else:
        y_clean = np.asarray(problem.flux(
            jnp.asarray(theta_true[:, problem.fit_idx], jnp.float32),
            jnp.asarray(theta_true, jnp.float32)), dtype=np.float64)

    sigma = np.full(problem.n_pixels, 1.0 / args.snr, dtype=np.float64)
    y_obs = y_clean if args.noiseless else y_clean + rng.normal(
        0.0, sigma, size=y_clean.shape)

    rows = (ds.row_idx[loc] if ds.row_idx is not None else loc).astype(np.int64)
    print(f"[inv] {n} targets  obs={args.obs}  snr={args.snr:g}"
          f"{'  (noiseless)' if args.noiseless else ''}", flush=True)
    return Targets(theta_true=theta_true, y_obs=y_obs, y_clean=y_clean,
                   sigma=sigma, rows=rows, obs_kind=args.obs)


# =========================================================================
# Likelihood / chi-square helpers
# =========================================================================

def make_chi2(problem: Problem, sigma):
    """Per-row chi-square and its scaled-space total, both jitted.

    ``chi2_rows(theta_free, theta_full, y_obs) -> (b,)``.

    ``total_scaled(s, theta_full, y_obs) -> scalar`` is ``sum_rows chi2 /
    n_pixels`` as a function of the unit-box coordinates. The rows are
    independent, so the gradient of that sum is each row's own gradient --
    which is what lets one Adam run optimise every spectrum at once.
    """
    inv_var = jnp.asarray(1.0 / np.asarray(sigma) ** 2, jnp.float32)
    lo = jnp.asarray(problem.lo, jnp.float32)
    span = jnp.asarray(problem.hi - problem.lo, jnp.float32)

    @jax.jit
    def chi2_rows(theta_free, theta_full, y_obs):
        d = problem.flux(theta_free, theta_full) - y_obs
        return jnp.sum(d ** 2 * inv_var, axis=1)

    @jax.jit
    def total_scaled(s, theta_full, y_obs):
        return jnp.sum(chi2_rows(lo + span * s, theta_full, y_obs)) \
            / problem.n_pixels

    return chi2_rows, total_scaled


def gauss_newton_cov(problem: Problem, theta_free, theta_full, sigma,
                     batch: int = 64) -> np.ndarray:
    """Gauss-Newton parameter covariance at ``theta_free``: ``(J^T W J)^-1``.

    The Hessian of the chi-square is approximated by dropping the second
    derivative of the model (exact for a linear model, standard for least
    squares), which keeps the result positive-definite. With ``k`` free
    parameters the Jacobian costs ``k`` forward-mode passes per row, so this
    is cheap enough to run for every target. Returns ``(n, k, k)``; rows whose
    normal matrix is singular come back as NaN.
    """
    inv_var = jnp.asarray(1.0 / np.asarray(sigma) ** 2, jnp.float32)

    def one_row(tf, tfull):
        J = jax.jacfwd(lambda t: problem.flux(t[None], tfull[None])[0])(tf)
        return J.T @ (J * inv_var[:, None])          # (k, k) normal matrix

    normal = jax.jit(jax.vmap(one_row))
    out = np.empty((theta_free.shape[0], problem.k, problem.k))
    for i in range(0, theta_free.shape[0], batch):
        sl = slice(i, i + batch)
        out[sl] = np.asarray(normal(jnp.asarray(theta_free[sl], jnp.float32),
                                    jnp.asarray(theta_full[sl], jnp.float32)),
                             dtype=np.float64)
    cov = np.full_like(out, np.nan)
    for i, A in enumerate(out):
        try:
            cov[i] = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            pass
    return cov


# =========================================================================
# MAP optimisation (used directly by run_gradient, and as NUTS init)
# =========================================================================

def initial_scaled(problem: Problem, targets: Targets, kind: str, rng,
                   n_restarts: int, guess=None) -> np.ndarray:
    """Starting points in the unit box, shape ``(n_restarts, n, k)``.

    ``mid`` is the box centre (deterministic, and deliberately far from the
    truth); ``random`` is uniform in the box; ``truth`` starts at the true
    labels, which is not a legitimate estimator but isolates "can the
    optimiser stay at the right answer" from "can it find it"; ``amortized``
    starts every target at the ``guess`` produced by an inverse net (a good
    first estimate), which pulls the heavy-tailed stragglers into the right
    basin. Extra restarts are jittered around the guess, and (with >=3) the
    last one is a uniform-random safety net.
    """
    n, k = targets.n, problem.k
    if kind == "mid":
        s = np.full((n_restarts, n, k), 0.5)
        # Only the first restart sits exactly at the centre; the rest are
        # jittered, or every restart would follow the same trajectory.
        if n_restarts > 1:
            s[1:] = rng.uniform(0.0, 1.0, size=(n_restarts - 1, n, k))
        return s
    if kind == "random":
        return rng.uniform(0.0, 1.0, size=(n_restarts, n, k))
    if kind == "truth":
        s0 = problem.to_scaled(targets.theta_true_free(problem))
        s = np.broadcast_to(s0, (n_restarts, n, k)).copy()
        if n_restarts > 1:
            s[1:] = np.clip(s[1:] + rng.normal(0, 0.02, s[1:].shape), 0, 1)
        return s
    if kind == "amortized":
        if guess is None:
            raise SystemExit("[inv] --init amortized needs --amortized-net")
        g = np.clip(np.asarray(guess, np.float64), 0.0, 1.0)   # (n, k) scaled
        s = np.broadcast_to(g, (n_restarts, n, k)).copy()
        if n_restarts > 1:
            s[1:] = np.clip(g[None] + rng.normal(0, 0.05, (n_restarts - 1, n, k)),
                            0.0, 1.0)
            if n_restarts >= 3:                    # one uniform-random safety net
                s[-1] = rng.uniform(0.0, 1.0, size=(n, k))
        return s
    raise SystemExit(f"[inv] unknown --init {kind!r}")


def load_inverse_net(path):
    """Load an amortized inverse net (from ``train_inverse_net.py``).

    Returns ``(guess_phys, core_labels)`` where ``guess_phys(y_obs)`` maps a
    batch of *physical* flux spectra ``(n, n_pixels)`` to *physical* core
    labels ``(n, len(core_labels))``.
    """
    import pickle
    from boss_emulator.emulator import build_model
    with open(path, "rb") as f:
        payload = pickle.load(f)
    model = build_model(payload["model"], payload["n_features"],
                        payload["n_pixels"], seed=payload.get("seed", 0),
                        hidden=tuple(payload["hidden"]),
                        activation=payload["activation"])
    graphdef, pstate, rest = nnx.split(model, nnx.Param, ...)
    nnx.replace_by_pure_dict(pstate, payload["params"])
    model = nnx.merge(graphdef, pstate, rest)
    model.eval()
    core = list(payload["core_labels"])
    lo = np.asarray(payload["lo"], np.float64)
    hi = np.asarray(payload["hi"], np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    ym = jnp.asarray(payload["y_mean"], jnp.float32)
    ys = jnp.asarray(payload["y_std"], jnp.float32)

    def guess_phys(y_obs):
        yn = (jnp.asarray(y_obs, jnp.float32) - ym) / ys
        sc = np.asarray(model(yn), np.float64)        # scaled labels in [0, 1]
        return lo + span * sc

    return guess_phys, core


def amortized_guess_scaled(problem: Problem, guess_phys, y_obs, core_labels):
    """Inverse-net guess for ``y_obs``, as scaled coords in the problem box.

    Maps the net's physical core-label prediction into the free-parameter order
    and the optimiser's unit box, so it can seed :func:`fit_map`.
    """
    missing = [nm for nm in problem.fit_names if nm not in core_labels]
    if missing:
        raise SystemExit(f"[inv] amortized net lacks fit labels {missing}; "
                         f"it predicts {core_labels}")
    col = [core_labels.index(nm) for nm in problem.fit_names]
    phys_fit = np.asarray(guess_phys(y_obs))[:, col]      # (n, k) physical
    return problem.to_scaled(phys_fit)                    # (n, k) scaled


def fit_map(problem: Problem, targets: Targets, *, steps: int, lr: float,
            n_restarts: int, init: str, seed: int, log_every: int = 0,
            guess=None):
    """Projected-Adam MAP fit of every target (and restart) at once.

    Bound handling is projection: an Adam step is taken in the unit box and
    the result clipped back into it. That is simpler and better-behaved near
    the edges than a sigmoid reparametrisation, whose gradients vanish exactly
    where a grid-edge truth lives.

    Returns ``(theta_free, chi2, history)`` -- the best restart per target,
    its chi-square, and the per-step best-chi2 trace for the loss curve.
    """
    rng = np.random.default_rng(seed)
    n, k = targets.n, problem.k
    _, total = make_chi2(problem, targets.sigma)

    # Restarts are just more independent rows in one big batch.
    s0 = initial_scaled(problem, targets, init, rng, n_restarts, guess=guess)
    s = jnp.asarray(s0.reshape(n_restarts * n, k), jnp.float32)
    theta_full = jnp.asarray(np.tile(targets.theta_true, (n_restarts, 1)),
                             jnp.float32)
    y_obs = jnp.asarray(np.tile(targets.y_obs, (n_restarts, 1)), jnp.float32)

    tx = optax.adam(lr)
    opt_state = tx.init(s)

    @jax.jit
    def step(s, opt_state):
        loss, g = jax.value_and_grad(total)(s, theta_full, y_obs)
        upd, opt_state = tx.update(g, opt_state)
        return jnp.clip(optax.apply_updates(s, upd), 0.0, 1.0), opt_state, loss

    history = []
    for it in range(steps):
        s, opt_state, loss = step(s, opt_state)
        if log_every and (it + 1) % log_every == 0:
            history.append({"step": it + 1, "mean_chi2_per_pixel":
                            float(loss) / (n_restarts * n)})
            print(f"[map] step {it + 1:6d}/{steps}  "
                  f"mean chi2/pix={float(loss) / (n_restarts * n):.5f}",
                  flush=True)

    chi2_rows, _ = make_chi2(problem, targets.sigma)
    theta_free = np.asarray(problem.to_phys(np.asarray(s, np.float64)))
    chi2 = np.asarray(chi2_rows(jnp.asarray(theta_free, jnp.float32),
                                theta_full, y_obs), np.float64)

    # Keep the lowest-chi2 restart for each target.
    chi2 = chi2.reshape(n_restarts, n)
    theta_free = theta_free.reshape(n_restarts, n, k)
    best = np.argmin(chi2, axis=0)
    return (theta_free[best, np.arange(n)], chi2[best, np.arange(n)], history)


# =========================================================================
# Reporting
# =========================================================================

def recovery_report(problem: Problem, targets: Targets, est: np.ndarray,
                    err: np.ndarray | None = None) -> dict:
    """Per-label recovery statistics.

    ``est`` is ``(n, k)`` point estimates; ``err`` optional ``(n, k)`` 1-sigma
    uncertainties. Errors are reported both in physical units and as a
    fraction of the prior box width, so labels on wildly different scales can
    be compared. When ``err`` is given, the pull ``(est - truth) / err`` is
    summarised too: its std should be ~1 and the 1-sigma coverage ~68% if the
    uncertainties are honest.
    """
    truth = targets.theta_true_free(problem)
    d = est - truth
    span = problem.hi - problem.lo
    out = {"n": int(targets.n), "obs": targets.obs_kind, "labels": {}}
    for j, name in enumerate(problem.fit_names):
        e = {
            "bias": float(np.mean(d[:, j])),
            "rmse": float(np.sqrt(np.mean(d[:, j] ** 2))),
            "mae": float(np.mean(np.abs(d[:, j]))),
            "median_abs_error": float(np.median(np.abs(d[:, j]))),
            "p90_abs_error": float(np.percentile(np.abs(d[:, j]), 90)),
            "max_abs_error": float(np.max(np.abs(d[:, j]))),
            "rmse_frac_of_range": float(
                np.sqrt(np.mean(d[:, j] ** 2)) / span[j]),
        }
        if err is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                pull = d[:, j] / err[:, j]
            good = np.isfinite(pull)
            e["median_sigma"] = float(np.nanmedian(err[:, j]))
            e["pull_mean"] = float(np.mean(pull[good])) if good.any() else np.nan
            e["pull_std"] = float(np.std(pull[good])) if good.any() else np.nan
            e["coverage_1sigma"] = (float(np.mean(np.abs(pull[good]) <= 1.0))
                                    if good.any() else np.nan)
        out["labels"][name] = e
    return out


def print_report(rep: dict, method: str) -> None:
    hdr = f"{'label':10s} {'bias':>11s} {'rmse':>11s} {'mae':>11s} {'rmse/range':>11s}"
    has_err = "pull_std" in next(iter(rep["labels"].values()))
    if has_err:
        hdr += f" {'pull_std':>9s} {'cov1sig':>8s}"
    print(f"[{method}] recovery over {rep['n']} targets (obs={rep['obs']}):")
    print("[" + method + "] " + hdr)
    for name, e in rep["labels"].items():
        line = (f"{name:10s} {e['bias']:11.4g} {e['rmse']:11.4g} "
                f"{e['mae']:11.4g} {e['rmse_frac_of_range']:11.4g}")
        if has_err:
            line += f" {e['pull_std']:9.3f} {e['coverage_1sigma']:8.3f}"
        print(f"[{method}] " + line, flush=True)


def plot_recovery(problem: Problem, targets: Targets, est: np.ndarray,
                  err: np.ndarray | None, rep: dict, title: str, path: Path):
    """Truth vs recovered, one panel per free label, residual strip below."""
    k = problem.k
    ncol = min(3, k)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(2 * nrow, ncol,
                             figsize=(4.6 * ncol, 5.0 * nrow),
                             gridspec_kw={"height_ratios": [3, 1.4] * nrow},
                             constrained_layout=True, squeeze=False)
    truth = targets.theta_true_free(problem)
    for j, name in enumerate(problem.fit_names):
        r, c = divmod(j, ncol)
        ax, rax = axes[2 * r, c], axes[2 * r + 1, c]
        t, e = truth[:, j], est[:, j]
        if err is not None:
            ax.errorbar(t, e, yerr=err[:, j], fmt="none", ecolor="0.7",
                        lw=0.6, alpha=0.7, zorder=1)
        ax.scatter(t, e, s=8, color="C0", alpha=0.7, zorder=2)
        lim = [problem.lo[j], problem.hi[j]]
        ax.plot(lim, lim, color="k", lw=0.8, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_ylabel(f"recovered {name}", fontsize=9)
        st = rep["labels"][name]
        ax.set_title(f"{name}: bias={st['bias']:.3g}  rmse={st['rmse']:.3g}"
                     f"  ({100 * st['rmse_frac_of_range']:.2f}% of range)",
                     fontsize=9, loc="left")

        rax.axhline(0.0, color="k", lw=0.7)
        rax.scatter(t, e - t, s=7, color="C3", alpha=0.7)
        rax.set_xlim(lim)
        rax.set_ylabel("resid", fontsize=8)
        rax.set_xlabel(f"true {name}", fontsize=9)
        rax.tick_params(labelsize=7)
    for j in range(k, nrow * ncol):          # blank any unused column
        r, c = divmod(j, ncol)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[inv] wrote", path, flush=True)


def plot_spectrum_fits(problem: Problem, targets: Targets, est: np.ndarray,
                       chi2: np.ndarray, title: str, path: Path, n_show: int = 4):
    """Observed vs best-fit spectra spanning the chi2 range, with residuals."""
    order = np.argsort(chi2)
    picks = [(f"best chi2/pix={chi2[order[0]] / problem.n_pixels:.3f}", order[0])]
    for q, nm in [(0.5, "median"), (0.9, "p90"), (1.0, "worst")]:
        i = order[min(int(q * (order.size - 1)), order.size - 1)]
        picks.append((f"{nm} chi2/pix={chi2[i] / problem.n_pixels:.3f}", i))
    picks = picks[:n_show]

    pred = np.asarray(problem.flux(
        jnp.asarray(est[[i for _, i in picks]], jnp.float32),
        jnp.asarray(targets.theta_true[[i for _, i in picks]], jnp.float32)),
        dtype=np.float64)

    pix = np.arange(problem.n_pixels)
    m = len(picks)
    fig, axes = plt.subplots(2 * m, 1, figsize=(14, 3.6 * m), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1] * m},
                             constrained_layout=True)
    for i, (name, row) in enumerate(picks):
        ax, rax = axes[2 * i], axes[2 * i + 1]
        ax.plot(pix, targets.y_obs[row], lw=0.6, color="k", label="observed")
        ax.plot(pix, pred[i], lw=0.6, color="C3", alpha=0.85, label="best fit")
        ax.set_ylabel("norm flux")
        dl = "  ".join(f"{n}: {t:.2f}->{e:.2f}" for n, t, e in zip(
            problem.fit_names, targets.theta_true[row][problem.fit_idx],
            est[row]))
        ax.set_title(f"[{name}]  {dl}", fontsize=8, loc="left")
        ax.legend(loc="lower right", fontsize=8)
        rax.axhline(0.0, color="k", lw=0.6)
        rax.plot(pix, pred[i] - targets.y_obs[row], lw=0.5, color="C0")
        # The noise band: residuals inside it are as good as the data allows.
        rax.axhspan(-targets.sigma[0], targets.sigma[0], color="C2", alpha=0.2,
                    label="1 sigma")
        rax.set_ylabel("resid", fontsize=8)
        rax.legend(loc="upper right", fontsize=7)
        rax.tick_params(labelsize=7)
    axes[-1].set_xlabel("pixel")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[inv] wrote", path, flush=True)


def save_results(outdir: Path, tag: str, method: str, problem: Problem,
                 targets: Targets, est: np.ndarray, chi2: np.ndarray,
                 rep: dict, err: np.ndarray | None = None,
                 extra: dict | None = None, **arrays) -> Path:
    """Write ``<tag>_<method>.npz`` + ``<tag>_<method>.json``.

    Every method writes the same fields, so a later comparison only has to
    load the npz files: ``truth``/``est`` (n, k), ``chi2`` (n,), ``rows`` for
    provenance, plus whatever the method adds (``cov``, ``samples``, ...).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"method": method, "tag": tag,
               "abund_mode": problem.abund_mode,
               "obs_kind": targets.obs_kind, "recovery": rep,
               "chi2_per_pixel_median": float(
                   np.median(chi2) / problem.n_pixels),
               "chi2_per_pixel_p90": float(
                   np.percentile(chi2, 90) / problem.n_pixels),
               **(extra or {})}
    (outdir / f"{tag}_{method}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        outdir / f"{tag}_{method}.npz",
        truth=targets.theta_true_free(problem), est=est, chi2=chi2,
        err=(err if err is not None else np.full_like(est, np.nan)),
        theta_true_full=targets.theta_true, rows=targets.rows,
        fit_names=np.array(problem.fit_names), lo=problem.lo, hi=problem.hi,
        sigma=targets.sigma, abund_mode=problem.abund_mode, **arrays)
    print(f"[inv] wrote {outdir / f'{tag}_{method}.npz'} (+ .json)", flush=True)
    return outdir / f"{tag}_{method}.json"


# =========================================================================
# CLI
# =========================================================================

def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--run", required=True,
                   help="run directory holding best.pkl (+ split_idx.npz); the "
                        "generalist when --specialist is also given")
    p.add_argument("--specialist", default=None,
                   help="optional cool-region specialist run dir or .pkl; when "
                        "given the forward model is the smooth Teff-gated blend "
                        "w*specialist + (1-w)*generalist (must share features)")
    p.add_argument("--gate-lo", type=float, default=3800.0, dest="gate_lo",
                   help="Teff below which the specialist fully takes over")
    p.add_argument("--gate-hi", type=float, default=4800.0, dest="gate_hi",
                   help="Teff above which only the generalist is used")
    p.add_argument("--gate-feature", default="Teff", dest="gate_feature",
                   help="label the gate ramps on (default Teff)")
    p.add_argument("--tag", default=None,
                   help="prefix for output files (default: run dir name)")
    p.add_argument("--outdir", type=Path,
                   default=Path("plots/inverse_modeling"),
                   help="where results and figures go")
    p.add_argument("--fit-labels", nargs="+", default=DEFAULT_FIT_LABELS,
                   dest="fit_labels", metavar="LABEL",
                   help="which emulator inputs to infer; the rest are held at "
                        f"their true values (default: {DEFAULT_FIT_LABELS})")
    p.add_argument("--abund-mode", dest="abund_mode",
                   choices=["slaved", "fixed", "free"], default="slaved",
                   help="what to do with the 21 A(X) inputs, which are an "
                        "exact affine function of m_H/alpha_m/carbon_m in the "
                        "grid. slaved (default): recompute them from the free "
                        "core labels, so queries stay on the grid's 6-D "
                        "manifold. fixed: hold them at the truth -- leaks the "
                        "metallicity and searches off-manifold. free: fit all "
                        "27 independently -- pure extrapolation, kept only for "
                        "comparison")
    p.add_argument("--n-targets", type=int, default=256, dest="n_targets",
                   help="how many test spectra to invert")
    p.add_argument("--target-seed", type=int, default=0, dest="target_seed",
                   help="seed for target selection and observational noise")
    p.add_argument("--obs", choices=["korg", "emulator"], default="korg",
                   help="fit the true Korg spectrum (includes emulator "
                        "misspecification) or the emulator's own prediction "
                        "at the true labels (pure identifiability test)")
    p.add_argument("--snr", type=float, default=100.0,
                   help="per-pixel SNR; likelihood sigma = 1/snr")
    p.add_argument("--noiseless", action="store_true",
                   help="keep sigma in the likelihood but do not perturb the "
                        "data (isolates bias from noise)")
    p.add_argument("--bound-pad", type=float, default=0.02, dest="bound_pad",
                   help="fractional padding on the label box taken from the "
                        "test split's min/max")
    p.add_argument("--full-data", action="store_true",
                   help="load the whole dataset instead of the test-only "
                        "cache (needed for runs whose split differs)")
    p.add_argument("--no-plots", action="store_true",
                   help="write results only, no figures")
    return p


def finalize_tag(args) -> None:
    """Default ``--tag`` to the run's directory name, as eval_payne does."""
    if args.tag:
        return
    d = Path(args.run)
    args.tag = d.parent.name if d.name.startswith(("payne_", "deeponet_",
                                                   "transformer_",
                                                   "singlelambda_")) else d.name
