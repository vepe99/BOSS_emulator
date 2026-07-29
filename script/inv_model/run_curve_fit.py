"""Inverse modelling with ``scipy.optimize.curve_fit`` (technique 1 of 3).

The classical approach: for each observed spectrum independently, run a
bounded Levenberg-Marquardt / trust-region least squares over the free labels,
with the emulator as the model function and its exact Jacobian supplied by
``jax.jacfwd``. ``curve_fit`` returns the parameter covariance, so this is the
cheapest of the three methods that also gives uncertainties -- linearised ones,
valid only if the chi-square is near-quadratic at the optimum.

Fitting happens in the unit-box coordinates of :class:`common.Problem` (all
free labels scaled to [0, 1]) so the trust region is isotropic across
parameters that differ by orders of magnitude; the covariance is transformed
back to physical units at the end.

Two costs to keep in mind. It is serial -- one ``least_squares`` per spectrum,
each iteration a separate host->device round trip -- so it is the slowest
method per target by a wide margin, and ``--n-targets`` should stay modest.
And it is local: a single start converges to whichever basin it lands in, so
``--n-restarts`` re-runs from fresh random points and keeps the lowest
chi-square (with ``--init mid`` the first start is the box centre).

Examples::

    # baseline: invert the true Korg spectra at SNR 100
    UV_CACHE_DIR=... uv run python script/inv_model/run_curve_fit.py \
        --run trained_model/payne_hpo/trial_0016/payne_* --n-targets 64

    # identifiability check -- no emulator misspecification, no noise
    UV_CACHE_DIR=... uv run python script/inv_model/run_curve_fit.py \
        --run trained_model/payne_hpo/trial_0016/payne_* \
        --obs emulator --noiseless --n-targets 64

autocvd claims a free GPU before JAX initialises (lab policy).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import least_squares

import common
import jax
import jax.numpy as jnp


def _residual_fns(problem: common.Problem, sigma: np.ndarray):
    """Jitted scaled-space residual and Jacobian for a single spectrum.

    ``least_squares`` minimises ``0.5 * sum(f**2)``, so the residual is the
    *whitened* one, ``(model - data) / sigma``; then ``sum(f**2)`` is exactly
    the chi-square and the Gauss-Newton covariance ``(J^T J)^-1`` needs no
    extra scaling.
    """
    inv_sig = jnp.asarray(1.0 / sigma, jnp.float32)
    lo = jnp.asarray(problem.lo, jnp.float32)
    span = jnp.asarray(problem.hi - problem.lo, jnp.float32)

    def resid(s, theta_full, y_obs):
        pred = problem.flux((lo + span * s)[None], theta_full[None])[0]
        return (pred - y_obs) * inv_sig

    return jax.jit(resid), jax.jit(jax.jacfwd(resid))


def fit_curve_fit(problem: common.Problem, targets: common.Targets, *,
                  n_restarts: int, init: str, seed: int, xtol: float,
                  max_nfev: int, verbose_every: int):
    """Bounded least squares per target; returns ``(est, err, chi2, cov, info)``."""
    resid, jac = _residual_fns(problem, targets.sigma)
    rng = np.random.default_rng(seed)
    starts = common.initial_scaled(problem, targets, init, rng, n_restarts)

    n, k = targets.n, problem.k
    est = np.empty((n, k))
    cov = np.full((n, k, k), np.nan)
    chi2 = np.full(n, np.inf)
    nfev = np.zeros(n, dtype=np.int64)
    span = problem.hi - problem.lo
    t0 = time.time()

    for i in range(n):
        tf = jnp.asarray(targets.theta_true[i], jnp.float32)
        yo = jnp.asarray(targets.y_obs[i], jnp.float32)
        f = lambda s: np.asarray(resid(jnp.asarray(s, jnp.float32), tf, yo),
                                 np.float64)
        jf = lambda s: np.asarray(jac(jnp.asarray(s, jnp.float32), tf, yo),
                                  np.float64)
        for r in range(n_restarts):
            sol = least_squares(
                f, starts[r, i], jac=jf, bounds=(0.0, 1.0), method="trf",
                xtol=xtol, max_nfev=max_nfev)
            nfev[i] += sol.nfev
            c2 = float(2.0 * sol.cost)          # cost = 0.5 * sum(resid**2)
            if c2 < chi2[i]:
                chi2[i] = c2
                est[i] = problem.to_phys(sol.x)
                # Gauss-Newton covariance in scaled space, from the final
                # Jacobian; sigma is already folded into the residual, so this
                # is absolute (curve_fit's absolute_sigma=True convention).
                try:
                    cov_s = np.linalg.inv(sol.jac.T @ sol.jac)
                except np.linalg.LinAlgError:
                    cov_s = np.full((k, k), np.nan)
                cov[i] = cov_s * np.outer(span, span)   # -> physical units
        if verbose_every and (i + 1) % verbose_every == 0:
            dt = time.time() - t0
            print(f"[curve_fit] {i + 1:5d}/{n} targets  "
                  f"chi2/pix={chi2[i] / problem.n_pixels:.4f}  "
                  f"({dt:.1f}s, {dt / (i + 1):.2f}s/target)", flush=True)

    err = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0.0, None))
    info = {"seconds": time.time() - t0, "nfev_mean": float(nfev.mean()),
            "n_restarts": n_restarts, "init": init}
    return est, err, chi2, cov, info


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--n-restarts", type=int, default=1, dest="n_restarts",
                   help="independent starts per target; the lowest chi2 wins")
    p.add_argument("--init", choices=["mid", "random", "truth"], default="mid",
                   help="starting point (see common.initial_scaled)")
    p.add_argument("--fit-seed", type=int, default=0, dest="fit_seed",
                   help="seed for the random starting points")
    p.add_argument("--xtol", type=float, default=1e-10,
                   help="least_squares step-size tolerance")
    p.add_argument("--max-nfev", type=int, default=2000, dest="max_nfev",
                   help="max residual evaluations per start")
    p.add_argument("--log-every", type=int, default=8, dest="log_every",
                   help="progress line every N targets (0 = silent)")
    args = p.parse_args()
    common.finalize_tag(args)

    problem, ds, test_idx = common.load_problem(args)
    targets = common.draw_targets(args, problem, ds, test_idx)

    est, err, chi2, cov, info = fit_curve_fit(
        problem, targets, n_restarts=args.n_restarts, init=args.init,
        seed=args.fit_seed, xtol=args.xtol, max_nfev=args.max_nfev,
        verbose_every=args.log_every)
    print(f"[curve_fit] done in {info['seconds']:.1f}s "
          f"({info['seconds'] / targets.n:.2f}s/target, "
          f"{info['nfev_mean']:.0f} nfev/target)", flush=True)

    rep = common.recovery_report(problem, targets, est, err)
    common.print_report(rep, "curve_fit")

    common.save_results(args.outdir, args.tag, "curve_fit", problem, targets,
                        est, chi2, rep, err=err, extra={**info, "snr": args.snr,
                                                       "noiseless": args.noiseless},
                        cov=cov)
    if not args.no_plots:
        ttl = (f"curve_fit ({args.tag}) obs={args.obs} snr={args.snr:g}"
               f"{' noiseless' if args.noiseless else ''}")
        common.plot_recovery(problem, targets, est, err, rep, ttl,
                             args.outdir / f"{args.tag}_curve_fit_recovery.png")
        common.plot_spectrum_fits(problem, targets, est, chi2, ttl,
                                  args.outdir / f"{args.tag}_curve_fit_spectra.png")


if __name__ == "__main__":
    main()
