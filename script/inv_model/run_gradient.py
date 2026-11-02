"""Inverse modelling by batched gradient descent (technique 2 of 3).

The emulator is differentiable in its *inputs*, so the chi-square can be
minimised by the same machinery that trained it. Because every spectrum's
chi-square depends only on its own labels, the gradient of the summed
objective is each target's own gradient -- so one Adam run optimises all
targets, and all restarts, simultaneously in a single jitted step. That makes
this by far the fastest of the three methods: thousands of targets cost about
the same wall clock as one.

Bounds are handled by projection (Adam step, then clip back into the unit
box), which behaves better at grid edges than a sigmoid reparametrisation
whose gradient vanishes exactly there.

Adam is a local method with a fixed step size, so it stops *near* the optimum
rather than at it. Two knobs address that:

  * ``--n-restarts`` runs independent random starts and keeps the lowest
    chi-square per target -- the defence against local minima.
  * ``--polish`` follows up with a few Gauss-Newton/Levenberg-Marquardt steps
    (also fully batched), which converges quadratically once Adam is in the
    right basin and typically drops the label RMSE noticeably.

Uncertainties come from the Gauss-Newton covariance at the optimum -- the same
linearised approximation ``curve_fit`` reports, computed here for every target
at once. Compare them against the NUTS posterior widths to see where the
linearisation breaks down.

Examples::

    # MAP labels for 4096 test spectra
    UV_CACHE_DIR=... uv run python script/inv_model/run_gradient.py \
        --run trained_model/payne_hpo/trial_0016/payne_* --n-targets 4096

    # harder: multi-start from random points, then polish
    UV_CACHE_DIR=... uv run python script/inv_model/run_gradient.py \
        --run trained_model/payne_hpo/trial_0016/payne_* \
        --init random --n-restarts 8 --steps 3000 --polish 30

autocvd claims a free GPU before JAX initialises (lab policy).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import common
import jax
import jax.numpy as jnp


def polish_gauss_newton(problem: common.Problem, targets: common.Targets,
                        theta_free: np.ndarray, *, steps: int,
                        lam0: float = 1e-3) -> np.ndarray:
    """Batched Levenberg-Marquardt refinement of an Adam solution.

    Each step solves the damped normal equations ``(J^T W J + lam*diag) dx =
    -J^T W r`` per target, in scaled coordinates, and accepts the step only
    where the chi-square improved (rejected rows keep their old value and get
    more damping, accepted rows less -- the standard LM trust-region logic,
    vectorised over targets).
    """
    inv_sig = jnp.asarray(1.0 / targets.sigma, jnp.float32)
    lo = jnp.asarray(problem.lo, jnp.float32)
    span = jnp.asarray(problem.hi - problem.lo, jnp.float32)
    theta_full = jnp.asarray(targets.theta_true, jnp.float32)
    y_obs = jnp.asarray(targets.y_obs, jnp.float32)

    def resid_one(s, tfull, yo):
        pred = problem.flux((lo + span * s)[None], tfull[None])[0]
        return (pred - yo) * inv_sig

    @jax.jit
    def lm_step(s, lam):
        r = jax.vmap(resid_one)(s, theta_full, y_obs)              # (n, p)
        J = jax.vmap(jax.jacfwd(resid_one))(s, theta_full, y_obs)  # (n, p, k)
        JtJ = jnp.einsum("npk,npl->nkl", J, J)
        g = jnp.einsum("npk,np->nk", J, r)
        eye = jnp.eye(problem.k)[None]
        A = JtJ + lam[:, None, None] * (eye * jnp.maximum(
            jnp.diagonal(JtJ, axis1=1, axis2=2), 1e-8)[:, None, :])
        dx = jnp.linalg.solve(A, -g[..., None])[..., 0]
        s_new = jnp.clip(s + dx, 0.0, 1.0)
        c_old = jnp.sum(r ** 2, axis=1)
        c_new = jnp.sum(jax.vmap(resid_one)(s_new, theta_full, y_obs) ** 2,
                        axis=1)
        better = c_new < c_old
        s_out = jnp.where(better[:, None], s_new, s)
        lam_out = jnp.where(better, jnp.maximum(lam / 3.0, 1e-9),
                            jnp.minimum(lam * 5.0, 1e6))
        return s_out, lam_out, jnp.where(better, c_new, c_old)

    s = jnp.asarray(problem.to_scaled(theta_free), jnp.float32)
    lam = jnp.full(targets.n, lam0, jnp.float32)
    for it in range(steps):
        s, lam, c = lm_step(s, lam)
    print(f"[polish] {steps} LM steps  median chi2/pix="
          f"{float(jnp.median(c)) / problem.n_pixels:.5f}", flush=True)
    return np.asarray(problem.to_phys(np.asarray(s, np.float64)))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--steps", type=int, default=2000,
                   help="Adam steps (all targets/restarts advance together)")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Adam learning rate, in unit-box coordinates")
    p.add_argument("--n-restarts", type=int, default=4, dest="n_restarts",
                   help="independent starts per target; lowest chi2 wins")
    p.add_argument("--init", choices=["mid", "random", "truth", "amortized"],
                   default="mid", help="starting point (see common.initial_scaled)")
    p.add_argument("--amortized-net", default=None, dest="amortized_net",
                   help="inverse-net checkpoint (train_inverse_net.py) used to "
                        "warm-start when --init amortized")
    p.add_argument("--fit-seed", type=int, default=0, dest="fit_seed",
                   help="seed for the random starting points")
    p.add_argument("--polish", type=int, default=20,
                   help="Levenberg-Marquardt refinement steps after Adam "
                        "(0 disables)")
    p.add_argument("--log-every", type=int, default=200, dest="log_every",
                   help="print the mean chi2 every N Adam steps (0 = silent)")
    p.add_argument("--no-errors", action="store_true", dest="no_errors",
                   help="skip the Gauss-Newton covariance")
    args = p.parse_args()
    common.finalize_tag(args)

    problem, ds, test_idx = common.load_problem(args)
    targets = common.draw_targets(args, problem, ds, test_idx)

    guess = None
    if args.init == "amortized":
        if not args.amortized_net:
            raise SystemExit("[inv] --init amortized needs --amortized-net PATH")
        guess_phys, core = common.load_inverse_net(args.amortized_net)
        guess = common.amortized_guess_scaled(problem, guess_phys,
                                              targets.y_obs, core)
        print(f"[inv] amortized warm-start from {args.amortized_net}", flush=True)

    t0 = time.time()
    est, chi2, history = common.fit_map(
        problem, targets, steps=args.steps, lr=args.lr,
        n_restarts=args.n_restarts, init=args.init, seed=args.fit_seed,
        log_every=args.log_every, guess=guess)
    if args.polish:
        est = polish_gauss_newton(problem, targets, est, steps=args.polish)
        chi2_rows, _ = common.make_chi2(problem, targets.sigma)
        chi2 = np.asarray(chi2_rows(jnp.asarray(est, jnp.float32),
                                    jnp.asarray(targets.theta_true, jnp.float32),
                                    jnp.asarray(targets.y_obs, jnp.float32)),
                          np.float64)
    secs = time.time() - t0
    print(f"[gradient] done in {secs:.1f}s for {targets.n} targets "
          f"({1000 * secs / targets.n:.1f}ms/target)", flush=True)

    err = cov = None
    if not args.no_errors:
        cov = common.gauss_newton_cov(problem, est, targets.theta_true,
                                      targets.sigma)
        err = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0.0, None))

    rep = common.recovery_report(problem, targets, est, err)
    common.print_report(rep, "gradient")

    extra = {"seconds": secs, "steps": args.steps, "lr": args.lr,
             "n_restarts": args.n_restarts, "init": args.init,
             "polish": args.polish, "snr": args.snr,
             "noiseless": args.noiseless, "history": history}
    common.save_results(args.outdir, args.tag, "gradient", problem, targets,
                        est, chi2, rep, err=err, extra=extra,
                        **({"cov": cov} if cov is not None else {}))
    if not args.no_plots:
        ttl = (f"gradient descent ({args.tag}) obs={args.obs} "
               f"snr={args.snr:g}{' noiseless' if args.noiseless else ''}")
        common.plot_recovery(problem, targets, est, err, rep, ttl,
                             args.outdir / f"{args.tag}_gradient_recovery.png")
        common.plot_spectrum_fits(problem, targets, est, chi2, ttl,
                                  args.outdir / f"{args.tag}_gradient_spectra.png")


if __name__ == "__main__":
    main()
