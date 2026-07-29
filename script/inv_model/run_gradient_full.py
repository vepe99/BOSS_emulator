"""Batched gradient-descent inversion over the ENTIRE test set, in chunks.

``common.fit_map`` optimises every target (and restart) in one jitted step,
which is ideal up to a few thousand spectra but will not fit the full ~225k
test set in a single shot. This driver runs the *same* Adam + Levenberg-
Marquardt MAP as ``run_gradient.py`` (imported, not re-implemented) over the
whole held-out set in fixed-size chunks and concatenates the results, so every
test spectrum gets a MAP label estimate under identical settings.

Supports the gated forward model (``--specialist`` etc.) exactly like
``run_gradient.py``. Error bars are skipped by default here (the linearised
Gauss-Newton covariance is unreliable on this degenerate posterior anyway --
see the README -- and it is the slow part at 225k rows); pass ``--errors`` to
compute them.

Example::

    BOSS_CACHE_DIR=/export/data/vgiusepp/BOSS_emulator/preprocessed_data \\
        .venv/bin/python script/inv_model/run_gradient_full.py \\
        --run trained_model/payne_hpo/trial_0077/payne_1785288314 \\
        --specialist trained_model/exp_specialist/specialist_1785327806 \\
        --obs korg --chunk-size 8192
"""
from __future__ import annotations

import argparse
import sys
import time
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import common
import run_gradient                     # reuse polish_gauss_newton
import jax.numpy as jnp


def make_targets(args, problem, ds, loc, rng) -> common.Targets:
    """Build observations for the explicit test rows ``loc`` (no subsampling)."""
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
    return common.Targets(theta_true=theta_true, y_obs=y_obs, y_clean=y_clean,
                          sigma=sigma, rows=rows, obs_kind=args.obs)


def slice_targets(t: common.Targets, sl: slice) -> common.Targets:
    return common.Targets(theta_true=t.theta_true[sl], y_obs=t.y_obs[sl],
                          y_clean=t.y_clean[sl], sigma=t.sigma,
                          rows=t.rows[sl], obs_kind=t.obs_kind)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-restarts", type=int, default=4, dest="n_restarts")
    p.add_argument("--init", choices=["mid", "random", "truth"], default="mid")
    p.add_argument("--fit-seed", type=int, default=0, dest="fit_seed")
    p.add_argument("--polish", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=8192, dest="chunk_size",
                   help="test spectra per jitted MAP call (times --n-restarts "
                        "rows are optimised together); lower it if the GPU OOMs")
    p.add_argument("--errors", action="store_true",
                   help="also compute the (unreliable) Gauss-Newton error bars")
    args = p.parse_args()
    common.finalize_tag(args)

    problem, ds, test_idx = common.load_problem(args)
    N = test_idx.size
    nchunks = ceil(N / args.chunk_size)
    print(f"[full] inverting the ENTIRE test set: {N:,} spectra in "
          f"{nchunks} chunks of {args.chunk_size} (x{args.n_restarts} restarts)",
          flush=True)

    rng = np.random.default_rng(args.target_seed)
    full = make_targets(args, problem, ds, test_idx, rng)   # all rows (host)

    est_parts, chi2_parts, err_parts = [], [], []
    t0 = time.time()
    for ci in range(nchunks):
        sl = slice(ci * args.chunk_size, min((ci + 1) * args.chunk_size, N))
        sub = slice_targets(full, sl)
        est, chi2, _ = common.fit_map(
            problem, sub, steps=args.steps, lr=args.lr,
            n_restarts=args.n_restarts, init=args.init, seed=args.fit_seed,
            log_every=0)
        if args.polish:
            est = run_gradient.polish_gauss_newton(
                problem, sub, est, steps=args.polish)
            chi2_rows, _ = common.make_chi2(problem, sub.sigma)
            chi2 = np.asarray(chi2_rows(
                jnp.asarray(est, jnp.float32),
                jnp.asarray(sub.theta_true, jnp.float32),
                jnp.asarray(sub.y_obs, jnp.float32)), np.float64)
        if args.errors:
            cov = common.gauss_newton_cov(problem, est, sub.theta_true,
                                          sub.sigma, batch=256)
            err_parts.append(np.sqrt(np.clip(
                np.diagonal(cov, axis1=1, axis2=2), 0.0, None)))
        est_parts.append(est)
        chi2_parts.append(chi2)
        el = time.time() - t0
        print(f"[full] chunk {ci + 1:3d}/{nchunks}  rows {sl.start:,}:{sl.stop:,}"
              f"  median chi2/pix={np.median(chi2) / problem.n_pixels:.4f}"
              f"  ({el:.0f}s elapsed)", flush=True)

    est = np.concatenate(est_parts)
    chi2 = np.concatenate(chi2_parts)
    err = np.concatenate(err_parts) if err_parts else None
    secs = time.time() - t0
    print(f"[full] done: {N:,} spectra in {secs:.0f}s "
          f"({1000 * secs / N:.2f} ms/spectrum)", flush=True)

    rep = common.recovery_report(problem, full, est, err)
    common.print_report(rep, "gradient_full")

    extra = {"seconds": secs, "steps": args.steps, "lr": args.lr,
             "n_restarts": args.n_restarts, "init": args.init,
             "polish": args.polish, "snr": args.snr,
             "noiseless": args.noiseless, "chunk_size": args.chunk_size,
             "n_spectra": int(N)}
    common.save_results(args.outdir, args.tag, "gradient_full", problem, full,
                        est, chi2, rep, err=err, extra=extra)
    if not args.no_plots:
        ttl = (f"gradient descent, FULL test set ({args.tag}) obs={args.obs} "
               f"snr={args.snr:g}{' noiseless' if args.noiseless else ''}  "
               f"n={N:,}")
        common.plot_recovery(problem, full, est, err, rep, ttl,
                             args.outdir / f"{args.tag}_gradient_full_recovery.png")
        common.plot_spectrum_fits(problem, full, est, chi2, ttl,
                                  args.outdir / f"{args.tag}_gradient_full_spectra.png")


if __name__ == "__main__":
    main()
