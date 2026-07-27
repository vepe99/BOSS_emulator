"""Inverse modelling with HMC/NUTS (technique 3 of 3).

The first two techniques return a point estimate plus a *linearised* error bar.
NUTS returns the posterior itself, which is what actually matters here: the
label likelihood of a stellar spectrum is famously non-Gaussian -- Teff/logg
and m_H/alpha_m/carbon_m trade off along curved ridges, and near a grid edge
the posterior is truncated rather than bell-shaped. Any place where the NUTS
credible interval disagrees with the Gauss-Newton sigma from
``run_gradient.py`` is a place where the cheap error bar is lying.

The model, per target:

    s ~ Uniform(0, 1)^k                       (uniform over the label box)
    y ~ Normal(emulator(theta(s)), sigma)     (sigma = 1/snr per pixel)

Sampling happens in the unit-box coordinates, which NumPyro then maps to an
unconstrained space internally -- so the box is respected exactly and the
adapted mass matrix sees comparably-scaled parameters.

Cost. One MCMC per target, because each posterior is its own geometry: with
``--chains`` run vectorised on the GPU, a 6-parameter fit at a few thousand
draws is seconds-to-minutes per spectrum. Keep ``--n-targets`` small (the
default is 16, versus thousands for gradient descent) and treat this as the
method that *audits* the other two on a subsample rather than one you run over
the whole test set.

Initialisation matters more than usual. 4479 pixels at ``sigma = 1/snr`` make
the posterior a needle inside a label box spanning 5000 K in Teff, so a chain
started from a prior draw typically never finds it -- ``r_hat`` in the
thousands and ESS of 1. The default therefore warm-starts every chain from a
batched Adam MAP fit (``--init map``); ``--init prior`` is a deliberate stress
test, not a sensible production setting. If ``r_hat`` stays bad even from the
MAP, raise ``--warmup`` and ``--target-accept`` before trusting anything.

Diagnostics are printed and stored: ``r_hat`` and ESS per target/parameter,
the fraction of divergent transitions, and 1-sigma/2-sigma coverage of the
truth by the credible intervals. Poor coverage with good r_hat means the model
is wrong (emulator misspecification, or an understated sigma), not that the
sampler failed -- which is exactly why ``--obs emulator --noiseless`` is worth
running as a control.

Examples::

    # posterior for 16 test spectra, 4 vectorised chains
    UV_CACHE_DIR=... uv run python script/inv_model/run_nuts.py \
        --run trained_model/payne_hpo/trial_0016/payne_* --n-targets 16

    # well-specified control: is the sampler + parametrisation sound?
    UV_CACHE_DIR=... uv run python script/inv_model/run_nuts.py \
        --run trained_model/payne_hpo/trial_0016/payne_* \
        --obs emulator --noiseless --n-targets 8

autocvd claims a free GPU before JAX initialises (lab policy).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_value


def make_model(problem: common.Problem):
    """NumPyro model for one spectrum, in unit-box coordinates."""
    lo = jnp.asarray(problem.lo, jnp.float32)
    span = jnp.asarray(problem.hi - problem.lo, jnp.float32)

    def model(theta_full, y_obs, sigma):
        s = numpyro.sample(
            "s", dist.Uniform(jnp.zeros(problem.k), jnp.ones(problem.k)))
        mu = problem.flux((lo + span * s)[None], theta_full[None])[0]
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)

    return model


def run_nuts(problem: common.Problem, targets: common.Targets, *,
             warmup: int, samples: int, chains: int, seed: int,
             target_accept: float, max_tree_depth: int, dense_mass: bool,
             init_values: np.ndarray | None):
    """One NUTS run per target. Returns per-target posteriors + diagnostics.

    ``init_values`` (optional, ``(n, k)`` physical) seeds each chain from a MAP
    solution instead of a draw from the prior.

    The kernel is rebuilt for every target, which costs a JAX retrace each
    time. That is not avoidable by hoisting it out of the loop: with
    ``chain_method='vectorized'``, ``NUTS.init`` *mutates* the kernel by
    wrapping its sample function in ``vmap``, so a reused kernel gets
    double-wrapped on the second target and vmap then fails on the scalar
    state fields.
    """
    model = make_model(problem)
    sigma = jnp.asarray(targets.sigma, jnp.float32)
    key = jax.random.key(seed)

    n, k = targets.n, problem.k
    med = np.empty((n, k))
    mean = np.empty((n, k))
    q16 = np.empty((n, k))
    q84 = np.empty((n, k))
    q025 = np.empty((n, k))
    q975 = np.empty((n, k))
    rhat = np.full((n, k), np.nan)
    ess = np.full((n, k), np.nan)
    divergences = np.zeros(n, dtype=np.int64)
    keep = []                                # samples for the corner plots
    t0 = time.time()

    for i in range(n):
        if init_values is None:
            init_strategy = init_to_uniform
        else:
            # A MAP solution can sit exactly on a box edge (Adam projects onto
            # it), and s=0 or 1 maps to -/+inf under Uniform's unconstrained
            # transform, which NumPyro rejects as an invalid initial point.
            s0 = np.clip(problem.to_scaled(init_values[i]), 1e-4, 1 - 1e-4)
            init_strategy = init_to_value(
                values={"s": jnp.asarray(s0, jnp.float32)})
        kernel = NUTS(model, target_accept_prob=target_accept,
                      max_tree_depth=max_tree_depth, dense_mass=dense_mass,
                      init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                    num_chains=chains, chain_method="vectorized",
                    progress_bar=False)
        key, sub = jax.random.split(key)
        mcmc.run(sub, jnp.asarray(targets.theta_true[i], jnp.float32),
                 jnp.asarray(targets.y_obs[i], jnp.float32), sigma,
                 extra_fields=("diverging",))

        by_chain = mcmc.get_samples(group_by_chain=True)["s"]   # (c, d, k)
        st = summary({"s": by_chain}, prob=0.68)["s"]
        rhat[i] = np.asarray(st["r_hat"])
        ess[i] = np.asarray(st["n_eff"])
        extra = mcmc.get_extra_fields()
        if "diverging" in extra:
            divergences[i] = int(np.asarray(extra["diverging"]).sum())

        s = np.asarray(by_chain).reshape(-1, k)
        theta = problem.to_phys(s.astype(np.float64))
        med[i] = np.median(theta, axis=0)
        mean[i] = theta.mean(axis=0)
        q16[i], q84[i] = np.percentile(theta, [16, 84], axis=0)
        q025[i], q975[i] = np.percentile(theta, [2.5, 97.5], axis=0)
        keep.append(theta)

        print(f"[nuts] {i + 1:4d}/{n}  max_rhat={rhat[i].max():.3f}  "
              f"min_ess={ess[i].min():7.0f}  div={divergences[i]:4d}  "
              f"({time.time() - t0:.1f}s elapsed)", flush=True)

    post = {"median": med, "mean": mean, "q16": q16, "q84": q84,
            "q025": q025, "q975": q975, "r_hat": rhat, "ess": ess,
            "divergences": divergences}
    return post, keep, time.time() - t0


def coverage_report(problem: common.Problem, targets: common.Targets,
                    post: dict) -> dict:
    """Fraction of targets whose truth lies inside each credible interval."""
    truth = targets.theta_true_free(problem)
    out = {}
    for name, (a, b), lvl in [("coverage_68", ("q16", "q84"), 0.68),
                              ("coverage_95", ("q025", "q975"), 0.95)]:
        inside = (truth >= post[a]) & (truth <= post[b])
        out[name] = {"nominal": lvl,
                     **{n: float(inside[:, j].mean())
                        for j, n in enumerate(problem.fit_names)},
                     "all_labels": float(inside.mean())}
    return out


def plot_corners(problem: common.Problem, targets: common.Targets,
                 keep: list[np.ndarray], outdir: Path, tag: str,
                 n_show: int, title: str) -> None:
    """Corner plot of the posterior for the first ``n_show`` targets.

    Truth is marked in every panel: a ridge that runs *through* the cross is a
    degeneracy the data cannot break; a ridge that misses it is bias.
    """
    for i in range(min(n_show, len(keep))):
        th = keep[i]
        truth = targets.theta_true_free(problem)[i]
        k = problem.k
        # Panel limits span the samples *and* the truth: when the chains are
        # stuck the sample range collapses, and autoscaling would then put the
        # truth marker off the axis and imply a spurious zoom level.
        rng_lo = np.minimum(th.min(axis=0), truth)
        rng_hi = np.maximum(th.max(axis=0), truth)
        margin = np.maximum(0.05 * (rng_hi - rng_lo), 1e-3 * (problem.hi - problem.lo))
        rng_lo, rng_hi = rng_lo - margin, rng_hi + margin
        fig, axes = plt.subplots(k, k, figsize=(2.3 * k, 2.3 * k))
        for a in range(k):
            for b in range(k):
                ax = axes[a, b]
                if b > a:
                    ax.axis("off")
                    continue
                if a == b:
                    ax.hist(th[:, a], bins=50, color="C0", alpha=0.85)
                    ax.axvline(truth[a], color="C3", lw=1.2)
                    ax.set_yticks([])
                    ax.set_xlim(rng_lo[a], rng_hi[a])
                else:
                    ax.hist2d(th[:, b], th[:, a], bins=45, cmap="viridis")
                    ax.plot(truth[b], truth[a], marker="+", color="C3",
                            ms=10, mew=1.6)
                    ax.set_xlim(rng_lo[b], rng_hi[b])
                    ax.set_ylim(rng_lo[a], rng_hi[a])
                if a == k - 1:
                    ax.set_xlabel(problem.fit_names[b], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if b == 0 and a != 0:
                    ax.set_ylabel(problem.fit_names[a], fontsize=8)
                elif b != 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=6)
        fig.suptitle(f"{title}\ntarget {i} (cache row {targets.rows[i]}), "
                     f"red = truth", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = outdir / f"{tag}_nuts_corner_{i:02d}.png"
        fig.savefig(path, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print("[inv] wrote", path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.set_defaults(n_targets=16)          # NUTS is per-target; keep it small
    p.add_argument("--warmup", type=int, default=1000,
                   help="NUTS warmup (adaptation) draws per chain")
    p.add_argument("--samples", type=int, default=1000,
                   help="post-warmup draws per chain")
    p.add_argument("--chains", type=int, default=4,
                   help="chains, run vectorised on one device (r_hat needs >1)")
    p.add_argument("--nuts-seed", type=int, default=0, dest="nuts_seed")
    p.add_argument("--target-accept", type=float, default=0.8,
                   dest="target_accept",
                   help="NUTS target acceptance; raise toward 0.95 if the run "
                        "reports divergences")
    p.add_argument("--max-tree-depth", type=int, default=10,
                   dest="max_tree_depth",
                   help="each draw costs up to 2**depth emulator "
                        "forward+backward passes, so this is the main runtime "
                        "knob; lower it (7-8) to trade mixing for speed")
    p.add_argument("--diag-mass", action="store_false", dest="dense_mass",
                   help="adapt a diagonal mass matrix instead of a dense one. "
                        "Dense is the default: with only a handful of free "
                        "labels it costs almost nothing, and the Teff/logg and "
                        "m_H/alpha_m/carbon_m directions are strongly enough "
                        "correlated that a diagonal metric mixes badly")
    p.add_argument("--init", choices=["map", "prior"], default="map",
                   help="seed the chains from a quick batched Adam MAP fit "
                        "(default) or from a prior draw. 4479 pixels at "
                        "sigma=1/snr make the posterior a needle in the label "
                        "box, so a prior draw usually leaves the chains stuck "
                        "and is only useful as a deliberate stress test")
    p.add_argument("--map-steps", type=int, default=1000, dest="map_steps",
                   help="Adam steps for --init-from-map")
    p.add_argument("--x64", action="store_true",
                   help="run the sampler in float64. The potential energy is "
                        "chi2/2 over 4479 pixels at 1/sigma^2 ~ 1e4, so its "
                        "gradients are large and stiff; float32 rounding there "
                        "is a plausible cause of bad step-size adaptation. "
                        "Costs GPU throughput -- try it when r_hat refuses to "
                        "come down")
    p.add_argument("--corner", type=int, default=4,
                   help="corner plots for the first N targets (0 = none)")
    p.add_argument("--save-samples", type=int, default=8, dest="save_samples",
                   help="store the raw chains for the first N targets in the "
                        "npz (0 = none; they are the bulky part)")
    args = p.parse_args()
    common.finalize_tag(args)
    if args.x64:
        # Must precede any array creation, hence before load_problem.
        jax.config.update("jax_enable_x64", True)
        print("[nuts] float64 enabled", flush=True)

    problem, ds, test_idx = common.load_problem(args)
    targets = common.draw_targets(args, problem, ds, test_idx)

    init_values = None
    if args.init == "map":
        init_values, map_chi2, _ = common.fit_map(
            problem, targets, steps=args.map_steps, lr=0.01, n_restarts=4,
            init="mid", seed=args.nuts_seed, log_every=0)
        print(f"[nuts] initialising chains from the Adam MAP solution "
              f"(median chi2/pix="
              f"{np.median(map_chi2) / problem.n_pixels:.4f})", flush=True)

    post, keep, secs = run_nuts(
        problem, targets, warmup=args.warmup, samples=args.samples,
        chains=args.chains, seed=args.nuts_seed,
        target_accept=args.target_accept, dense_mass=args.dense_mass,
        max_tree_depth=args.max_tree_depth, init_values=init_values)
    print(f"[nuts] done in {secs:.1f}s for {targets.n} targets "
          f"({secs / targets.n:.1f}s/target)", flush=True)

    # Point estimate = posterior median; error bar = half the 68% interval.
    est = post["median"]
    err = 0.5 * (post["q84"] - post["q16"])
    chi2_rows, _ = common.make_chi2(problem, targets.sigma)
    chi2 = np.asarray(chi2_rows(jnp.asarray(est, jnp.float32),
                                jnp.asarray(targets.theta_true, jnp.float32),
                                jnp.asarray(targets.y_obs, jnp.float32)),
                      np.float64)

    rep = common.recovery_report(problem, targets, est, err)
    common.print_report(rep, "nuts")
    cov_rep = coverage_report(problem, targets, post)
    for name, d in cov_rep.items():
        print(f"[nuts] {name} (nominal {d['nominal']:.2f}): "
              f"all={d['all_labels']:.3f}  " +
              "  ".join(f"{n}={d[n]:.2f}" for n in problem.fit_names),
              flush=True)
    print(f"[nuts] max r_hat over all targets/params = "
          f"{np.nanmax(post['r_hat']):.4f}   min ESS = "
          f"{np.nanmin(post['ess']):.0f}   total divergences = "
          f"{int(post['divergences'].sum())}", flush=True)

    extra = {"seconds": secs, "warmup": args.warmup, "samples": args.samples,
             "chains": args.chains, "target_accept": args.target_accept,
             "init": args.init, "dense_mass": args.dense_mass,
             "max_tree_depth": args.max_tree_depth, "x64": args.x64,
             "snr": args.snr,
             "noiseless": args.noiseless, "coverage": cov_rep,
             "max_r_hat": float(np.nanmax(post["r_hat"])),
             "min_ess": float(np.nanmin(post["ess"])),
             "total_divergences": int(post["divergences"].sum())}
    arrays = {f"post_{k}": v for k, v in post.items()}
    if args.save_samples:
        arrays["samples"] = np.stack(keep[:args.save_samples])
    common.save_results(args.outdir, args.tag, "nuts", problem, targets, est,
                        chi2, rep, err=err, extra=extra, **arrays)

    if not args.no_plots:
        ttl = (f"NUTS ({args.tag}) obs={args.obs} snr={args.snr:g}"
               f"{' noiseless' if args.noiseless else ''}")
        common.plot_recovery(problem, targets, est, err, rep, ttl,
                             args.outdir / f"{args.tag}_nuts_recovery.png")
        common.plot_spectrum_fits(problem, targets, est, chi2, ttl,
                                  args.outdir / f"{args.tag}_nuts_spectra.png")
        if args.corner:
            args.outdir.mkdir(parents=True, exist_ok=True)
            plot_corners(problem, targets, keep, args.outdir, args.tag,
                         args.corner, ttl)


if __name__ == "__main__":
    main()
