"""Self-consistency / "learned-regularity" test for the inverse model.

The emulator and the whole inversion have only ever been *evaluated* on
held-out points of the **factorial Korg grid**: the same discrete lattice of
Teff / logg / m_H / ... values, in unseen combinations. Strong recovery there
is ambiguous. It could mean the amortized inverse net learned a genuine,
continuous ``spectrum -> labels`` map -- or that it learned the *grid
regularity* and essentially snaps a spectrum onto the nearest lattice node
(a near-classifier), which would look perfect on-grid and fall apart between
nodes.

A **Latin-hypercube** draw samples the *continuous interior* of the label box,
so every target sits off every grid node. Feeding those through the emulator
and inverting them separates the two hypotheses:

* ``obs=emulator`` -- the observation is the emulator's own forward prediction
  at the (off-grid) LHS labels, so emulator-vs-Korg interpolation error is NOT
  a confound. The forward model used to invert is exactly the one that made the
  data; any label error is identifiability + estimator, not misspecification.
* **Gradient descent** is the control: it minimises chi2 against that same
  forward model, so it must recover the off-grid labels ~perfectly (up to the
  genuine carbon/vmic degeneracies). If GD nails the LHS points but the net
  does not, the gap is precisely the regularity the net leaned on.
* An equal-sized **on-grid** control set (drawn from the real test rows, same
  emulator obs, same noise) is inverted by the same net, so "does the net do
  worse off-grid?" is apples-to-apples.

Diagnostics written to ``--outdir`` (default ``plots/lhs_selfconsistency``):

* ``<tag>_lhs_net.png`` / ``<tag>_lhs_gradient.png`` -- truth vs recovered.
* ``<tag>_regularity.png`` -- per-label predicted-value histograms with grid
  nodes overlaid (snapping shows as spikes on the nodes), the net's off-grid
  vs on-grid error, and net-error vs distance-to-nearest-grid-node.
* ``<tag>_lhs_*.npz`` / ``.json`` -- the usual results bundle per method.

Nothing here is public; it is a local diagnostic.

Example::

    BOSS_CACHE_DIR=/path/to/preprocessed_data \\
        .venv/bin/python script/inv_model/run_lhs.py \\
        --run trained_model/payne_hpo/trial_0077/payne_1785288314 \\
        --specialist trained_model/exp_specialist/specialist_1785327806 \\
        --amortized-net trained_model/inverse_net/inv.pkl --n-targets 4096

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
from run_gradient import polish_gauss_newton
import jax
import jax.numpy as jnp
from scipy.stats import qmc


def lhs_targets(args, problem: common.Problem, ds, test_idx: np.ndarray,
                *, seed: int,
                discrete_labels=("vmic",)) -> tuple[common.Targets, np.ndarray]:
    """Latin-hypercube targets in the grid interior, plus the grid node table.

    Samples ``n`` points uniformly (LHS) inside the per-label [min, max] of the
    *actual* grid (unpadded, so we test interpolation, not extrapolation), then
    generates their spectra with the (gated) emulator. Also returns the list of
    unique grid values per fit label (for the snapping diagnostic).

    **Physically-discrete labels** (``discrete_labels``, default ``vmic``) are
    drawn FROM their grid nodes, not the continuous interval: the training grid
    only ever sampled those values, so the emulator's response between them is
    unvalidated interpolation -- an "observation" at an off-node vmic is not a
    spectrum any real star / Korg run would produce. The remaining labels stay
    continuous (genuinely off-grid), which is what the interpolation test needs.
    """
    Xt = ds.X[test_idx]
    grid_nodes = [np.unique(Xt[:, c]) for c in problem.fit_idx]     # per label
    lo = np.array([g.min() for g in grid_nodes], dtype=np.float64)
    hi = np.array([g.max() for g in grid_nodes], dtype=np.float64)

    n = args.n_targets
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=problem.k, seed=seed)
    unit = sampler.random(n)                                        # (n, k) in [0,1]
    theta_free = lo + (hi - lo) * unit                              # physical
    # Replace discrete labels' continuous draw with a uniform pick over nodes.
    disc = []
    for j, nm in enumerate(problem.fit_names):
        if nm in discrete_labels:
            theta_free[:, j] = rng.choice(grid_nodes[j], size=n)
            disc.append(nm)

    # theta_true carries all n_features; the fit columns hold the LHS values,
    # any non-fit columns are irrelevant (they are overwritten by flux()).
    theta_true = np.zeros((n, len(problem.names)), dtype=np.float64)
    # seed the non-fit columns with a grid-typical value (median) for safety
    for c in range(len(problem.names)):
        theta_true[:, c] = np.median(Xt[:, c])
    theta_true[:, problem.fit_idx] = theta_free

    y_clean = np.asarray(problem.flux(
        jnp.asarray(theta_free, jnp.float32),
        jnp.asarray(theta_true, jnp.float32)), dtype=np.float64)
    sigma = np.full(problem.n_pixels, 1.0 / args.snr, dtype=np.float64)
    y_obs = y_clean if args.noiseless else y_clean + rng.normal(
        0.0, sigma, size=y_clean.shape)

    print(f"[lhs] {n} Latin-hypercube targets (obs=emulator; continuous except "
          f"discrete {disc}) snr={args.snr:g}"
          f"{'  (noiseless)' if args.noiseless else ''}", flush=True)
    tg = common.Targets(theta_true=theta_true, y_obs=y_obs, y_clean=y_clean,
                        sigma=sigma, rows=-np.ones(n, np.int64),
                        obs_kind="emulator")
    return tg, grid_nodes


def grid_control_targets(args, problem: common.Problem, ds,
                         test_idx: np.ndarray, *, seed: int) -> common.Targets:
    """On-grid control: real test rows, emulator obs, same N/noise as the LHS
    set -- so the net's on- vs off-grid error is compared like-for-like."""
    rng = np.random.default_rng(seed)
    n = min(args.n_targets, test_idx.size)
    loc = test_idx[np.sort(rng.choice(test_idx.size, size=n, replace=False))]
    theta_true = np.ascontiguousarray(ds.X[loc], dtype=np.float64)
    y_clean = np.asarray(problem.flux(
        jnp.asarray(theta_true[:, problem.fit_idx], jnp.float32),
        jnp.asarray(theta_true, jnp.float32)), dtype=np.float64)
    sigma = np.full(problem.n_pixels, 1.0 / args.snr, dtype=np.float64)
    y_obs = y_clean if args.noiseless else y_clean + rng.normal(
        0.0, sigma, size=y_clean.shape)
    rows = (ds.row_idx[loc] if ds.row_idx is not None else loc).astype(np.int64)
    return common.Targets(theta_true=theta_true, y_obs=y_obs, y_clean=y_clean,
                          sigma=sigma, rows=rows, obs_kind="emulator")


def net_estimate(problem, targets, guess_phys, core) -> np.ndarray:
    """Amortized-net point estimate (physical, in fit-label order)."""
    s = common.amortized_guess_scaled(problem, guess_phys, targets.y_obs, core)
    return problem.to_phys(np.clip(s, 0.0, 1.0))


def chi2_of(problem, targets, est) -> np.ndarray:
    chi2_rows, _ = common.make_chi2(problem, targets.sigma)
    return np.asarray(chi2_rows(
        jnp.asarray(est, jnp.float32),
        jnp.asarray(targets.theta_true, jnp.float32),
        jnp.asarray(targets.y_obs, jnp.float32)), np.float64)


def dist_to_nearest_node(theta_free, grid_nodes) -> np.ndarray:
    """Per-target, per-label |value - nearest grid node|, normalised by the
    local node spacing (so all labels are comparable)."""
    out = np.empty_like(theta_free)
    for j, nodes in enumerate(grid_nodes):
        d = np.abs(theta_free[:, j][:, None] - nodes[None, :])
        out[:, j] = d.min(axis=1)
        step = np.median(np.diff(nodes)) if nodes.size > 1 else 1.0
        out[:, j] /= step
    return out


def plot_regularity(problem, lhs_tg, net_lhs, gd_lhs, grid_nodes,
                    grid_tg, net_grid, title, path):
    """Three-row diagnostic: (1) predicted-value histograms with grid nodes,
    (2) off- vs on-grid per-label |error|, (3) net |error| vs distance to the
    nearest node (LHS)."""
    k = problem.k
    fig, axes = plt.subplots(3, k, figsize=(3.1 * k, 9.0),
                             constrained_layout=True, squeeze=False)
    truth_lhs = lhs_tg.theta_true[:, problem.fit_idx]
    dist = dist_to_nearest_node(truth_lhs, grid_nodes)
    span = problem.hi - problem.lo

    for j, name in enumerate(problem.fit_names):
        nodes = grid_nodes[j]
        # Row 0: net predictions off-grid; grid nodes as vertical lines. If the
        # net snapped to the lattice, mass piles up on the lines.
        ax = axes[0, j]
        ax.hist(net_lhs[:, j], bins=60, color="C0", alpha=0.8,
                label="net pred (LHS)")
        for v in nodes:
            ax.axvline(v, color="0.6", lw=0.5, alpha=0.7)
        ax.set_title(name, fontsize=10)
        if j == 0:
            ax.set_ylabel("count\n(vlines = grid nodes)", fontsize=8)
            ax.legend(fontsize=7)

        # Row 1: per-label |error| distribution, net off-grid vs on-grid vs GD.
        ax = axes[1, j]
        e_net = np.abs(net_lhs[:, j] - truth_lhs[:, j]) / span[j]
        e_grid = np.abs(net_grid[:, j]
                        - grid_tg.theta_true[:, problem.fit_idx][:, j]) / span[j]
        e_gd = np.abs(gd_lhs[:, j] - truth_lhs[:, j]) / span[j]
        bins = np.linspace(0, np.percentile(
            np.concatenate([e_net, e_grid]), 99) + 1e-9, 50)
        ax.hist(e_grid, bins=bins, color="C2", alpha=0.6, density=True,
                label=f"net on-grid ({np.median(e_grid):.3g})")
        ax.hist(e_net, bins=bins, color="C0", alpha=0.6, density=True,
                label=f"net LHS ({np.median(e_net):.3g})")
        ax.hist(e_gd, bins=bins, color="C3", alpha=0.5, density=True,
                label=f"GD LHS ({np.median(e_gd):.3g})")
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("|err|/range density", fontsize=8)
        ax.legend(fontsize=6, title="median", title_fontsize=6)

        # Row 2: net |error| vs distance to nearest node (LHS). A rising trend
        # = the net relies on being near a lattice node.
        ax = axes[2, j]
        ax.scatter(dist[:, j], np.abs(net_lhs[:, j] - truth_lhs[:, j]),
                   s=4, alpha=0.3, color="C0")
        ax.set_xlabel("dist to nearest node\n(in node steps)", fontsize=8)
        if j == 0:
            ax.set_ylabel("net |error| (physical)", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[lhs] wrote", path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--amortized-net", required=True, dest="amortized_net",
                   help="inverse-net checkpoint (train_inverse_net.py)")
    p.add_argument("--lhs-seed", type=int, default=0, dest="lhs_seed",
                   help="seed for the Latin-hypercube draw + noise")
    p.add_argument("--steps", type=int, default=2000,
                   help="Adam steps for the GD control")
    p.add_argument("--lr", type=float, default=0.01, help="Adam lr (unit box)")
    p.add_argument("--n-restarts", type=int, default=4, dest="n_restarts")
    p.add_argument("--polish", type=int, default=20,
                   help="Levenberg-Marquardt refinement steps after Adam")
    p.add_argument("--gd-init", choices=["mid", "random", "amortized"],
                   default="amortized", dest="gd_init",
                   help="GD start; 'amortized' warm-starts from the net")
    args = p.parse_args()
    # obs is forced to emulator for a self-consistent forward model.
    args.obs = "emulator"
    if not args.tag:
        args.tag = Path(args.run).parent.name
    args.outdir = (args.outdir if args.outdir != Path("plots/inverse_modeling")
                   else Path("plots/lhs_selfconsistency"))

    problem, ds, test_idx = common.load_problem(args)
    guess_phys, core = common.load_inverse_net(args.amortized_net)

    lhs_tg, grid_nodes = lhs_targets(args, problem, ds, test_idx,
                                     seed=args.lhs_seed)
    grid_tg = grid_control_targets(args, problem, ds, test_idx,
                                   seed=args.lhs_seed)

    # --- Amortized net alone (the suspect) -------------------------------
    net_lhs = net_estimate(problem, lhs_tg, guess_phys, core)
    net_grid = net_estimate(problem, grid_tg, guess_phys, core)
    chi2_net = chi2_of(problem, lhs_tg, net_lhs)
    rep_net = common.recovery_report(problem, lhs_tg, net_lhs)
    common.print_report(rep_net, "net-LHS")
    rep_net_grid = common.recovery_report(problem, grid_tg, net_grid)
    common.print_report(rep_net_grid, "net-grid")

    # --- Gradient descent control on the LHS targets ----------------------
    guess = (common.amortized_guess_scaled(problem, guess_phys, lhs_tg.y_obs,
                                           core) if args.gd_init == "amortized"
             else None)
    t0 = time.time()
    gd_lhs, chi2_gd, _ = common.fit_map(
        problem, lhs_tg, steps=args.steps, lr=args.lr,
        n_restarts=args.n_restarts, init=args.gd_init, seed=0,
        log_every=0, guess=guess)
    if args.polish:
        gd_lhs = polish_gauss_newton(problem, lhs_tg, gd_lhs, steps=args.polish)
        chi2_gd = chi2_of(problem, lhs_tg, gd_lhs)
    rep_gd = common.recovery_report(problem, lhs_tg, gd_lhs)
    common.print_report(rep_gd, "gd-LHS")
    print(f"[lhs] GD control done in {time.time()-t0:.1f}s", flush=True)

    # --- Save + plot ------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    common.save_results(args.outdir, args.tag, "lhs_net", problem, lhs_tg,
                        net_lhs, chi2_net, rep_net,
                        extra={"set": "lhs", "estimator": "amortized_net",
                               "snr": args.snr, "noiseless": args.noiseless,
                               "on_grid_recovery": rep_net_grid})
    common.save_results(args.outdir, args.tag, "lhs_gradient", problem, lhs_tg,
                        gd_lhs, chi2_gd, rep_gd,
                        extra={"set": "lhs", "estimator": f"gd_{args.gd_init}",
                               "snr": args.snr, "noiseless": args.noiseless})
    if not args.no_plots:
        base = (f"LHS self-consistency ({args.tag}) obs=emulator snr={args.snr:g}"
                f"{' noiseless' if args.noiseless else ''}")
        common.plot_recovery(problem, lhs_tg, net_lhs, None, rep_net,
                             f"amortized net on LHS -- {base}",
                             args.outdir / f"{args.tag}_lhs_net.png")
        common.plot_recovery(problem, lhs_tg, gd_lhs, None, rep_gd,
                             f"gradient descent on LHS -- {base}",
                             args.outdir / f"{args.tag}_lhs_gradient.png")
        plot_regularity(problem, lhs_tg, net_lhs, gd_lhs, grid_nodes,
                        grid_tg, net_grid,
                        f"learned-regularity diagnostics -- {base}",
                        args.outdir / f"{args.tag}_regularity.png")


if __name__ == "__main__":
    main()
