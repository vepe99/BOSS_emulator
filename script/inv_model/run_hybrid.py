"""Hybrid discrete-vmic + continuous inference.

The LHS self-consistency test (``run_lhs.py``) showed the inverse estimator's
weak point is **vmic**: it has only 4 grid nodes, a weak spectral signal, and
the amortized net snaps it to the lattice. Continuous gradient descent over
vmic wanders in a near-flat direction and, worse, optimises against emulator
*interpolation* in vmic that has no Korg ground truth -- the grid only ever
sampled 4 vmic values, so the emulator only truly "knows" those.

This pipeline treats vmic honestly as a **discrete** label. For each candidate
vmic node ``v_g`` it fixes ``vmic = v_g`` and fits the other five labels
continuously (projected Adam + Levenberg-Marquardt, every target and every node
batched together on the GPU), then per target:

  * **point estimate** -- the node with the lowest chi2, plus its five
    continuous labels (a profile-likelihood MAP);
  * **discrete posterior** -- ``w_g proportional to exp(-0.5 (chi2_g - min chi2))``,
    the honest weight on each node (spread out when nodes are degenerate).

The continuous optimiser never touches the flat vmic direction, so it is more
stable; vmic is reported on the grid the emulator was actually trained on.

Compared here against the plain continuous GD (all six free) on the same
targets, so the cost of discretising vmic -- any bias it induces on the other
five -- is measured directly. ``--targets lhs`` (default) uses off-grid Latin-
hypercube points (continuous truth, so vmic quantisation is real and visible);
``--targets grid`` uses real test rows (vmic truth already on a node).

Example::

    BOSS_CACHE_DIR=/path/to/preprocessed_data \\
        .venv/bin/python script/inv_model/run_hybrid.py \\
        --run trained_model/payne_hpo/trial_0077/payne_1785288314 \\
        --specialist trained_model/exp_specialist/specialist_1785327806 \\
        --n-targets 4096 --targets lhs

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
from matplotlib.colors import LogNorm

import common
from run_gradient import polish_gauss_newton
from run_lhs import lhs_targets, grid_control_targets
import jax
import jax.numpy as jnp

DISCRETE_LABEL = "vmic"


def fit_at_fixed_node(problem, targets, node_value, node_col, *, steps, lr,
                      n_restarts, polish):
    """Continuous MAP of the free labels with one label pinned to ``node_value``.

    ``problem.fit_idx`` already excludes the discrete label; ``node_col`` is its
    column in the full feature vector. We overwrite that column in a COPY of the
    targets' ``theta_true`` so the fixed value fed to the forward model is the
    candidate node, not the (unknown) truth -- then reuse the standard fitter.
    Returns ``(est_free, chi2)``.
    """
    tt = targets.theta_true.copy()
    tt[:, node_col] = node_value
    pinned = common.Targets(theta_true=tt, y_obs=targets.y_obs,
                            y_clean=targets.y_clean, sigma=targets.sigma,
                            rows=targets.rows, obs_kind=targets.obs_kind)
    est, chi2, _ = common.fit_map(problem, pinned, steps=steps, lr=lr,
                                  n_restarts=n_restarts, init="mid", seed=0,
                                  log_every=0)
    if polish:
        est = polish_gauss_newton(problem, pinned, est, steps=polish)
        chi2_rows, _ = common.make_chi2(problem, targets.sigma)
        chi2 = np.asarray(chi2_rows(
            jnp.asarray(est, jnp.float32), jnp.asarray(tt, jnp.float32),
            jnp.asarray(targets.y_obs, jnp.float32)), np.float64)
    return est, chi2


def continuous_baseline(problem_all, targets, *, steps, lr, n_restarts, polish):
    """Plain GD with all six labels free (the current pipeline)."""
    est, chi2, _ = common.fit_map(problem_all, targets, steps=steps, lr=lr,
                                  n_restarts=n_restarts, init="mid", seed=0,
                                  log_every=0)
    if polish:
        est = polish_gauss_newton(problem_all, targets, est, steps=polish)
        chi2_rows, _ = common.make_chi2(problem_all, targets.sigma)
        chi2 = np.asarray(chi2_rows(
            jnp.asarray(est, jnp.float32),
            jnp.asarray(targets.theta_true, jnp.float32),
            jnp.asarray(targets.y_obs, jnp.float32)), np.float64)
    return est, chi2


def plot_vmic_discrete(nodes, true_vmic, node_hat, w, title, path):
    """Discrete-vmic diagnostics: picked node vs true (with quantisation band),
    posterior-weight heat by true vmic, and node-selection accuracy."""
    nearest = nodes[np.argmin(np.abs(true_vmic[:, None] - nodes[None, :]),
                              axis=1)]
    acc = float(np.mean(node_hat == nearest))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    # Left: picked node value vs true continuous vmic.
    ax[0].scatter(true_vmic, node_hat, s=6, alpha=0.35, color="C0",
                  label="picked node")
    lim = [nodes.min() - 0.3, nodes.max() + 0.3]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    for v in nodes:
        ax[0].axhline(v, color="0.7", lw=0.5)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("true vmic (continuous)")
    ax[0].set_ylabel("chosen vmic node")
    ax[0].set_title(f"node selection  (acc vs nearest node = {acc:.2f})",
                    fontsize=10)

    # Right: mean posterior weight per node, in bins of true vmic.
    edges = np.linspace(true_vmic.min(), true_vmic.max(), 24)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(true_vmic, edges) - 1, 0, len(ctr) - 1)
    wmap = np.zeros((len(nodes), len(ctr)))
    for b in range(len(ctr)):
        m = which == b
        if m.any():
            wmap[:, b] = w[m].mean(axis=0)
    im = ax[1].imshow(wmap, aspect="auto", origin="lower", cmap="viridis",
                      extent=[ctr[0], ctr[-1], -0.5, len(nodes) - 0.5],
                      vmin=0, vmax=1)
    ax[1].set_yticks(range(len(nodes)))
    ax[1].set_yticklabels([f"{v:.2f}" for v in nodes])
    ax[1].set_xlabel("true vmic (continuous)")
    ax[1].set_ylabel("vmic node")
    ax[1].set_title("mean posterior weight per node", fontsize=10)
    fig.colorbar(im, ax=ax[1], label="weight")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[hybrid] wrote", path, flush=True)


def plot_recovery_density(names, truth, est, title, path):
    """Truth-vs-recovered 2D log-density (PDF), one panel per label, for the
    discrete-vmic estimate. Reads cleaner than a scatter at N in the thousands;
    vmic shows as horizontal bands on its grid nodes."""
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.3 * nrow),
                             constrained_layout=True, squeeze=False)
    for j, nm in enumerate(names):
        ax = axes[divmod(j, ncol)]
        t, e = truth[:, j], est[:, j]
        lo, hi = float(t.min()), float(t.max())
        pad = 0.03 * (hi - lo) if hi > lo else 1.0
        lim = [lo - pad, hi + pad]
        h = ax.hist2d(t, e, bins=90, range=[lim, lim], density=True,
                      cmap="viridis", norm=LogNorm())
        ax.plot(lim, lim, color="w", lw=1.0, ls="--", alpha=0.9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        u = " [K]" if nm == "Teff" else ""
        ax.set_xlabel(f"true {nm}{u}", fontsize=9)
        ax.set_ylabel(f"recovered {nm}{u}", fontsize=9)
        d = e - t
        ax.set_title(f"{nm}:  RMSE={np.sqrt(np.mean(d**2)):.3g}  "
                     f"MAE={np.mean(np.abs(d)):.3g}  bias={np.mean(d):+.2g}",
                     fontsize=9, loc="left")
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.02, label="density")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("[hybrid] wrote", path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--targets", choices=["lhs", "grid"], default="lhs",
                   help="off-grid Latin-hypercube (continuous vmic) or on-grid "
                        "test rows (vmic already on a node)")
    p.add_argument("--lhs-seed", type=int, default=0, dest="lhs_seed")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-restarts", type=int, default=4, dest="n_restarts")
    p.add_argument("--polish", type=int, default=20)
    args = p.parse_args()
    args.obs = "emulator"
    if not args.tag:
        args.tag = Path(args.run).parent.name + f"_{args.targets}"
    if args.outdir == Path("plots/inverse_modeling"):
        args.outdir = Path("plots/lhs_selfconsistency/hybrid")

    # Full 6-free problem FIRST: it generates the targets (so vmic is sampled
    # CONTINUOUSLY, off every node -- the whole point) and serves as the
    # continuous baseline. It also gives us ds/test_idx and the vmic nodes.
    args_all = argparse.Namespace(**vars(args))
    args_all.fit_labels = list(common.DEFAULT_FIT_LABELS)       # all 6 incl vmic
    problem_all, ds, test_idx = common.load_problem(args_all)
    node_col = problem_all.names.index(DISCRETE_LABEL)
    nodes = np.unique(ds.X[test_idx][:, node_col]).astype(np.float64)
    print(f"[hybrid] discrete {DISCRETE_LABEL} nodes = "
          f"{np.array2string(nodes, precision=3)}", flush=True)

    if args.targets == "lhs":
        targets, _ = lhs_targets(args, problem_all, ds, test_idx,
                                 seed=args.lhs_seed)
    else:
        targets = grid_control_targets(args, problem_all, ds, test_idx,
                                       seed=args.lhs_seed)
    true_vmic = targets.theta_true[:, node_col]

    # 5-free problem (vmic pinned per node during inference) for the profile.
    args.fit_labels = [l for l in args.fit_labels if l != DISCRETE_LABEL]
    problem, _, _ = common.load_problem(args)

    # --- profile over the discrete nodes --------------------------------
    t0 = time.time()
    ests, chi2s = [], []
    for g, v in enumerate(nodes):
        e, c = fit_at_fixed_node(problem, targets, v, node_col,
                                 steps=args.steps, lr=args.lr,
                                 n_restarts=args.n_restarts, polish=args.polish)
        ests.append(e); chi2s.append(c)
        print(f"[hybrid] node {g+1}/{len(nodes)} vmic={v:.3f}  "
              f"median chi2/pix={np.median(c)/problem.n_pixels:.4f}", flush=True)
    ests = np.stack(ests)           # (G, n, 5)
    chi2s = np.stack(chi2s)         # (G, n)

    best = np.argmin(chi2s, axis=0)                       # (n,) chosen node idx
    n = targets.n
    est5 = ests[best, np.arange(n)]                       # (n, 5) continuous
    chi2_hyb = chi2s[best, np.arange(n)]
    node_hat = nodes[best]                                # (n,) discrete vmic
    # discrete posterior weights over nodes
    dchi = chi2s - chi2s.min(axis=0, keepdims=True)
    w = np.exp(-0.5 * dchi); w /= w.sum(axis=0, keepdims=True)   # (G, n)
    w = w.T                                                       # (n, G)
    print(f"[hybrid] profile over {len(nodes)} nodes done in "
          f"{time.time()-t0:.1f}s", flush=True)

    # assemble a full 6-label estimate (5 continuous + chosen node)
    est6 = np.empty((n, problem.k + 1))
    fit_names6 = list(problem.fit_names) + [DISCRETE_LABEL]
    est6[:, :problem.k] = est5
    est6[:, problem.k] = node_hat

    # --- continuous baseline (all six free) -----------------------------
    # problem_all was loaded up front (it generated the targets).
    base_est, base_chi2 = continuous_baseline(problem_all, targets,
                                              steps=args.steps, lr=args.lr,
                                              n_restarts=args.n_restarts,
                                              polish=args.polish)

    # --- report: per-label RMSE/MAE, hybrid vs continuous ---------------
    truth6 = targets.theta_true[:, problem_all.fit_idx]      # (n, 6) same order
    def stats(est, names, colmap):
        d = est - truth6[:, colmap]
        return {nm: (float(np.sqrt(np.mean(d[:, j]**2))),
                     float(np.mean(np.abs(d[:, j]))))
                for j, nm in enumerate(names)}
    order = list(problem_all.fit_names)
    hyb_map = [order.index(nm) for nm in fit_names6]
    hyb = stats(est6, fit_names6, hyb_map)
    base = stats(base_est, order, list(range(6)))
    print(f"\n[hybrid] {args.targets} targets={n}  "
          f"(RMSE | MAE)   hybrid-discrete-vmic  vs  continuous", flush=True)
    print(f"[hybrid] {'label':10s} {'hyb RMSE':>10s} {'hyb MAE':>10s} "
          f"{'cont RMSE':>10s} {'cont MAE':>10s}", flush=True)
    for nm in order:
        hr, hm = hyb[nm]; br, bm = base[nm]
        print(f"[hybrid] {nm:10s} {hr:10.4g} {hm:10.4g} {br:10.4g} {bm:10.4g}",
              flush=True)
    nearest = nodes[np.argmin(np.abs(true_vmic[:, None] - nodes[None, :]),
                              axis=1)]
    print(f"[hybrid] vmic node-selection accuracy (vs nearest node) = "
          f"{np.mean(node_hat == nearest):.3f}", flush=True)

    # --- save + plot ----------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outdir / f"{args.tag}_hybrid.npz",
        truth6=truth6, est_hybrid=est6, est_continuous=base_est,
        chi2_hybrid=chi2_hyb, chi2_continuous=base_chi2,
        nodes=nodes, node_hat=node_hat, node_weights=w,
        fit_names=np.array(fit_names6), true_vmic=true_vmic)
    import json
    (args.outdir / f"{args.tag}_hybrid.json").write_text(json.dumps(
        {"targets": args.targets, "n": int(n), "nodes": nodes.tolist(),
         "node_accuracy_vs_nearest": float(np.mean(node_hat == nearest)),
         "hybrid": {k: {"rmse": v[0], "mae": v[1]} for k, v in hyb.items()},
         "continuous": {k: {"rmse": v[0], "mae": v[1]}
                        for k, v in base.items()}}, indent=2))
    print(f"[hybrid] wrote {args.outdir / f'{args.tag}_hybrid.npz'} (+ .json)",
          flush=True)
    if not args.no_plots:
        base = (f"({args.tag}) obs=emulator snr={args.snr:g}"
                f"{' noiseless' if args.noiseless else ''}")
        plot_vmic_discrete(
            nodes, true_vmic, node_hat, w, f"discrete vmic {base}",
            args.outdir / f"{args.tag}_vmic_discrete.png")
        plot_recovery_density(
            fit_names6, truth6, est6,
            f"discrete-vmic parameter recovery -- 2D density {base}",
            args.outdir / f"{args.tag}_recovery_density.png")


if __name__ == "__main__":
    main()
