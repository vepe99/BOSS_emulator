"""Where does the emulator keep its metallicity sensitivity?

The Korg grid satisfies, exactly,

    A(E) = A_sun(E) + m_H + alpha_m * [E is alpha] + carbon_m * [E == C]

so the emulator's 27 inputs lie on a 6-D manifold and the 21 abundance inputs
duplicate information already carried by m_H / alpha_m / carbon_m. A network
trained on that parametrisation is free to split each duplicated signal across
the redundant channels however it likes -- and it does.

This script measures the split, as two directional derivatives of the flux:

  partial : d flux / d l holding the 21 A(X) fixed. This is the direction an
            inverse fit moves along under ``--abund-mode fixed``.
  total   : d flux / d l with A(X) slaved to l, i.e. along the manifold. The
            only direction the emulator ever saw varying in training, and what
            ``--abund-mode slaved`` uses.

Their ratio is the justification for the ``slaved`` default: a ratio far below
1 means the free label is nearly inert once the abundances are pinned (the fit
leans on the leaked channels instead), and a ratio above 1 means the direct and
duplicated responses partially cancel on the manifold.

Both are reported as an RMS-over-pixels flux change per unit label, medianed
over spectra, so they compare directly with the noise floor sigma = 1/snr.

    UV_CACHE_DIR=... uv run python script/inv_model/diagnose_manifold.py \
        --run trained_model/payne_hpo/trial_0016/payne_* --n-targets 256
"""
from __future__ import annotations

import argparse

import numpy as np

import common                      # claims a GPU on import (lab policy)

import jax
import jax.numpy as jnp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.set_defaults(n_targets=256)
    args = p.parse_args()
    common.finalize_tag(args)
    # Every input must be in the free block so the forward map is
    # differentiable in all 27 directions; the slaving is applied by hand here.
    args.abund_mode = "free"
    problem, ds, test_idx = common.load_problem(args)
    targets = common.draw_targets(args, problem, ds, test_idx)

    names = problem.names
    mask, const, coef, worst = common.abundance_slaving(names, ds.X[test_idx])
    print(f"[diag] abundance relation verified, max residual {worst:.2g}")

    theta = jnp.asarray(targets.theta_true, jnp.float32)

    def flux_full(th):
        return problem.flux(th[:, problem.fit_idx], th)

    def rms_dirderiv(vec: np.ndarray) -> float:
        """Median over spectra of the RMS-over-pixels flux change per unit."""
        v = jnp.broadcast_to(jnp.asarray(vec, jnp.float32), theta.shape)
        _, d = jax.jvp(flux_full, (theta,), (v,))
        return float(np.median(np.sqrt(np.mean(np.asarray(d) ** 2, axis=1))))

    print(f"\n[diag] {targets.n} test spectra, {problem.n_pixels} pixels")
    print(f"\n{'label':>9} {'A(X) fixed':>12} {'on-manifold':>13} {'ratio':>8}")
    for lab in common.DRIVER_LABELS:
        j = names.index(lab)
        direct = np.zeros(len(names))
        direct[j] = 1.0
        # On-manifold tangent: the label plus every abundance it drives.
        tangent = direct.copy()
        tangent[mask] = coef[mask, common.DRIVER_LABELS.index(lab)]
        rp, rt = rms_dirderiv(direct), rms_dirderiv(tangent)
        print(f"{lab:>9} {rp:12.3e} {rt:13.3e} {rp / rt:8.3f}")

    print(f"\n[diag] noise floor: sigma = 1/snr = {1.0 / args.snr:.3e} per pixel")


if __name__ == "__main__":
    main()
