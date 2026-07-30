"""Experiment: a cool-star SPECIALIST + smooth label-GATE on top of the
frozen best generalist (trial 77), to improve the region where the emulator
fails without touching the 95% it already nails.

Rationale (see the target-transform + autopsy memories)
-------------------------------------------------------
The error concentrates in cool, carbon-rich, high-microturbulence stars whose
spectra are heavily molecule-veiled -- a small, physically-extreme, grid-edge
region. Reparameterising the target did NOT help and hurt the bulk. So instead
of one model that must be good everywhere, we keep the excellent generalist
frozen and add a **specialist** that spends all its capacity on the cool
region, then blend the two by a **smooth gate on Teff** so the combined
emulator stays continuous in parameter space (important for downstream
gradient-based spectral fitting -- a hard switch would inject a discontinuity).

    output(labels) = w(Teff)*specialist + (1-w(Teff))*generalist
    w(Teff) = clip((gate_hi - Teff) / (gate_hi - gate_lo), 0, 1)

Bulk-safe by construction: for Teff >= gate_hi, w=0 => exactly the generalist,
so hot stars are untouched. Acceptance rule: the gated model's full-set median
flux RMSE must stay <= the generalist's, while the cool region improves.

The specialist is WARM-STARTED from the generalist (inherits good features)
and then trained only on rows with Teff <= region_teff_max, using the SAME
per-pixel flux standardisation as the generalist so their outputs blend
consistently in flux space.

Example
-------
    PYTHONUNBUFFERED=1 BOSS_CACHE_DIR=/export/data/vgiusepp/BOSS_emulator/preprocessed_data \\
        .venv/bin/python script/exp_specialist_gate.py \\
        --generalist trained_model/payne_hpo/trial_0077/payne_1785288314 \\
        --epochs 400
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from boss_emulator import data as data_mod, training
from boss_emulator.emulator import load_checkpoint
from boss_emulator.training import save_checkpoint

MIN_FLUX = 1e-2


def gate_weight(teff, lo, hi):
    """Smooth specialist weight: 1 for Teff<=lo, 0 for Teff>=hi, linear between."""
    return np.clip((hi - teff) / (hi - lo), 0.0, 1.0).astype("float32")


def predict_flux(model, Xn, idx, ym, ys, batch):
    """De-standardised flux predictions for rows ``idx`` (host float32 array)."""
    out = np.empty((idx.size, ym.shape[0]), np.float32)
    w = 0
    for b in data_mod.iter_batches(idx, batch, shuffle=False):
        pred = np.asarray(model(jnp.asarray(Xn[b])) * ys + ym)
        out[w:w + b.size] = pred
        w += b.size
    return out


def per_spectrum_rmse(pred_flux, true_flux):
    d = pred_flux - true_flux
    return np.sqrt((d ** 2).mean(axis=1)).astype("float32")


def regime_report(tag, persp, T, C, G):
    print(f"\n--- {tag}: error vs regime ---", flush=True)
    for name, m in [("ALL", np.ones(persp.size, bool)),
                    ("Teff<=4000", T <= 4000),
                    ("Teff<=3500", T <= 3500),
                    ("Teff<=3500 & C>=0.5", (T <= 3500) & (C >= 0.5)),
                    ("Teff<=3000 & C>=0.5 & logg<=0.5",
                     (T <= 3000) & (C >= 0.5) & (G <= 0.5))]:
        if m.any():
            print(f"  {name:34s} n={m.sum():>7,}  med={np.median(persp[m]):.4e}  "
                  f"p99={np.percentile(persp[m],99):.4e}  max={persp[m].max():.4e}",
                  flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generalist", required=True,
                   help="run dir of the frozen generalist (holds best.pkl)")
    p.add_argument("--region-teff-max", type=float, default=5000.0,
                   dest="region_teff_max",
                   help="specialist trains on rows with Teff <= this")
    p.add_argument("--gate-lo", type=float, default=3800.0, dest="gate_lo")
    p.add_argument("--gate-hi", type=float, default=4800.0, dest="gate_hi")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4096, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="peak LR for the (warm-started) specialist")
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--eval-batch-size", type=int, default=8192, dest="eval_batch_size")
    p.add_argument("--outdir", default="trained_model/exp_specialist")
    p.add_argument("--n-worst", type=int, default=10, dest="n_worst")
    args = p.parse_args()

    training.claim_gpu()
    gen_dir = Path(args.generalist)
    outdir = Path(args.outdir) / f"specialist_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[spec] writing to {outdir}", flush=True)

    # --- Data: full split, same config/standardisation as the generalist -----
    base = training.build_parser().parse_args([])
    base.model = "mlp"; base.labels = list(data_mod.CORE_LABELS)
    base.abundances = []; base.val_frac = 0.1; base.test_frac = 0.1
    base.seed = 0; base.max_rows = None
    ds, Xn, Yn = training.prepare_data(base)
    Yflux = ds.Y
    y_mean = np.asarray(ds.y_mean, np.float32)
    y_std = np.asarray(ds.y_std, np.float32)
    ym = jnp.asarray(y_mean); ys = jnp.asarray(y_std)
    y_std_dev = jnp.asarray(ds.y_std)
    feat = ds.feature_names
    ci = {k: feat.index(k) for k in ("Teff", "carbon_m", "logg")}

    # --- Generalist (frozen) --------------------------------------------------
    gen_model, gpay = load_checkpoint(gen_dir / "best.pkl")
    gen_model.eval()
    # sanity: the generalist must share our standardisation for the blend
    assert np.allclose(np.asarray(gpay["y_mean"], np.float32), y_mean, atol=1e-4), \
        "generalist y_mean != dataset y_mean (different split/config)"
    # The specialist has the SAME architecture as the generalist (warm-started
    # from it), so save_checkpoint must record THAT arch, not the parser
    # defaults -- otherwise reload rebuilds the wrong shape. Copy it into `base`.
    base.hidden = list(gpay["hidden"])
    base.dropout = gpay.get("dropout", 0.0)
    _mk = gpay.get("model_kwargs") or {}
    base.activation = _mk.get("activation", base.activation)
    base.residual = _mk.get("residual", base.residual)
    base.layernorm = _mk.get("layernorm", base.layernorm)

    # --- Specialist: warm-start from the generalist, train on the cool region -
    spec_model, _ = load_checkpoint(gen_dir / "best.pkl")
    spec_model.train()

    Ttr = ds.X[ds.train_idx][:, ci["Teff"]]
    reg_tr = ds.train_idx[Ttr <= args.region_teff_max]
    Tva = ds.X[ds.val_idx][:, ci["Teff"]]
    reg_va = ds.val_idx[Tva <= args.region_teff_max]
    print(f"[spec] specialist train rows (Teff<={args.region_teff_max:.0f}): "
          f"{reg_tr.size:,}  val rows: {reg_va.size:,}", flush=True)

    Xtr = jax.device_put(Xn[reg_tr]); Ytr = jax.device_put(Yn[reg_tr])
    Xva = jax.device_put(Xn[reg_va]); Yva = jax.device_put(Yn[reg_va])
    tr_pos = np.arange(reg_tr.size); va_pos = np.arange(reg_va.size)

    steps = max(1, reg_tr.size // args.batch_size)
    total = steps * args.epochs
    sched = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1, peak_value=args.lr,
        warmup_steps=min(300, total // 20 + 1), decay_steps=total,
        end_value=args.lr * 0.01)
    opt = nnx.Optimizer(spec_model, optax.adamw(sched, weight_decay=args.weight_decay),
                        wrt=nnx.Param)

    best = np.inf
    for epoch in range(args.epochs):
        t0 = time.time()
        for b in data_mod.iter_batches(tr_pos, args.batch_size, seed=epoch):
            training.train_step(spec_model, opt, Xtr[b], Ytr[b])
        spec_model.eval()
        _, persp = training.evaluate(spec_model, Xva, Yva, va_pos, y_std_dev,
                                     args.batch_size)
        spec_model.train()
        obj = float(np.median(persp))
        if obj < best:
            best = obj
            save_checkpoint(outdir / "best.pkl", spec_model, ds, base)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"[spec] epoch {epoch+1:4d}/{args.epochs}  "
                  f"region_val_medRMSE={obj:.4e}  best={best:.4e}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # reload best specialist for evaluation
    spec_model, _ = load_checkpoint(outdir / "best.pkl")
    spec_model.eval()

    # --- Gated evaluation on the test set ------------------------------------
    idx = ds.test_idx
    T = ds.X[idx][:, ci["Teff"]]; C = ds.X[idx][:, ci["carbon_m"]]
    G = ds.X[idx][:, ci["logg"]]
    true = Yflux[idx]
    gen_flux = predict_flux(gen_model, Xn, idx, ym, ys, args.eval_batch_size)
    spec_flux = predict_flux(spec_model, Xn, idx, ym, ys, args.eval_batch_size)
    w = gate_weight(T, args.gate_lo, args.gate_hi)[:, None]
    gated_flux = w * spec_flux + (1.0 - w) * gen_flux

    persp_gen = per_spectrum_rmse(gen_flux, true)
    persp_gated = per_spectrum_rmse(gated_flux, true)

    def summ(persp):
        return {
            "median": float(np.median(persp)),
            "mean": float(persp.mean()),
            "p90": float(np.percentile(persp, 90)),
            "p99": float(np.percentile(persp, 99)),
            "max": float(persp.max()),
        }

    result = {
        "gate": {"lo": args.gate_lo, "hi": args.gate_hi,
                 "region_teff_max": args.region_teff_max},
        "generalist_only": summ(persp_gen),
        "gated": summ(persp_gated),
        "generalist_dir": str(gen_dir),
        "specialist_dir": str(outdir),
    }
    (outdir / "gated_eval.json").write_text(json.dumps(result, indent=2))

    print("\n================ SPECIALIST + GATE  (test set) ================")
    print(f"{'':16s} {'median':>11s} {'p90':>11s} {'p99':>11s} {'max':>11s}")
    for name, s in [("generalist", result["generalist_only"]),
                    ("gated", result["gated"])]:
        print(f"{name:16s} {s['median']:11.4e} {s['p90']:11.4e} "
              f"{s['p99']:11.4e} {s['max']:11.4e}")
    regime_report("generalist", persp_gen, T, C, G)
    regime_report("gated", persp_gated, T, C, G)

    # accept?
    ok = result["gated"]["median"] <= result["generalist_only"]["median"] * 1.001
    print(f"\n[spec] bulk-safe (gated median <= generalist median): {ok}", flush=True)

    # --- worst-N reconstructions of the GATED model --------------------------
    order = np.argsort(persp_gated)[::-1][:args.n_worst]
    pix = np.arange(ds.n_pixels)
    n = args.n_worst
    fig, axes = plt.subplots(2 * n, 1, figsize=(14, 3.4 * n), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1] * n},
                             constrained_layout=True)
    for i, loc in enumerate(order):
        ax, rax = axes[2 * i], axes[2 * i + 1]
        pr = gated_flux[loc]; tr = true[loc]
        lab = "  ".join(f"{feat[k]}={ds.X[idx[loc]][k]:.2f}" for k in range(6))
        ax.plot(pix, tr, lw=0.6, color="k", label="Korg")
        ax.plot(pix, pr, lw=0.6, color="C3", alpha=0.8,
                label=f"gated (w={float(w[loc,0]):.2f})")
        ax.set_ylabel("norm flux")
        ax.set_title(f"[worst #{i+1}] RMSE={persp_gated[loc]:.4f}   {lab}",
                     fontsize=9, loc="left")
        ax.legend(loc="lower right", fontsize=8)
        denom = np.where(np.abs(tr) < MIN_FLUX, np.nan, tr)
        pct = 100.0 * (pr - tr) / denom
        rax.axhline(0, color="k", lw=0.6); rax.plot(pix, pct, lw=0.5, color="C0")
        lim = max(np.nanpercentile(np.abs(pct), 99.5), 0.05)
        rax.set_ylim(-lim, lim); rax.set_ylabel("resid [%]", fontsize=8)
        rax.tick_params(labelsize=7)
    axes[-1].set_xlabel("pixel")
    fig.suptitle("Specialist+gate: 10 worst test reconstructions", fontsize=12)
    fig.savefig(outdir / "gated_worst10.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[spec] wrote", outdir / "gated_worst10.png", flush=True)


if __name__ == "__main__":
    main()
