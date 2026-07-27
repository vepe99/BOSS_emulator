"""Prototype: Teff/carbon-weighted training of the current-best Payne (trial 16).

Runs the SAME architecture/optimizer as trial #16 but with a per-sample
weighted MSE loss that upweights the hard cool / carbon-rich corner. Run twice
(``--weighted`` on/off) with identical seed + epochs for a controlled A/B, then
compare cool-region vs global test error.

    UV_CACHE_DIR=... uv run python script/train_payne_weighted.py \
        --tag weighted --weighted --epochs 400
    UV_CACHE_DIR=... uv run python script/train_payne_weighted.py \
        --tag baseline           --epochs 400

autocvd claims one free GPU at import (lab policy); stagger the two launches so
they land on different GPUs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from flax import nnx
import optax
import jax.numpy as jnp

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import build_model, load_checkpoint

# --- trial #16 (current best) architecture + optimisation ------------------
CFG = dict(hidden=[2048] * 8, activation="silu", residual=False,
           layernorm=True, dropout=0.0024043533677179374,
           lr=0.0008382480067418546, weight_decay=3.4740558796803107e-06,
           batch_size=4096, seed=0)


@nnx.jit
def train_step_w(model, optimizer, x, y, w):
    def loss_fn(m):
        ps = jnp.mean((m(x) - y) ** 2, axis=1)      # per-sample MSE
        return jnp.sum(w * ps) / jnp.sum(w)          # weighted mean
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def make_weights(Teff, carbon_m, train_idx):
    """Upweight cool / carbon-rich rows; normalise to mean 1 on the train split."""
    w = (1.0
         + 3.0 * np.clip((4500.0 - Teff) / 1500.0, 0.0, 1.0)   # cool ramp
         + 1.5 * np.clip(carbon_m / 1.0, 0.0, 1.0))            # carbon ramp
    w = w.astype(np.float32)
    w *= 1.0 / w[train_idx].mean()                             # mean(train)=1
    return w


def build_args(tag, epochs, outdir):
    a = training.build_parser().parse_args([])
    a.model = "payne"
    a.labels = list(data_mod.CORE_LABELS)
    a.abundances = list(data_mod.ABUNDANCE_ELEMENTS)
    a.val_frac = 0.1; a.test_frac = 0.1
    a.hidden = CFG["hidden"]; a.activation = CFG["activation"]
    a.residual = CFG["residual"]; a.layernorm = CFG["layernorm"]
    a.dropout = CFG["dropout"]; a.lr = CFG["lr"]
    a.weight_decay = CFG["weight_decay"]; a.batch_size = CFG["batch_size"]
    a.seed = CFG["seed"]; a.epochs = epochs
    a.outdir = str(Path(outdir) / tag)
    return a


def cool_median(persp, Teff, idx):
    m = Teff[idx] <= 4000.0
    return float(np.median(persp[m])) if m.any() else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--weighted", action="store_true")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--outdir", default="trained_model/payne_weighted")
    cli = p.parse_args()

    training.claim_gpu()
    args = build_args(cli.tag, cli.epochs, cli.outdir)
    ds, Xn, Yn = training.prepare_data(args)
    y_std_dev = jnp.asarray(ds.y_std)

    feat = ds.feature_names
    Teff = ds.X[:, feat.index("Teff")]
    carbon = ds.X[:, feat.index("carbon_m")]
    if cli.weighted:
        W = make_weights(Teff, carbon, ds.train_idx)
    else:
        W = np.ones(ds.X.shape[0], np.float32)
    print(f"[w] tag={cli.tag} weighted={cli.weighted}  "
          f"train weight range [{W[ds.train_idx].min():.2f},"
          f"{W[ds.train_idx].max():.2f}]", flush=True)

    model = build_model("payne", ds.n_features, ds.n_pixels, seed=args.seed,
                        hidden=tuple(CFG["hidden"]), activation=CFG["activation"],
                        residual=CFG["residual"], layernorm=CFG["layernorm"],
                        dropout=CFG["dropout"])
    model.train()

    steps = max(1, ds.train_idx.size // args.batch_size)
    total = steps * args.epochs
    sched = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1, peak_value=args.lr,
        warmup_steps=min(500, total // 20 + 1),
        decay_steps=total, end_value=args.lr * 0.01)
    opt = nnx.Optimizer(model, optax.adamw(sched, weight_decay=args.weight_decay),
                        wrt=nnx.Param)

    outdir = Path(args.outdir) / f"payne_{cli.tag}_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "split_idx.npz", train_idx=ds.train_idx,
             val_idx=ds.val_idx, test_idx=ds.test_idx)
    print(f"[w] writing to {outdir}", flush=True)

    best_obj, hist = np.inf, []
    for epoch in range(args.epochs):
        t0 = time.time()
        losses = []
        for batch in data_mod.iter_batches(ds.train_idx, args.batch_size,
                                           seed=args.seed + epoch):
            xb = jnp.asarray(Xn[batch]); yb = jnp.asarray(Yn[batch])
            wb = jnp.asarray(W[batch])
            losses.append(float(train_step_w(model, opt, xb, yb, wb)))
        model.eval()
        va, persp = training.evaluate(model, Xn, Yn, ds.val_idx, y_std_dev,
                                      args.batch_size)
        model.train()
        obj = float(np.median(persp))
        cool = cool_median(persp, Teff, ds.val_idx)
        dt = time.time() - t0
        print(f"[{cli.tag}] ep {epoch+1:4d}/{args.epochs} "
              f"train={np.mean(losses):.4e} val_med={obj:.4e} "
              f"val_cool_med={cool:.4e} ({dt:.1f}s)", flush=True)
        hist.append({"epoch": epoch + 1, "val_median_rmse": obj,
                     "val_cool_median_rmse": cool, "val_mse": float(va)})
        if obj < best_obj:
            best_obj = obj
            training.save_checkpoint(outdir / "best.pkl", model, ds, args)
    training.save_checkpoint(outdir / "final.pkl", model, ds, args)
    (outdir / "history.json").write_text(json.dumps(hist, indent=2))

    # --- test-set evaluation, stratified -----------------------------------
    model_b, payload = load_checkpoint(outdir / "best.pkl")
    ym = np.asarray(payload["y_mean"], np.float32)
    ys = np.asarray(payload["y_std"], np.float32)
    ti = ds.test_idx
    tp = np.empty(ti.size, np.float32); wr = 0
    for s in range(0, ti.size, 8192):
        b = ti[s:s + 8192]
        pred = np.asarray(model_b(jnp.asarray(Xn[b]))) * ys + ym
        tp[wr:wr + b.size] = np.sqrt(np.mean((pred - ds.Y[b]) ** 2, axis=1))
        wr += b.size
    Tt = Teff[ti]; Ct = carbon[ti]
    strat = {
        "tag": cli.tag, "weighted": cli.weighted, "epochs": args.epochs,
        "global_median": float(np.median(tp)),
        "global_p99": float(np.percentile(tp, 99)),
        "global_max": float(tp.max()),
        "cool_le4000_median": float(np.median(tp[Tt <= 4000])),
        "cool_le4000_p99": float(np.percentile(tp[Tt <= 4000], 99)),
        "coolcarbon_median": float(np.median(tp[(Tt <= 4000) & (Ct >= 0.5)])),
        "coolcarbon_p99": float(np.percentile(tp[(Tt <= 4000) & (Ct >= 0.5)], 99)),
        "coolcarbon_max": float(tp[(Tt <= 4000) & (Ct >= 0.5)].max()),
        "hot_gt6000_median": float(np.median(tp[Tt > 6000])),
    }
    (outdir / "test_stratified.json").write_text(json.dumps(strat, indent=2))
    np.savez(outdir / "test_persp.npz", persp=tp, Teff=Tt, carbon_m=Ct)
    print("[w] RESULT", json.dumps(strat), flush=True)


if __name__ == "__main__":
    main()
