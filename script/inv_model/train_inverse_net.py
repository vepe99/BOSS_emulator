"""Train an amortized inverse net: spectrum -> scaled stellar labels.

This is the warm-start for the gradient-descent inversion (strategy 2). A
direct spectrum->labels regressor gives a good first guess for every target in
one forward pass, so the projected-Adam MAP starts near the truth instead of at
the box centre -- which is what pulls the heavy-tailed stragglers out of wrong
basins. It is NOT the final estimator (it has no notion of chi2 / the forward
model); it only initialises the optimiser.

Reuses the MLP (``build_model('payne', n_in=n_pixels, n_out=6)``) and the same
train/val split as everything else. Input is the standardised flux; output is
the six core labels scaled to [0,1] by the training-split min/max. Saved as a
pickle whose params load exactly like a forward checkpoint.

Example::

    BOSS_CACHE_DIR=/export/data/vgiusepp/BOSS_emulator/preprocessed_data \\
        .venv/bin/python script/inv_model/train_inverse_net.py \\
        --epochs 300 --out trained_model/inverse_net/inv.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from boss_emulator import data as data_mod, training
from boss_emulator.emulator import build_model


@nnx.jit
def _step(model, opt, x, y):
    def loss_fn(m):
        return jnp.mean((m(x) - y) ** 2)
    loss, g = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, g)
    return loss


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", nargs="+", type=int, default=[2048, 1024, 256])
    p.add_argument("--activation", default="gelu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4096, dest="batch_size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5, dest="weight_decay")
    p.add_argument("--abundances", nargs="*", default=[])
    p.add_argument("--snr", type=float, default=None,
                   help="noise-augment training: add fresh N(0, 1/snr) flux "
                        "noise to each input batch so the net inverts noisy "
                        "spectra (also evaluated at this SNR). Omit for clean.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="trained_model/inverse_net/inv.pkl")
    args = p.parse_args()

    training.claim_gpu()
    base = training.build_parser().parse_args([])
    base.model = "payne"; base.labels = list(data_mod.CORE_LABELS)
    base.abundances = list(args.abundances); base.val_frac = 0.1
    base.test_frac = 0.1; base.seed = args.seed; base.max_rows = None
    ds, _, _ = training.prepare_data(base)

    core = list(data_mod.CORE_LABELS)
    ci = [ds.feature_names.index(c) for c in core]
    Xcore = ds.X[:, ci].astype("float32")                # physical core labels
    lo = Xcore[ds.train_idx].min(0)
    hi = Xcore[ds.train_idx].max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    S = ((Xcore - lo) / span).astype("float32")          # scaled labels in [0,1]

    # Net input = flux standardised by a FLOORED per-pixel std. The raw per-
    # pixel std is ~0 at the continuum; dividing by it blows those pixels up
    # (huge inputs) and collapses training to the label mean. Flooring keeps
    # every pixel well-scaled.
    xmean = ds.y_mean.astype("float32")
    xstd = np.maximum(ds.y_std, 0.02).astype("float32")
    Xin = ((ds.Y - xmean) / xstd).astype("float32")

    n_in, n_out = ds.n_pixels, len(core)
    model = build_model("payne", n_in, n_out, seed=args.seed,
                        hidden=tuple(args.hidden), activation=args.activation)
    model.train()

    Xtr = jax.device_put(Xin[ds.train_idx]); Ytr = jax.device_put(S[ds.train_idx])
    Xva = jax.device_put(Xin[ds.val_idx]);   Yva = jax.device_put(S[ds.val_idx])
    tr_pos = np.arange(ds.train_idx.size)

    # Noise augmentation. sigma = 1/snr is in physical flux; in the (floored-
    # std) standardised input space it becomes sigma/xstd per pixel. Fresh
    # noise is drawn per batch during training; the validation set gets one
    # fixed noisy realisation so the best-checkpoint objective is stable.
    noise_scale = key = None
    Xva_eval = Xva
    if args.snr:
        noise_scale = jnp.asarray((1.0 / args.snr) / xstd, jnp.float32)
        key = jax.random.PRNGKey(args.seed)
        kv, key = jax.random.split(key)
        Xva_eval = Xva + jax.random.normal(kv, Xva.shape) * noise_scale
        print(f"[invnet] noise-augmenting training at SNR={args.snr:g} "
              f"(sigma={1.0/args.snr:.3g} flux)", flush=True)

    steps = max(1, ds.train_idx.size // args.batch_size)
    total = steps * args.epochs
    sched = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.1, peak_value=args.lr,
        warmup_steps=min(500, total // 20 + 1), decay_steps=total,
        end_value=args.lr * 0.01)
    opt = nnx.Optimizer(model, optax.adamw(sched, weight_decay=args.weight_decay),
                        wrt=nnx.Param)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    best = np.inf
    span_j = jnp.asarray(span)
    for epoch in range(args.epochs):
        t0 = time.time()
        for b in data_mod.iter_batches(tr_pos, args.batch_size, seed=epoch):
            xb = Xtr[b]
            if noise_scale is not None:
                key, sub = jax.random.split(key)
                xb = xb + jax.random.normal(sub, xb.shape) * noise_scale
            _step(model, opt, xb, Ytr[b])
        model.eval()
        # validation: select on the unitless mean scaled MAE (comparable across
        # labels); print per-label PHYSICAL MAE for readability.
        pv = np.asarray(model(Xva_eval)); yv = np.asarray(Yva)
        mae_scaled = np.mean(np.abs(pv - yv), 0)           # per-label, [0,1] units
        mae_phys = mae_scaled * np.asarray(span)           # per-label, physical
        model.train()
        obj = float(np.mean(mae_scaled))
        if obj < best:
            best = obj
            _, pstate, _ = nnx.split(model, nnx.Param, ...)
            params = jax.tree.map(lambda x: np.asarray(x),
                                  nnx.to_pure_dict(pstate))
            payload = {"kind": "inverse_net", "model": "payne",
                       "hidden": list(args.hidden), "activation": args.activation,
                       "n_features": n_in, "n_pixels": n_out, "seed": args.seed,
                       "core_labels": core, "lo": lo, "hi": hi,
                       "y_mean": xmean, "y_std": xstd, "params": params,
                       "train_snr": args.snr}
            with open(out, "wb") as f:
                pickle.dump(payload, f)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            mp = "  ".join(f"{c}={v:.3g}" for c, v in zip(core, mae_phys))
            print(f"[invnet] epoch {epoch+1:4d}/{args.epochs}  "
                  f"val_meanMAE_scaled={obj:.4f}  best={best:.4f}  "
                  f"({time.time()-t0:.1f}s)  [{mp}]", flush=True)

    print(f"[invnet] done. best mean scaled MAE={best:.4f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
