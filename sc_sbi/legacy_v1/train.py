"""Train the amortized posterior, then fine-tune it with self-consistency (SC).

Phase A  -- ``amortized``      : NPE on Korg spectra with real BOSS noise
                                  (labeled loss only).
Phase B  -- ``amortized_sc``   : continue from A with labeled loss + SC loss
                                  on unlabeled real BOSS spectra (train split).
Control  -- ``amortized_cont`` : continue from A for the same number of
                                  epochs *without* SC, so any gain of
                                  ``amortized_sc`` is attributable to SC and
                                  not to extra training.

Every run directory holds ``config.json``, ``<model>.weights.h5``,
``splits.npz``, ``history.json`` and ``timing.json``; ``evaluate.py`` reads
them.

Examples::

    # quick end-to-end check on a data subset (CPU)
    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/legacy_v1/train.py --smoke

    # full run (GPU picked by autocvd)
    .venv/bin/python sc_sbi/legacy_v1/train.py --epochs-a 200 --epochs-b 50 --tag full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (sets KERAS_BACKEND before keras import)

import h5py  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import bayesflow.experimental.self_consistency as sc  # noqa: E402


class StepTimer(keras.callbacks.Callback):
    """Wall time per training step (first `skip` steps = compilation, excluded)."""

    def __init__(self, skip: int = 2):
        super().__init__()
        self.skip, self.times, self._t = skip, [], None

    def on_train_batch_begin(self, batch, logs=None):
        self._t = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        self.times.append(time.perf_counter() - self._t)

    def summary(self) -> dict:
        t = np.array(self.times[self.skip:] or self.times)
        return {"steps": len(self.times), "first_step_s": float(self.times[0]) if self.times else None,
                "median_step_s": float(np.median(t)) if t.size else None,
                "mean_step_s": float(t.mean()) if t.size else None}


def label_box(cache: Path) -> dict:
    """Label box of the *full* Korg cache (independent of --max-rows)."""
    with h5py.File(cache, "r") as f:
        return {k: (float(f[k][:].min()), float(f[k][:].max())) for k in C.LABELS}


def val_nll(posterior, adapter, labels, flux, templates, seed=123) -> float:
    """Mean -log q(theta | y) on a fixed noisy Korg validation set (adapted space)."""
    rng = np.random.default_rng(seed)
    y, iv, lsnr = templates.apply(flux, rng)
    data = adapter(C.label_dict(labels) | {"flux": y, "ivar": iv, "log_snr": lsnr})
    lp = np.concatenate([np.asarray(posterior.log_prob({k: v[s:s + 512] for k, v in data.items()}))
                         for s in range(0, labels.shape[0], 512)])
    return float(-lp.mean())


def run_phase(estimator, dataset, *, epochs, lr, steps_per_epoch, sc_loss=None, label=""):
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=max(1, epochs * steps_per_epoch))
    kw = {"composite_metrics": {"unlabeled": sc_loss}} if sc_loss is not None else {}
    estimator.compile(keras.optimizers.AdamW(sched, clipvalue=1.0),
                      estimator_metrics={"labeled": ["posterior"]}, **kw)
    timer = StepTimer()
    C.log(f"phase {label}: {epochs} epochs x {steps_per_epoch} steps, lr={lr:g}"
          f"{'  + SC' if sc_loss is not None else ''}")
    t0 = time.perf_counter()
    hist = estimator.fit(dataset=dataset, epochs=epochs, verbose=2, callbacks=[timer])
    wall = time.perf_counter() - t0
    hist = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return hist, timer.summary() | {"wall_s": wall}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--korg-cache", default=str(C.KORG_CACHE))
    p.add_argument("--max-rows", type=int, default=None,
                   help="subsample the Korg cache (re-splits it; tests only)")
    p.add_argument("--boss-max-rows", type=int, default=None,
                   help="cap the unlabeled BOSS train rows used for SC")
    p.add_argument("--emulator", default=str(C.EMULATOR_CKPT))
    p.add_argument("--epochs-a", type=int, default=200)
    p.add_argument("--epochs-b", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--unlabeled-batch", type=int, default=32)
    p.add_argument("--lr-a", type=float, default=5e-4)
    p.add_argument("--lr-b", type=float, default=1e-4)
    p.add_argument("--sc-samples", type=int, default=16)
    p.add_argument("--sc-weight", type=float, default=1.0)
    p.add_argument("--sigma-floor", type=float, default=0.0)
    p.add_argument("--no-control", action="store_true",
                   help="skip the amortized_cont (no-SC) control")
    p.add_argument("--n-val", type=int, default=4096)
    p.add_argument("--net", type=json.loads, default={},
                   help=f"JSON overrides of {C.DEFAULT_NET}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(C.PROJECT_ROOT / "trained_model/sc_sbi"))
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 20k Korg rows, 3k BOSS rows, 1 epoch x 5 steps per phase")
    args = p.parse_args()
    if args.smoke:
        args.max_rows = args.max_rows or 20000
        args.boss_max_rows = args.boss_max_rows or 3000
        args.epochs_a = args.epochs_b = 1
        args.steps_per_epoch = 5
        args.n_val = min(args.n_val, 512)
        args.tag = args.tag or "smoke"
    tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    C.log(f"run dir {run_dir}  backend={keras.backend.backend()}  "
          f"devices={C.jax.devices()}")

    # ---- data -------------------------------------------------------------
    box = label_box(Path(args.korg_cache))
    pbox = C.padded_box(box)
    C.log("loading Korg cache ...")
    korg, kidx, _ = C.load_korg(args.korg_cache, max_rows=args.max_rows, seed=0,
                                splits=("train", "val"))
    (Ltr, Ftr), (Lva, Fva) = korg["train"], korg["val"]
    C.log(f"Korg train {Ltr.shape[0]:,}  val {Lva.shape[0]:,}")

    boss = C.load_boss()
    ok = C.boss_in_box(boss, box)
    tmpl_tr = np.flatnonzero((boss["split"] == 0) & np.isfinite(boss["snr"]) & (boss["snr"] > 0))
    tmpl_va = np.flatnonzero((boss["split"] == 1) & np.isfinite(boss["snr"]) & (boss["snr"] > 0))
    sc_rows = np.flatnonzero((boss["split"] == 0) & ok)
    rng = np.random.default_rng(args.seed)
    if args.boss_max_rows and sc_rows.size > args.boss_max_rows:
        sc_rows = np.sort(rng.choice(sc_rows, args.boss_max_rows, replace=False))
        tmpl_tr = np.sort(rng.choice(tmpl_tr, min(tmpl_tr.size, args.boss_max_rows), replace=False))
    C.log(f"BOSS: noise templates {tmpl_tr.size:,} (train split), "
          f"SC unlabeled {sc_rows.size:,} (train split, in box)")
    templates = C.NoiseTemplates(boss["ivar"][tmpl_tr], boss["snr"][tmpl_tr])
    templates_val = C.NoiseTemplates(boss["ivar"][tmpl_va], boss["snr"][tmpl_va])

    adapter = C.shared_adapter(pbox)
    lab = C.KorgNoisyDataset(Ltr, Ftr, templates, args.batch_size, args.steps_per_epoch,
                             adapter, seed=args.seed)
    unl = C.BossDataset(boss["flux"][sc_rows], boss["ivar"][sc_rows], boss["snr"][sc_rows],
                        args.unlabeled_batch, args.steps_per_epoch, adapter, seed=args.seed + 1)
    dataset = sc.MultiDataset(labeled=lab, unlabeled=unl)

    nv = min(args.n_val, Lva.shape[0])
    Lval, Fval = Lva[:nv], Fva[:nv]

    C.log(f"loading emulator {args.emulator}")
    likelihood = C.make_likelihood(C.make_emulator_fn(args.emulator), args.sigma_floor)

    config = {"args": vars(args), "box": box, "padded_box": pbox,
              "net": C.DEFAULT_NET | args.net, "labels": C.LABELS,
              "korg_cache": args.korg_cache, "emulator": args.emulator}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    np.savez(run_dir / "splits.npz", korg_test_idx=kidx["test"], korg_val_idx=kidx["val"],
             boss_split=boss["split"], boss_sc_rows=sc_rows, boss_template_rows=tmpl_tr,
             max_rows=-1 if args.max_rows is None else args.max_rows)

    history, timing = {}, {}

    # ---- phase A: amortized ----------------------------------------------
    posterior = C.build_posterior(config["net"])
    est = C.build_estimator(posterior, adapter, likelihood)
    history["amortized"], timing["amortized"] = run_phase(
        est, dataset, epochs=args.epochs_a, lr=args.lr_a,
        steps_per_epoch=args.steps_per_epoch, label="A (amortized)")
    C.save_posterior(posterior, run_dir, "amortized")
    history["amortized"]["val_nll_end"] = val_nll(posterior, adapter, Lval, Fval, templates_val)
    C.log(f"amortized: val NLL {history['amortized']['val_nll_end']:.3f}")
    example = lab[0]

    # ---- control: same extra epochs, no SC -------------------------------
    if not args.no_control:
        post_c = C.load_posterior(run_dir, "amortized", example)
        est_c = C.build_estimator(post_c, adapter, likelihood)
        history["amortized_cont"], timing["amortized_cont"] = run_phase(
            est_c, dataset, epochs=args.epochs_b, lr=args.lr_b,
            steps_per_epoch=args.steps_per_epoch, label="control (no SC)")
        C.save_posterior(post_c, run_dir, "amortized_cont")
        history["amortized_cont"]["val_nll_end"] = val_nll(post_c, adapter, Lval, Fval, templates_val)
        C.log(f"amortized_cont: val NLL {history['amortized_cont']['val_nll_end']:.3f}")

    # ---- phase B: + self-consistency -------------------------------------
    post_s = C.load_posterior(run_dir, "amortized", example)
    est_s = C.build_estimator(post_s, adapter, likelihood)
    history["amortized_sc"], timing["amortized_sc"] = run_phase(
        est_s, dataset, epochs=args.epochs_b, lr=args.lr_b,
        steps_per_epoch=args.steps_per_epoch,
        sc_loss=C.WeightedSC(num_samples=args.sc_samples, weight=args.sc_weight),
        label="B (amortized_sc)")
    C.save_posterior(post_s, run_dir, "amortized_sc")
    history["amortized_sc"]["val_nll_end"] = val_nll(post_s, adapter, Lval, Fval, templates_val)
    C.log(f"amortized_sc: val NLL {history['amortized_sc']['val_nll_end']:.3f}")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    for k, t in timing.items():
        C.log(f"timing {k:15s} median {t['median_step_s']:.3f} s/step  "
              f"(first step {t['first_step_s']:.1f} s, wall {t['wall_s']:.0f} s)")
    C.log(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
