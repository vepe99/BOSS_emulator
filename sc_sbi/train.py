"""Train the amortized posterior, then fine-tune it with self-consistency (SC).

Phase A  -- ``amortized``      : NPE on Korg spectra with real BOSS noise
                                  (labeled loss only).
Phase B  -- ``amortized_sc``   : continue from A with labeled loss + SC loss
                                  on unlabeled spectra (real BOSS train split,
                                  or noisy Korg with ``--sc-unlabeled korg`` as
                                  a sanity check where the likelihood is exact).
Control  -- ``amortized_cont`` : continue from A for the same number of
                                  epochs *without* SC, so any gain of
                                  ``amortized_sc`` is attributable to SC and
                                  not to extra training.

Every run directory holds ``config.json``, ``<model>.weights.h5`` (last finite
epoch), ``<model>_bestval.weights.h5`` (best Korg val NLL), ``splits.npz``,
``history.json`` (incl. per-epoch ``val_nll``) and ``timing.json``;
``evaluate.py`` reads them.

Presets
-------
``--preset v1`` (default) reproduces the original experiment (template noise,
catalog SNR, flux-only summary input, plain variance SC loss, clipvalue);
the only differences are safety nets that do not change a healthy run
(TerminateOnNaN, last-finite / best-val checkpoints, per-epoch val NLL).
``sc_sbi/legacy_v1/train.py`` is a frozen copy of the original script.

``--preset v2`` (needs ``--mismatch`` from ``diagnose_mismatch.py``):
template + ``s_m`` noise, measured SNR, per-pixel sigma channel, SC
likelihood with ``sigma^2 + s_m^2``, log-variance SC loss with trimming and a
weight warm-up, global-norm clipping, lr_b 3e-5, 32 SC samples.
Any explicit flag overrides the preset.

Examples::

    # quick end-to-end check (CPU)
    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/train.py --smoke
    JAX_PLATFORMS=cpu .venv/bin/python sc_sbi/train.py --smoke --preset v2 \\
        --mismatch trained_model/sc_sbi/mismatch/smoke/mismatch.npz --tag smoke_v2

    # v2 full run (GPU picked by autocvd)
    .venv/bin/python sc_sbi/train.py --preset v2 \\
        --mismatch trained_model/sc_sbi/mismatch/v1/mismatch.npz --tag v2_full

    # only phase B / control, re-using the phase-A model of an earlier run
    .venv/bin/python sc_sbi/train.py --preset v2 --mismatch ... \\
        --init-run trained_model/sc_sbi/v2_full --tag v2_sc_try2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (sets KERAS_BACKEND before keras import)

import keras  # noqa: E402
import numpy as np  # noqa: E402
import bayesflow.experimental.self_consistency as sc  # noqa: E402

PRESETS = {
    "v1": dict(noise_mode="template", snr_mode="catalog", noise_channel=False,
               likelihood_sm=False, sc_loss="var", sc_trim=0.0, sc_warmup_epochs=0,
               clip="value", lr_b=1e-4, sc_samples=16, sc_weight=1.0),
    "v2": dict(noise_mode="sm", snr_mode="measured", noise_channel=True,
               likelihood_sm=True, sc_loss="logvar", sc_trim=1 / 32, sc_warmup_epochs=5,
               clip="norm", lr_b=3e-5, sc_samples=32, sc_weight=1.0),
}


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


class EpochMonitor(keras.callbacks.Callback):
    """Per-epoch Korg val NLL + checkpoints that never hold NaN weights.

    ``<name>_last.weights.h5``    -- last epoch whose loss and weights are finite
    ``<name>_bestval.weights.h5`` -- lowest val NLL so far
    """

    def __init__(self, posterior, val_data, run_dir: Path, name: str):
        super().__init__()
        self.posterior, self.val_data = posterior, val_data
        self.run_dir, self.name = Path(run_dir), name
        self.val_nll, self.best, self.last_finite_epoch = [], np.inf, None
        self.epoch_offset = 0

    def on_epoch_end(self, epoch, logs=None):
        e = self.epoch_offset + epoch
        loss = float((logs or {}).get("loss", np.nan))
        finite = np.isfinite(loss) and C.weights_finite(self.posterior)
        v = val_nll_data(self.posterior, self.val_data) if finite else float("nan")
        self.val_nll.append(v)
        if finite and np.isfinite(v):
            self.posterior.save_weights(self.run_dir / f"{self.name}_last.weights.h5")
            self.last_finite_epoch = e
            if v < self.best:
                self.best = v
                self.posterior.save_weights(self.run_dir / f"{self.name}_bestval.weights.h5")
        C.log(f"  [{self.name}] epoch {e + 1}: loss {loss:.4g}  val NLL {v:.4f}"
              + ("" if finite else "  (non-finite: checkpoint skipped)"))


def val_batch(adapter, labels, flux, noise, seed=123) -> dict:
    """A fixed noisy Korg validation set in adapted space."""
    rng = np.random.default_rng(seed)
    b = noise.make_batch(flux, labels, rng)
    return adapter(C.label_dict(labels) | b)


def val_nll_data(posterior, data, batch=512) -> float:
    n = next(iter(data.values())).shape[0]
    lp = np.concatenate([np.asarray(posterior.log_prob({k: v[s:s + batch] for k, v in data.items()}))
                         for s in range(0, n, batch)])
    return float(-lp.mean())


def make_optimizer(lr, epochs, steps_per_epoch, clip, clip_value):
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=max(1, epochs * steps_per_epoch))
    kw = {"clipvalue": clip_value} if clip == "value" else {"global_clipnorm": clip_value}
    return keras.optimizers.AdamW(sched, **kw)


def run_phase(estimator, posterior, dataset, *, epochs, lr, steps_per_epoch, args,
              val_data, run_dir, name, sc_factory=None, warmup=0):
    """Train ``epochs`` epochs. With ``sc_factory(weight)`` the SC loss is added;
    during the first ``warmup`` epochs its weight ramps linearly up to
    ``args.sc_weight`` (one compile per warm-up epoch, shared optimizer)."""
    opt = make_optimizer(lr, epochs, steps_per_epoch, args.clip, args.clip_value)
    timer = StepTimer()
    mon = EpochMonitor(posterior, val_data, run_dir, name)
    cbs = [timer, mon, keras.callbacks.TerminateOnNaN()]
    C.log(f"phase {name}: {epochs} epochs x {steps_per_epoch} steps, lr={lr:g}, "
          f"clip={args.clip}:{args.clip_value}"
          + (f"  + SC ({args.sc_loss}, trim {args.sc_trim:.3g}, warm-up {warmup})"
             if sc_factory else ""))
    if sc_factory is None:
        segments = [(epochs, None)]
    else:
        w = min(warmup, epochs)
        segments = [(1, args.sc_weight * (e + 1) / (w + 1)) for e in range(w)]
        segments += [(epochs - w, args.sc_weight)] if epochs > w else []
    hist, t0, done, planned = {}, time.perf_counter(), 0, 0
    for n_ep, weight in segments:
        planned += n_ep
        kw = {"composite_metrics": {"unlabeled": sc_factory(weight)}} if sc_factory else {}
        estimator.compile(opt, estimator_metrics={"labeled": ["posterior"]}, **kw)
        mon.epoch_offset = done
        h = estimator.fit(dataset=dataset, epochs=n_ep, verbose=2, callbacks=cbs)
        for k, vals in h.history.items():
            hist.setdefault(k, []).extend(float(v) for v in vals)
        if weight is not None:
            hist.setdefault("sc_weight", []).extend([weight] * len(h.history.get("loss", [])))
        done += len(h.history.get("loss", []))
        if done < planned:
            break                                   # TerminateOnNaN fired
    wall = time.perf_counter() - t0
    hist["val_nll"] = mon.val_nll
    # final weights = last finite epoch
    if not C.weights_finite(posterior) or not np.isfinite(hist["loss"][-1]):
        last = run_dir / f"{name}_last.weights.h5"
        if last.exists():
            C.log(f"  [{name}] non-finite at the end -> restoring epoch "
                  f"{mon.last_finite_epoch + 1}")
            posterior.load_weights(last)
        hist["diverged_at_epoch"] = done
    C.save_posterior(posterior, run_dir, name)
    hist["val_nll_end"] = val_nll_data(posterior, val_data)
    hist["last_finite_epoch"] = None if mon.last_finite_epoch is None else mon.last_finite_epoch + 1
    C.log(f"{name}: val NLL {hist['val_nll_end']:.3f}")
    return hist, timer.summary() | {"wall_s": wall}


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS), default="v1")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=__doc__, parents=[pre],
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
    p.add_argument("--lr-b", type=float)
    p.add_argument("--sigma-floor", type=float, default=0.0)
    # --- v2: noise model / inputs ---
    p.add_argument("--mismatch", default=None, help="mismatch.npz from diagnose_mismatch.py")
    p.add_argument("--noise-mode", choices=["template", "sm", "residual"])
    p.add_argument("--snr-mode", choices=["catalog", "measured"])
    p.add_argument("--noise-channel", action=argparse.BooleanOptionalAction)
    p.add_argument("--likelihood-sm", action=argparse.BooleanOptionalAction)
    p.add_argument("--sm-scale", type=float, default=1.0)
    p.add_argument("--sm-mask-thresh", type=float, default=None,
                   help="mask pixels with s_m above this everywhere")
    p.add_argument("--resid-frac", type=float, default=0.5)
    # --- v2: SC loss / optimiser ---
    p.add_argument("--sc-loss", choices=["var", "logvar", "normvar"])
    p.add_argument("--sc-trim", type=float)
    p.add_argument("--sc-warmup-epochs", type=int)
    p.add_argument("--sc-samples", type=int)
    p.add_argument("--sc-weight", type=float)
    p.add_argument("--sc-unlabeled", choices=["boss", "korg"], default="boss",
                   help="korg: SC on noisy Korg spectra (exact likelihood) as a sanity check")
    p.add_argument("--clip", choices=["value", "norm"])
    p.add_argument("--clip-value", type=float, default=1.0)
    # --- run control ---
    p.add_argument("--init-run", default=None,
                   help="skip phase A: copy amortized.weights.h5 from this run dir")
    p.add_argument("--no-control", action="store_true",
                   help="skip the amortized_cont (no-SC) control")
    p.add_argument("--no-sc", action="store_true", help="skip phase B")
    p.add_argument("--n-val", type=int, default=4096)
    p.add_argument("--net", type=json.loads, default={},
                   help=f"JSON overrides of {C.DEFAULT_NET}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(C.PROJECT_ROOT / "trained_model/sc_sbi"))
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 20k Korg rows, 3k BOSS rows, 2 epochs x 5 steps per phase")
    p.set_defaults(**PRESETS[pre_args.preset])
    args = p.parse_args()
    if args.smoke:
        args.max_rows = args.max_rows or 20000
        args.boss_max_rows = args.boss_max_rows or 3000
        args.epochs_a = args.epochs_b = 2
        args.sc_warmup_epochs = min(args.sc_warmup_epochs, 1)
        args.steps_per_epoch = 5
        args.n_val = min(args.n_val, 512)
        args.tag = args.tag or "smoke"
    return args


def main():
    args = parse_args()
    tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    C.log(f"run dir {run_dir}  preset={args.preset}  backend={keras.backend.backend()}  "
          f"devices={C.jax.devices()}")

    ncfg = C.noise_config({"mode": args.noise_mode, "snr_mode": args.snr_mode,
                           "noise_channel": bool(args.noise_channel),
                           "mismatch": args.mismatch, "sm_scale": args.sm_scale,
                           "sm_mask_thresh": args.sm_mask_thresh,
                           "likelihood_sm": bool(args.likelihood_sm),
                           "resid_frac": args.resid_frac})
    mm = C.load_mismatch(ncfg)
    C.log(f"noise config: {ncfg}")

    # ---- data -------------------------------------------------------------
    box = C.label_box(args.korg_cache)
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
    bank_kw = {}
    if ncfg["mode"] == "residual":
        assert np.all(boss["split"][mm.bank_rows] == 0), "residual bank must be train split"
        bank_kw = dict(bank_ivar=boss["ivar"][mm.bank_rows], bank_snr=boss["snr"][mm.bank_rows])
    noise = C.NoiseModel(C.NoiseTemplates(boss["ivar"][tmpl_tr], boss["snr"][tmpl_tr]),
                         ncfg, mm, **bank_kw)
    # validation noise: val-split templates, no residual bank (template [+ s_m])
    noise_val = C.NoiseModel(C.NoiseTemplates(boss["ivar"][tmpl_va], boss["snr"][tmpl_va]),
                             ncfg | {"mode": "sm" if ncfg["mode"] == "residual" else ncfg["mode"]}, mm)

    adapter = C.shared_adapter(pbox)
    lab = C.KorgNoisyDataset(Ltr, Ftr, noise, args.batch_size, args.steps_per_epoch,
                             adapter, seed=args.seed)
    if args.sc_unlabeled == "korg":
        unl = C.KorgNoisyDataset(Lva, Fva, noise_val, args.unlabeled_batch, args.steps_per_epoch,
                                 adapter, seed=args.seed + 1, drop_labels=True)
    else:
        unl = C.BossDataset(boss["flux"][sc_rows], boss["ivar"][sc_rows], boss["snr"][sc_rows],
                            args.unlabeled_batch, args.steps_per_epoch, adapter,
                            seed=args.seed + 1, ncfg=ncfg, mm=mm)
    dataset = sc.MultiDataset(labeled=lab, unlabeled=unl)

    nv = min(args.n_val, Lva.shape[0])
    val_data = val_batch(adapter, Lva[:nv], Fva[:nv], noise_val)

    C.log(f"loading emulator {args.emulator}")
    likelihood = C.make_likelihood(
        C.make_emulator_fn(args.emulator), args.sigma_floor,
        s_m=mm.s_m if (mm is not None and ncfg["likelihood_sm"]) else None,
        pixel_mask=mm.mask if (mm is not None and mm.mask.any()) else None)

    config = {"args": vars(args), "box": box, "padded_box": pbox,
              "net": C.DEFAULT_NET | args.net, "labels": C.LABELS, "noise": ncfg,
              "korg_cache": args.korg_cache, "emulator": args.emulator, "code_version": 2}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    np.savez(run_dir / "splits.npz", korg_test_idx=kidx["test"], korg_val_idx=kidx["val"],
             boss_split=boss["split"], boss_sc_rows=sc_rows, boss_template_rows=tmpl_tr,
             max_rows=-1 if args.max_rows is None else args.max_rows)

    history, timing = {}, {}
    example = lab[0]
    common_kw = dict(steps_per_epoch=args.steps_per_epoch, args=args, val_data=val_data,
                     run_dir=run_dir)

    def fresh(name_from="amortized"):
        post = C.load_posterior(run_dir, name_from, example)
        return post, C.build_estimator(post, adapter, likelihood)

    # ---- phase A: amortized ----------------------------------------------
    if args.init_run:
        src = Path(args.init_run)
        src_cfg = C.load_run_config(src)
        assert src_cfg["net"] == config["net"], "net config differs from --init-run"
        assert (C.noise_config(src_cfg.get("noise"))["noise_channel"] == ncfg["noise_channel"]), \
            "noise_channel differs from --init-run"
        shutil.copy(src / "amortized.weights.h5", run_dir / "amortized.weights.h5")
        for f in ("history.json", "timing.json"):
            if (src / f).exists():
                d = json.loads((src / f).read_text()).get("amortized")
                (history if f == "history.json" else timing)["amortized"] = d
        C.log(f"phase A skipped: amortized weights from {src}")
    else:
        posterior = C.build_posterior(config["net"], ncfg["noise_channel"])
        est = C.build_estimator(posterior, adapter, likelihood)
        history["amortized"], timing["amortized"] = run_phase(
            est, posterior, dataset, epochs=args.epochs_a, lr=args.lr_a,
            name="amortized", **common_kw)

    def save_logs():
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    save_logs()

    # ---- control: same extra epochs, no SC -------------------------------
    if not args.no_control:
        post_c, est_c = fresh()
        history["amortized_cont"], timing["amortized_cont"] = run_phase(
            est_c, post_c, dataset, epochs=args.epochs_b, lr=args.lr_b,
            name="amortized_cont", **common_kw)
        save_logs()

    # ---- phase B: + self-consistency -------------------------------------
    if not args.no_sc:
        def sc_factory(weight):
            if args.sc_loss == "var" and args.sc_trim == 0:
                return C.WeightedSC(num_samples=args.sc_samples, weight=weight)   # v1
            return C.StabilizedSC(num_samples=args.sc_samples, weight=weight,
                                  mode=args.sc_loss, trim=args.sc_trim)
        post_s, est_s = fresh()
        history["amortized_sc"], timing["amortized_sc"] = run_phase(
            est_s, post_s, dataset, epochs=args.epochs_b, lr=args.lr_b,
            name="amortized_sc", sc_factory=sc_factory, warmup=args.sc_warmup_epochs,
            **common_kw)
        save_logs()

    for k, t in timing.items():
        if t and t.get("median_step_s") is not None:
            C.log(f"timing {k:15s} median {t['median_step_s']:.3f} s/step  "
                  f"(first step {t['first_step_s']:.1f} s, wall {t['wall_s']:.0f} s)")
    C.log(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
