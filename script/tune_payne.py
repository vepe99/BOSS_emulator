"""Optuna hyper-parameter search for the Payne MLP emulator.

This is *one worker*. Launch several (one per GPU) pointing at the same
``JournalStorage`` file and they cooperate on a single shared study -- each
process pulls the next trial, trains it on its own GPU, and writes results back
to the journal. See ``script/run_payne_hpo.sh`` for the 3-GPU launcher.

What a trial does
-----------------
* Samples a Payne configuration (depth/width, activation, residual + LayerNorm
  toggles, dropout, LR, weight decay, batch size).
* Trains it for ``--epochs`` (>=1000) on the *full* dataset, saving models
  while training (``best.pkl`` on the objective, periodic ``epoch_XXXX.pkl``,
  and ``final.pkl``).
* Objective = per-spectrum **median** reconstruction RMSE on the validation
  set, in physical flux units (robust to tail spectra). It is reported every
  epoch so Optuna's ``MedianPruner`` can kill unpromising trials early -- the
  only reason 1000-epoch trials are affordable at this trial count.
* For trials that run to completion, the best checkpoint is reloaded and
  evaluated on the held-out **test set**; the metrics are written to
  ``reconstruction_stats.json`` in the run directory and attached to the trial.

The dataset (~38 GB) is loaded **once** per worker and reused across all its
trials.

GPU
---
Importing ``boss_emulator.training`` runs ``autocvd(num_gpus=1)``, so each
worker claims a distinct free GPU before JAX initialises (lab policy). Stagger
worker launches (the launcher does) so two workers don't race for the same
device before it looks busy.

Example
-------
    UV_CACHE_DIR=/export/data/vgiusepp/.uv_cache \
    uv run python script/tune_payne.py --n-trials 40 --epochs 1000
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

# Importing training selects a free GPU via autocvd (unless JAX_PLATFORMS=cpu).
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import jax.numpy as jnp

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint


# =========================================================================
# Search space
# =========================================================================

def suggest_config(trial: optuna.Trial) -> dict:
    """Sample one Payne configuration from the search space."""
    n_layers = trial.suggest_int("n_layers", 3, 8)
    width = trial.suggest_categorical("width", [512, 1024, 2048])
    return {
        "hidden": [width] * n_layers,
        "activation": trial.suggest_categorical(
            "activation", ["gelu", "silu", "relu"]),
        "residual": trial.suggest_categorical("residual", [True, False]),
        "layernorm": trial.suggest_categorical("layernorm", [True, False]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical(
            "batch_size", [512, 1024, 2048, 4096, 8192, 16384]),
    }


# =========================================================================
# Test-set evaluation of a trial's best checkpoint
# =========================================================================

def evaluate_best_on_test(run_dir: Path, ds, Xn, Yn, eval_batch_size) -> dict:
    """Reload ``best.pkl`` and score it on the held-out test set.

    Returns a summary dict (also written to ``reconstruction_stats.json``).
    Uses the physical per-spectrum RMSE from :func:`training.evaluate`.
    """
    model, _payload = load_checkpoint(run_dir / "best.pkl")
    summary, _persp = training.full_reconstruction_metrics(
        model, Xn, ds.Y, Yn, ds.test_idx, ds.y_std, ds.y_mean, eval_batch_size)
    summary["n_test"] = summary.pop("n")   # keep the historical field name
    (run_dir / "reconstruction_stats.json").write_text(
        json.dumps({"test_set": summary}, indent=2))
    return summary


# =========================================================================
# Objective
# =========================================================================

def make_objective(base_args, prepared, study_outdir: Path):
    ds, Xn, Yn = prepared

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)

        # Per-trial training args: copy the base config, apply the sample.
        a = copy.deepcopy(base_args)
        a.model = "payne"
        a.hidden = cfg["hidden"]
        a.activation = cfg["activation"]
        a.residual = cfg["residual"]
        a.layernorm = cfg["layernorm"]
        a.dropout = cfg["dropout"]
        a.lr = cfg["lr"]
        a.weight_decay = cfg["weight_decay"]
        a.batch_size = cfg["batch_size"]
        # Unique per-trial run directory (trial.number is unique across workers
        # via shared storage); train() makes a timestamped subdir beneath it.
        a.outdir = str(study_outdir / f"trial_{trial.number:04d}")

        # Train (reusing the preloaded dataset); reports + prunes via `trial`.
        run_dir, _ds, best_obj = training.train(a, prepared=prepared, trial=trial)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("val_median_rmse", float(best_obj))

        # Completed trial: score the best checkpoint on the held-out test set.
        test = evaluate_best_on_test(run_dir, ds, Xn, Yn, base_args.eval_batch_size)
        trial.set_user_attr("test_median_rmse",
                            test["per_spectrum_rmse_median"])
        print(f"[trial {trial.number}] done  val_medRMSE={best_obj:.4e}  "
              f"test_medRMSE={test['per_spectrum_rmse_median']:.4e}  "
              f"-> {run_dir}", flush=True)
        return best_obj

    return objective


# =========================================================================
# CLI / main
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna HPO worker for the Payne MLP emulator.")
    p.add_argument("--study-name", default="payne_hpo", dest="study_name")
    p.add_argument("--n-trials", type=int, default=40, dest="n_trials",
                   help="trials THIS worker runs (3 workers x 40 = 120)")
    p.add_argument("--epochs", type=int, default=1000,
                   help="epochs per trial (>=1000 as requested)")
    p.add_argument("--ckpt-every", type=int, default=250, dest="ckpt_every",
                   help="snapshot every N epochs during training")
    p.add_argument("--eval-batch-size", type=int, default=8192,
                   dest="eval_batch_size")
    # Data config (shared across workers so the split/test set is identical).
    p.add_argument("--labels", nargs="+", default=list(data_mod.CORE_LABELS))
    p.add_argument("--abundances", nargs="*",
                   default=list(data_mod.ABUNDANCE_ELEMENTS))
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    p.add_argument("--test-frac", type=float, default=0.1, dest="test_frac")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None, dest="max_rows",
                   help="use only N rows (quick end-to-end dry run)")
    p.add_argument("--outdir",
                   default=str(data_mod.PROJECT_ROOT / "trained_model" / "payne_hpo"))
    p.add_argument("--storage",
                   default=None,
                   help="JournalStorage log path "
                        "(default: <outdir>/<study_name>.log)")
    return p.parse_args()


def main():
    args = parse_args()

    # 0. Claim the autocvd-selected GPU up front (before the CPU-heavy load).
    training.claim_gpu()

    study_outdir = Path(args.outdir)
    study_outdir.mkdir(parents=True, exist_ok=True)
    storage_path = Path(args.storage) if args.storage else (
        study_outdir / f"{args.study_name}.log")

    # 1. Shared file-based study storage (multi-process safe, unlike SQLite/NFS).
    storage = JournalStorage(JournalFileBackend(str(storage_path)))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,   # all workers join the same study
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        # Report every epoch; don't prune before a trial has had 30 epochs, and
        # only once a few trials have completed to form a baseline median.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=30, interval_steps=5),
    )
    print(f"[hpo] study={args.study_name!r}  storage={storage_path}  "
          f"n_trials(this worker)={args.n_trials}", flush=True)

    # 2. Base training args (defaults from the training parser, then overrides).
    base_args = training.build_parser().parse_args([])
    base_args.model = "payne"
    base_args.epochs = args.epochs
    base_args.ckpt_every = args.ckpt_every
    base_args.labels = args.labels
    base_args.abundances = args.abundances
    base_args.val_frac = args.val_frac
    base_args.test_frac = args.test_frac
    base_args.seed = args.seed
    base_args.max_rows = args.max_rows
    base_args.eval_batch_size = args.eval_batch_size

    # 3. Load the dataset ONCE for this worker and reuse across all its trials.
    prepared = training.prepare_data(base_args)

    # 4. Run trials. `catch` keeps one crashing trial (e.g. transient OOM) from
    #    killing the worker; TrialPruned is handled by Optuna itself.
    objective = make_objective(base_args, prepared, study_outdir)
    study.optimize(objective, n_trials=args.n_trials,
                   catch=(Exception,), gc_after_trial=True)

    print("\n[hpo] best trial so far:")
    bt = study.best_trial
    print(f"  value(val_medRMSE)={bt.value:.4e}")
    print(f"  params={bt.params}")
    print(f"  run_dir={bt.user_attrs.get('run_dir')}")
    print(f"  test_medRMSE={bt.user_attrs.get('test_median_rmse')}")


if __name__ == "__main__":
    main()
