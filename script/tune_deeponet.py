"""Optuna hyper-parameter search for the DeepONet emulator.

The DeepONet analogue of ``script/tune_payne.py`` -- see that file's docstring
for the full description of the multi-worker journal-storage design, the
per-spectrum median-RMSE objective, the ``MedianPruner`` early-stopping, and the
load-dataset-once-per-worker pattern. Everything here is identical except the
sampled architecture: this searches the *branch* net (depth/width), the shared
``latent`` basis dimension, dropout on the branch, and the usual optimisation
knobs. The trunk net keeps its constructor default (it only sees the fixed pixel
coordinates, so it is not label-dependent and matters far less).

Example
-------
    BOSS_CACHE_DIR=/export/data/vgiusepp/BOSS_emulator/preprocessed_data \
    .venv/bin/python script/tune_deeponet.py --n-trials 30 --epochs 1000
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

from boss_emulator import data as data_mod
from boss_emulator import training
from boss_emulator.emulator import load_checkpoint


# =========================================================================
# Search space
# =========================================================================

def suggest_config(trial: optuna.Trial) -> dict:
    """Sample one DeepONet configuration from the search space."""
    n_layers = trial.suggest_int("n_branch_layers", 2, 5)
    width = trial.suggest_categorical("branch_width", [256, 512, 1024])
    return {
        "hidden": [width] * n_layers,               # branch net widths
        "latent": trial.suggest_categorical("latent", [128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical(
            "batch_size", [4096, 8192, 16384]),
    }


# =========================================================================
# Test-set evaluation of a trial's best checkpoint
# =========================================================================

def evaluate_best_on_test(run_dir: Path, ds, Xn, Yn, eval_batch_size) -> dict:
    """Reload ``best.pkl`` and score it on the held-out test set."""
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

        a = copy.deepcopy(base_args)
        a.model = "deeponet"
        a.hidden = cfg["hidden"]          # forwarded as branch_hidden
        a.latent = cfg["latent"]
        a.dropout = cfg["dropout"]
        a.lr = cfg["lr"]
        a.weight_decay = cfg["weight_decay"]
        a.batch_size = cfg["batch_size"]
        a.outdir = str(study_outdir / f"trial_{trial.number:04d}")

        run_dir, _ds, best_obj = training.train(a, prepared=prepared, trial=trial)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("val_median_rmse", float(best_obj))

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
        description="Optuna HPO worker for the DeepONet emulator.")
    p.add_argument("--study-name", default="deeponet_hpo", dest="study_name")
    p.add_argument("--n-trials", type=int, default=30, dest="n_trials")
    p.add_argument("--epochs", type=int, default=1000,
                   help="epochs per trial")
    p.add_argument("--ckpt-every", type=int, default=250, dest="ckpt_every",
                   help="snapshot every N epochs during training")
    p.add_argument("--eval-batch-size", type=int, default=8192,
                   dest="eval_batch_size")
    # Data config (shared across workers so the split/test set is identical).
    p.add_argument("--labels", nargs="+", default=list(data_mod.CORE_LABELS))
    p.add_argument("--abundances", nargs="*", default=[],
                   help="abundance elements to add as inputs (default: none, "
                        "i.e. the 6 stellar parameters only)")
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    p.add_argument("--test-frac", type=float, default=0.1, dest="test_frac")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None, dest="max_rows",
                   help="use only N rows (quick end-to-end dry run)")
    p.add_argument("--outdir",
                   default=str(data_mod.PROJECT_ROOT / "trained_model" / "deeponet_hpo"))
    p.add_argument("--storage", default=None,
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
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=30, interval_steps=5),
    )
    print(f"[hpo] study={args.study_name!r}  storage={storage_path}  "
          f"n_trials(this worker)={args.n_trials}", flush=True)

    # 2. Base training args (defaults from the training parser, then overrides).
    base_args = training.build_parser().parse_args([])
    base_args.model = "deeponet"
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

    # 4. Run trials.
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
