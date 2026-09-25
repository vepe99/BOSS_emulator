"""Print a leaderboard of an Optuna JournalStorage study.

Example::

    .venv/bin/python script/hpo_status.py trained_model/combined_hpo/combined_hpo.log
"""
from __future__ import annotations

import argparse
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

optuna.logging.set_verbosity(optuna.logging.WARNING)

p = argparse.ArgumentParser()
p.add_argument("journal")
p.add_argument("--study-name", default=None)
p.add_argument("--top", type=int, default=10)
a = p.parse_args()

storage = JournalStorage(JournalFileBackend(a.journal))
name = a.study_name or optuna.get_all_study_names(storage)[0]
st = optuna.load_study(study_name=name, storage=storage)
tr = st.get_trials(deepcopy=False)
states = {}
for t in tr:
    states[t.state.name] = states.get(t.state.name, 0) + 1
print(f"study={name}  trials={len(tr)}  " + "  ".join(f"{k}={v}" for k, v in states.items()))
done = [t for t in tr if t.state.name == "COMPLETE"]
done.sort(key=lambda t: t.value)
print(f"{'#':>3s} {'val_medRMSE':>12s} {'test_med':>10s} {'layers':>6s} {'width':>5s} {'act':>5s} {'res':>3s} {'ln':>3s} "
      f"{'drop':>6s} {'lr':>8s} {'wd':>8s} {'batch':>6s}")
for t in done[: a.top]:
    q = t.params
    tm = t.user_attrs.get("test_median_rmse")
    print(f"{t.number:3d} {t.value:12.4e} {(f'{tm:.3e}' if tm else '-'):>10s} {q['n_layers']:6d} {q['width']:5d} "
          f"{q['activation']:>5s} {str(q['residual'])[0]:>3s} {str(q['layernorm'])[0]:>3s} "
          f"{q['dropout']:6.3f} {q['lr']:8.2e} {q['weight_decay']:8.2e} {q['batch_size']:6d}")
run = [t for t in tr if t.state.name == "RUNNING"]
for t in run:
    q = t.params
    print(f"running #{t.number}: {q.get('n_layers')}x{q.get('width')} {q.get('activation')} "
          f"res={q.get('residual')} ln={q.get('layernorm')} drop={q.get('dropout', 0):.3f} "
          f"lr={q.get('lr', 0):.2e} wd={q.get('weight_decay', 0):.2e} bs={q.get('batch_size')}")
