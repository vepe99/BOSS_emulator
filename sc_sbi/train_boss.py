"""Supervised NPE trained purely on BOSS-CLAM (no Korg, no emulator):
q(Teff, logg, [M/H], [alpha/M] | BOSS spectrum, log SNR) with the CLAM labels as targets.

Same posterior as v3 (MLP summary on [flux | sigma_feat], measured log SNR, CouplingFlow),
same star-level split of the prepared cache (0 train / 1 val / 2 test), and by default the
same label box as the Korg-trained models (so test metrics are comparable with evaluate.py).
Keeps the best-val-NLL weights; at the end evaluates them on the test split.

    .venv/bin/python sc_sbi/train_boss.py --tag boss_only
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                      # KERAS_BACKEND=jax + autocvd GPU pick
import train as T                       # EpochMonitor, StepTimer, make_optimizer, val_nll_data
import numpy as np, jax, jax.numpy as jnp, keras, bayesflow as bf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--steps-per-epoch", type=int, default=None, help="default: one pass over the train rows")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--clip", choices=["value", "norm"], default="norm")
p.add_argument("--clip-value", type=float, default=1.0)
p.add_argument("--max-masked", type=float, default=0.3, help="drop spectra with more masked pixels")
p.add_argument("--full-range", action="store_true", help="use the data's label range, not the Korg box")
p.add_argument("--net", type=json.loads, default={"dropout": 0.1},
               help='overrides of common.DEFAULT_NET, e.g. \'{"dropout": 0.1, "summary_dim": 64}\'')
p.add_argument("--n-samples", type=int, default=200, help="posterior draws per test star")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--outdir", default=str(C.PROJECT_ROOT / "trained_model/sc_sbi"))
p.add_argument("--tag", default="boss_only")
a = p.parse_args()
names = C.BOSS_LABELS
ncfg = C.noise_config(dict(noise_mode="template", snr_mode="measured", noise_channel=True))
run = Path(a.outdir) / a.tag; run.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(a.seed)

# ---- data ---------------------------------------------------------------------
boss = C.load_boss(); L = boss["labels"]
ok = np.isfinite(L).all(1) & np.isfinite(boss["snr"]) & (boss["snr"] > 0) & ((boss["ivar"] <= 0).mean(1) <= a.max_masked)
if a.full_range:
    box = {k: (float(L[ok, i].min()), float(L[ok, i].max())) for i, k in enumerate(names)}
else:
    kb = C.label_box(); box = {k: kb[k] for k in names}
    ok &= C.boss_in_box(boss, kb)
pbox = C.padded_box(box)
rows = {s: np.flatnonzero(ok & (boss["split"] == i)) for i, s in enumerate(("train", "val", "test"))}
C.log("rows " + "  ".join(f"{s} {r.size:,}" for s, r in rows.items()) + f"   box {box}")

adapter = bf.Adapter()
for k in names:
    adapter = adapter.constrain(k, lower=float(pbox[k][0]), upper=float(pbox[k][1]))


def feats(r):
    f = C.spectrum_features(boss["flux"][r], boss["ivar"][r], boss["snr"][r], ncfg, None)
    return f | {k: L[r, i:i + 1].astype("float32") for i, k in enumerate(names)}


class BossLabeled(keras.utils.PyDataset):
    """Random train batches, gathered on the GPU (flux + sigma_feat + labels live there)."""

    def __init__(self, d, batch, steps, post_adapter, seed):
        super().__init__()
        self.d = {k: jax.device_put(v) for k, v in d.items() if k != "ivar"}
        self.n, self.batch, self.steps, self.pa = len(d["flux"]), batch, steps, post_adapter
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, item):
        i = jnp.asarray(self.rng.integers(0, self.n, self.batch))
        return self.pa(adapter({k: v[i] for k, v in self.d.items()}))

    def __len__(self):
        return self.steps


net = C.DEFAULT_NET | a.net
posterior = C.build_posterior(net, True, names)
steps = a.steps_per_epoch or max(1, rows["train"].size // a.batch_size)
ds = BossLabeled(feats(rows["train"]), a.batch_size, steps, posterior.adapter, a.seed)
val = adapter(feats(rows["val"]))
(run / "config.json").write_text(json.dumps({"args": vars(a), "labels": names, "box": box, "padded_box": pbox,
                                             "net": net, "noise": ncfg, "trained_on": "boss_clam"}, indent=2))

# ---- train ----------------------------------------------------------------------
posterior.compile(T.make_optimizer(a.lr, a.epochs, steps, a.clip, a.clip_value))
mon = T.EpochMonitor(posterior, val, run, "boss"); timer = T.StepTimer()
t0 = time.perf_counter()
h = posterior.fit(dataset=ds, epochs=a.epochs, verbose=2, callbacks=[timer, mon, keras.callbacks.TerminateOnNaN()])
mon.publish(); C.save_posterior(posterior, run, "boss")
hist = {k: [float(x) for x in v] for k, v in h.history.items()} | {"val_nll": mon.val_nll}
(run / "history.json").write_text(json.dumps(hist, indent=2))
(run / "timing.json").write_text(json.dumps(timer.summary() | {"wall_s": time.perf_counter() - t0}, indent=2))

# ---- test: best-val weights ------------------------------------------------------------
posterior.load_weights(run / "boss_bestval.weights.h5")
te = feats(rows["test"]); Lt = L[rows["test"]]
smp = C.sample_posterior(posterior, adapter, te, a.n_samples, names=names)      # (n, S, 4)
med = np.median(smp, 1); lo, hi = np.percentile(smp, [16, 84], 1); std = smp.std(1)
giant = Lt[:, 1] < 3.5
metrics = {"val_nll_best": float(mon.best), "n_test": int(len(Lt))}
for j, k in enumerate(names):
    e = med[:, j] - Lt[:, j]
    metrics[k] = {"mae": float(np.mean(np.abs(e))), "bias": float(np.median(e)),
                  "mae_giants": float(np.mean(np.abs(e[giant]))), "mae_dwarfs": float(np.mean(np.abs(e[~giant]))),
                  "cov68": float(np.mean((Lt[:, j] >= lo[:, j]) & (Lt[:, j] <= hi[:, j]))),
                  "median_post_std": float(np.median(std[:, j]))}
    C.log(f"test {k:8s} MAE {metrics[k]['mae']:.4g} (giants {metrics[k]['mae_giants']:.4g}, dwarfs "
          f"{metrics[k]['mae_dwarfs']:.4g})  bias {metrics[k]['bias']:+.3g}  68% coverage {metrics[k]['cov68']:.2f}")
(run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
np.savez(run / "test_posterior.npz", rows=rows["test"], median=med, p16=lo, p84=hi, std=std)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
for j, k in enumerate(names):
    ax[j].scatter(Lt[:, j], med[:, j], s=1, alpha=0.3, c=np.where(giant, "C0", "C1"))
    r = [np.percentile(Lt[:, j], 0.5), np.percentile(Lt[:, j], 99.5)]
    ax[j].plot(r, r, "k:"); ax[j].set_xlim(r); ax[j].set_ylim(r)
    ax[j].set_xlabel(f"CLAM {k}"); ax[j].set_ylabel("posterior median")
    ax[j].set_title(f"{k}: MAE {metrics[k]['mae']:.3g}, cov68 {metrics[k]['cov68']:.2f}", fontsize=9)
fig.suptitle(f"{a.tag}: NPE trained on BOSS-CLAM only, test split ({len(Lt):,} spectra; blue giants, orange dwarfs)")
fig.tight_layout(); fig.savefig(run / "test_scatter.png", dpi=110)
C.log(f"done -> {run}")
