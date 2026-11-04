"""4-parameter NPE trained on Korg spectra forward-modelled to BOSS-CLAM with the bridge.

Simulator (fresh every batch, on the GPU):
    Korg grid spectrum (train split; [C/M], vmic are nuisances already sampled in the grid)
    -> bridge: y = mu(x) + s(x) * eps                       (bridge/train_bridge.py)
    -> + photon noise and mask of a random BOSS *train-split* ivar template
    -> features exactly as boss_only: [flux | sigma_feat], measured log SNR.
Posterior q(Teff, logg, [M/H], [alpha/M] | spectrum): same network, optimiser, label box
and noise config as trained_model/sc_sbi/boss_only, so the two are directly comparable.

Evaluations (best-val weights):
  1. Korg test split, simulated through the bridge   -> sim_test/ (truth = Korg labels)
  2. BOSS-CLAM test split (star-level split 2). It is the boss_only test set AND the set held
     out from bridge training (asserted below)        -> eval/ (sc_sbi/eval_boss.py, unchanged)
     plus a side-by-side table with boss_only          -> eval/compare_boss_only.md

    uv run --no-sync python bridge/train_sbi_bridge.py --tag draw_v1
"""
import argparse, json, pickle, runpy, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sc_sbi")); sys.path.insert(0, str(ROOT / "bridge"))
import common as C                      # KERAS_BACKEND=jax + autocvd GPU pick
import train as T                       # EpochMonitor, StepTimer, make_optimizer
import evaluate as E                    # point_metrics, interval_coverage
from train_bridge import make_bridge_fn
import numpy as np, jax, jax.numpy as jnp, keras, bayesflow as bf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--bridge", default=str(ROOT / "trained_model/bridge/draw_v1/best.pkl"))
p.add_argument("--compare", default=str(ROOT / "trained_model/sc_sbi/boss_only"), help="boss_only run to compare with")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--steps-per-epoch", type=int, default=1000)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--clip", choices=["value", "norm"], default="norm")
p.add_argument("--clip-value", type=float, default=1.0)
p.add_argument("--net", type=json.loads, default={"dropout": 0.1}, help="overrides of common.DEFAULT_NET")
p.add_argument("--max-masked", type=float, default=0.3, help="BOSS row cut (as boss_only / the bridge)")
p.add_argument("--n-val", type=int, default=10000, help="fixed simulated Korg val spectra")
p.add_argument("--n-test-sim", type=int, default=5000, help="simulated Korg test spectra")
p.add_argument("--n-samples", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--outdir", default=str(ROOT / "trained_model/bridge_sbi"))
p.add_argument("--tag", default="draw_v1")
a = p.parse_args()
run = Path(a.outdir) / a.tag; run.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(a.seed); rng = np.random.default_rng(a.seed)
names = C.BOSS_LABELS                                                   # Teff, logg, m_H, alpha_m
ncfg = C.noise_config(dict(snr_mode="measured", noise_channel=True))    # = boss_only
box = C.label_box(); box4 = {k: box[k] for k in names}; pbox = C.padded_box(box4)
adapter = bf.Adapter()
for k in names:
    adapter = adapter.constrain(k, lower=float(pbox[k][0]), upper=float(pbox[k][1]))

# ---- Korg spectra (host memory; ~8.5 GB train) ------------------------------------------
korg, _, _ = C.load_korg()
(Ltr, Ftr), (Lva, Fva), (Lte, Fte) = korg["train"], korg["val"], korg["test"]
C.log(f"Korg train {len(Ltr):,}  val {len(Lva):,}  test {len(Lte):,}")

# ---- BOSS rows: same cut as boss_only and the bridge ------------------------------------------
boss = C.load_boss(); L = boss["labels"]
ok = np.isfinite(L).all(1) & C.boss_in_box(boss, box) & ((boss["ivar"] <= 0).mean(1) <= a.max_masked)
rows = {s: np.flatnonzero(ok & (boss["split"] == i)) for i, s in enumerate(("train", "val", "test"))}
assert np.array_equal(rows["test"], np.load(Path(a.compare) / "test_posterior.npz")["rows"]), "BOSS test != boss_only test"
with open(a.bridge, "rb") as f:
    bcfg = pickle.load(f)["config"]
assert bcfg["max_masked"] == a.max_masked and not bcfg.get("max_rows"), f"bridge used other rows: {bcfg}"
C.log(f"BOSS test {rows['test'].size:,} spectra = boss_only test = held out from bridge training")

# noise templates: BOSS train split only (never the test stars)
tiv = boss["ivar"][rows["train"]]
dev = dict(ivar_t=jax.device_put(tiv),
           sig_t=jax.device_put(np.where(tiv > 0, 1 / np.sqrt(np.maximum(tiv, 1e-30)), 0).astype("float32")),
           snr_t=jax.device_put(boss["snr"][rows["train"]].astype("float32")),
           s_m_noise=jnp.zeros(C.N_PIX), s_m_feat=jnp.zeros(C.N_PIX),
           mask=jnp.zeros(C.N_PIX, bool), measured=jnp.asarray(True))
bridge = make_bridge_fn(a.bridge)


@jax.jit
def dirty(key, x):
    mu, s = bridge(x)
    return mu + s * jax.random.normal(key, mu.shape)


def simulate(flux, key, r):
    """Korg flux (n, P) -> BOSS-like features dict (flux, ivar, log_snr, sigma_feat)."""
    k1, k2 = jax.random.split(key)
    x = dirty(k1, jnp.asarray(flux))
    j = r.integers(0, len(tiv), len(flux))
    return C._gpu_noisy_batch(k2, x, i=jnp.arange(len(flux)), j=jnp.asarray(j), **dev)


def labels4(Lk):
    return {k: Lk[:, i:i + 1].astype("float32") for i, k in enumerate(names)}


def simulate_set(Lk, Fk, seed, bs=512):
    """Fixed simulated set (numpy), e.g. for validation / testing."""
    r, key, out = np.random.default_rng(seed), jax.random.key(seed), []
    for s in range(0, len(Fk), bs):
        out.append({k: np.asarray(v) for k, v in simulate(Fk[s:s + bs], jax.random.fold_in(key, s), r).items()})
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]} | labels4(Lk)


class KorgBridged(keras.utils.PyDataset):
    def __init__(self, steps, post_adapter):
        super().__init__()
        self.steps, self.pa, self.key, self.calls = steps, post_adapter, jax.random.key(a.seed), 0

    def __getitem__(self, item):
        i = np.sort(rng.integers(0, len(Ltr), a.batch_size))
        self.calls += 1
        b = simulate(Ftr[i], jax.random.fold_in(self.key, self.calls), rng)
        return self.pa(adapter(b | labels4(Ltr[i])))

    def __len__(self):
        return self.steps


# ---- train --------------------------------------------------------------------------------
net = C.DEFAULT_NET | a.net
posterior = C.build_posterior(net, True, names)
ds = KorgBridged(a.steps_per_epoch, posterior.adapter)
val = adapter(simulate_set(Lva[:a.n_val], Fva[:a.n_val], seed=123))
(run / "config.json").write_text(json.dumps({"args": vars(a), "labels": names, "box": box4, "padded_box": pbox,
                                             "net": net, "noise": ncfg, "trained_on": "korg+bridge",
                                             "bridge": a.bridge}, indent=2))
posterior.compile(T.make_optimizer(a.lr, a.epochs, a.steps_per_epoch, a.clip, a.clip_value))
mon = T.EpochMonitor(posterior, val, run, "bridge"); timer = T.StepTimer()
t0 = time.perf_counter()
h = posterior.fit(dataset=ds, epochs=a.epochs, verbose=2, callbacks=[timer, mon, keras.callbacks.TerminateOnNaN()])
mon.publish(); C.save_posterior(posterior, run, "bridge")
hist = {k: [float(x) for x in v] for k, v in h.history.items()} | {"val_nll": mon.val_nll}
(run / "history.json").write_text(json.dumps(hist, indent=2))
(run / "timing.json").write_text(json.dumps(timer.summary() | {"wall_s": time.perf_counter() - t0}, indent=2))
posterior.load_weights(run / "bridge_bestval.weights.h5")

# ---- eval 1: simulated Korg test (truth = Korg labels) ---------------------------------------------
out1 = run / "sim_test"; out1.mkdir(exist_ok=True)
te = simulate_set(Lte[:a.n_test_sim], Fte[:a.n_test_sim], seed=456)
smp = C.sample_posterior(posterior, adapter, te, a.n_samples, names=names)
med, tru = np.median(smp, 1), Lte[:a.n_test_sim, :4]
m1 = {k: E.point_metrics(med[:, j], tru[:, j]) | E.interval_coverage(smp[:, :, j], tru[:, j])
      | {"median_post_std": float(np.median(smp[:, :, j].std(1)))} for j, k in enumerate(names)}
(out1 / "metrics.json").write_text(json.dumps(m1, indent=2))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
for j, k in enumerate(names):
    ax[j].hexbin(tru[:, j], med[:, j], gridsize=60, bins="log", mincnt=1)
    lo, hi = np.percentile(tru[:, j], [0.5, 99.5]); ax[j].plot([lo, hi], [lo, hi], "r-", lw=0.8)
    ax[j].set_title(f"{k}: MAE {m1[k]['mae']:.3g}  cov68 {m1[k]['cov68']:.2f}", fontsize=9)
    ax[j].set_xlabel(f"Korg {k}")
ax[0].set_ylabel("posterior median"); fig.suptitle(f"{a.tag}: simulated Korg test ({len(tru):,}, bridged + BOSS noise)")
fig.tight_layout(); fig.savefig(out1 / "recovery.png", dpi=110); plt.close(fig)
C.log("sim test  " + "  ".join(f"{k}: MAE {m1[k]['mae']:.3g} cov68 {m1[k]['cov68']:.2f}" for k in names))

# ---- eval 2: BOSS-CLAM test with sc_sbi/eval_boss.py (unchanged, in-process: same GPU) -------------------
np.savez(run / "test_posterior.npz", rows=rows["test"])
sys.argv = ["eval_boss.py", str(run), "--weights", "bridge_bestval", "--n-samples", str(a.n_samples)]
runpy.run_path(str(ROOT / "sc_sbi/eval_boss.py"), run_name="__main__")

mine = json.loads((run / "eval/metrics.json").read_text())["bridge_bestval"]
ref = json.loads((Path(a.compare) / "eval/metrics.json").read_text())["boss_bestval"]
cols = ["mae", "rmse", "bias", "nmad", "median_post_std", "cov68", "cov95", "z_std"]
lines = [f"# {a.tag} (Korg + bridge) vs boss_only on the BOSS-CLAM test set ({rows['test'].size:,} spectra)", "",
         "| label | model | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
for k in names:
    for nm, m in ((a.tag, mine), ("boss_only", ref)):
        lines.append(f"| {k} | {nm} | " + " | ".join(f"{m[k][c]:.4g}" for c in cols) + " |")
lines += ["", "| region | model | " + " | ".join(f"{k} MAE / cov68" for k in names) + " |", "|---|---|" + "---|" * len(names)]
for g in E.HR_REGIONS:
    for nm, m in ((a.tag, mine), ("boss_only", ref)):
        lines.append(f"| {g} | {nm} | " + " | ".join(
            f"{m[k]['by_region'][g].get('mae', np.nan):.3g} / {m[k]['by_region'][g]['cov68']:.2f}" for k in names) + " |")
(run / "eval/compare_boss_only.md").write_text("\n".join(lines)); print("\n".join(lines))
C.log(f"done -> {run}")
