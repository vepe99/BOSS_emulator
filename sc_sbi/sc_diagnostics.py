"""Is the SC loss doing anything sensible? For a run's checkpoints, on a FIXED set of
BOSS *validation* giants (never used by SC training):

1. SC objective: per-spectrum Var_s(log-evidence) over S posterior draws (trimmed like
   training) -> median Var and mean log(Var + eps) (= the logvar SC loss).
2. Posterior movement vs a reference checkpoint: median shift of each parameter, and
   the change of |error| against the CLAM labels.
3. Loss balance at each checkpoint: gradient norm of the NPE term (noisy Korg batch)
   vs lambda * SC term (BOSS batch) and their cosine -> is lambda (sc_weight) sensible?

    .venv/bin/python sc_sbi/sc_diagnostics.py trained_model/sc_sbi/v3_giants \\
        --models amortized amortized_cont amortized_sc_last --ref amortized_cont
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                      # sets KERAS_BACKEND=jax before keras is imported
import numpy as np, h5py, jax, jax.numpy as jnp, keras

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--models", nargs="+", default=["amortized", "amortized_cont", "amortized_sc_last"])
p.add_argument("--ref", default="amortized_cont")
p.add_argument("--n-boss", type=int, default=512)
p.add_argument("--n-grad", type=int, default=64, help="spectra per batch for the gradient balance")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
run = Path(a.run_dir)
cfg = C.load_run_config(run); ncfg = C.noise_config(cfg["noise"]); args = cfg["args"]
names = C.params(ncfg); mm = C.load_mismatch(ncfg)
adapter = C.shared_adapter({k: tuple(v) for k, v in cfg["padded_box"].items()}, cfg.get("log_params"))
box = {k: tuple(v) for k, v in cfg["box"].items()}
S, trim, eps = args["sc_samples"], int(round(args["sc_trim"] * args["sc_samples"])), 1.0
lam = args["sc_weight"]
rng = np.random.default_rng(a.seed)

# ---- fixed BOSS validation giants, same cuts as the SC training rows ----------
boss = C.load_boss()
ok = (C.boss_in_box(boss, box) & (boss["split"] == 1) & (boss["labels"][:, 1] < args["sc_logg_max"])
      & ((boss["ivar"] <= 0).mean(1) <= args.get("sc_max_masked", 1.0)))
rows = np.sort(rng.choice(np.flatnonzero(ok), min(a.n_boss, ok.sum()), replace=False))
feats = C.spectrum_features(boss["flux"][rows], boss["ivar"][rows], boss["snr"][rows], ncfg, mm)
Lb = boss["labels"][rows]
# ---- small noisy Korg batch (labeled) for the NPE gradient -------------------------
with h5py.File(cfg["korg_cache"], "r") as f:
    kr = np.sort(rng.choice(f["norm_flux"].shape[0], a.n_grad, replace=False))
    Fk = f["norm_flux"][kr].astype("float32")
    Lk = np.stack([f[k][kr] for k in C.LABELS], 1).astype("float32")
tm = C.NoiseTemplates(boss["ivar"][boss["split"] == 1][:5000], boss["snr"][boss["split"] == 1][:5000])
lab = adapter(C.label_dict(Lk) | C.NoiseModel(tm, ncfg, mm).make_batch(Fk, Lk, rng))
lik = C.make_likelihood(C.make_emulator_fn(cfg["emulator"]), args["sigma_floor"],
                        s_m=mm.s_m if (mm is not None and ncfg["likelihood_sm"]) else None,
                        lsf=ncfg["lsf"], cont_deg=ncfg["cont_deg"])


def lml_terms(post, d, n):
    """Draws (fixed) + the parameter-independent part of log-evidence for spectra d."""
    cond = {k: v for k, v in d.items() if k != "ivar"}
    smp = post.sample(num_samples=S, conditions=cond, to_numpy=True)
    smp = {k: v.reshape(n * S, -1) for k, v in smp.items()}
    rep = {k: np.repeat(np.asarray(v), S, axis=0) for k, v in d.items()}
    ad = rep | smp
    phys, ldj = adapter(ad, inverse=True, log_det_jac=True)
    th = {k: jnp.asarray(np.asarray(phys[k])) for k in names}
    const = (np.asarray(C.prior_log_prob(**th)).ravel() + sum(np.asarray(ldj[k]).ravel() for k in names)
             + np.concatenate([np.asarray(lik(jnp.asarray(rep["flux"][s:s + 4096]), jnp.asarray(rep["ivar"][s:s + 4096]),
                                              **{k: v[s:s + 4096] for k, v in th.items()})).ravel()
                               for s in range(0, n * S, 4096)]))
    return ad, const, {k: np.asarray(phys[k]).reshape(n, S) for k in names}


def sc_from_lml(lml):
    l = jnp.sort(lml, axis=1)[:, trim:S - trim]
    return jnp.var(l, axis=1)


def grad_norms(post, dB, nB):
    """||g|| of NPE loss (Korg) and of the SC logvar loss (BOSS), and their cosine."""
    tv = post.trainable_variables
    vals = [v.value for v in tv]
    def npe(vs):
        with keras.StatelessScope(state_mapping=list(zip(tv, vs))):
            return -jnp.mean(post.log_prob({k: jnp.asarray(x) for k, x in lab.items()}, to_numpy=False))
    ad, const, _ = lml_terms(post, dB, nB)
    def scl(vs):
        with keras.StatelessScope(state_mapping=list(zip(tv, vs))):
            q = post.log_prob({k: jnp.asarray(x) for k, x in ad.items()}, to_numpy=False)
        v = sc_from_lml(jnp.reshape(jnp.asarray(const) - jnp.ravel(q), (nB, S)))
        return jnp.mean(jnp.log(v + eps))
    g1 = jax.grad(npe)(vals); g2 = jax.grad(scl)(vals)
    f1 = jnp.concatenate([x.ravel() for x in g1]); f2 = jnp.concatenate([x.ravel() for x in g2])
    n1, n2 = float(jnp.linalg.norm(f1)), float(jnp.linalg.norm(f2))
    return n1, n2, float(jnp.dot(f1, f2) / (n1 * n2 + 1e-30))


example = lab
out, med = {}, {}
for m in a.models:
    w = run / f"{m}.weights.h5"
    if not w.exists():
        print(f"[{m}] missing, skipped"); continue
    post = C.load_posterior(run, m, example)
    V, meds = [], {k: [] for k in names}
    for s in range(0, len(rows), 128):
        d = {k: v[s:s + 128] for k, v in feats.items()}; n = d["flux"].shape[0]
        ad, const, phys = lml_terms(post, d, n)
        q = np.asarray(post.log_prob(ad, to_numpy=False)).ravel()
        V.append(np.asarray(sc_from_lml(jnp.asarray((const - q).reshape(n, S)))))
        for k in names:
            meds[k].append(np.median(phys[k], 1))
    V = np.concatenate(V); med[m] = {k: np.concatenate(v) for k, v in meds.items()}
    gi = rng.choice(len(rows), a.n_grad, replace=False)
    n_npe, n_sc, cos = grad_norms(post, {k: v[gi] for k, v in feats.items()}, a.n_grad)
    out[m] = {"median_var": float(np.median(V)), "mean_logvar": float(np.mean(np.log(V + eps))),
              "p90_var": float(np.percentile(V, 90)),
              "grad_npe": n_npe, "grad_sc": n_sc, "lambda_x_ratio": lam * n_sc / n_npe, "cos": cos}
    err = {k: float(np.nanmedian(np.abs(med[m][k] - Lb[:, j]))) for j, k in enumerate(C.BOSS_LABELS)}
    out[m]["median_abs_err_vs_clam"] = err
    out[m]["lsf_px_median"] = float(np.median(med[m][C.LSF])) if C.LSF in names else None
    print(f"[{m}] SC objective on {len(rows)} BOSS val giants: median Var {out[m]['median_var']:.4g}  "
          f"mean log(Var+1) {out[m]['mean_logvar']:.3f}  p90 Var {out[m]['p90_var']:.4g}")
    print(f"[{m}]   grads: |g_NPE| {n_npe:.3g}  |g_SC| {n_sc:.3g}  lambda*|g_SC|/|g_NPE| {lam * n_sc / n_npe:.3g}  cos {cos:+.3f}")
    print(f"[{m}]   median |err| vs CLAM: " + "  ".join(f"{k} {v:.3g}" for k, v in err.items())
          + (f"   lsf_px median {out[m]['lsf_px_median']:.2f}" if C.LSF in names else ""))

if a.ref in med:
    for m in med:
        if m == a.ref:
            continue
        sh = {k: float(np.median(med[m][k] - med[a.ref][k])) for k in names}
        ash = {k: float(np.median(np.abs(med[m][k] - med[a.ref][k]))) for k in names}
        out[m]["shift_vs_ref"] = {"ref": a.ref, "median": sh, "median_abs": ash}
        print(f"[{m}] shift vs {a.ref} (median / median|.|): "
              + "  ".join(f"{k} {sh[k]:+.3g}/{ash[k]:.3g}" for k in names))
(run / "sc_diagnostics.json").write_text(json.dumps(out, indent=2))
print(f"wrote {run / 'sc_diagnostics.json'}")
