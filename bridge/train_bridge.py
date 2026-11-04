"""Bridge network: clean Korg-emulator spectrum -> "dirty" boss_clam spectrum.

Input  x = emu(CLAM Teff, logg, [M/H], [alpha/M], [C/M] ~ U(box), vmic ~ logU(box)),
       either one fresh prior draw per row and step (--input draw, default; what a 6-label
       SBI simulator feeds it) or the median over 64 draws (--input median, precomputed).
Target y = the real boss_clam spectrum with its ivar (same star-level split as sc_sbi).
Model  p(y | x) = N(mu(x), 1/ivar + s(x)^2), mu = x + h[:P], s = exp(h[P:]),
       h = MLP(standardised x). Loss = Gaussian NLL over good pixels. The bridge never
       sees labels, so it can follow the emulator in a 6-label simulator
       (load with ``make_bridge_fn`` below; importing this module does not train).

    .venv/bin/python bridge/train_bridge.py --tag draw_v1
"""
import argparse, json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sc_sbi"))
import common as C                      # KERAS_BACKEND=jax + autocvd GPU pick
import numpy as np, jax, jax.numpy as jnp, optax
from flax import nnx
from boss_emulator.emulator import build_model, load_checkpoint
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def make_bridge_fn(ckpt):
    """Clean emulator flux (n, P) -> BOSS-like ``(mu, s)``: mean spectrum and per-pixel extra
    scatter on top of photon noise. Dirty sample: ``mu + s*eps1 + eps2/sqrt(ivar)``."""
    model, payload = load_checkpoint(ckpt)
    graphdef, state = nnx.split(model)
    x_mean = jnp.asarray(payload["x_mean"], jnp.float32)
    x_std = jnp.asarray(payload["x_std"], jnp.float32)
    P = payload["n_features"]

    def bridge(x):
        h = nnx.merge(graphdef, state)((x - x_mean) / x_std)
        return x + h[:, :P], jnp.exp(h[:, P:])

    return bridge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", choices=["draw", "median"], default="draw")
    p.add_argument("--n-median", type=int, default=64, help="prior draws per row for --input median")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-masked", type=float, default=0.3)
    p.add_argument("--max-rows", type=int, default=None, help="subsample each split (smoke tests)")
    p.add_argument("--emulator", default=str(C.EMULATOR_CKPT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(C.PROJECT_ROOT / "trained_model/bridge"))
    p.add_argument("--tag", default="draw_v1")
    a = p.parse_args()
    run = Path(a.outdir) / a.tag; run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1))
    rng = np.random.default_rng(a.seed); P = C.N_PIX

    # ---- data -----------------------------------------------------------------------------
    boss = C.load_boss(); L = boss["labels"]; box = C.label_box()
    ok = (np.isfinite(L).all(1) & C.boss_in_box(boss, box) & ((boss["ivar"] <= 0).mean(1) <= a.max_masked))
    rows = {s: np.flatnonzero(ok & (boss["split"] == i)) for i, s in enumerate(("train", "val", "test"))}
    if a.max_rows:
        rows = {s: np.sort(rng.choice(r, min(a.max_rows, r.size), replace=False)) for s, r in rows.items()}
    C.log("rows " + "  ".join(f"{s} {r.size:,}" for s, r in rows.items()) + f"  input={a.input}")

    emu = C.make_emulator_fn(a.emulator)
    c_lo, c_hi = box["carbon_m"]; v_lo, v_hi = np.log10(box["vmic"])


    def theta6(lab4, key):
        """(n, 4) CLAM labels -> (n, 6) with [C/M], vmic drawn from the Korg-grid prior."""
        k1, k2 = jax.random.split(key)
        n = lab4.shape[0]
        c = jax.random.uniform(k1, (n, 1), minval=c_lo, maxval=c_hi)
        v = 10 ** jax.random.uniform(k2, (n, 1), minval=v_lo, maxval=v_hi)
        return jnp.concatenate([lab4, c, v], 1)


    @jax.jit
    def clean_draw(lab4, key):
        return emu(theta6(lab4, key))


    @jax.jit
    def clean_median(lab4, key):
        keys = jax.random.split(key, a.n_median)
        return jnp.median(jax.vmap(lambda k: emu(theta6(lab4, k)))(keys), axis=0)


    def clean(lab4, key):
        return (clean_draw if a.input == "draw" else clean_median)(jnp.asarray(lab4, jnp.float32), key)


    key = jax.random.PRNGKey(a.seed)
    Xmed = {}
    if a.input == "median":                  # precompute once, host memory
        for s, r in rows.items():
            Xmed[s] = np.concatenate([np.asarray(clean(L[r[i:i + 64]], jax.random.fold_in(key, i)))
                                      for i in range(0, r.size, 64)])
            C.log(f"median inputs {s} {Xmed[s].shape}")

    # input standardisation from train inputs
    r0 = rows["train"][:4096]
    X0 = Xmed["train"][:4096] if a.input == "median" else np.asarray(clean(L[r0], jax.random.PRNGKey(123)))
    x_mean = X0.mean(0).astype("float32"); x_std = np.maximum(X0.std(0), 1e-3).astype("float32")

    # train targets on the device
    tr = rows["train"]
    Ytr, IVtr, Ltr = (jnp.asarray(boss["flux"][tr]), jnp.asarray(boss["ivar"][tr]), jnp.asarray(L[tr], jnp.float32))
    Xtr = jnp.asarray(Xmed["train"]) if a.input == "median" else None

    # ---- model ----------------------------------------------------------------------------
    model = build_model("mlp", P, 2 * P, seed=a.seed, hidden=tuple(a.hidden))
    last = model.layers[-1]                   # start at mu = x, s = 0.01
    last.kernel[...] = last.kernel[...] * 1e-2
    last.bias[...] = jnp.concatenate([jnp.zeros(P), jnp.full(P, np.log(0.01))]).astype(jnp.float32)
    xm, xs = jnp.asarray(x_mean), jnp.asarray(x_std)


    def forward(m, x):
        h = m((x - xm) / xs)
        return x + h[:, :P], h[:, P:]


    def nll(mu, log_s, y, iv):
        good = iv > 0
        v = jnp.where(good, 1.0 / jnp.maximum(iv, 1e-30), 1.0) + jnp.exp(2 * log_s)
        term = jnp.where(good, 0.5 * ((y - mu) ** 2 / v + jnp.log(v)), 0.0)
        return term.sum() / jnp.maximum(good.sum(), 1)


    steps_per_epoch = max(tr.size // a.batch_size, 1)
    sched = optax.cosine_decay_schedule(a.lr, a.epochs * steps_per_epoch, alpha=0.05)
    opt = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0),
                                           optax.adamw(sched, weight_decay=a.weight_decay)), wrt=nnx.Param)


    @nnx.jit
    def train_step(m, opt, x, y, iv):
        loss, g = nnx.value_and_grad(lambda m: nll(*forward(m, x), y, iv))(m)
        opt.update(m, g)
        return loss


    @nnx.jit
    def predict(m, x):
        mu, log_s = forward(m, x)
        return mu, jnp.exp(log_s)


    def batches(split, bs=512):
        """(rows, x, mu, s) over a split; draw-mode inputs use a fixed key -> reproducible."""
        r = rows[split]
        for i in range(0, r.size, bs):
            x = (jnp.asarray(Xmed[split][i:i + bs]) if a.input == "median"
                 else clean(L[r[i:i + bs]], jax.random.fold_in(jax.random.PRNGKey(999), i)))
            mu, s = predict(model, x)
            yield r[i:i + bs], np.asarray(x), np.asarray(mu), np.asarray(s)


    def split_nll(split):
        tot = n = 0
        for r, x, mu, s in batches(split):
            iv = boss["ivar"][r]; good = iv > 0
            v = np.where(good, 1 / np.maximum(iv, 1e-30), 1) + s ** 2
            tot += np.where(good, 0.5 * ((boss["flux"][r] - mu) ** 2 / v + np.log(v)), 0).sum(); n += good.sum()
        return float(tot / n)


    def save(path):
        _, ps, _ = nnx.split(model, nnx.Param, ...)
        payload = dict(model="mlp", model_kwargs={}, hidden=list(a.hidden), dropout=0.0, n_features=P,
                       n_pixels=2 * P, seed=a.seed, x_mean=x_mean, x_std=x_std, config=vars(a),
                       params=jax.tree.map(np.asarray, nnx.to_pure_dict(ps)))
        with open(path, "wb") as f:
            pickle.dump(payload, f)


    # ---- train ------------------------------------------------------------------------------
    hist, best, bad = [], np.inf, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); perm = rng.permutation(tr.size); ls = []
        for i in range(steps_per_epoch):
            key, k = jax.random.split(key)
            idx = jnp.asarray(perm[i * a.batch_size:(i + 1) * a.batch_size])
            x = Xtr[idx] if Xtr is not None else clean_draw(Ltr[idx], k)
            ls.append(train_step(model, opt, x, Ytr[idx], IVtr[idx]))
        tl = float(np.mean(jax.device_get(ls))); vl = split_nll("val")
        hist.append(dict(epoch=ep, train_nll=tl, val_nll=vl))
        flag = ""
        if vl < best:
            best, bad, flag = vl, 0, " *"; save(run / "best.pkl")
        else:
            bad += 1
        C.log(f"epoch {ep:4d}  train {tl:.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s){flag}")
        (run / "history.json").write_text(json.dumps(hist))
        if not np.isfinite(tl) or bad >= a.patience:
            C.log("stopping: " + ("non-finite loss" if not np.isfinite(tl) else f"no val gain in {a.patience}"))
            break

    # ---- test ---------------------------------------------------------------------------------
    model = load_checkpoint(run / "best.pkl")[0]
    rms_x, rms_mu, chi2, in1, in2, ngood = [], [], [], 0, 0, 0
    zsum, z2sum, zn = np.zeros(P), np.zeros(P), np.zeros(P); s_sum = np.zeros(P); ex = None
    for r, x, mu, s in batches("test"):
        y, iv = boss["flux"][r], boss["ivar"][r]; good = iv > 0
        sig = np.sqrt(np.where(good, 1 / np.maximum(iv, 1e-30), 1) + s ** 2)
        z = np.where(good, (y - mu) / sig, 0.0); ng = np.maximum(good.sum(1), 1)
        rms_x += list(np.sqrt((np.where(good, y - x, 0) ** 2).sum(1) / ng))
        rms_mu += list(np.sqrt((np.where(good, y - mu, 0) ** 2).sum(1) / ng))
        chi2 += list((z ** 2).sum(1) / ng)
        in1 += ((np.abs(z) < 1) & good).sum(); in2 += ((np.abs(z) < 2) & good).sum(); ngood += good.sum()
        zsum += z.sum(0); z2sum += (z ** 2).sum(0); zn += good.sum(0); s_sum += s.sum(0)
        if ex is None:
            ex = (r, x, mu, s)
    rms_x, rms_mu, chi2 = map(np.array, (rms_x, rms_mu, chi2))
    zstd = np.sqrt(np.maximum(z2sum / np.maximum(zn, 1) - (zsum / np.maximum(zn, 1)) ** 2, 0))
    met = dict(test_nll=split_nll("test"), n_test=int(rows["test"].size),
               rms_clean_median=float(np.median(rms_x)), rms_bridge_median=float(np.median(rms_mu)),
               chi2_px_median=float(np.median(chi2)), chi2_px_mean=float(chi2.mean()),
               frac_in_1sigma=float(in1 / ngood), frac_in_2sigma=float(in2 / ngood),
               zstd_pixel_median=float(np.median(zstd)), s_median=float(np.median(s_sum / rows["test"].size)))
    (run / "test_metrics.json").write_text(json.dumps(met, indent=1)); C.log(f"test {met}")

    # ---- plots ------------------------------------------------------------------------------------
    pix = np.arange(P); r, x, mu, s = ex
    fig, axs = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for ax, j in zip(axs[:3], rng.choice(len(r), 3, replace=False)):
        iv = boss["ivar"][r[j]]; good = iv > 0; y = np.where(good, boss["flux"][r[j]], np.nan)
        sim = mu[j] + s[j] * rng.standard_normal(P) + rng.standard_normal(P) / np.sqrt(np.maximum(iv, 1e-30))
        ax.plot(pix, y, "k", lw=0.4, label="BOSS")
        ax.plot(pix, np.where(good, sim, np.nan) - 0.3, "C2", lw=0.4, label="bridge sample (offset -0.3)")
        ax.plot(pix, x[j], "C0", lw=0.5, label="clean emu")
        ax.fill_between(pix, mu[j] - s[j], mu[j] + s[j], color="C3", alpha=0.4, lw=0)
        ax.plot(pix, mu[j], "C3", lw=0.5, label=r"bridge $\mu \pm s$")
        ax.set_title(f"sdss_id {boss['sdss_id'][r[j]]}  Teff {L[r[j], 0]:.0f}  logg {L[r[j], 1]:.2f}  "
                     f"SNR {boss['snr'][r[j]]:.0f}", fontsize=9)
    axs[0].legend(fontsize=7, ncol=4)
    axs[3].plot(pix, s_sum / rows["test"].size, "C3", lw=0.6, label="mean s")
    axs[3].plot(pix, zstd, "k", lw=0.4, label="std of z per pixel"); axs[3].axhline(1, color="gray", lw=0.5)
    axs[3].set_yscale("log"); axs[3].legend(fontsize=7); axs[3].set_xlabel("pixel")
    fig.tight_layout(); fig.savefig(run / "test_examples.png", dpi=110)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    h = [e["train_nll"] for e in hist], [e["val_nll"] for e in hist]
    axs[0].plot(h[0], label="train"); axs[0].plot(h[1], label="val"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("NLL"); axs[0].legend()
    axs[1].hist(np.log10(chi2), 100, color="k", histtype="step"); axs[1].axvline(0, color="C3")
    axs[1].set_xlabel(r"log$_{10}$ $\chi^2$/px  (test, z = (y-$\mu$)/$\sqrt{1/iv+s^2}$)")
    bins = np.logspace(-4, 0, 80)
    axs[2].hist(rms_x, bins, histtype="step", label="BOSS - clean"); axs[2].hist(rms_mu, bins, histtype="step", label=r"BOSS - $\mu$")
    axs[2].set_xscale("log"); axs[2].set_xlabel("per-spectrum RMS"); axs[2].legend()
    fig.tight_layout(); fig.savefig(run / "test_summary.png", dpi=110); C.log(f"done -> {run}")


if __name__ == "__main__":
    main()
