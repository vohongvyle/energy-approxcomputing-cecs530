"""
Backend for Energy-Efficient Approximate Computing in ML (EvoApproxLib 8x8 multipliers).

Three improvements over run_experiment.py:
  1. Fairer energy model: one MAC = multiplier + exact 16-bit accumulator adder
     (EvoApproxLib add16u_1E2, 0.072 mW), not the multiplier alone.
  2. Bias compensation at zero hardware cost: the expected error of each approximate
     product is estimated on calibration data and folded into the neuron's bias.
  3. Layer-wise search: pick a (possibly different) multiplier per layer on a validation
     set, then report on an untouched test set. Results are averaged over several seeds.

Usage:
    python build_luts.py --lib evoapproxlib --out luts      # once
    python approx_backend.py                                 # digits, 3 seeds
    python approx_backend.py --dataset mnist --seeds 1       # needs internet
"""
import argparse, csv, json, os, itertools
import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

EXACT = "mul8u_1JFF"
ADDER_PWR = 0.072   # mW, exact 16-bit adder add16u_1E2 (EvoApproxLib, PDK45)


# --------------------------------------------------------------------- data
def load_data(name, seed):
    if name == "digits":
        X, y = load_digits(return_X_y=True); X = X / 16.0
    else:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X, y = X / 255.0, y.astype(int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    return Xtr, Xval, Xte, ytr, yval, yte


def load_luts(folder):
    meta = json.load(open(os.path.join(folder, "circuits.json")))
    names = sorted(meta, key=lambda n: -meta[n]["pwr"])
    luts = np.stack([np.load(os.path.join(folder, n + ".npy")) for n in names]).astype(np.int64)
    return names, luts, meta


# --------------------------------------------------------------------- model
class QuantMLP:
    """8-bit MLP: uint8 activations, weights as sign + 8-bit magnitude, any depth."""

    def __init__(self, clf, Xcal, luts):
        self.luts = luts                                             # (C, 256, 256)
        self.layers = []
        a = Xcal
        for k, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            sa = max(np.percentile(a, 99.9), 1e-8) / 255.0          # activation scale
            sw = np.abs(W).max() / 255.0                             # weight scale
            q = np.clip(np.round(W / sw), -255, 255).astype(np.int64)
            self.layers.append(dict(sa=sa, sw=sw, sign=np.sign(q), mag=np.abs(q), b=b))
            a = a @ W + b
            if k < len(clf.coefs_) - 1:
                a = np.maximum(a, 0)
        self.macs = [L["mag"].size for L in self.layers]

    @staticmethod
    def qact(a, s):
        return np.clip(np.round(a / s), 0, 255).astype(np.int64)

    @staticmethod
    def matmul(xq, sign, mag, lut, batch=128):
        out = np.empty((len(xq), mag.shape[1]), dtype=np.int64)
        for i in range(0, len(xq), batch):
            p = lut[xq[i:i + batch, :, None], mag[None]]           # (B, in, out) table lookups
            out[i:i + batch] = (p * sign[None]).sum(1)
        return out

    def layer(self, k, a, c, comp):
        """Layer k using circuit index c; comp=True subtracts the calibrated error."""
        L = self.layers[k]
        xq = self.qact(a, L["sa"])
        acc = self.matmul(xq, L["sign"], L["mag"], self.luts[c])
        if comp:
            acc = acc - L["corr"][c]
        z = acc * (L["sa"] * L["sw"]) + L["b"]
        return z if k == len(self.layers) - 1 else np.maximum(z, 0)

    def forward(self, X, circuits, comp=False):
        a = X
        for k, c in enumerate(circuits):
            a = self.layer(k, a, c, comp)
        return a

    def calibrate(self, Xcal, exact_idx):
        """Bias compensation: for every layer and candidate circuit, estimate the mean
        product error E_x[lut(x,m) - x*m] using the real distribution of each input x_i
        (seen with exact upstream layers), and store sum_i sign_io * E_i(m_io) per neuron.
        It is a constant per neuron, so in hardware it simply changes the bias."""
        exact = np.outer(np.arange(256), np.arange(256))
        a = Xcal
        for k, L in enumerate(self.layers):
            xq = self.qact(a, L["sa"])                               # (N, in)
            hist = np.stack([np.bincount(col, minlength=256) for col in xq.T]) / len(xq)  # (in, 256)
            Eim = np.einsum("ix,cxm->cim", hist, self.luts - exact)  # (C, in, 256) mean error per weight m
            idx = np.arange(L["mag"].shape[0])[:, None]
            L["corr"] = (L["sign"][None] * Eim[:, idx, L["mag"]]).sum(1)  # (C, out)
            a = self.layer(k, a, exact_idx, comp=False)


# --------------------------------------------------------------------- energy
def energy(model, pwrs):
    """Relative energy per inference: sum_l MACs_l * (mult_pwr_l + adder_pwr)."""
    return sum(m * (p + ADDER_PWR) for m, p in zip(model.macs, pwrs))


# --------------------------------------------------------------------- experiments
def run_seed(args, seed, names, luts, meta):
    Xtr, Xval, Xte, ytr, yval, yte = load_data(args.dataset, seed)
    clf = MLPClassifier(hidden_layer_sizes=(args.hidden,), max_iter=400, random_state=seed).fit(Xtr, ytr)
    lab = clf.classes_
    model = QuantMLP(clf, Xtr, luts)
    ex = names.index(EXACT)
    model.calibrate(Xtr, ex)
    e_exact = energy(model, [meta[EXACT]["pwr"]] * len(model.layers))
    acc = lambda X, y, ls, c: float((lab[model.forward(X, ls, c).argmax(1)] == y).mean())
    base = dict(float=float((clf.predict(Xte) == yte).mean()), int8=acc(Xte, yte, [ex] * 2, False))

    # 1) same circuit in every layer: without compensation, with it, and "auto"
    #    (compensation switched on only if it helps on the validation set)
    sweep = []
    for i, n in enumerate(names):
        ls = [i] * len(model.layers)
        use = acc(Xval, yval, ls, True) > acc(Xval, yval, ls, False)
        a, ac = acc(Xte, yte, ls, False), acc(Xte, yte, ls, True)
        sweep.append(dict(circuit=n, pwr=meta[n]["pwr"], mae_pct=meta[n]["mae%"],
                          rel_energy=energy(model, [meta[n]["pwr"]] * 2) / e_exact,
                          acc=a, acc_comp=ac, acc_auto=ac if use else a))

    # 2) layer-wise search on validation over (circuit, compensation on/off) per layer.
    #    Layer-1 outputs are cached, so each candidate pair only re-runs the small layer 2.
    opts = [(i, c) for i in range(len(names)) for c in (False, True)]
    h_val = {o: model.layer(0, Xval, *o) for o in opts}
    h_te = {o: model.layer(0, Xte, *o) for o in opts}
    val_exact = acc(Xval, yval, [ex] * 2, False)
    combos = []
    for o1, o2 in itertools.product(opts, repeat=2):
        va = float((lab[model.layer(1, h_val[o1], *o2).argmax(1)] == yval).mean())
        en = energy(model, [meta[names[o1[0]]]["pwr"], meta[names[o2[0]]]["pwr"]]) / e_exact
        combos.append((o1, o2, va, en))
    picks = []
    for budget in args.budgets:
        ok = [c for c in combos if c[2] >= val_exact - budget / 100 - 1e-9]
        o1, o2, va, en = min(ok, key=lambda c: (c[3], -c[2]))
        te = float((lab[model.layer(1, h_te[o1], *o2).argmax(1)] == yte).mean())
        picks.append(dict(budget_pp=budget, layer1=names[o1[0]], comp1=o1[1],
                          layer2=names[o2[0]], comp2=o2[1], rel_energy=en, val_acc=va, test_acc=te))
    return base, sweep, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                    help="allowed validation accuracy drop, percentage points")
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names, luts, meta = load_luts(args.luts)

    bases, sweeps, picks = [], [], []
    for s in range(args.seeds):
        b, sw, pk = run_seed(args, s, names, luts, meta)
        bases.append(b); sweeps.append(sw); picks.append(pk)
        print(f"seed {s}: float {b['float']:.4f}  int8 exact {b['int8']:.4f}")

    # average the sweep over seeds
    rows = []
    for k, n in enumerate(names):
        a, c, u = (np.array([sw[k][f] for sw in sweeps]) for f in ("acc", "acc_comp", "acc_auto"))
        rows.append(dict(circuit=n, pwr_mW=meta[n]["pwr"], mae_pct=meta[n]["mae%"],
                         rel_energy=round(sweeps[0][k]["rel_energy"], 4),
                         acc_mean=round(a.mean(), 4), acc_std=round(a.std(), 4),
                         acc_comp_mean=round(c.mean(), 4), acc_comp_std=round(c.std(), 4),
                         acc_auto_mean=round(u.mean(), 4), acc_auto_std=round(u.std(), 4)))
    int8 = np.mean([b["int8"] for b in bases])
    write_csv(os.path.join(args.out, f"sweep_{args.dataset}.csv"), rows)
    write_csv(os.path.join(args.out, f"layer_search_{args.dataset}.csv"),
              [dict(seed=s, **p) for s, pk in enumerate(picks) for p in pk])

    print(f"\nMean 8-bit exact test accuracy: {int8:.4f}  (energy = multiplier + {ADDER_PWR} mW adder per MAC)")
    print(f"\n{'circuit':12s} {'energy':>7s} {'plain':>15s} {'compensated':>15s} {'auto (val)':>15s}")
    for r in rows:
        print(f"{r['circuit']:12s} {r['rel_energy']:7.3f} {r['acc_mean']:8.4f}±{r['acc_std']:.4f} "
              f"{r['acc_comp_mean']:8.4f}±{r['acc_comp_std']:.4f} {r['acc_auto_mean']:8.4f}±{r['acc_auto_std']:.4f}")
    print("\nLayer-wise search (chosen on validation, reported on test), mean over seeds:")
    for bi, bud in enumerate(args.budgets):
        ps = [pk[bi] for pk in picks]
        print(f"  budget {bud:3.1f} pp: energy {np.mean([p['rel_energy'] for p in ps]):.3f}  "
              f"test acc {np.mean([p['test_acc'] for p in ps]):.4f}  "
              f"e.g. {ps[0]['layer1']}{'+comp' if ps[0]['comp1'] else ''} / {ps[0]['layer2']}{'+comp' if ps[0]['comp2'] else ''}")
    plot(rows, int8, picks, args)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def plot(rows, int8, picks, args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    e = [r["rel_energy"] for r in rows]
    ax.scatter(e, [100 * r["acc_mean"] for r in rows], s=30, color="#2a78d6", edgecolor="white",
               label="Same circuit in both layers", zorder=3)
    ax.scatter(e, [100 * r["acc_auto_mean"] for r in rows], s=30, color="#eb6834", marker="s",
               edgecolor="white", label="… with bias compensation where it helps (chosen on validation)", zorder=3)
    for bi, bud in enumerate(args.budgets):
        ps = [pk[bi] for pk in picks]
        ax.scatter(np.mean([p["rel_energy"] for p in ps]), 100 * np.mean([p["test_acc"] for p in ps]),
                   s=70, marker="D", color="#1baf7a", edgecolor="white", zorder=4,
                   label="Layer-wise search (budgets)" if bi == 0 else None)
    ax.axhline(100 * int8, color="#888", ls="--", lw=1)
    ax.text(1.0, 100 * int8 + 0.3, "8-bit exact", ha="right", fontsize=8, color="#555")
    ax.set_ylim(88, 100); ax.set_xlim(0, 1.04)
    ax.set_xlabel("MAC energy per inference (multiplier + adder), relative to exact")
    ax.set_ylabel(f"Test accuracy (%), mean of {args.seeds} seeds")
    ax.set_title("Accuracy vs energy with bias compensation and layer-wise search", loc="left", fontsize=11)
    ax.grid(True, color="#e6e6e6"); [ax.spines[s].set_visible(False) for s in ["top", "right"]]
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"backend_{args.dataset}.png"))


if __name__ == "__main__":
    main()
