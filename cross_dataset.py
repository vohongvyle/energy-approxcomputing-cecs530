"""
Cross-dataset study: how much energy do approximate multipliers save on MNIST, Fashion-MNIST,
CIFAR-10 and CIFAR-100, and what does it cost in accuracy?

For each dataset and seed:
  1. Train a float MLP (input -> 256 ReLU -> classes) with scikit-learn on the official training
     set minus a 5,000-image validation set. Pixels are scaled to [0, 1].
  2. Quantize it to 8 bits as in approx_backend.py: unsigned 8-bit activations, weights as a
     sign plus an 8-bit magnitude, every product read from a multiplier's lookup table.
  3. Run every one of the 36 EvoApproxLib multipliers in both layers, with and without bias
     compensation, on the validation set and on the official 10,000-image test set.

Outputs (results/):
    cross_baselines.csv   float and 8-bit exact accuracy per dataset and seed
    cross_sweep.csv       one row per dataset x seed x multiplier
    cross_summary.csv     energy saved at a fixed accuracy budget, chosen on validation

Usage:
    bash get_data.sh                                  # once, about 400 MB
    python3 build_luts.py --lib evoapproxlib --out luts
    python3 cross_dataset.py                          # all four datasets, 3 seeds (about 1 hour)
    python3 cross_dataset.py --datasets mnist --seeds 1
    python3 plot_cross_dataset.py
"""
import argparse, csv, os, time, warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

import datasets
from approx_backend import ADDER_PWR, EXACT, load_luts
from fastlut import lut_matmul

warnings.filterwarnings("ignore", category=ConvergenceWarning)
BUDGETS = [0.5, 1.0, 2.0]          # allowed accuracy loss, percentage points


class QuantMLP:
    """8-bit MLP whose every multiplication is read from a multiplier's lookup table."""

    def __init__(self, clf, Xcal):
        self.layers, a = [], Xcal
        for k, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            sa = max(np.percentile(a, 99.9), 1e-8) / 255.0
            sw = np.abs(W).max() / 255.0
            q = np.clip(np.round(W / sw), -255, 255).astype(np.int64)
            self.layers.append(dict(sa=sa, sw=sw, sgn=np.sign(q).astype(np.int8),
                                    mag=np.abs(q).astype(np.uint8), b=b))
            a = a @ W + b
            if k < len(clf.coefs_) - 1:
                a = np.maximum(a, 0)
        self.macs = [L["mag"].size for L in self.layers]

    def accumulate(self, k, a, lut):
        """Integer accumulators of layer k: sum_i sign * T[a_q, m]."""
        L = self.layers[k]
        xq = np.clip(np.round(a / L["sa"]), 0, 255).astype(np.uint8)
        return lut_matmul(xq, L["mag"], L["sgn"], lut).astype(np.float64)

    def finish(self, k, acc, corr=None):
        """Scale, add the bias (minus the compensation term, if any) and apply ReLU."""
        L = self.layers[k]
        if corr is not None:
            acc = acc - corr[k]
        z = acc * (L["sa"] * L["sw"]) + L["b"]
        return z if k == len(self.layers) - 1 else np.maximum(z, 0)

    def forward_both(self, X, lut, corr):
        """Logits without and with bias compensation. Layer 1 accumulators are shared."""
        acc1 = self.accumulate(0, X, lut)
        out = []
        for c in (None, corr):
            a = self.finish(0, acc1, c)
            for k in range(1, len(self.layers)):
                a = self.finish(k, self.accumulate(k, a, lut), c)
            out.append(a)
        return out

    def forward(self, X, lut):
        a = X
        for k in range(len(self.layers)):
            a = self.finish(k, self.accumulate(k, a, lut))
        return a

    def input_histograms(self, Xcal, exact_lut):
        """Distribution of every layer input on calibration data, with exact upstream layers."""
        self.hists, a = [], Xcal
        for k, L in enumerate(self.layers):
            xq = np.clip(np.round(a / L["sa"]), 0, 255).astype(np.int64)
            self.hists.append(np.stack([np.bincount(col, minlength=256) for col in xq.T]) / len(xq))
            a = self.finish(k, self.accumulate(k, a, exact_lut))

    def corrections(self, lut):
        """Bias compensation: the expected error of each product, summed per neuron. It is a
        constant per neuron, so in hardware it only changes the bias."""
        err = lut.astype(np.float64) - np.outer(np.arange(256), np.arange(256))
        corr = []
        for L, hist in zip(self.layers, self.hists):
            Em = hist @ err                                         # (in, 256): mean error per weight
            idx = np.arange(L["mag"].shape[0])[:, None]
            corr.append((L["sgn"] * Em[idx, L["mag"]]).sum(0))     # (out,)
        return corr


def run(name, seed, names, luts, meta, hidden, epochs, n_val=5000):
    Xtr, ytr, Xte, yte = datasets.load(name)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    va, tr = perm[:n_val], perm[n_val:]
    f = lambda X: X.astype(np.float64) / 255.0
    Xtrain, ytrain, Xval, yval, Xtest = f(Xtr[tr]), ytr[tr], f(Xtr[va]), ytr[va], f(Xte)

    t0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), batch_size=256, max_iter=epochs,
                        alpha=1e-4, random_state=seed).fit(Xtrain, ytrain)
    t_train = time.time() - t0
    Xcal = Xtrain[rng.choice(len(Xtrain), 5000, replace=False)]
    model = QuantMLP(clf, Xcal)
    ex = names.index(EXACT)
    acc = lambda logits, y: float((clf.classes_[logits.argmax(1)] == y).mean())

    exact_val = model.forward(Xval, luts[ex])
    exact_te = model.forward(Xtest, luts[ex])
    base = dict(dataset=name, seed=seed, train_s=round(t_train, 1), epochs=clf.n_iter_,
                float_test=float((clf.predict(Xtest) == yte).mean()),
                int8_val=acc(exact_val, yval), int8_test=acc(exact_te, yte))
    pred_exact = exact_te.argmax(1)
    model.input_histograms(Xcal, luts[ex])

    rows, e_exact = [], meta[EXACT]["pwr"] + ADDER_PWR
    for i, n in enumerate(names):
        corr = model.corrections(luts[i])
        r = dict(dataset=name, seed=seed, circuit=n, pwr_mW=meta[n]["pwr"], mae_pct=meta[n]["mae%"],
                 mult_energy=meta[n]["pwr"] / meta[EXACT]["pwr"],
                 mac_energy=(meta[n]["pwr"] + ADDER_PWR) / e_exact)
        val = model.forward_both(Xval, luts[i], corr)
        te = model.forward_both(Xtest, luts[i], corr)
        for tag, lv, lt in (("", val[0], te[0]), ("_comp", val[1], te[1])):
            r["val_acc" + tag], r["test_acc" + tag] = acc(lv, yval), acc(lt, yte)
            r["flip" + tag] = float((lt.argmax(1) != pred_exact).mean())   # share of test predictions changed
        rows.append(r)
    return base, rows


def summarize(base_rows, sweep):
    """Per dataset, seed and budget: the cheapest multiplier whose validation accuracy stays within
    the budget of the 8-bit exact model (compensation on if it helped on validation), then its test
    numbers. Averaged over seeds."""
    out = []
    for name in dict.fromkeys(r["dataset"] for r in base_rows):
        for budget in BUDGETS:
            picks = []
            for b in (b for b in base_rows if b["dataset"] == name):
                cand = []
                for r in (r for r in sweep if r["dataset"] == name and r["seed"] == b["seed"]):
                    comp = r["val_acc_comp"] > r["val_acc"]
                    v = r["val_acc_comp"] if comp else r["val_acc"]
                    if v >= b["int8_val"] - budget / 100 - 1e-12:
                        cand.append((r["mac_energy"], r, comp))
                e, r, comp = min(cand, key=lambda c: c[0])
                t = r["test_acc_comp"] if comp else r["test_acc"]
                picks.append(dict(circuit=r["circuit"], comp=comp, mac=e, mult=r["mult_energy"],
                                  drop=100 * (b["int8_test"] - t)))
            out.append(dict(dataset=name, budget_pp=budget,
                            circuits="/".join(sorted({p["circuit"] + ("+c" if p["comp"] else "") for p in picks})),
                            mult_saved_pct=round(100 * (1 - np.mean([p["mult"] for p in picks])), 1),
                            mac_saved_pct=round(100 * (1 - np.mean([p["mac"] for p in picks])), 1),
                            test_drop_pp=round(np.mean([p["drop"] for p in picks]), 2),
                            test_drop_sd=round(np.std([p["drop"] for p in picks]), 2)))
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=datasets.NAMES, choices=datasets.NAMES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names, luts, meta = load_luts(args.luts)

    bases, sweep = [], []
    for name in args.datasets:
        for s in range(args.seeds):
            t0 = time.time()
            b, rows = run(name, s, names, luts, meta, args.hidden, args.epochs)
            bases.append(b); sweep += rows
            print(f"{datasets.PRETTY[name]:14s} seed {s}: float {b['float_test']:.4f}  8-bit exact {b['int8_test']:.4f}"
                  f"  ({b['epochs']} epochs, {time.time() - t0:.0f} s)", flush=True)
            write_csv(os.path.join(args.out, "cross_baselines.csv"), bases)     # saved as it goes
            write_csv(os.path.join(args.out, "cross_sweep.csv"), sweep)

    summary = summarize(bases, sweep)
    write_csv(os.path.join(args.out, "cross_summary.csv"), summary)
    print(f"\nEnergy saved at a fixed accuracy budget (chosen on validation, test drop shown), mean of {args.seeds} seeds")
    print(f"{'dataset':14s} {'budget':>6s} {'mult saved':>10s} {'MAC saved':>9s} {'test drop':>10s}  multipliers")
    for r in summary:
        print(f"{datasets.PRETTY[r['dataset']]:14s} {r['budget_pp']:5.1f}p {r['mult_saved_pct']:9.1f}% {r['mac_saved_pct']:8.1f}%"
              f" {r['test_drop_pp']:7.2f} pp  {r['circuits']}")


if __name__ == "__main__":
    main()
