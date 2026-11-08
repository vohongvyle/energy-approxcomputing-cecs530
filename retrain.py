"""
Retraining with approximate multipliers: how much accuracy does a few epochs of fine-tuning
recover, compared with bias compensation, which costs nothing?

For each dataset and seed we train the same MLP as cross_dataset.py (input -> 256 -> classes), then
for each multiplier compare on the test set:
    plain          the trained network, approximate multiplier dropped in
    compensated    plus bias compensation (expected product error removed from each neuron's bias)
    retrained      fine-tuned for a few epochs with the approximate multiplier in the forward pass
    both           retrained, then bias-compensated
A control fine-tunes the exact 8-bit network for the same epochs, so that any gain from simply
training longer is not credited to retraining.

Fine-tuning uses a straight-through estimator: the forward pass is the real 8-bit approximate
computation (every product from the lookup table), and the backward pass treats it as the ordinary
float layer y = x W + b. Quantization scales stay fixed during fine-tuning.

Usage:
    python3 retrain.py                                    # 4 datasets x 2 seeds, about 1 hour
    python3 retrain.py --datasets mnist --seeds 1 --epochs 1
Outputs: results/retrain.csv, results/retrain.png
"""
import argparse, csv, os, time, warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

import datasets
from approx_backend import EXACT, load_luts
from fastlut import lut_matmul

warnings.filterwarnings("ignore", category=ConvergenceWarning)
MULTIPLIERS = ["mul8u_QKX", "mul8u_13QR", "mul8u_17QU"]


class QNet:
    """8-bit MLP with float master weights, for straight-through fine-tuning."""

    def __init__(self, clf, Xcal):
        self.W = [w.astype(np.float32).copy() for w in clf.coefs_]
        self.b = [b.astype(np.float32).copy() for b in clf.intercepts_]
        self.sa, self.sw, a = [], [], Xcal
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            self.sa.append(max(float(np.percentile(a, 99.9)), 1e-8) / 255.0)
            self.sw.append(float(np.abs(W).max()) / 255.0)
            a = a @ W + b
            if k < len(self.W) - 1:
                a = np.maximum(a, 0)

    def copy(self):
        new = QNet.__new__(QNet)
        new.W, new.b = [w.copy() for w in self.W], [b.copy() for b in self.b]
        new.sa, new.sw = list(self.sa), list(self.sw)
        return new

    def qweights(self, k):
        q = np.clip(np.round(self.W[k] / self.sw[k]), -255, 255)
        return np.sign(q).astype(np.int8), np.abs(q).astype(np.uint8)

    def forward(self, X, lut, corr=None, keep=False):
        a, cache = X, []
        for k in range(len(self.W)):
            xq = np.clip(np.round(a / self.sa[k]), 0, 255).astype(np.uint8)
            sgn, mag = self.qweights(k)
            acc = lut_matmul(xq, mag, sgn, lut).astype(np.float32)
            if corr is not None:
                acc -= corr[k]
            z = acc * np.float32(self.sa[k] * self.sw[k]) + self.b[k]
            cache.append((xq.astype(np.float32) * np.float32(self.sa[k]), z))
            a = z if k == len(self.W) - 1 else np.maximum(z, 0)
        return (a, cache) if keep else a

    def corrections(self, Xcal, lut, exact_lut):
        err = lut.astype(np.float64) - np.outer(np.arange(256), np.arange(256))
        corr, a = [], Xcal
        for k in range(len(self.W)):
            xq = np.clip(np.round(a / self.sa[k]), 0, 255).astype(np.int64)
            hist = np.stack([np.bincount(col, minlength=256) for col in xq.T]) / len(xq)
            sgn, mag = self.qweights(k)
            idx = np.arange(mag.shape[0])[:, None]
            corr.append(((sgn * (hist @ err)[idx, mag]).sum(0)).astype(np.float32))
            sgn_e, mag_e = sgn, mag
            z = lut_matmul(xq.astype(np.uint8), mag_e, sgn_e, exact_lut) * (self.sa[k] * self.sw[k]) + self.b[k]
            a = z if k == len(self.W) - 1 else np.maximum(z, 0)
        return corr

    def finetune(self, X, y, lut, epochs, lr=3e-4, batch=256, seed=0):
        rng = np.random.default_rng(seed)
        m = [np.zeros_like(p) for p in self.W + self.b]; v = [np.zeros_like(p) for p in self.W + self.b]
        t = 0
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                bi = order[i:i + batch]
                z, cache = self.forward(X[bi], lut, keep=True)
                z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
                d = p; d[np.arange(len(bi)), y[bi]] -= 1; d /= len(bi)
                grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
                for k in reversed(range(len(self.W))):
                    a_in, _ = cache[k]
                    grads_W[k] = a_in.T @ d; grads_b[k] = d.sum(0)
                    if k > 0:
                        d = (d @ self.W[k].T) * (cache[k - 1][1] > 0)
                t += 1
                for j, (prm, g) in enumerate(zip(self.W + self.b, grads_W + grads_b)):
                    m[j] = 0.9 * m[j] + 0.1 * g; v[j] = 0.999 * v[j] + 0.001 * g * g
                    prm -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
                for k in range(len(self.W)):          # keep weights inside the fixed 8-bit range
                    np.clip(self.W[k], -255 * self.sw[k], 255 * self.sw[k], out=self.W[k])
        return self


def run(name, seed, names, luts, epochs, hidden=256, n_val=5000):
    Xtr, ytr, Xte, yte = datasets.load(name)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    tr = perm[n_val:]
    f = lambda X: (X / 255.0).astype(np.float32)
    Xtrain, ytrain, Xtest = f(Xtr[tr]), ytr[tr], f(Xte)
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), batch_size=256, max_iter=30, alpha=1e-4,
                        random_state=seed).fit(Xtrain, ytrain)
    Xcal = Xtrain[rng.choice(len(Xtrain), 5000, replace=False)]
    base = QNet(clf, Xcal)
    ex = luts[names.index(EXACT)]
    acc = lambda net, lut, corr=None: 100 * float((net.forward(Xtest, lut, corr).argmax(1) == yte).mean())

    rows = []
    a_exact = acc(base, ex)
    control = base.copy().finetune(Xtrain, ytrain, ex, epochs, seed=seed)
    rows.append(dict(dataset=name, seed=seed, circuit=EXACT, exact=a_exact, plain=a_exact, compensated=a_exact,
                     retrained=acc(control, ex), both=acc(control, ex)))
    for n in MULTIPLIERS:
        lut = luts[names.index(n)]
        net = base.copy().finetune(Xtrain, ytrain, lut, epochs, seed=seed)
        rows.append(dict(dataset=name, seed=seed, circuit=n, exact=a_exact,
                         plain=acc(base, lut), compensated=acc(base, lut, base.corrections(Xcal, lut, ex)),
                         retrained=acc(net, lut), both=acc(net, lut, net.corrections(Xcal, lut, ex))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=datasets.NAMES, choices=datasets.NAMES)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/retrain.csv")
    args = ap.parse_args()
    if args.plot_only:
        rows = [{k: (v if k in ("dataset", "circuit") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(open(os.path.join(args.out, "retrain.csv")))]
        return plot(rows, args.out)
    names, luts, _ = load_luts(args.luts)
    rows = []
    for d in args.datasets:
        for s in range(args.seeds):
            t0 = time.time()
            rs = run(d, s, names, luts, args.epochs)
            rows += rs
            for r in rs:
                print(f"{datasets.PRETTY[d]:14s} seed {s} {r['circuit']:11s} exact {r['exact']:6.2f}  plain {r['plain']:6.2f}"
                      f"  comp {r['compensated']:6.2f}  retrained {r['retrained']:6.2f}  both {r['both']:6.2f}", flush=True)
            print(f"  ({time.time() - t0:.0f} s)", flush=True)
            with open(os.path.join(args.out, "retrain.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    plot(rows, args.out)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK = "#1f2a37"
    sets = [d for d in datasets.NAMES if any(r["dataset"] == d for r in rows)]
    variants = [("plain", "plain", "#c9d2dc"), ("compensated", "bias-compensated", "#8fa3b8"),
                ("retrained", "retrained", "#2f6fb0"), ("both", "retrained + compensated", "#173d66")]
    fig, axes = plt.subplots(1, len(MULTIPLIERS), figsize=(4.4 * len(MULTIPLIERS), 3.8), dpi=150, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), MULTIPLIERS):
        w = 0.8 / len(variants)
        for j, (k, lab, col) in enumerate(variants):
            vals = [np.mean([r["exact"] - r[k] for r in rows if r["dataset"] == d and r["circuit"] == n]) for d in sets]
            ax.bar(np.arange(len(sets)) + (j - 1.5) * w, np.maximum(vals, 0), w * 0.92, color=col, label=lab)
        ax.set_xticks(range(len(sets)), [datasets.PRETTY[d] for d in sets], fontsize=8)
        ax.set_title(n, loc="left", fontsize=10, color=INK)
        ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 100)
        ax.set_yticks([0, 1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50"])
        ax.grid(axis="y", color="#eceef0"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("accuracy drop vs 8-bit exact (points)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Retraining recovers much of the loss on MNIST and Fashion-MNIST, little on CIFAR-100; compensating a retrained network hurts",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, "retrain.png"), bbox_inches="tight")
    print("saved results/retrain.csv, results/retrain.png")


if __name__ == "__main__":
    main()
