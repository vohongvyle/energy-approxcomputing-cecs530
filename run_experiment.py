"""
Step 2: exact vs approximate inference of an 8-bit quantized MLP.

Pipeline
  1. Train a small float MLP (scikit-learn).
  2. Quantize it: activations -> uint8 (0..255), weights -> sign + 8-bit magnitude.
  3. Run integer inference where every multiplication x*|w| is a lookup in the
     256x256 table of an EvoApproxLib multiplier (exact one = mul8u_1JFF).
  4. Record accuracy and multiplier power for every circuit, and for three
     placements: approximate in layer 1 only, layer 2 only, or both.

Energy model (deliberately simple, state it in your report):
  energy per inference  ~  sum over layers of (#multiplications x multiplier power)
  using the PDK45 power numbers published with EvoApproxLib. This covers the
  multipliers only, not adders, memory or data movement.

Usage:
    python run_experiment.py                 # sklearn digits (bundled, offline)
    python run_experiment.py --dataset mnist # needs internet to download MNIST
"""
import argparse, json, os, csv
import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------- data + model
def load_data(name, seed):
    if name == "digits":
        X, y = load_digits(return_X_y=True)
        X = X / 16.0
    else:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X, y = X / 255.0, y.astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def train(Xtr, ytr, hidden, seed):
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu",
                        max_iter=300, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


# ---------------------------------------------------------------- quantization
def quantize_weights(W):
    """Symmetric per-layer: W ~= s * sign * mag, with mag in 0..255."""
    s = np.abs(W).max() / 255.0
    q = np.clip(np.round(W / s), -255, 255).astype(np.int64)
    return np.sign(q), np.abs(q), s


def quantize_act(a, scale):
    return np.clip(np.round(a / scale), 0, 255).astype(np.int64)


class QuantMLP:
    def __init__(self, clf, Xcal):
        W1, W2 = clf.coefs_
        self.b1, self.b2 = clf.intercepts_
        self.sg1, self.m1, self.sw1 = quantize_weights(W1)
        self.sg2, self.m2, self.sw2 = quantize_weights(W2)
        # activation scales calibrated on training data (99.9th percentile)
        self.sx = np.percentile(Xcal, 99.9) / 255.0 or 1 / 255.0
        h = np.maximum(Xcal @ W1 + self.b1, 0)
        self.sh = np.percentile(h, 99.9) / 255.0
        self.macs = (W1.size, W2.size)  # multiplications per inference

    @staticmethod
    def layer(xq, sign, mag, lut, batch=256):
        """Integer matmul: sum_i sign_ij * LUT[x_i, |w_ij|]."""
        out = np.empty((xq.shape[0], mag.shape[1]), dtype=np.int64)
        for i in range(0, xq.shape[0], batch):
            xb = xq[i:i + batch]
            prod = lut[xb[:, :, None], mag[None, :, :]]       # (B, in, out)
            out[i:i + batch] = (prod * sign[None]).sum(axis=1)
        return out

    def predict(self, X, lut1, lut2):
        xq = quantize_act(X, self.sx)
        acc1 = self.layer(xq, self.sg1, self.m1, lut1)
        h = np.maximum(acc1 * (self.sx * self.sw1) + self.b1, 0)
        hq = quantize_act(h, self.sh)
        acc2 = self.layer(hq, self.sg2, self.m2, lut2)
        logits = acc2 * (self.sh * self.sw2) + self.b2
        return logits.argmax(axis=1)


# ---------------------------------------------------------------- experiment
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Xtr, Xte, ytr, yte = load_data(args.dataset, args.seed)
    clf = train(Xtr, ytr, args.hidden, args.seed)
    labels = clf.classes_
    float_acc = (clf.predict(Xte) == yte).mean()

    q = QuantMLP(clf, Xtr)
    circuits = json.load(open(os.path.join(args.luts, "circuits.json")))
    luts = {n: np.load(os.path.join(args.luts, n + ".npy")) for n in circuits}
    exact_name = "mul8u_1JFF"
    p_exact = circuits[exact_name]["pwr"]
    exact_lut = luts[exact_name]

    def acc(l1, l2):
        return (labels[q.predict(Xte, l1, l2)] == yte).mean()

    int8_acc = acc(exact_lut, exact_lut)
    print(f"Dataset {args.dataset}: {len(Xtr)} train / {len(Xte)} test, MLP {Xtr.shape[1]}-{args.hidden}-{len(labels)}")
    print(f"Float accuracy        : {float_acc:.4f}")
    print(f"8-bit exact accuracy  : {int8_acc:.4f}   (multiplications/inference: {sum(q.macs)})\n")

    m1, m2 = q.macs
    rows = []
    for name, c in sorted(circuits.items(), key=lambda kv: -kv[1]["pwr"]):
        p = c["pwr"]
        for where, l1, l2, energy in [
            ("both",    luts[name], luts[name], (m1 + m2) * p),
            ("layer1",  luts[name], exact_lut,  m1 * p + m2 * p_exact),
            ("layer2",  exact_lut,  luts[name], m1 * p_exact + m2 * p),
        ]:
            a = acc(l1, l2)
            rows.append(dict(circuit=name, approx_in=where, mult_power_mW=p,
                             mae_pct=c["mae%"], wce_pct=c["wce%"], accuracy=round(a, 4),
                             acc_drop=round(int8_acc - a, 4),
                             rel_mult_energy=round(energy / ((m1 + m2) * p_exact), 4)))
        r = rows[-3]
        print(f"{name:12s} pwr={p:.4f}  MAE%={c['mae%']:<7}  acc(both)={r['accuracy']:.4f}  "
              f"rel.energy={r['rel_mult_energy']:.3f}")

    with open(os.path.join(args.out, f"results_{args.dataset}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    json.dump(dict(dataset=args.dataset, float_acc=float_acc, int8_acc=int8_acc,
                   macs=q.macs, hidden=args.hidden),
              open(os.path.join(args.out, f"baseline_{args.dataset}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
