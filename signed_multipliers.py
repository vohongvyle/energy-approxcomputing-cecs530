"""
Signed 8x8 multipliers (mul8s_*) instead of unsigned multipliers with sign-magnitude weights.

Our main experiments feed an unsigned 8x8 multiplier with an 8-bit activation (0..255) and an 8-bit
weight magnitude, and handle the weight's sign separately. A signed multiplier takes two's-complement
inputs (-128..127) directly. Activations after ReLU are never negative, so with a signed multiplier
they get only 7 bits (0..127) and weights 8 bits (-127..127).

This script builds lookup tables for EvoApproxLib's 13 signed 8x8 multipliers and runs them in the
MLP of cross_dataset.py (seed 0), next to the unsigned results for the same network.

Usage:
    python3 signed_multipliers.py --lib evoapproxlib
Outputs: luts_signed/, results/signed.csv, results/signed.png
"""
import argparse, csv, ctypes, json, os, subprocess, tempfile
import numpy as np

import datasets
from approx_backend import ADDER_PWR, EXACT, load_luts
from cross_dataset import QuantMLP
from fastlut import lut_matmul
from mlp_cache import trained_mlp


def find_signed(lib):
    meta = json.load(open(os.path.join(lib, "meta.json")))
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if "instances" in o and "8x8_signed" in o.get("folder", ""):
                for inst in o["instances"]:
                    c = next(f["file"] for f in inst["files"] if f["file"].endswith(".c"))
                    found.setdefault(inst["name"], dict(params={k: float(v) for k, v in inst["params"].items()},
                                                        c=os.path.join(lib, o["folder"], c)))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(meta)
    return found


def build_signed_luts(lib, out):
    """lut[a][b] = product of the int8 values whose bit patterns are a and b (0..255)."""
    circuits = find_signed(lib)
    os.makedirs(out, exist_ok=True)
    vals = np.arange(256).astype(np.uint8).view(np.int8).astype(int)
    info = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n, c in circuits.items():
            path = os.path.join(out, n + ".npy")
            if not os.path.exists(path):
                so = os.path.join(tmp, n + ".so")
                subprocess.run(["gcc", "-O2", "-shared", "-fPIC", c["c"], "-o", so], check=True)
                f = getattr(ctypes.CDLL(so), n)
                f.restype = ctypes.c_int16; f.argtypes = [ctypes.c_int8, ctypes.c_int8]
                lut = np.array([[f(int(a), int(b)) for b in vals] for a in vals], np.int32)
                np.save(path, lut)
            lut = np.load(path)
            err = lut - np.outer(vals, vals)
            info[n] = dict(c["params"], measured_mae=float(np.abs(err).mean()), measured_bias=float(err.mean()))
    json.dump(info, open(os.path.join(out, "circuits.json"), "w"), indent=1)
    return info


class SignedMLP:
    """7-bit activations (0..127) and 8-bit two's-complement weights (-127..127)."""

    def __init__(self, clf, Xcal):
        self.layers, a = [], Xcal
        for k, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            sa = max(np.percentile(a, 99.9), 1e-8) / 127.0
            sw = np.abs(W).max() / 127.0
            q = np.clip(np.round(W / sw), -127, 127).astype(np.int8)
            self.layers.append(dict(sa=sa, sw=sw, bits=q.view(np.uint8), ones=np.ones(q.shape, np.int8), b=b))
            a = a @ W + b
            if k < len(clf.coefs_) - 1:
                a = np.maximum(a, 0)
        self.macs = [L["bits"].size for L in self.layers]

    def forward(self, X, lut):
        a = X
        for k, L in enumerate(self.layers):
            xq = np.clip(np.round(a / L["sa"]), 0, 127).astype(np.uint8)
            z = lut_matmul(xq, L["bits"], L["ones"], lut) * (L["sa"] * L["sw"]) + L["b"]
            a = z if k == len(self.layers) - 1 else np.maximum(z, 0)
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="evoapproxlib")
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--datasets", nargs="+", default=datasets.NAMES, choices=datasets.NAMES)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    smeta = build_signed_luts(args.lib, "luts_signed")
    snames = sorted(smeta, key=lambda n: -smeta[n]["pwr"])
    s_exact = next(n for n in snames if smeta[n]["mae%"] == 0)
    names, luts, umeta = load_luts(args.luts)
    e0 = umeta[EXACT]["pwr"] + ADDER_PWR                 # unsigned exact MAC = reference energy

    rows = []
    for d in args.datasets:
        clf, (Xtrain, ytrain, Xval, yval, Xtest, ytest) = trained_mlp(d, 0)
        Xcal = Xtrain[np.random.default_rng(0).choice(len(Xtrain), 5000, replace=False)]
        acc = lambda logits: float((clf.classes_[logits.argmax(1)] == ytest).mean())
        um, sm = QuantMLP(clf, Xcal), SignedMLP(clf, Xcal)
        u_exact = acc(um.forward(Xtest, luts[names.index(EXACT)]))
        for n in names:
            rows.append(dict(dataset=d, kind="unsigned (sign-magnitude)", circuit=n, mac_energy=(umeta[n]["pwr"] + ADDER_PWR) / e0,
                             test_acc=acc(um.forward(Xtest, luts[names.index(n)])), ref_acc=u_exact))
        for n in snames:
            lut = np.load(os.path.join("luts_signed", n + ".npy"))
            rows.append(dict(dataset=d, kind="signed", circuit=n, mac_energy=(smeta[n]["pwr"] + ADDER_PWR) / e0,
                             test_acc=acc(sm.forward(Xtest, lut)), ref_acc=u_exact))
        s0 = next(r for r in rows if r["dataset"] == d and r["circuit"] == s_exact)
        print(f"{datasets.PRETTY[d]:14s} exact unsigned {100 * u_exact:.2f}%  exact signed ({s_exact}) {100 * s0['test_acc']:.2f}%"
              f"  signed MAC energy {s0['mac_energy']:.3f}", flush=True)
    for r in rows:
        r["test_drop_pp"] = round(100 * (r["ref_acc"] - r["test_acc"]), 2)
    with open(os.path.join(args.out, "signed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r in rows:
        if r["kind"] == "signed":
            print(f"  {datasets.PRETTY[r['dataset']]:14s} {r['circuit']:11s} MAC energy {r['mac_energy']:.3f}  drop {r['test_drop_pp']:6.2f}")
    plot(rows, args.out)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, GREY, BLUE = "#1f2a37", "#9aa5b1", "#2f6fb0"
    sets = list(dict.fromkeys(r["dataset"] for r in rows))
    fig, axes = plt.subplots(1, len(sets), figsize=(3.6 * len(sets), 3.6), dpi=150, sharey=True)
    for ax, d in zip(np.atleast_1d(axes), sets):
        for kind, col, mk in (("unsigned (sign-magnitude)", GREY, "o"), ("signed", BLUE, "s")):
            rs = [r for r in rows if r["dataset"] == d and r["kind"] == kind]
            ax.scatter([100 * (1 - r["mac_energy"]) for r in rs], [max(r["test_drop_pp"], 0) for r in rs], s=14,
                       color=col, marker=mk, label=kind, lw=0)
        ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 100); ax.set_xlim(-12, 86)
        ax.set_yticks([0, 1, 5, 20, 50]); ax.set_yticklabels(["0", "1", "5", "20", "50"])
        ax.axvline(0, color="#c4c8cc", lw=0.8)
        ax.set_title(datasets.PRETTY[d], loc="left", fontsize=10, color=INK)
        ax.set_xlabel("MAC energy saved (%)")
        ax.grid(color="#eceef0"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("test accuracy drop (points)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Signed multipliers (mul8s) against unsigned ones with sign-magnitude weights (MLP, seed 0)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, "signed.png"), bbox_inches="tight")
    print("saved results/signed.csv, results/signed.png")


if __name__ == "__main__":
    main()
