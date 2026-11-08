"""
Approximate adders in the accumulator, not only approximate multipliers.

Every multiply-accumulate also needs an addition. Here the 16-bit adder of the accumulator is
replaced by each of the 30 approximate 16-bit unsigned adders of EvoApproxLib (see lutadd.c for the
accumulator design), first with the exact multiplier and then together with mul8u_1AGV, a
multiplier that cost no accuracy on MNIST or CIFAR-10.

Network: the MLP of cross_dataset.py, seed 0 (one seed: an exploratory study). The adders are
simulated gate by gate in C, which is slow, so each dataset is scored on a fixed random subset of
2,000 test images (the same images for every adder, so differences between adders are paired).
Adders that already cost more than 5 points with the exact multiplier are not retried with mul8u_1AGV.
Usage:
    python3 approx_adders.py --lib evoapproxlib
Outputs: results/adders.csv, results/adders.png
"""
import argparse, csv, ctypes, json, os, subprocess
import numpy as np

import datasets
from approx_backend import EXACT, load_luts
from mlp_cache import trained_mlp
from cross_dataset import QuantMLP

HERE = os.path.dirname(os.path.abspath(__file__))
EXACT_ADDER = "add16u_1E2"
PARTNER = "mul8u_1AGV"


def find_adders(lib):
    meta = json.load(open(os.path.join(lib, "meta.json")))
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if "instances" in o and "adders/16_unsigned" in o.get("folder", ""):
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


def load_adder_kernel(name, c_path):
    so = os.path.join(HERE, f"_lutadd_{name}.so")
    if not os.path.exists(so):
        subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
                        f"-DADDER_FILE=\"{os.path.abspath(c_path)}\"", f"-DADDER={name}",
                        os.path.join(HERE, "lutadd.c"), "-o", so], check=True)
    lib = ctypes.CDLL(so)
    P = np.ctypeslib.ndpointer
    lib.lut_matmul_add.argtypes = [P(np.uint8, flags="C"), P(np.uint8, flags="C"), P(np.int8, flags="C"),
                                   P(np.int32, flags="C"), P(np.int64, flags="C"),
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int]
    f = getattr(lib, name)
    f.restype = ctypes.c_uint64; f.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    return lib, f


def forward(model, X, lut, kernel):
    a = X
    for k, L in enumerate(model.layers):
        xq = np.ascontiguousarray(np.clip(np.round(a / L["sa"]), 0, 255).astype(np.uint8))
        out = np.empty((len(xq), L["mag"].shape[1]), np.int64)
        kernel.lut_matmul_add(xq, np.ascontiguousarray(L["mag"]), np.ascontiguousarray(L["sgn"]),
                              np.ascontiguousarray(lut, np.int32), out, len(xq), xq.shape[1], out.shape[1])
        z = out * (L["sa"] * L["sw"]) + L["b"]
        a = z if k == len(model.layers) - 1 else np.maximum(z, 0)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="evoapproxlib")
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--datasets", nargs="+", default=datasets.NAMES, choices=datasets.NAMES)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/adders.csv")
    args = ap.parse_args()
    if args.plot_only:
        rows = [{k: (v if k in ("dataset", "multiplier", "adder") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(open(os.path.join(args.out, "adders.csv")))]
        return plot(rows, args.out)
    names, luts, mmeta = load_luts(args.luts)
    adders = find_adders(args.lib)
    order = sorted(adders, key=lambda n: -adders[n]["params"]["pwr"])
    kernels = {}
    for n in order:
        lib, f = load_adder_kernel(n, adders[n]["c"])
        kernels[n] = lib
        if n == EXACT_ADDER:
            assert f(65535, 1) == 65536 and f(1234, 4321) == 5555, "exact adder check failed"
    e0 = mmeta[EXACT]["pwr"] + adders[EXACT_ADDER]["params"]["pwr"]

    rows = []
    for d in args.datasets:
        clf, (Xtrain, ytrain, Xval, yval, Xtest, ytest) = trained_mlp(d, 0)
        model = QuantMLP(clf, Xtrain[np.random.default_rng(0).choice(len(Xtrain), 5000, replace=False)])
        sub = np.random.default_rng(1).choice(len(Xtest), min(args.n_test, len(Xtest)), replace=False)
        Xs, ys = Xtest[sub], ytest[sub]
        ref, failed = None, set()
        for mult in (EXACT, PARTNER):
            lut = luts[names.index(mult)]
            for n in order:
                if mult == PARTNER and n in failed:
                    continue
                r = dict(dataset=d, multiplier=mult, adder=n, adder_pwr_mW=adders[n]["params"]["pwr"],
                         adder_mae_pct=adders[n]["params"]["mae%"],
                         mac_energy=(mmeta[mult]["pwr"] + adders[n]["params"]["pwr"]) / e0,
                         test_acc=float((clf.classes_[forward(model, Xs, lut, kernels[n]).argmax(1)] == ys).mean()))
                if ref is None:
                    ref = r
                r["test_drop_pp"] = round(100 * (ref["test_acc"] - r["test_acc"]), 2)
                if mult == EXACT and r["test_drop_pp"] > 5:
                    failed.add(n)
                rows.append(r)
                print(f"{datasets.PRETTY[d]:14s} {mult:11s} {n:11s} adder {r['adder_pwr_mW']:.4f} mW  "
                      f"MAC energy {r['mac_energy']:.3f}  test {100 * r['test_acc']:.2f}%  drop {r['test_drop_pp']:6.2f}", flush=True)
        with open(os.path.join(args.out, "adders.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    plot(rows, args.out)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COLORS = {"mnist": "#9aa5b1", "fashion": "#5b7a99", "cifar10": "#2f6fb0", "cifar100": "#c2362f"}
    INK = "#1f2a37"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, sharey=True)
    for ax, mult, title in ((axes[0], EXACT, "A  Exact multiplier + approximate adder"),
                            (axes[1], PARTNER, f"B  {PARTNER} + approximate adder")):
        for d in dict.fromkeys(r["dataset"] for r in rows):
            rs = sorted((r for r in rows if r["dataset"] == d and r["multiplier"] == mult), key=lambda r: r["mac_energy"])
            ax.scatter([100 * (1 - r["mac_energy"]) for r in rs], [max(r["test_drop_pp"], 0) for r in rs], s=16,
                       lw=0, alpha=0.85, color=COLORS[d], label=datasets.PRETTY[d])
        ax.axhline(1, color="#c4c8cc", lw=0.8, ls="--")
        ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 100)
        ax.set_yticks([0, 1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50"])
        ax.set_xlabel("MAC energy saved (%), multiplier + adder"); ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(color="#eceef0"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("test accuracy drop (points)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Approximate adders in the accumulator: each dot is one of 30 EvoApproxLib 16-bit adders (MLP, seed 0, 2,000 test images)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, "adders.png"), bbox_inches="tight")
    print("saved results/adders.csv, results/adders.png")


if __name__ == "__main__":
    main()
