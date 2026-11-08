"""
Check the library's hardware numbers for the 36 multipliers, three ways:

1. Equivalence: simulate each gate-level netlist on all 65,536 inputs and compare with the lookup
   table used in the experiments (they must be identical).
2. Synthesis: optimize and map each netlist with the ABC logic synthesis tool onto a small generic
   cell library (cells.genlib), and report cell count, area and logic depth.
3. Switching activity, the main driver of dynamic power: simulate how many gate outputs change per
   multiplication, for random inputs (as the library assumes) and for the real operand stream of a
   trained 8-bit MLP's first layer on MNIST and CIFAR-10.

Then compare each measure with the library's power figure (Spearman rank correlation), and check
whether the cheapest multipliers keep their ranking.

Usage (from the repository root, after build_luts.py; ABC from github.com/berkeley-abc/abc):
    python3 hardware/check_hardware.py --lib evoapproxlib --abc path/to/abc
Outputs: results/hardware.csv, results/hardware_check.png
"""
import argparse, csv, json, os, re, subprocess, sys, tempfile
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from netlist import Netlist, find_netlist
import datasets

EXACT = "mul8u_1JFF"


def abc_synth(nl, abc, tmp):
    blif = os.path.join(tmp, nl.name + ".blif")
    nl.to_blif(blif)
    resyn2 = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance"
    script = (f"read_blif {blif}; strash; dc2; dc2; {resyn2}; print_stats; "
              f"read_library {os.path.join(HERE, 'cells.genlib')}; map -a; print_stats; cec {blif}")
    out = subprocess.run([abc, "-c", script], capture_output=True, text=True, check=True).stdout
    aig = int(re.search(r"and\s*=\s*(\d+)", out).group(1))
    gates = int(re.findall(r"nd\s*=\s*(\d+)", out)[-1])
    area = float(re.search(r"area\s*=\s*([\d.]+)", out).group(1))
    depth = int(re.findall(r"lev\s*=\s*(\d+)", out)[-1])
    assert "Networks are equivalent" in out, f"ABC changed the function of {nl.name}"
    return dict(aig_nodes=aig, mapped_cells=gates, mapped_area=area, mapped_depth=depth)


def nn_stream(name, n_pairs, seed=0):
    """Operand pairs a real MLP's first layer feeds one multiplier, in output-stationary order:
    for one neuron j and one image, the products x_k * |w_kj| for k = 1, 2, ... in sequence."""
    from sklearn.neural_network import MLPClassifier
    import warnings
    warnings.filterwarnings("ignore")
    Xtr, ytr, Xte, _ = datasets.load(name)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), 10000, replace=False)
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=5, batch_size=256, random_state=seed).fit(Xtr[idx] / 255.0, ytr[idx])
    W = clf.coefs_[0]
    mag = np.clip(np.round(np.abs(W) / (np.abs(W).max() / 255)), 0, 255).astype(np.uint8)   # (in, 256)
    act, wt = [], []
    while sum(len(x) for x in act) < n_pairs:
        x = Xte[rng.integers(len(Xte))]                     # pixels are already 8-bit activations
        j = rng.integers(mag.shape[1])
        act.append(x); wt.append(mag[:, j])
    return np.concatenate(act)[:n_pairs], np.concatenate(wt)[:n_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="evoapproxlib")
    ap.add_argument("--abc", default="abc")
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--pairs", type=int, default=30000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/hardware.csv")
    args = ap.parse_args()
    if args.plot_only:
        rows = [{k: (v if k == "circuit" else float(v) if v not in ("True", "False") else v == "True") for k, v in r.items()}
                for r in csv.DictReader(open(os.path.join(args.out, "hardware.csv")))]
        return plot(rows, args.out)
    meta = json.load(open(os.path.join(args.luts, "circuits.json")))
    names = sorted(meta, key=lambda n: -meta[n]["pwr"])

    rng = np.random.default_rng(0)
    streams = {"random": (rng.integers(0, 256, args.pairs).astype(np.uint8), rng.integers(0, 256, args.pairs).astype(np.uint8))}
    for d in ("mnist", "cifar10"):
        streams[d] = nn_stream(d, args.pairs)
        a, w = streams[d]
        print(f"{d}: {100 * np.mean(a == 0):.0f}% of activations are 0, mean weight magnitude {w.mean():.1f}")

    a_all = np.repeat(np.arange(256), 256).astype(np.uint8); b_all = np.tile(np.arange(256), 256).astype(np.uint8)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in names:
            nl = Netlist(find_netlist(args.lib, n))
            out, _ = nl.simulate(a_all, b_all)           # netlist port A = weight, B = activation
            same = np.array_equal(out.reshape(256, 256), np.load(os.path.join(args.luts, n + ".npy")).T)
            r = dict(circuit=n, lib_pwr_mW=meta[n]["pwr"], lib_area_um2=meta[n]["area"], netlist_matches_table=same,
                     netlist_gates=len(nl.gates), **abc_synth(nl, args.abc, tmp))
            for s, (act, wt) in streams.items():
                r[f"toggles_{s}"] = round(nl.switching(wt, act), 2)
            rows.append(r)
            print(f"{n:11s} pwr {r['lib_pwr_mW']:.4f}  gates {r['netlist_gates']:4d}  ABC cells {r['mapped_cells']:4d}"
                  f"  area {r['mapped_area']:7.1f}  toggles random {r['toggles_random']:6.1f}  MNIST {r['toggles_mnist']:6.1f}"
                  f"  CIFAR-10 {r['toggles_cifar10']:6.1f}  {'ok' if same else 'MISMATCH'}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "hardware.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    pw = [r["lib_pwr_mW"] for r in rows]
    print("\nSpearman rank correlation with the library's power figure (36 multipliers):")
    for k in ("lib_area_um2", "netlist_gates", "mapped_area", "toggles_random", "toggles_mnist", "toggles_cifar10"):
        print(f"  {k:16s} {spearmanr(pw, [r[k] for r in rows])[0]:.3f}")
    plot(rows, args.out)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, RED = "#1f2a37", "#6b7280", "#2f6fb0", "#c2362f"
    ex = next(r for r in rows if r["circuit"] == EXACT)
    pw = np.array([r["lib_pwr_mW"] / ex["lib_pwr_mW"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)
    panels = [("mapped_area", "A  Area after ABC synthesis", "mapped area, relative to exact"),
              ("toggles_random", "B  Switching, random inputs", "gate switches per multiply, rel. to exact"),
              ("toggles_cifar10", "C  Switching, real CIFAR-10 operands", "gate switches per multiply, rel. to exact")]
    for ax, (k, title, ylab) in zip(axes, panels):
        y = np.array([r[k] / ex[k] for r in rows])
        rho = spearmanr(pw, y)[0]
        ax.plot([0, 1.05], [0, 1.05], color=MUTED, lw=0.8, ls="--")
        ax.scatter(pw, y, s=16, color=BLUE, lw=0)
        offsets = {"mul8u_YX7": (6, 0), "mul8u_QKX": (6, 2), "mul8u_1AGV": (6, -8), "mul8u_13QR": (6, 4)}
        for r, x_, y_ in zip(rows, pw, y):
            if r["circuit"] in offsets:
                ax.annotate(r["circuit"].replace("mul8u_", ""), (x_, y_), xytext=offsets[r["circuit"]],
                            textcoords="offset points", fontsize=7.5, color=INK)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, max(1.15, y.max() * 1.05))
        ax.set_xlabel("library power, relative to exact"); ax.set_ylabel(ylab)
        ax.set_title(f"{title}\nSpearman ρ = {rho:.2f}", loc="left", fontsize=10, color=INK)
        ax.grid(color="#eceef0", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Checking the library's power figures for 36 multipliers (dashed line: perfect agreement)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, "hardware_check.png"), bbox_inches="tight")
    print("saved results/hardware.csv, results/hardware_check.png")


if __name__ == "__main__":
    main()
