"""
CNN against MLP: energy saved at a 1-point accuracy budget, and the accuracy drop of a few
multipliers on each dataset. Reads results/cross_*.csv and results/cnn_*.csv.

Usage: python3 plot_cnn_vs_mlp.py      Output: results/cnn_vs_mlp.png
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datasets

R = "results"
INK, GREY, BLUE = "#1f2a37", "#9aa5b1", "#2f6fb0"
CIRCUITS = ["mul8u_12N4", "mul8u_1AGV", "mul8u_FTA", "mul8u_QKX"]


def rows(f):
    return list(csv.DictReader(open(os.path.join(R, f))))


def drops(base_file, sweep_file, acc_key):
    base = {(b["dataset"], b["seed"]): float(b[acc_key]) for b in rows(base_file)}
    out = {}
    for r in rows(sweep_file):
        if r["circuit"] in CIRCUITS:
            out.setdefault((r["dataset"], r["circuit"]), []).append(100 * (base[(r["dataset"], r["seed"])] - float(r["test_acc"])))
    return {k: np.mean(v) for k, v in out.items()}


def main():
    sets = datasets.NAMES
    saved = {}
    for net, f in (("MLP", "cross_summary.csv"), ("CNN", "cnn_summary.csv")):
        for r in rows(f):
            if float(r["budget_pp"]) == 1.0:
                saved[(net, r["dataset"])] = float(r["mac_saved_pct"])
    d_mlp = drops("cross_baselines.csv", "cross_sweep.csv", "int8_test")
    d_cnn = drops("cnn_baselines.csv", "cnn_sweep.csv", "int8_test")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.5]))
    ax = axes[0]
    x = np.arange(len(sets)); w = 0.38
    for j, (net, col) in enumerate((("MLP", GREY), ("CNN", BLUE))):
        v = [saved[(net, d)] for d in sets]
        b = ax.bar(x + (j - 0.5) * w, v, w * 0.92, color=col, label=net)
        ax.bar_label(b, fmt="%.0f", fontsize=8, color=INK, padding=2)
    ax.set_xticks(x, [datasets.PRETTY[d] for d in sets], fontsize=8.5)
    ax.set_ylim(0, 90); ax.set_ylabel("MAC energy saved (%)")
    ax.set_title("A  Energy saved at a 1-point budget", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    w = 0.8 / len(CIRCUITS)
    shades = ["#c9d2dc", "#8fa3b8", "#2f6fb0", "#173d66"]
    xs = np.arange(len(sets))
    for j, c in enumerate(CIRCUITS):
        m = [max(d_mlp[(d, c)], 0) for d in sets]; n = [max(d_cnn[(d, c)], 0) for d in sets]
        pos = xs + (j - (len(CIRCUITS) - 1) / 2) * w
        ax.bar(pos, n, w * 0.9, color=shades[j], label=c.replace("mul8u_", ""))
        ax.scatter(pos, m, marker="_", s=120, color=INK, lw=1.6, zorder=3, label="MLP" if j == 0 else None)
    ax.set_xticks(xs, [datasets.PRETTY[d] for d in sets], fontsize=8.5)
    ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 60)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50"])
    ax.set_ylabel("test accuracy drop (points)")
    ax.set_title("B  Drop per multiplier: CNN bars, MLP black ticks", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    for a in axes:
        a.grid(axis="y", color="#eceef0"); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("A CNN is more accurate but tolerates approximate multipliers less than the MLP",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(R, "cnn_vs_mlp.png"), bbox_inches="tight")
    print("saved results/cnn_vs_mlp.png")


if __name__ == "__main__":
    main()
