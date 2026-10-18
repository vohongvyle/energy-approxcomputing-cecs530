"""
Step 3: accuracy vs multiplier energy plot (the Pareto front of the project).

Usage:
    python plot_results.py --dataset digits
"""
import argparse, csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"both": "#2a78d6", "layer1": "#eb6834", "layer2": "#1baf7a"}
MARKERS = {"both": "o", "layer1": "s", "layer2": "^"}
LABELS = {"both": "Approximate in both layers",
          "layer1": "Approximate in layer 1 only",
          "layer2": "Approximate in layer 2 only"}


def pareto(points):
    """Points not beaten on both lower energy and higher accuracy."""
    pts = sorted(points)
    front, best = [], -1
    for e, a, n in pts:
        if a > best:
            front.append((e, a, n)); best = a
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.out, f"results_{args.dataset}.csv"))))
    base = json.load(open(os.path.join(args.out, f"baseline_{args.dataset}.json")))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for where in ["both", "layer1", "layer2"]:
        pts = [(float(r["rel_mult_energy"]), float(r["accuracy"]) * 100, r["circuit"])
               for r in rows if r["approx_in"] == where]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, marker=MARKERS[where],
                   color=COLORS[where], edgecolor="white", linewidth=1, label=LABELS[where], zorder=3)
        if where == "both":
            f = [p for p in pareto(pts) if p[1] >= 88]
            ax.plot([p[0] for p in f], [p[1] for p in f], color=COLORS[where], lw=1.5,
                    alpha=0.6, zorder=2, drawstyle="steps-post")
            for e, a, n in f:
                if a >= 90 and e < 0.6:
                    ax.annotate(n.replace("mul8u_", ""), (e, a), xytext=(5, -11),
                                textcoords="offset points", fontsize=7.5, color="#555")

    ax.axhline(base["int8_acc"] * 100, color="#888", lw=1, ls="--", zorder=1)
    ax.text(1.0, base["int8_acc"] * 100 + 0.3, "8-bit exact baseline",
            ha="right", fontsize=8, color="#555")
    ax.set_xlabel("Multiplier energy per inference (relative to exact 8-bit multiplier)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Accuracy vs multiplier energy, EvoApproxLib 8x8 multipliers ({args.dataset})",
                 fontsize=11, loc="left")
    ax.set_ylim(88, 100)
    ax.set_xlim(-0.02, 1.04)
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(0.01, 0.01, "Circuits below 88% accuracy are cut off the y-axis; see the CSV for every point.",
             fontsize=7.5, color="#777")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(args.out, f"tradeoff_{args.dataset}.png")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
