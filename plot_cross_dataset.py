"""Figures for the cross-dataset study (run cross_dataset.py first)."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import NAMES, PRETTY

OUT = "results"
COLORS = {"mnist": "#9aa5b1", "fashion": "#5b7a99", "cifar10": "#2f6fb0", "cifar100": "#c2362f"}
INK, MUTED = "#1f2a37", "#6b7280"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c4c8cc", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def rows(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#eceef0", lw=0.8)
    ax.set_axisbelow(True)


base = rows("cross_baselines.csv")
sweep = rows("cross_sweep.csv")
summ = rows("cross_summary.csv")
names = [n for n in NAMES if any(b["dataset"] == n for b in base)]


def mean_points(name, relative=False):
    """Per multiplier: MAC energy saved (%) and mean test accuracy drop over seeds, in points or
    as a share of the 8-bit exact accuracy (%)."""
    ex = {b["seed"]: float(b["int8_test"]) for b in base if b["dataset"] == name}
    by = {}
    for r in sweep:
        if r["dataset"] == name:
            d = ex[r["seed"]] - float(r["test_acc"])
            by.setdefault(r["circuit"], []).append((float(r["mac_energy"]), 100 * (d / ex[r["seed"]] if relative else d)))
    return np.array(sorted((100 * (1 - v[0][0]), max(np.mean([d for _, d in v]), 0)) for v in by.values()))


def front(p):
    """Pareto front: the smallest drop reachable with at least each energy saving."""
    out, best = [], np.inf
    for s, d in sorted(p.tolist(), key=lambda t: -t[0]):
        best = min(best, d)
        out.append((s, best))
    return np.array(out[::-1])


# Figure 1: the best accuracy drop reachable at each energy saving, one line per dataset
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
for ax, rel, title, unit in ((axes[0], False, "A  Accuracy drop in points", "points"),
                             (axes[1], True, "B  Drop as a share of the exact model's accuracy", "%")):
    for n in names:
        p = mean_points(n, rel)
        ax.scatter(p[:, 0], p[:, 1], s=8, color=COLORS[n], alpha=0.35, lw=0)
        f = front(p)
        ax.step(f[:, 0], f[:, 1], where="post", lw=1.8, color=COLORS[n], label=PRETTY[n])
    ax.axhline(1, color=MUTED, ls="--", lw=0.8)
    ax.text(1, 1.25, "1 point" if unit == "points" else "1%", color=MUTED, fontsize=8)
    ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 100)
    ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_xlim(0, 85)
    ax.set_xlabel("MAC energy saved (%), multiplier + adder")
    ax.set_ylabel(f"test accuracy drop ({unit}, mean of 3 seeds)")
    ax.set_title(title); clean(ax)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Harder datasets lose accuracy at lower energy savings (lines: best multiplier at each saving)",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cross_drop_vs_energy.png"), bbox_inches="tight")

# Figure 2: energy saved at each accuracy budget, chosen on validation
fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160)
budgets = sorted({float(r["budget_pp"]) for r in summ})
w = 0.8 / len(names)
for i, n in enumerate(names):
    vals = [float(next(r["mac_saved_pct"] for r in summ if r["dataset"] == n and float(r["budget_pp"]) == b)) for b in budgets]
    bars = ax.bar(np.arange(len(budgets)) + (i - (len(names) - 1) / 2) * w, vals, w * 0.92, color=COLORS[n], label=PRETTY[n])
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(len(budgets)), [f"≤ {b:g} point loss" for b in budgets])
ax.set_ylabel("MAC energy saved (%)"); ax.set_ylim(0, 90)
ax.set_title("Energy saved at a fixed accuracy budget (chosen on validation)")
ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left"); clean(ax); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cross_budget_savings.png"), bbox_inches="tight")

# Figure 3: changed predictions against accuracy drop
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=160)
for n in names:
    ex = {b["seed"]: float(b["int8_test"]) for b in base if b["dataset"] == n}
    rs = [r for r in sweep if r["dataset"] == n]
    ax.scatter([100 * float(r["flip"]) for r in rs], [100 * (ex[r["seed"]] - float(r["test_acc"])) for r in rs],
               s=12, color=COLORS[n], alpha=0.75, lw=0, label=PRETTY[n])
ax.set_xlim(0, 100); ax.set_ylim(-2, 92)
ax.set_xlabel("test predictions that change vs the exact network (%)")
ax.set_ylabel("accuracy drop (points)")
ax.set_title("Each dot is one multiplier and seed: hard datasets change many predictions before accuracy drops")
ax.title.set_fontsize(9.5); ax.legend(frameon=False, fontsize=8, loc="upper left"); clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cross_flips.png"), bbox_inches="tight")
print("saved results/cross_drop_vs_energy.png, cross_budget_savings.png, cross_flips.png")
