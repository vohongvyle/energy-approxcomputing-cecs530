"""
Which error metric of a multiplier predicts how much accuracy the network loses?

The library describes each multiplier with error metrics measured over all 65,536 input pairs:
mean absolute error (MAE%), worst-case error (WCE%), error probability (EP%) and mean relative
error (MRE%). We add the bias: the mean signed error, as a % of the largest product (65,025).
For each metric we compute the Spearman rank correlation with the mean test accuracy drop of the
35 approximate multipliers, separately for each dataset and network.

Usage (after cross_dataset.py, and run_cnn.py for the CNN rows):
    python3 analyze_metrics.py
Outputs: results/metric_correlations.csv, results/metric_scatter.png
"""
import csv, json, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import NAMES, PRETTY

OUT = "results"
COLORS = {"mnist": "#9aa5b1", "fashion": "#5b7a99", "cifar10": "#2f6fb0", "cifar100": "#c2362f"}
INK, MUTED = "#1f2a37", "#6b7280"
meta = json.load(open("luts/circuits.json"))
METRICS = {
    "MAE (%)": lambda m: m["mae%"],
    "Worst-case error (%)": lambda m: m["wce%"],
    "Bias, |mean error| (%)": lambda m: 100 * abs(m["measured_bias"]) / 65025,
    "Error probability (%)": lambda m: m["ep%"],
    "Mean relative error (%)": lambda m: m["mre%"],
}


def drops(prefix):
    """Mean test accuracy drop (points) per dataset and multiplier."""
    if not os.path.exists(os.path.join(OUT, f"{prefix}_sweep.csv")):
        return {}
    base = list(csv.DictReader(open(os.path.join(OUT, f"{prefix}_baselines.csv"))))
    sweep = list(csv.DictReader(open(os.path.join(OUT, f"{prefix}_sweep.csv"))))
    out = {}
    for n in NAMES:
        ex = {b["seed"]: float(b["int8_test"]) for b in base if b["dataset"] == n}
        if not ex:
            continue
        by = {}
        for r in sweep:
            if r["dataset"] == n:
                by.setdefault(r["circuit"], []).append(100 * (ex[r["seed"]] - float(r["test_acc"])))
        out[n] = {c: float(np.mean(v)) for c, v in by.items() if meta[c]["mae%"] > 0}
    return out


rows = []
results = {"MLP": drops("cross"), "CNN": drops("cnn")}
for net, per_ds in results.items():
    for n, d in per_ds.items():
        cs = sorted(d)
        r = dict(network=net, dataset=n, multipliers=len(cs))
        for name, f in METRICS.items():
            rho, p = spearmanr([f(meta[c]) for c in cs], [d[c] for c in cs])
            r[name] = round(float(rho), 2)
        rows.append(r)

with open(os.path.join(OUT, "metric_correlations.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

print("Spearman rank correlation between each metric and the accuracy drop (35 approximate multipliers)")
print(f"{'network':8s}{'dataset':15s}" + "".join(f"{k[:22]:>24s}" for k in METRICS))
for r in rows:
    print(f"{r['network']:8s}{PRETTY[r['dataset']]:15s}" + "".join(f"{r[k]:24.2f}" for k in METRICS))

# scatter: four main metrics, MLP results
d = results["MLP"]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=150, sharey=True)
for ax, name in zip(axes, list(METRICS)[:4]):
    f = METRICS[name]
    for n in d:
        cs = sorted(d[n])
        ax.scatter([max(f(meta[c]), 1e-3) for c in cs], [max(d[n][c], 0) for c in cs], s=14, alpha=0.8,
                   color=COLORS[n], lw=0, label=PRETTY[n])
    rhos = [r[name] for r in rows if r["network"] == "MLP"]
    ax.set_title(f"{name}\nmean Spearman ρ = {np.mean(rhos):.2f}", loc="left", fontsize=9.5, color=INK)
    if "probability" not in name:
        ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 100)
    ax.set_yticks([0, 1, 5, 20, 50]); ax.set_yticklabels(["0", "1", "5", "20", "50"])
    ax.grid(color="#eceef0", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("test accuracy drop (points)")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("Which error metric predicts the accuracy drop? (MLP, 35 approximate multipliers, each dot = one multiplier)",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "metric_scatter.png"), bbox_inches="tight")
print("saved results/metric_correlations.csv, results/metric_scatter.png")
