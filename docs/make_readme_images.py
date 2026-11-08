"""Extra README images: dataset samples and a multiplier x dataset heatmap.
Run from the repository root after cross_dataset.py:  python3 docs/make_readme_images.py"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
import datasets

OUT = "docs/images"
INK, MUTED = "#1f2a37", "#6b7280"
os.makedirs(OUT, exist_ok=True)

# 1. eight test images from each dataset
fig, axes = plt.subplots(4, 8, figsize=(9, 5), dpi=150)
for row, name in enumerate(datasets.NAMES):
    _, _, Xte, yte = datasets.load(name)
    idx = np.random.default_rng(1).choice(len(Xte), 8, replace=False)
    for col, i in enumerate(idx):
        ax = axes[row, col]
        img = Xte[i].reshape(28, 28) if Xte.shape[1] == 784 else Xte[i].reshape(32, 32, 3)
        ax.imshow(img, cmap="gray_r" if img.ndim == 2 else None, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[row, 0].set_ylabel(f"{datasets.PRETTY[name]}\n{datasets.N_CLASSES[name]} classes", rotation=0,
                            ha="right", va="center", fontsize=9, color=INK, labelpad=12)
fig.suptitle("The four datasets, from easiest (top) to hardest (bottom)", x=0.02, ha="left",
             fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dataset_samples.png"), bbox_inches="tight")

# 2. accuracy drop of every cheap multiplier on every dataset
base = list(csv.DictReader(open("results/cross_baselines.csv")))
sweep = list(csv.DictReader(open("results/cross_sweep.csv")))
names = [n for n in datasets.NAMES if any(b["dataset"] == n for b in base)]
circ = {}
for r in sweep:
    circ.setdefault(r["circuit"], float(r["mult_energy"]))
circuits = [c for c, e in sorted(circ.items(), key=lambda t: -t[1]) if e < 0.55]
drop = np.zeros((len(circuits), len(names))); flip = np.zeros_like(drop)
for j, n in enumerate(names):
    ex = {b["seed"]: float(b["int8_test"]) for b in base if b["dataset"] == n}
    for i, c in enumerate(circuits):
        rs = [r for r in sweep if r["dataset"] == n and r["circuit"] == c]
        drop[i, j] = np.mean([100 * (ex[r["seed"]] - float(r["test_acc"])) for r in rs])
        flip[i, j] = np.mean([100 * float(r["flip"]) for r in rs])

fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
shown = np.clip(drop, 0, None)
ax.imshow(np.log1p(shown), cmap="Blues", aspect="auto", vmin=0, vmax=np.log1p(60))
for i in range(len(circuits)):
    for j in range(len(names)):
        v = shown[i, j]
        ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                color="white" if v > 8 else INK)
ax.set_xticks(range(len(names)), [datasets.PRETTY[n] for n in names])
ax.set_yticks(range(len(circuits)), [f"{c}  ({100 * circ[c]:.1f}%)" for c in circuits], fontsize=8.5)
ax.xaxis.tick_top(); ax.tick_params(length=0, colors=INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Accuracy drop (points) of each multiplier on each dataset\n(multiplier energy in brackets, mean of 3 seeds)",
             loc="left", fontsize=10, color=INK, pad=28)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "multiplier_heatmap.png"), bbox_inches="tight")
print("saved docs/images/dataset_samples.png, multiplier_heatmap.png")
