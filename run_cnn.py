"""
The four-dataset study again, with a small CNN instead of the MLP.

For each dataset and seed: train the CNN in cnn.py (float, NumPy), quantize it to 8 bits, and run
all 36 EvoApproxLib multipliers in all three layers on the 5,000-image validation set and the
official test set. Same split, budgets and energy model as cross_dataset.py, so the results can be
compared directly with the MLP.

Outputs (results/):
    cnn_baselines.csv   float and 8-bit exact accuracy
    cnn_sweep.csv       one row per dataset x seed x multiplier
    cnn_summary.csv     energy saved at a fixed accuracy budget, chosen on validation

Usage:
    python3 run_cnn.py                         # 4 datasets x 2 seeds, about 1.5 hours on 2 cores
    python3 run_cnn.py --datasets mnist --seeds 1 --epochs 2
"""
import argparse, csv, os, time
import numpy as np

import datasets
from approx_backend import ADDER_PWR, EXACT, load_luts
from cnn import CNN, QuantCNN
from cross_dataset import BUDGETS, write_csv

EPOCHS = {"mnist": 8, "fashion": 10, "cifar10": 15, "cifar100": 15}


def to_images(X, name):
    shape = (28, 28, 1) if X.shape[1] == 784 else (32, 32, 3)
    return (X.reshape(-1, *shape) / 255.0).astype(np.float32), shape


def run(name, seed, names, luts, meta, epochs, n_val=5000):
    Xtr, ytr, Xte, yte = datasets.load(name)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    va, tr = perm[:n_val], perm[n_val:]
    Xtrain, shape = to_images(Xtr[tr], name)
    Xval, _ = to_images(Xtr[va], name)
    Xtest, _ = to_images(Xte, name)
    ytrain, yval = ytr[tr], ytr[va]

    t0 = time.time()
    net = CNN(shape, datasets.N_CLASSES[name], seed=seed).fit(Xtrain, ytrain, epochs, seed=seed)
    t_train = time.time() - t0
    q = QuantCNN(net, Xtrain[rng.choice(len(Xtrain), 2000, replace=False)])
    ex = names.index(EXACT)
    acc = lambda logits, y: float((logits.argmax(1) == y).mean())

    Pval, Ptest = q.prepare(Xval), q.prepare(Xtest)
    exact_val = q.forward(Pval, [luts[ex]] * 3)
    exact_te = q.forward(Ptest, [luts[ex]] * 3)
    base = dict(dataset=name, seed=seed, epochs=epochs, train_s=round(t_train),
                float_test=acc(net.predict_logits(Xtest), yte),
                int8_val=acc(exact_val, yval), int8_test=acc(exact_te, yte), macs=sum(q.macs))
    pred_exact = exact_te.argmax(1)
    e_exact = meta[EXACT]["pwr"] + ADDER_PWR
    rows = []
    for i, n in enumerate(names):
        lv, lt = q.forward(Pval, [luts[i]] * 3), q.forward(Ptest, [luts[i]] * 3)
        rows.append(dict(dataset=name, seed=seed, circuit=n, pwr_mW=meta[n]["pwr"], mae_pct=meta[n]["mae%"],
                         mult_energy=meta[n]["pwr"] / meta[EXACT]["pwr"],
                         mac_energy=(meta[n]["pwr"] + ADDER_PWR) / e_exact,
                         val_acc=acc(lv, yval), test_acc=acc(lt, yte),
                         flip=float((lt.argmax(1) != pred_exact).mean())))
    return base, rows


def summarize(bases, sweep):
    out = []
    for name in dict.fromkeys(b["dataset"] for b in bases):
        for budget in BUDGETS:
            picks = []
            for b in (b for b in bases if b["dataset"] == name):
                ok = [r for r in sweep if r["dataset"] == name and r["seed"] == b["seed"]
                      and r["val_acc"] >= b["int8_val"] - budget / 100 - 1e-12]
                r = min(ok, key=lambda r: r["mac_energy"])
                picks.append(dict(circuit=r["circuit"], mac=r["mac_energy"], mult=r["mult_energy"],
                                  drop=100 * (b["int8_test"] - r["test_acc"])))
            out.append(dict(dataset=name, budget_pp=budget, circuits="/".join(sorted({p["circuit"] for p in picks})),
                            mult_saved_pct=round(100 * (1 - np.mean([p["mult"] for p in picks])), 1),
                            mac_saved_pct=round(100 * (1 - np.mean([p["mac"] for p in picks])), 1),
                            test_drop_pp=round(np.mean([p["drop"] for p in picks]), 2),
                            test_drop_sd=round(np.std([p["drop"] for p in picks]), 2)))
    return out


def read_csv(path):
    def conv(v):
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
        return {"True": True, "False": False}.get(v, v)
    return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(open(path))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=datasets.NAMES, choices=datasets.NAMES)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None, help="override the per-dataset epoch count")
    ap.add_argument("--luts", default="luts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true", help="keep finished runs from the CSVs in --out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names, luts, meta = load_luts(args.luts)

    bases, sweep = [], []
    if args.resume and os.path.exists(os.path.join(args.out, "cnn_baselines.csv")):
        bases = read_csv(os.path.join(args.out, "cnn_baselines.csv"))
        sweep = read_csv(os.path.join(args.out, "cnn_sweep.csv"))
    for name in args.datasets:
        for s in range(args.seeds):
            if any(b["dataset"] == name and b["seed"] == s for b in bases):
                continue
            t0 = time.time()
            b, rows = run(name, s, names, luts, meta, args.epochs or EPOCHS[name])
            bases.append(b); sweep += rows
            print(f"{datasets.PRETTY[name]:14s} seed {s}: float {b['float_test']:.4f}  8-bit exact {b['int8_test']:.4f}"
                  f"  ({b['epochs']} epochs, {time.time() - t0:.0f} s)", flush=True)
            write_csv(os.path.join(args.out, "cnn_baselines.csv"), bases)
            write_csv(os.path.join(args.out, "cnn_sweep.csv"), sweep)

    summary = summarize(bases, sweep)
    write_csv(os.path.join(args.out, "cnn_summary.csv"), summary)
    print(f"\nCNN: energy saved at a fixed accuracy budget (chosen on validation), mean of {args.seeds} seeds")
    print(f"{'dataset':14s} {'budget':>6s} {'mult saved':>10s} {'MAC saved':>9s} {'test drop':>10s}  multipliers")
    for r in summary:
        print(f"{datasets.PRETTY[r['dataset']]:14s} {r['budget_pp']:5.1f}p {r['mult_saved_pct']:9.1f}% {r['mac_saved_pct']:8.1f}%"
              f" {r['test_drop_pp']:7.2f} pp  {r['circuits']}")


if __name__ == "__main__":
    main()
