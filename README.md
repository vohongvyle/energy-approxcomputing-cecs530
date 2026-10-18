# Energy-Efficient Approximate Computing for ML — starter project

This project measures how much multiplier energy a neural network can save when exact 8-bit
multipliers are replaced with approximate ones from
[EvoApproxLib](https://github.com/ehw-fit/evoapproxlib), and how much accuracy that costs.

## How it works

1. **`build_luts.py`** compiles the C model of every 8x8 unsigned multiplier in EvoApproxLib
   (36 circuits) and evaluates all 256 x 256 input pairs into a lookup table. Each circuit's
   published PDK45 power, MAE and WCE come from the library's `meta.json`.
2. **`run_experiment.py`** trains a small MLP (64-64-10 on scikit-learn's handwritten digits),
   then quantizes it to 8 bits:
   - Activations become `uint8` (inputs and post-ReLU values are non-negative).
   - Weights become a sign plus an 8-bit magnitude, so the unsigned multiplier can be used.
   - Integer inference then does every `x * |w|` as a lookup in a multiplier's table.

   The script tries every circuit in three placements: both layers, layer 1 only, and layer 2 only.
3. **`plot_results.py`** draws accuracy against relative multiplier energy, with the Pareto front.

**Energy model:** energy per inference ≈ Σ (multiplications in layer × multiplier power).
This counts multipliers only, not adders, memory or data movement. Say so in your report.

## Run it

```bash
pip install numpy scikit-learn matplotlib
git clone https://github.com/ehw-fit/evoapproxlib
python build_luts.py --lib evoapproxlib --out luts
python run_experiment.py                  # digits dataset, runs in seconds
python run_experiment.py --dataset mnist  # MNIST (downloads ~50 MB), slower
python plot_results.py --dataset digits
```

## First results (digits, seed 0, 540 test images)

| Setup | Accuracy | Multiplier energy |
|---|---|---|
| Float model | 97.78% | — |
| 8-bit exact (`mul8u_1JFF`) | 97.78% | 1.00 |
| `mul8u_1AGV`, both layers | 97.78% | 0.24 |
| `mul8u_FTA`, both layers | 97.59% | 0.21 |
| `mul8u_QKX`, both layers | 97.22% | 0.07 |
| `mul8u_13QR`, both layers | 93.89% | 0.02 |
| `mul8u_17QU`, both layers | 72.22% | 0.004 |

What this shows:
- **About 75% of multiplier energy can be saved with no accuracy loss on this task.**
  With about 1% loss, the saving rises above 90%.
- **Low average error is not the same as low DNN error.** Several circuits with similar MAE behave
  differently, because consistent negative *bias* (always rounding down) accumulates across a
  dot product. `build_luts.py` prints each circuit's mean error so you can study this.
- **Layer 1 holds 86% of the multiplications** (64×64 vs 64×10), so approximating only layer 2
  barely saves energy. Approximation pays off most in the largest layers.

**Caveat:** with 540 test images, one image is 0.19 percentage points. Differences under ~0.5 pp
are noise. Repeat with several `--seed` values (or use MNIST) before drawing conclusions.

## Improved backend: `approx_backend.py`

```bash
python approx_backend.py              # digits, 3 seeds, ~45 s
python approx_backend.py --seeds 5 --budgets 0 0.5 1 2
```

What it adds over `run_experiment.py`:

1. **Fairer energy model.** One MAC = multiplier + exact 16-bit accumulator adder
   (`add16u_1E2`, 0.072 mW). Savings shrink compared with multiplier-only numbers.
   For example, `mul8u_1AGV` goes from 24% to 36% of exact energy.
2. **Bias compensation at zero hardware cost.** Approximate multipliers mostly round down.
   The expected error of every product is estimated from calibration activations and folded
   into each neuron's bias. On layer 1 it removes 99% of the accumulator bias and 25–35% of the
   RMS error. It is switched on per circuit only when it helps on the validation set, because it
   does not always help accuracy.
3. **Layer-wise search.** It tries every (circuit, compensation on/off) pair for each layer,
   picks the cheapest under an accuracy budget on validation, and reports on an untouched test set.
4. **Train / validation / test split and several seeds**, reported as mean ± std.

Results (digits, 3 seeds, MAC energy relative to exact):

| Setup | Energy | Test accuracy |
|---|---|---|
| 8-bit exact | 1.00 | 97.31% |
| `mul8u_1AGV` in both layers | 0.36 | 97.31% |
| `mul8u_13QR` plain → with compensation | 0.17 | 91.85% → 92.87% |
| `mul8u_17QU` plain → with compensation | 0.16 | 66.39% → 79.07% |
| Layer-wise search, 0.5 pp budget | 0.23 | 96.39% |

Compensation helps most for the most aggressive circuits. The layer-wise search finds cheaper
designs, but the 360-image validation set is small, so its picks lose about 0.5–1 pp on test.
A larger dataset (MNIST) makes the search more reliable.

## Where to take it next

1. **Bigger, harder model.** Use a CNN on MNIST or CIFAR-10 (PyTorch). Errors matter more in deep
   networks, and the trade-off becomes more interesting.
2. **Approximation-aware retraining.** Fine-tune with the approximate multiplier in the forward
   pass (straight-through gradient). This usually recovers most of the lost accuracy.
3. **Per-layer assignment.** Search for the cheapest circuit per layer under an accuracy budget.
4. **Bias compensation.** Subtract each circuit's mean error from the accumulator and check
   whether accuracy recovers for free.
5. **Real hardware numbers.** Synthesize a MAC unit from the library's Verilog with Yosys,
   or try an FPGA, to confirm the power savings beyond the published figures.

## References

- V. Mrazek et al., "EvoApprox8b: Library of approximate adders and multipliers for circuit
  design and benchmarking of approximation methods", DATE 2017.
- V. Mrazek et al., "ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network
  Accelerators without Retraining", ICCAD 2019 (a good model for the per-layer idea).

## Health application: arrhythmia detection

See `ecg/README.md`. Run `python ecg/ecg_experiment.py` for the synthetic test bench, or
`python ecg/ecg_experiment.py --data mitbih` for the real MIT-BIH data (`pip install wfdb`).
