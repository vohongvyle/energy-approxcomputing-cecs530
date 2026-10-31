<<<<<<< HEAD
# Energy-Efficient Approximate Computing for ML

I built this project for CECS530 to see how much energy a neural network could save by replacing exact 8-bit multipliers with approximate ones from [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib), and what that change does to accuracy.
=======
# Risk-aware approximate computing for wearable heart monitoring

When a wearable heart monitor saves energy with approximate arithmetic, **which heartbeats
does it start to miss?** Overall accuracy hides the answer, because abnormal beats are rare.
This project measures it directly and adds two fixes.
>>>>>>> 9ed8701 (Added my files for personal upgrade project)

It extends my CECS 530 group project,
[approx-computing-ml](https://github.com/vohongvyle/approx-computing-ml), from handwritten
digits to heartbeat classification. Everything is plain NumPy, so it runs on a laptop.

<<<<<<< HEAD
1. **`build_luts.py`** compiles the C model for each of EvoApproxLib’s 36 unsigned 8×8 multiplier circuits. It tests all 256 × 256 input pairs and saves the results in lookup tables. The published PDK45 power, MAE, and WCE values come from each circuit’s `meta.json`.

2. **`run_experiment.py`** trains a small 64-64-10 MLP on scikit-learn’s handwritten digits dataset, then quantizes it to 8 bits:

   - Activations use `uint8`, since the inputs and post-ReLU values are non-negative.
   - Weights are stored as a sign and an 8-bit magnitude, so the model can use unsigned multipliers.
   - During integer inference, each `x * |w|` operation is looked up in the selected multiplier’s table.

   The script tests each circuit in three placements: both layers, layer 1 only, and layer 2 only.

3. **`plot_results.py`** plots accuracy against relative multiplier energy and shows the Pareto front.

**Energy model:** Energy per inference is estimated as the sum of the number of multiplications in each layer times the multiplier’s power. This estimate counts multipliers only; it does not include adders, memory, or data movement.
=======
![results](ecg/results/ecg_synthetic.png)

## What it does

- A small 1D CNN classifies each heartbeat into the AAMI classes N (normal),
  S (supraventricular), V (ventricular) and F (fusion).
- The network runs in 8 bits, and every multiplication goes through the lookup table of one
  of 36 approximate multipliers from [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib).
- Patients are split between training and testing, so the model is always tested on people it
  has not seen.
- For each design it reports accuracy, sensitivity per class and the number of abnormal beats
  called normal.

Two additions:

1. **A clinically constrained search.** It picks a multiplier for each layer, keeping
   S and V sensitivity close to the exact design instead of only overall accuracy.
2. **A confidence-gated fallback.** A beat the approximate model calls "normal" with low
   confidence is re-checked on the exact multiplier. Abnormal calls are never overruled.

## Results so far (simulated heartbeats)

These come from a synthetic test bench used to validate the pipeline. They are not clinical
results; the MIT-BIH run is next.

- Accuracy does not rank designs by risk: one multiplier kept 82% accuracy but missed 111
  abnormal beats, against 28 for the exact design.
- The fallback re-checks 1.9% of beats, uses 31% of the exact design's arithmetic energy, and
  misses 23 abnormal beats, fewer than the exact model alone.
>>>>>>> 9ed8701 (Added my files for personal upgrade project)

## Run it

```bash
pip install -r requirements.txt
git clone https://github.com/ehw-fit/evoapproxlib
<<<<<<< HEAD
python build_luts.py --lib evoapproxlib --out luts
python run_experiment.py                  # digits dataset, runs in seconds
python run_experiment.py --dataset mnist  # MNIST, downloads about 50 MB and takes longer
python plot_results.py --dataset digits
```

## First results

These results are from the digits dataset with seed 0 and 540 test images.

| Setup | Accuracy | Multiplier energy |
|---|---:|---:|
| Float model | 97.78% | — |
| 8-bit exact (`mul8u_1JFF`) | 97.78% | 1.00 |
| `mul8u_1AGV`, both layers | 97.78% | 0.24 |
| `mul8u_FTA`, both layers | 97.59% | 0.21 |
| `mul8u_QKX`, both layers | 97.22% | 0.07 |
| `mul8u_13QR`, both layers | 93.89% | 0.02 |
| `mul8u_17QU`, both layers | 72.22% | 0.004 |

In this experiment, `mul8u_1AGV` saved about 75% of multiplier energy without reducing accuracy. With about a one percentage point accuracy drop, the savings went above 90%.

I also found that low average error does not always mean low error in the neural network. Circuits with similar MAE can behave differently because a consistent negative bias—where products tend to round down—can build up across a dot product. `build_luts.py` prints each circuit’s mean error so you can compare them.

Layer 1 accounts for 86% of the multiplications because it uses 64 × 64 operations, compared with 64 × 10 in layer 2. That’s why approximating only layer 2 saves relatively little energy.

**A note on the results:** There are only 540 test images, so one image changes the accuracy by about 0.19 percentage points. Differences under roughly 0.5 percentage points may be noise. I’d repeat the experiment with several `--seed` values or use MNIST before drawing firm conclusions.

## Improved backend: `approx_backend.py`

I added a second backend with a more detailed energy estimate and a layer-wise search.

```bash
python approx_backend.py              # digits, 3 seeds, about 45 seconds
python approx_backend.py --seeds 5 --budgets 0 0.5 1 2
```

Here’s what it adds:

1. **A more complete energy estimate.** Each multiply-accumulate (MAC) includes a multiplier and an exact 16-bit accumulator adder (`add16u_1E2`, 0.072 mW). This raises the relative energy compared with the multiplier-only estimate. For example, `mul8u_1AGV` goes from 24% to 36% of exact energy.

2. **Bias compensation with no added hardware cost.** Approximate multipliers tend to round down. I estimate each product’s expected error using calibration activations, then fold that estimate into each neuron’s bias. In layer 1, this removes 99% of the accumulator bias and 25–35% of the RMS error. Since compensation does not always improve accuracy, the script enables it for a circuit only when it helps on the validation set.

3. **A layer-wise search.** The script tries each circuit with compensation on and off for each layer. It then chooses the lowest-energy option that stays within the accuracy budget on the validation set.

4. **Multiple data splits and seeds.** The results use separate training, validation, and test sets, and report averages across seeds.

These results use the digits dataset with three seeds. Energy is relative to the exact MAC setup.

| Setup | Energy | Test accuracy |
|---|---:|---:|
| 8-bit exact | 1.00 | 97.31% |
| `mul8u_1AGV` in both layers | 0.36 | 97.31% |
| `mul8u_13QR`, plain → compensated | 0.17 | 91.85% → 92.87% |
| `mul8u_17QU`, plain → compensated | 0.16 | 66.39% → 79.07% |
| Layer-wise search, 0.5 pp budget | 0.23 | 96.39% |

Compensation helped most with the more aggressive circuits. The layer-wise search also found cheaper designs, but its choices sometimes lost about 0.5–1 percentage point on the test set. The validation set has only 360 images, so a larger dataset like MNIST could make the search more reliable.

## What I’d try next

1. **Test a larger model.** I could use a CNN with MNIST or CIFAR-10 to see how approximation errors affect a deeper network.
2. **Retrain with approximate multipliers.** Fine-tuning with approximate multiplication in the forward pass and a straight-through gradient may recover some of the lost accuracy.
3. **Search for a circuit per layer.** The current backend explores layer-wise choices; I could extend it to search for the cheapest circuit in each layer under an accuracy budget.
4. **Test bias compensation further.** I could subtract each circuit’s mean error from the accumulator and measure how much accuracy comes back.
5. **Check the estimates on hardware.** I could synthesize a MAC unit from the library’s Verilog with Yosys or test it on an FPGA to compare real power use with the published estimates.

## References

- V. Mrazek et al., “EvoApprox8b: Library of approximate adders and multipliers for circuit design and benchmarking of approximation methods,” DATE 2017.
- V. Mrazek et al., “ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators without Retraining,” ICCAD 2019. This paper is a useful reference for the per-layer approach.

## ECG application: arrhythmia detection

I also added an ECG example. See [`ecg/README.md`](ecg/README.md) for details.

```bash
python ecg/ecg_experiment.py              # synthetic test bench
python ecg/ecg_experiment.py --data mitbih  # real MIT-BIH data; install wfdb first
```

=======
python3 build_luts.py --lib evoapproxlib --out luts    # multiplier lookup tables (needs a C compiler)

python3 ecg/ecg_experiment.py                  # synthetic heartbeats, about 5 minutes
python3 ecg/mitbih.py                          # downloads MIT-BIH (~100 MB), prints beat counts
python3 ecg/ecg_experiment.py --data mitbih    # real data
python3 ecg/export_web.py --data mitbih        # interactive page: ecg/web/arrhythmia_lab_mitbih.html
```

Options: `--acc-budget 0.5` (allowed drop in overall accuracy, points), `--se-budget 1.0`
(allowed drop in S and V sensitivity, points), `--seed`, `--epochs`.

## Files

| File | What it does |
|---|---|
| `build_luts.py` | Builds a 256×256 lookup table for each EvoApproxLib 8×8 multiplier |
| `ecg/synth_ecg.py` | Synthetic heartbeats from simulated patients, for testing without downloads |
| `ecg/mitbih.py` | Loads MIT-BIH: lead MLII, AAMI classes, DS1/DS2 patient split |
| `ecg/cnn.py` | 1D CNN trained in NumPy, its 8-bit approximate version, bias compensation |
| `ecg/ecg_experiment.py` | Multiplier sweep, layer-wise searches, gated fallback, figure |
| `ecg/export_web.py` | Builds a self-contained interactive page from the results |

## Limitations

- Results so far are on simulated beats, which are cleaner than real ECG.
- Energy counts the multiplier and accumulator adder only (45 nm figures), not memory or radio.
- One small CNN, post-training approximation only.

## References

- V. Mrazek et al., "EvoApprox8b: Library of approximate adders and multipliers for circuit design and benchmarking of approximation methods," DATE 2017.
- P. de Chazal, M. O'Dwyer, R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," IEEE TBME, 2004.
- G. B. Moody, R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," IEEE EMB Magazine, 2001.
- F. Bulten et al., "Toward Energy-Efficient and Low-Power Arrhythmia Detection for Wearable Devices," arXiv:2607.14747, 2026.
>>>>>>> 9ed8701 (Added my files for personal upgrade project)
