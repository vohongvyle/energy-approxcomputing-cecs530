# Risk-Aware Approximate Computing for Wearable Heart Monitoring

I started this project in CECS 530 as a group project about using approximate multipliers to save energy in a neural network. For my personal upgrade, I extended the idea from handwritten digits to heartbeat classification.

I wanted to look beyond overall accuracy and ask a more specific question: **When a wearable heart monitor saves energy with approximate arithmetic, which abnormal heartbeats might it miss?** Since abnormal beats are less common, overall accuracy can hide those mistakes.

The project uses plain NumPy, so it can run on a laptop. It uses approximate multiplier circuits from [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib).

## What the ECG version does

- A small 1D CNN classifies heartbeats as N (normal), S (supraventricular), V (ventricular), or F (fusion).
- The network runs with 8-bit values, and each multiplication uses a lookup table for one of EvoApproxLib’s 36 approximate multipliers.
- Patients are split between training and testing, so the model is tested on patients it has not seen before.
- For each design, the experiment reports accuracy, sensitivity for each class, and how many abnormal beats were classified as normal.

I also added two ways to make the results more useful:

1. **Clinically constrained search:** It chooses a multiplier for each layer while keeping S and V sensitivity close to the exact model’s sensitivity.
2. **Confidence-gated fallback:** If the approximate model predicts “normal” with low confidence, the beat is checked again using the exact multiplier. Predictions of an abnormal beat are not changed.

## ECG results so far

These results use simulated heartbeats to check that the pipeline works. They are not clinical results. Running the experiment on MIT-BIH is the next step.

- Overall accuracy did not show which designs were safest. One multiplier reached 82% accuracy but missed 111 abnormal beats, compared with 28 for the exact design.
- The fallback checked 1.9% of the beats, used 31% of the exact design’s arithmetic energy, and missed 23 abnormal beats.

## The original CECS 530 project

The original group project, [approx-computing-ml](https://github.com/vohongvyle/approx-computing-ml), uses a small 64-64-10 MLP trained on scikit-learn’s handwritten digits dataset. It replaces exact 8-bit multipliers with approximate ones and compares energy use with accuracy.

### How it works

1. `build_luts.py` compiles the C model for each of EvoApproxLib’s 36 unsigned 8×8 multiplier circuits. It tests all 256 × 256 input pairs and saves the results in lookup tables. The published PDK45 power, MAE, and WCE values come from each circuit’s `meta.json`.

2. `run_experiment.py` trains the MLP and quantizes it to 8 bits:
   - Activations use `uint8`, since the inputs and post-ReLU values are non-negative.
   - Weights are stored as a sign and an 8-bit magnitude, so the model can use unsigned multipliers.
   - During integer inference, each `x * |w|` operation is looked up in the selected multiplier’s table.

   The script tests each circuit in three placements: both layers, layer 1 only, and layer 2 only.

3. `plot_results.py` plots accuracy against relative multiplier energy and shows the Pareto front.

The original energy estimate counts the multiplications in each layer and multiplies that number by the selected circuit’s power. It only estimates multiplier energy; it does not include adders, memory, or data movement.

### First results

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

In this run, `mul8u_1AGV` saved about 75% of multiplier energy without reducing accuracy. The savings went above 90% with about a one percentage point drop in accuracy.

I also found that a circuit’s average error does not always predict how it will affect the network. Circuits with similar MAE can behave differently because a consistent negative bias, where products tend to round down, can build up across a dot product. `build_luts.py` prints each circuit’s mean error so I can compare them.

Layer 1 accounts for 86% of the multiplications because it uses 64 × 64 operations, compared with 64 × 10 in layer 2. That is why approximating only layer 2 saves relatively little energy.

There were only 540 test images, so one image changes the accuracy by about 0.19 percentage points. Differences under roughly 0.5 percentage points may be noise. I would want to repeat the experiment with several `--seed` values or use MNIST before drawing firm conclusions.

## Improved backend: `approx_backend.py`

I also added a second backend with a more detailed energy estimate and a layer-wise search.

```bash
python approx_backend.py
python approx_backend.py --seeds 5 --budgets 0 0.5 1 2
