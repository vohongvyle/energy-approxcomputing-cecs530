# Energy-Efficient Approximate Computing for ML

For our CECS 530 group project at California State University, Long Beach, we looked at what happens when a neural network uses approximate 8-bit multipliers instead of exact ones. How much multiplier energy can we save, and how much accuracy do we give up?

We tested all 36 unsigned 8-bit multipliers in [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib). We started with a small handwritten-digits dataset, then tested MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100.

![Energy saved at a fixed accuracy budget on four datasets](results/cross_budget_savings.png)

## Main results

- On the small handwritten-digits task, `mul8u_1AGV` reduced multiplier energy by about **75% with no accuracy loss**.
- With an accuracy-loss limit of one percentage point, multiplier-plus-adder energy fell by **78% on MNIST**, 71% on Fashion-MNIST, 66% on CIFAR-10, and **57% on CIFAR-100**.
- The best multiplier depends on the dataset. A multiplier that works well on MNIST can cause a much larger accuracy drop on CIFAR-10.
- Accuracy alone does not show the full effect: predictions can change even when overall accuracy barely moves.
- Our CNN was much more accurate than the MLP on CIFAR, but it tolerated approximation less well. At the one-point accuracy budget, it saved 54% on CIFAR-10, compared with 66% for the MLP.
- Retraining recovered much of the loss on MNIST and Fashion-MNIST, but little on CIFAR-100.
- Our synthesis results followed EvoApproxLib’s power rankings closely, with a rank correlation of 0.997.

## How we ran the experiments

![The experiment pipeline in five steps](docs/images/pipeline.png)

1. **Build lookup tables.** `build_luts.py` compiles each multiplier’s C model and records its output for all 65,536 input pairs in a 256 × 256 table. Power and error figures come from the library.
2. **Train and quantize the network.** We train an MLP in floating point, then quantize activations to unsigned 8-bit values. ReLU outputs are nonnegative. We represent each weight as a sign and an 8-bit magnitude, so we can use an unsigned multiplier.
3. **Run approximate inference.** Each product, `x · |w|`, is read from the selected multiplier’s table. Our C kernel, `lutmm.c`, handles about 2.4 billion lookups per second on two CPU cores and matches our NumPy implementation.
4. **Estimate energy.** For each multiply-accumulate (MAC), we count the multiplier plus an exact 16-bit adder, `add16u_1E2`, which costs 0.072 mW. Since the adder stays exact, MAC energy savings are capped at 84%.

## What an approximate multiplier does

An approximate multiplier trades arithmetic accuracy for a smaller, lower-power circuit. Each map shows the error for all 65,536 possible pairs of 8-bit inputs.

![Error maps of four multipliers](docs/images/multiplier_error_maps.png)

The exact multiplier on the left has no error. Cheaper designs can produce larger errors, and many tend to return results that are too small. Across hundreds of products in a neuron, that pattern can shift the neuron’s output.

## Part 1: Small handwritten-digits dataset

We began with scikit-learn’s 8×8 handwritten-digits dataset and a 64-64-10 network.

![Accuracy against multiplier energy for 36 approximate multipliers](docs/images/accuracy_vs_energy.png)

Each point represents one multiplier used in both layers. The staircase marks the Pareto front: no other multiplier is both cheaper and more accurate.

| Setup (seed 0, 540 test images) | Accuracy | Multiplier energy |
|---|---:|---:|
| 8-bit exact (`mul8u_1JFF`) | 97.78% | 1.00 |
| `mul8u_1AGV`, both layers | 97.78% | 0.24 |
| `mul8u_FTA`, both layers | 97.59% | 0.21 |
| `mul8u_QKX`, both layers | 97.22% | 0.07 |
| `mul8u_13QR`, both layers | 93.89% | 0.02 |
| `mul8u_17QU`, both layers | 72.22% | 0.004 |

### Improved backend: `approx_backend.py`

![Accuracy against energy with bias compensation and layer-wise search](docs/images/backend_results.png)

We added bias compensation and layer-wise multiplier selection. For compensation, we estimate each multiplier’s average error on calibration data and fold it into the neuron biases. We enable it only when it helps on validation data, so it adds no hardware cost. The layer-wise search picks the cheapest multiplier for each layer that meets the validation accuracy budget. These results use separate train, validation, and test sets and three seeds.

| Setup (mean of 3 seeds) | MAC energy | Test accuracy |
|---|---:|---:|
| 8-bit exact | 1.00 | 97.31% |
| `mul8u_1AGV`, both layers | 0.36 | 97.31% |
| `mul8u_13QR`, plain → compensated | 0.17 | 91.85% → 92.87% |
| `mul8u_17QU`, plain → compensated | 0.16 | 66.39% → 79.07% |
| Layer-wise search, 0.5-point budget | 0.23 | 96.39% |

The search met its budget on validation. On test, accuracy was 0.9 points lower than the exact model.

## Part 2: Four datasets

We tested all 36 multipliers on four standard datasets using an MLP with one 256-neuron hidden layer and three seeds. We chose multipliers on a 5,000-image validation set and evaluated them on each dataset’s official 10,000-image test set.

![Eight test images from each dataset](docs/images/dataset_samples.png)

| Dataset | Float accuracy | 8-bit exact | MAC energy saved, ≤ 1-point loss | Multiplier energy saved | Test drop |
|---|---:|---:|---:|---:|---:|
| MNIST | 97.94% | 97.94% | 78.2% | 92.6% | 0.21 pp |
| Fashion-MNIST | 89.14% | 89.13% | 71.3% | 84.4% | 0.76 pp |
| CIFAR-10 | 46.62% | 46.52% | 66.3% | 78.5% | 0.56 pp |
| CIFAR-100 | 20.64% | 20.42% | 57.2% | 67.7% | 0.72 pp |

### Harder datasets lose accuracy sooner

![Accuracy drop against energy saved on four datasets](results/cross_drop_vs_energy.png)

The lines show the best multiplier at each energy-saving level; faint dots show all multipliers. Panel B scales the accuracy drop relative to the exact model’s accuracy. By that measure, CIFAR-100 takes the biggest hit.

### The same multiplier can behave differently across datasets

![Accuracy drop of each multiplier on each dataset](docs/images/multiplier_heatmap.png)

`mul8u_QKX` is a good choice for MNIST: it uses 7.4% of the exact multiplier’s energy and loses 0.2 accuracy points. On CIFAR-10, the same multiplier loses 3.0 points. Below about 2% of exact energy, every dataset’s accuracy collapses; this includes `mul8u_13QR` and cheaper multipliers.

### Accuracy does not show every changed prediction

![Changed predictions against accuracy drop](results/cross_flips.png)

With `mul8u_1AGV`, predictions changed for 0.6% of MNIST images, 4.8% of Fashion-MNIST images, 12.5% of CIFAR-10 images, and 41% of CIFAR-100 images. On harder datasets, many predictions can change before overall accuracy drops much. Some of those predictions were already incorrect.

**Bias compensation had mixed results.** It helped some aggressive multipliers. For example, on CIFAR-10 it reduced the loss for `mul8u_13QR` from 12.1 to 9.1 points. But it often hurt with more useful multipliers: on MNIST, `mul8u_QKX` lost 2.30 points with compensation, compared with 0.21 without it.

**Limitations.** An MLP is not a strong model for CIFAR; its 8-bit accuracy was 46.5% on CIFAR-10 and 20.4% on CIFAR-100. Part 3 repeats the study with a CNN. Our energy figures are estimates based on EvoApproxLib’s 45 nm data, not direct measurements, and they do not include memory energy.

## Part 3: Follow-up studies

We ran six follow-up studies. The CNN experiments use two seeds; the other studies are smaller exploratory runs.

### A CNN on all four datasets: `run_cnn.py`

Our NumPy CNN has two 3×3 convolution layers (16 and 32 channels), each followed by ReLU and 2×2 max pooling, and one dense layer. We trained it with Adam for 8–15 epochs. All three layers use 8-bit arithmetic and the same multiplier lookup tables. We converted convolutions to matrix multiplications with im2col. As in Part 2, we chose multipliers using validation data and reported test results.

| Dataset | MLP, 8-bit | CNN, 8-bit | MLP MAC saved at ≤ 1-point loss | CNN |
|---|---:|---:|---:|---:|
| MNIST | 97.94% | 98.82% | 78.2% | 78.2% |
| Fashion-MNIST | 89.13% | 89.87% | 71.3% | 57.9% |
| CIFAR-10 | 46.52% | 67.70% | 66.3% | 53.8% |
| CIFAR-100 | 20.42% | 37.13% | 57.2% | 53.8% |

![CNN against MLP](results/cnn_vs_mlp.png)

The CNN did much better on CIFAR, but it was also more sensitive to approximation. For example, `mul8u_QKX` cost the CNN 16 accuracy points on CIFAR-10, compared with 3 points for the MLP. One possible reason is that errors in the first convolution pass through two more approximate layers. Each convolution output also sums only 9–144 products, giving errors fewer terms to average out than in the MLP’s first layer, which sums 784–3,072 products per neuron.

### Retraining with approximate multipliers: `retrain.py`

We fine-tuned the MLP for three epochs with an approximate multiplier in the forward pass. Products still came from the multiplier’s lookup table, and we used a straight-through estimator for gradients. The table shows test accuracy averaged over two seeds.

| Dataset | Multiplier | 8-bit exact | Plain | Compensated | Retrained | Retrained + compensated |
|---|---|---:|---:|---:|---:|---:|
| MNIST | `mul8u_13QR` | 97.94 | 82.70 | 83.08 | **97.08** | 71.84 |
| MNIST | `mul8u_17QU` | 97.94 | 45.29 | 47.14 | **92.82** | 45.92 |
| Fashion-MNIST | `mul8u_13QR` | 89.06 | 41.43 | 28.96 | **80.59** | 23.66 |
| Fashion-MNIST | `mul8u_17QU` | 89.06 | 15.88 | 14.65 | **71.61** | 12.42 |
| CIFAR-10 | `mul8u_13QR` | 47.11 | 34.40 | 37.71 | **39.98** | 22.01 |
| CIFAR-100 | `mul8u_13QR` | 20.78 | 3.64 | 4.36 | 4.82 | 1.60 |

![Retraining against bias compensation](results/retrain.png)

Retraining recovered much of the lost accuracy on MNIST and Fashion-MNIST. It helped less on CIFAR-10 and barely helped on CIFAR-100, where the multipliers left too little signal for the network to learn from.

Compensation hurt retrained models. Their weights had already adapted to the multiplier’s bias, so correcting that bias again over-corrected the network. As a control, fine-tuning the exact network for the same three epochs improved accuracy by 0.1–2.4 points (`retrain.csv`, rows for `mul8u_1JFF`). Some of the CIFAR-10 gain may therefore come from extra training. This retraining baseline uses float32 inputs, which explains why its exact accuracy differs slightly from Part 2.

### Checking the hardware estimates: `hardware/`

Yosys was not available in our environment, so we used [ABC](https://github.com/berkeley-abc/abc), the synthesis engine that Yosys calls. For each multiplier, `hardware/check_hardware.py`:

1. Simulates the gate-level Verilog netlist on all 65,536 input pairs. All 36 outputs match their lookup tables exactly.
2. Optimizes and maps the netlist to the generic cell library in `hardware/cells.genlib`, then uses ABC’s `cec` command to confirm the function did not change.
3. Counts gate output switches per multiplication for random inputs and for operand streams from trained MNIST and CIFAR-10 networks.

| Measure | Spearman rank correlation with library power |
|---|---:|
| Library area | 0.992 |
| Gates in the netlist | 0.985 |
| Area after ABC synthesis | 0.997 |
| Switching, random inputs | 0.986 |
| Switching, MNIST operands | 0.987 |
| Switching, CIFAR-10 operands | 0.989 |

![Library power against our synthesis and switching measurements](results/hardware_check.png)

The rankings stayed close. Two low-power multipliers switched more than their library power figures might suggest: `mul8u_YX7` used 15.6% of exact power and switched about 44% as much as the exact multiplier; `mul8u_QKX` used 7.4% of exact power and switched about 20% as much.

Real data also caused less switching than random inputs. Since 80% of MNIST pixels are zero, the exact multiplier toggled 20.9 gates per multiplication on MNIST, compared with 105.9 for random inputs.

`hardware/synth.sh` runs the same ABC flow on one multiplier.

### Which error metric best predicts accuracy loss? `analyze_metrics.py`

We calculated Spearman rank correlation between each error metric and accuracy drop across the 35 approximate multipliers.

| Network | Dataset | MAE | Worst-case error | Bias | Error probability |
|---|---|---:|---:|---:|---:|
| MLP | MNIST | 0.59 | 0.58 | 0.53 | 0.40 |
| MLP | Fashion-MNIST | 0.94 | 0.90 | 0.91 | 0.79 |
| MLP | CIFAR-10 | 0.80 | 0.81 | 0.78 | 0.53 |
| MLP | CIFAR-100 | 0.95 | 0.91 | 0.91 | 0.74 |
| CNN | MNIST | 0.88 | 0.84 | 0.84 | 0.72 |
| CNN | Fashion-MNIST | 0.94 | 0.92 | 0.92 | 0.70 |
| CNN | CIFAR-10 | 0.84 | 0.85 | 0.83 | 0.54 |
| CNN | CIFAR-100 | 0.96 | 0.94 | 0.92 | 0.69 |

![Error metrics against accuracy drop](results/metric_scatter.png)

Mean absolute error (MAE) was the best or tied-best predictor in nearly every case. Bias was not better. Error probability—the fraction of input pairs where the multiplier returns an incorrect result—was the weakest metric. Many circuits are wrong on nearly every input, but their errors can still be small.

### Approximate adders in the accumulator: `approx_adders.py`, `lutadd.c`

We tested all 30 of EvoApproxLib’s approximate 16-bit adders as replacements for the exact accumulator adder. We ran the tests first with an exact multiplier, then with `mul8u_1AGV`. These MLP results use seed 0 and the same 2,000 randomly selected test images for every adder. Gate-by-gate simulation made this experiment slow.

| Dataset | Exact multiplier: MAC saved at ≤ 1-point loss | `mul8u_1AGV` alone | `mul8u_1AGV` + best adder at ≤ 1-point loss |
|---|---:|---:|---:|
| MNIST | 11.7% (`add16u_0GK`) | 63.9% | 75.6% (`add16u_0GK`) |
| Fashion-MNIST | 9.9% (`add16u_00G`) | 63.9% | 73.9% (`add16u_00G`) |
| CIFAR-10 | 10.8% (`add16u_0B4`) | 63.9% | 73.9% (`add16u_00G`) |
| CIFAR-100 | 7.8% (`add16u_0M0`) | 63.9%, but with a 2.2-point loss | None within 1 point |

![Approximate adders](results/adders.png)

The adder is a small part of an exact MAC: 0.072 mW out of 0.463 mW. That limits the savings from approximating the adder alone. Once the multiplier is cheaper, though, the adder takes up a larger share of the remaining energy. A good approximate adder adds about 10 percentage points of savings.

Some adders were too inaccurate for this use. For example, `add16u_1X9` caused a large accuracy loss on every dataset despite saving only 4.3% of energy. Its 2% MAE builds up across the many additions in each neuron.

### Signed multipliers: `signed_multipliers.py`

The library also has 13 signed 8×8 multipliers (`mul8s_*`) that take two’s-complement inputs directly. Since ReLU activations are nonnegative, we represented them with 7 bits (0–127) and weights with 8 bits. These are MLP results from seed 0, with the best multiplier selected on the test set.

| Dataset | Unsigned + sign-magnitude: MAC saved at ≤ 1-point loss | Signed: MAC saved at ≤ 1-point loss |
|---|---:|---:|
| MNIST | 78.2% | 41.3% |
| Fashion-MNIST | 71.3% | 33.3% |
| CIFAR-10 | 66.3% | 33.3% |
| CIFAR-100 | 53.8% | 33.3% |

![Signed against unsigned multipliers](results/signed.png)

The signed multipliers performed worse in our experiments. The library has fewer signed options, its exact signed multiplier uses more energy than the unsigned one (0.425 versus 0.391 mW), and the activations have one fewer bit. For these ReLU networks, representing weights as sign plus magnitude and using an unsigned multiplier worked better.

## Running the project

```bash
pip install -r requirements.txt          # numpy, scipy, scikit-learn, matplotlib, pillow
git clone https://github.com/ehw-fit/evoapproxlib
python3 build_luts.py --lib evoapproxlib --out luts    # needs gcc or clang

# Part 1: small digits dataset
python3 run_experiment.py
python3 plot_results.py --dataset digits
python3 approx_backend.py                              # about 45 seconds

# Part 2: four datasets
bash get_data.sh                                       # downloads about 400 MB into data/
python3 cross_dataset.py                               # about 45 minutes on 2 CPU cores
python3 cross_dataset.py --datasets mnist --seeds 1    # quick run, about 1 minute
python3 plot_cross_dataset.py
python3 docs/make_readme_images.py

# Part 3
python3 run_cnn.py                                     # about 2.5 hours on 2 CPU cores; --resume continues
python3 plot_cnn_vs_mlp.py
python3 retrain.py                                     # about 1.5 hours
python3 analyze_metrics.py                             # after cross_dataset.py and run_cnn.py
python3 hardware/check_hardware.py --abc path/to/abc
python3 approx_adders.py                               # about 50 minutes
python3 signed_multipliers.py                          # about 10 minutes
```

A full run of `cross_dataset.py` prints:

![Terminal output of cross_dataset.py](docs/images/terminal_run.png)

## Project checklist

- [x] Built lookup tables for all 36 unsigned 8×8 multipliers (`build_luts.py`)
- [x] Trained an 8-bit network and ran exact and approximate inference (`run_experiment.py`)
- [x] Plotted accuracy against energy and identified the Pareto front (`plot_results.py`)
- [x] Added a multiplier-plus-adder energy model (`approx_backend.py`)
- [x] Added bias compensation and layer-wise multiplier selection
- [x] Used separate train, validation, and test sets across three seeds
- [x] Added a fast C lookup-table kernel (`lutmm.c`)
- [x] Tested the MLP on MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 (`cross_dataset.py`)
- [x] Trained and compared a small 8-bit CNN on the four datasets (`run_cnn.py`)
- [x] Compared plain inference, compensation, retraining, and both together (`retrain.py`)
- [x] Checked netlist equivalence, ABC synthesis, and switching activity (`hardware/`)
- [x] Compared error metrics with accuracy drop (`analyze_metrics.py`)
- [x] Tested approximate accumulator adders (`approx_adders.py`)
- [x] Compared signed and unsigned multipliers (`signed_multipliers.py`)
- [x] Finished the report and slides

Yosys was unavailable in our environment, so we used ABC directly for the hardware checks.

## References

- V. Mrazek et al., “EvoApprox8b: Library of approximate adders and multipliers for circuit design and benchmarking of approximation methods,” DATE 2017.
- V. Mrazek, L. Sekanina, and Z. Vasicek, “Libraries of approximate circuits: Automated design and application in CNN accelerators,” IEEE JETCAS, 2020.
- V. Mrazek et al., “ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators without Retraining,” ICCAD 2019.
- Y. LeCun et al., “Gradient-based learning applied to document recognition,” *Proceedings of the IEEE*, 1998.
- H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST,” arXiv:1708.07747, 2017.
- A. Krizhevsky, “Learning multiple layers of features from tiny images,” technical report, 2009.
