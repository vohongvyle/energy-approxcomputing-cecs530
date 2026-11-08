## Energy-Efficient Approximate Computing for ML

Group project for CECS 530 (Fall 2026).

How much energy can a neural network save by replacing its exact 8-bit multipliers with
approximate ones from [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib), and what does it
cost in accuracy? We tested all 36 approximate multipliers of the library, first on a small digits
dataset, then on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100.

![Energy saved at a fixed accuracy budget on four datasets](results/cross_budget_savings.png)

**Main results**

- On small handwritten digits, one multiplier (`mul8u_1AGV`) cut multiplier energy by **about 75%
  with no loss in accuracy**.
- On harder data there is less room. Allowing at most a 1-point accuracy loss, the energy saved
  (multiplier + adder) falls from **78% on MNIST** to 71% on Fashion-MNIST, 66% on CIFAR-10 and
  **57% on CIFAR-100**.
- The best multiplier for one dataset is often not safe on another, and accuracy alone hides how
  many predictions change.

## Contents

1. [How it works](#how-it-works)
2. [What an approximate multiplier does](#what-an-approximate-multiplier-does)
3. [Part 1: digits dataset](#part-1-digits-dataset)
4. [Part 2: four datasets](#part-2-four-datasets)
5. [Run it](#run-it)
6. [Files](#files)
7. [Roadmap](#roadmap)

## How it works

![The experiment pipeline in five steps](docs/images/pipeline.png)

1. **Lookup tables.** `build_luts.py` compiles the C model of each of EvoApproxLib's 36 unsigned
   8×8 multipliers and stores its result for all 65,536 input pairs in a 256 × 256 table. Power and
   error figures come from the library.
2. **8-bit network.** A small MLP is trained in floating point, then quantized: activations become
   unsigned 8-bit values (inputs and ReLU outputs are never negative), and weights become a sign
   plus an 8-bit magnitude, so an unsigned multiplier can be used.
3. **Approximate inference.** Every product `x · |w|` is read from the chosen multiplier's table.
   A small C kernel (`lutmm.c`) does about 2.4 billion lookups per second on 2 CPU cores, and gives
   exactly the same result as a plain NumPy version.
4. **Energy.** Each multiply-accumulate (MAC) counts the multiplier plus an exact 16-bit adder
   (`add16u_1E2`, 0.072 mW). Because the adder always costs energy, no multiplier can save more than
   84% of MAC energy.

## What an approximate multiplier does

An approximate multiplier gives up some exactness in return for a smaller, lower-power circuit.
Each map below shows its error for every one of the 65,536 pairs of 8-bit inputs.

![Error maps of four multipliers](docs/images/multiplier_error_maps.png)

*The exact multiplier (left) has no error. Cheaper circuits make larger errors, and most of them
lean toward results that are too small (blue). Summed over hundreds of products in a neuron, that
lean builds up into a bias.*

## Part 1: digits dataset

scikit-learn's 8×8 handwritten digits, a 64-64-10 network.

![Accuracy against multiplier energy for 36 approximate multipliers](docs/images/accuracy_vs_energy.png)

*Each dot is one multiplier used in both layers. The staircase is the Pareto front: no other
multiplier is both cheaper and more accurate.*

| Setup (seed 0, 540 test images) | Accuracy | Multiplier energy |
|---|---|---|
| 8-bit exact (`mul8u_1JFF`) | 97.78% | 1.00 |
| `mul8u_1AGV`, both layers | 97.78% | 0.24 |
| `mul8u_FTA`, both layers | 97.59% | 0.21 |
| `mul8u_QKX`, both layers | 97.22% | 0.07 |
| `mul8u_13QR`, both layers | 93.89% | 0.02 |
| `mul8u_17QU`, both layers | 72.22% | 0.004 |

### Improved backend: `approx_backend.py`

![Accuracy against energy with bias compensation and layer-wise search](docs/images/backend_results.png)

- **Bias compensation at no hardware cost.** Each product's expected error is estimated on
  calibration data and folded into the neuron biases. It is switched on only when it helps on the
  validation set.
- **Layer-wise search.** Each layer gets its own multiplier: the cheapest one that stays within an
  accuracy budget on validation.
- **Separate train, validation and test sets, three seeds.**

| Setup (mean of 3 seeds) | MAC energy | Test accuracy |
|---|---|---|
| 8-bit exact | 1.00 | 97.31% |
| `mul8u_1AGV`, both layers | 0.36 | 97.31% |
| `mul8u_13QR`, plain → compensated | 0.17 | 91.85% → 92.87% |
| `mul8u_17QU`, plain → compensated | 0.16 | 66.39% → 79.07% |
| Layer-wise search, 0.5-point budget | 0.23 | 96.39% |

The search stays within the budget on validation; on test it lost 0.9 points.

## Part 2: four datasets

Do these savings hold on harder data? We ran all 36 multipliers in the same kind of network, an
MLP with one hidden layer of 256 neurons, on four standard datasets, 3 seeds each. The multiplier is
chosen on a 5,000-image validation set and reported on the official 10,000-image test set.

![Eight test images from each dataset](docs/images/dataset_samples.png)

| Dataset | Float accuracy | 8-bit exact | MAC energy saved, ≤ 1-point loss | Multiplier energy saved | Test drop |
|---|---|---|---|---|---|
| MNIST | 97.94% | 97.94% | 78.2% | 92.6% | 0.21 pp |
| Fashion-MNIST | 89.14% | 89.13% | 71.3% | 84.4% | 0.76 pp |
| CIFAR-10 | 46.62% | 46.52% | 66.3% | 78.5% | 0.56 pp |
| CIFAR-100 | 20.64% | 20.42% | 57.2% | 67.7% | 0.72 pp |

### Harder datasets lose accuracy at lower savings

![Accuracy drop against energy saved on four datasets](results/cross_drop_vs_energy.png)

*Lines show the best multiplier at each energy saving; faint dots are all multipliers. Panel B
divides the drop by the exact model's accuracy: relative to what it had, CIFAR-100 loses the most.*

### The same multiplier costs different amounts on each dataset

![Accuracy drop of each multiplier on each dataset](docs/images/multiplier_heatmap.png)

*`mul8u_QKX` is the best choice for MNIST (0.2 points lost at 7.4% of the energy) but loses 3.0
points on CIFAR-10. Below about 2% of exact energy (`mul8u_13QR` and cheaper), every dataset
collapses.*

### Accuracy hides how many predictions change

![Changed predictions against accuracy drop](results/cross_flips.png)

*With `mul8u_1AGV`, 0.6% of MNIST predictions change, against 4.8% on Fashion-MNIST, 12.5% on
CIFAR-10 and 41% on CIFAR-100. On hard tasks many predictions change before accuracy drops,
because many of them were already wrong.*

**Bias compensation rarely helps here.** It helped the most aggressive multipliers (for example
`mul8u_13QR` on CIFAR-10, 12.1 → 9.1 points lost), but often hurt the useful ones: `mul8u_QKX` lost
2.30 points on MNIST with it, against 0.21 without.

**Limits.** An MLP is a weak model for CIFAR (46.5% and 20.4%), so a CNN may behave differently.
Energy is estimated from the library's 45 nm figures, not measured, and does not count memory.

## Run it

```bash
pip install -r requirements.txt          # numpy, scikit-learn, matplotlib (+ pillow for image folders)
git clone https://github.com/ehw-fit/evoapproxlib
python3 build_luts.py --lib evoapproxlib --out luts    # needs a C compiler (gcc or clang)

# Part 1: digits (seconds)
python3 run_experiment.py
python3 plot_results.py --dataset digits
python3 approx_backend.py                               # about 45 s

# Part 2: four datasets
bash get_data.sh                                        # about 400 MB into data/
python3 cross_dataset.py                                # about 45 min on 2 CPU cores
python3 cross_dataset.py --datasets mnist --seeds 1     # quick test, about 1 min
python3 plot_cross_dataset.py
python3 docs/make_readme_images.py
```

What a full run of `cross_dataset.py` prints:

![Terminal output of cross_dataset.py](docs/images/terminal_run.png)

## Files

| File | What it does |
|---|---|
| `build_luts.py` | Lookup tables for the 36 EvoApproxLib multipliers |
| `run_experiment.py` | Part 1: quantized network, exact vs approximate inference on digits |
| `plot_results.py` | Part 1: accuracy vs energy plot |
| `approx_backend.py` | Part 1: fuller energy model, bias compensation, layer-wise search |
| `datasets.py` | Loaders for MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 |
| `lutmm.c`, `fastlut.py` | Fast lookup-table matrix multiply in C, with a Python wrapper |
| `cross_dataset.py` | Part 2: all 36 multipliers on four datasets |
| `plot_cross_dataset.py` | Part 2: figures in `results/` |
| `get_data.sh` | Downloads the four datasets |
| `docs/make_readme_images.py` | Dataset samples and heatmap for this README |
| `results/` | CSV results and figures from our runs |

## Roadmap

Each open item is one GitHub issue with one owner. Tick it when its pull request is merged.

### Done

- [x] Lookup tables for all 36 unsigned 8×8 multipliers in EvoApproxLib (`build_luts.py`)
- [x] 8-bit quantized network with exact and approximate inference (`run_experiment.py`)
- [x] Accuracy vs energy plot and Pareto front (`plot_results.py`)
- [x] Multiplier-plus-adder energy model (`approx_backend.py`)
- [x] Bias compensation folded into neuron biases
- [x] Layer-wise multiplier search with an accuracy budget
- [x] Separate train, validation and test sets over three seeds
- [x] Fast C lookup-table kernel (`lutmm.c`)
- [x] Study on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 (`cross_dataset.py`)

### Next: scale up the computation

- [ ] **A CNN on the four datasets** · owner: @teammate-A
  - [ ] Small CNN (2 conv layers + 1 dense) quantized to 8 bits, convolutions as im2col + `fastlut.lut_matmul` (`run_cnn.py`)
  - [ ] Rerun the four-dataset study with it and compare with the MLP
  - Question: does a CNN, much more accurate on CIFAR, tolerate approximation better or worse?

- [ ] **Retraining with approximate multipliers** · owner: @teammate-B
  - [ ] Fine-tune with the approximate multiplier in the forward pass and a straight-through estimator for gradients (`retrain.py`)
  - [ ] Compare plain, bias-compensated, retrained, and both, for `mul8u_13QR` and `mul8u_17QU`
  - Question: how much accuracy does retraining recover compared with free bias compensation?

- [ ] **Check the hardware numbers** · owner: @teammate-C
  - [ ] Synthesize the exact multiplier and 3–4 approximate ones from EvoApproxLib's Verilog with Yosys (`hardware/synth.sh`)
  - [ ] Table of cell count and area against the library's power figures
  - Question: do the cheapest multipliers keep the same ranking after synthesis?

- [ ] **Which error metric predicts the damage** · owner: @teammate-D
  - [ ] For all 36 multipliers, correlate mean absolute error, worst-case error, bias and error probability with the accuracy drop (`analyze_metrics.py`)
  - [ ] Scatter plot with correlation values for each metric
  - Question: is bias a better predictor than average error?

### Later

- [ ] Layer-wise search on the four datasets
- [ ] Approximate adders in the accumulator, not only multipliers
- [ ] Signed multipliers (`mul8s_*`) instead of sign-magnitude weights
- [ ] Final report and slides with all results

## References

- V. Mrazek et al., "EvoApprox8b: Library of approximate adders and multipliers for circuit design
  and benchmarking of approximation methods," DATE 2017.
- V. Mrazek, L. Sekanina and Z. Vasicek, "Libraries of approximate circuits: Automated design and
  application in CNN accelerators," IEEE JETCAS, 2020.
- V. Mrazek et al., "ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators
  without Retraining," ICCAD 2019.
- Y. LeCun et al., "Gradient-based learning applied to document recognition," Proc. IEEE, 1998 (MNIST).
- H. Xiao, K. Rasul and R. Vollgraf, "Fashion-MNIST," arXiv:1708.07747, 2017.
- A. Krizhevsky, "Learning multiple layers of features from tiny images," tech. report, 2009 (CIFAR).