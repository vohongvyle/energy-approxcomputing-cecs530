# Energy-Efficient Approximate Computing for ML

Group project for CECS 530 (Fall 2026).

We tested how much energy a neural network could save by replacing exact 8-bit multipliers with
approximate ones from [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib), and what that change
did to accuracy. In our experiments, one approximate multiplier cut multiplier energy by about 75%
without reducing accuracy.

![Accuracy against multiplier energy for 36 approximate multipliers](docs/images/accuracy_vs_energy.png)

*Each dot is one approximate multiplier used in the network. The staircase is the Pareto front:
no other multiplier is both cheaper and more accurate.*

## What an approximate multiplier does

An approximate multiplier gives up some exactness in return for a smaller, lower-power circuit.
Each map below shows its error for every one of the 65,536 pairs of 8-bit inputs.

![Error maps of four multipliers](docs/images/multiplier_error_maps.png)

*The exact multiplier (left) has no error. Cheaper circuits make larger errors, and most of them
lean toward results that are too small (blue). That lean matters: summed over hundreds of
products in a neural network, it builds up into a bias.*

## Run it

```bash
pip install -r requirements.txt
git clone https://github.com/ehw-fit/evoapproxlib
python3 build_luts.py --lib evoapproxlib --out luts    # needs a C compiler (gcc or clang)
python3 run_experiment.py                               # digits dataset, a few seconds
python3 plot_results.py --dataset digits
python3 approx_backend.py                               # improved backend, about 45 s
```

## First results

Digits dataset, seed 0, 540 test images.

| Setup | Accuracy | Multiplier energy |
|---|---|---|
| Float model | 97.78% | – |
| 8-bit exact (`mul8u_1JFF`) | 97.78% | 1.00 |
| `mul8u_1AGV`, both layers | 97.78% | 0.24 |
| `mul8u_FTA`, both layers | 97.59% | 0.21 |
| `mul8u_QKX`, both layers | 97.22% | 0.07 |
| `mul8u_13QR`, both layers | 93.89% | 0.02 |
| `mul8u_17QU`, both layers | 72.22% | 0.004 |

- `mul8u_1AGV` saved about 75% of multiplier energy with no loss in accuracy. With about one point
  of accuracy loss, savings went above 90%.
- Low average error does not mean low error in the network. Multipliers that consistently round
  down build up bias across a dot product.
- Layer 1 holds 86% of the multiplications (64 × 64 against 64 × 10), so approximating only
  layer 2 saves little energy.

With 540 test images, one image is about 0.19 points of accuracy, so differences under about half
a point may be noise.

## Improved backend: `approx_backend.py`

![Accuracy against energy with bias compensation and layer-wise search](docs/images/backend_results.png)

1. **A fuller energy model.** Each multiply-accumulate counts the multiplier plus an exact 16-bit
   adder (`add16u_1E2`, 0.072 mW). `mul8u_1AGV` then uses 36% of the exact energy instead of 24%.
2. **Bias compensation at no hardware cost.** Each product's expected error is estimated on
   calibration data and folded into the neuron biases. In layer 1 this removes 99% of the
   accumulated bias. It is switched on for a multiplier only when it helps on the validation set.
3. **Layer-wise search.** Each layer gets its own multiplier, chosen as the cheapest option that
   stays within an accuracy budget on the validation set.
4. **Separate training, validation and test sets, averaged over three seeds.**

| Setup | Energy | Test accuracy |
|---|---|---|
| 8-bit exact | 1.00 | 97.31% |
| `mul8u_1AGV`, both layers | 0.36 | 97.31% |
| `mul8u_13QR`, plain → compensated | 0.17 | 91.85% → 92.87% |
| `mul8u_17QU`, plain → compensated | 0.16 | 66.39% → 79.07% |
| Layer-wise search, 0.5-point budget | 0.23 | 96.39% |

The search keeps within the budget on the validation set; on the test set it lost 0.9 points.

## Cross-dataset study: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100

Do these savings hold on harder data? We ran all 36 multipliers in the same 8-bit MLP
(input → 256 → classes) on four datasets, 3 seeds each. The multiplier is chosen on a validation
set and reported on the official test set.

```bash
bash get_data.sh                  # downloads the four datasets into data/ (about 400 MB)
python3 cross_dataset.py          # about 45 minutes on 2 CPU cores; fast C kernel in lutmm.c
python3 plot_cross_dataset.py
```

![Accuracy drop against energy saved on four datasets](results/cross_drop_vs_energy.png)

| Dataset | 8-bit exact accuracy | MAC energy saved, ≤ 1-point loss | Multiplier energy saved | Test drop |
|---|---|---|---|---|
| MNIST | 97.94% | 78.2% | 92.6% | 0.21 pp |
| Fashion-MNIST | 89.13% | 71.3% | 84.4% | 0.76 pp |
| CIFAR-10 | 46.52% | 66.3% | 78.5% | 0.56 pp |
| CIFAR-100 | 20.42% | 57.2% | 67.7% | 0.72 pp |

- **Harder datasets leave less room.** At a 1-point budget, MAC energy saved falls from 78% to 57%.
- **The best multiplier for one dataset is not safe on another.** `mul8u_QKX` costs 0.21 points on
  MNIST but 3.0 on CIFAR-10.
- **Accuracy hides changed predictions.** With `mul8u_1AGV`, 0.6% of MNIST predictions change,
  against 41% on CIFAR-100.
- **Bias compensation rarely helps here**, unlike on the small digits dataset.

The MLP is a weak model for CIFAR (46.5% and 20.4%),, so a CNN may behave differently: that is
the next step in the roadmap.

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
- [x] Fast C lookup-table kernel (`lutmm.c`, about 2.4 billion lookups per second on 2 cores)
- [x] Cross-dataset study on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 (`cross_dataset.py`)

### Next: scale up the computation

- [ ] **A CNN on the four datasets** · owner: @teammate-A
  - [ ] Small CNN (2 conv layers + 1 dense) quantized to 8 bits, convolutions as im2col + `fastlut.lut_matmul` (`run_cnn.py`)
  - [ ] Rerun the cross-dataset study with it and compare with the MLP results
  - Question: does a CNN, much more accurate on CIFAR, tolerate approximation better or worse than the MLP?

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

- [ ] Approximate adders in the accumulator, not only multipliers
- [ ] Signed multipliers (`mul8s_*`) instead of sign-magnitude weights
- [ ] Final report and slides with all results

## References

- V. Mrazek et al., "EvoApprox8b: Library of approximate adders and multipliers for circuit design
  and benchmarking of approximation methods," DATE 2017.
- V. Mrazek et al., "ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators
  without Retraining," ICCAD 2019.
