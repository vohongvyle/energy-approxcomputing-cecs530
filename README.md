### Energy-Efficient Approximate Computing for ML

Group project for CECS 530 (Fall 2026).

How much energy can a neural network save by replacing exact 8-bit multipliers with
approximate ones from [EvoApproxLib](https://github.com/ehw-fit/evoapproxlib), and what does it
cost in accuracy? In our experiments, one approximate multiplier cut multiplier energy by
**about 75% with no loss in accuracy**.

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

## References

- V. Mrazek et al., "EvoApprox8b: Library of approximate adders and multipliers for circuit design
  and benchmarking of approximation methods," DATE 2017.
- V. Mrazek et al., "ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators
  without Retraining," ICCAD 2019.
