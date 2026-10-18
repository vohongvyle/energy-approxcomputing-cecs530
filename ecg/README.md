# Risk-aware approximate computing for wearable arrhythmia detection

This part of the project asks a health question of the approximate-multiplier work:
**when a wearable heart monitor saves energy with approximate arithmetic, which heartbeats
does it start to miss?** Average accuracy hides the answer, because abnormal beats are rare.

Everything is plain NumPy (no PyTorch), so it runs on a laptop in a few minutes.

## Files

| File | What it does |
|---|---|
| `synth_ecg.py` | Synthetic heartbeats (N, S, V, F) from simulated patients, for testing without downloads |
| `mitbih.py` | Loads the real MIT-BIH Arrhythmia Database in the same format (needs `pip install wfdb`) |
| `cnn.py` | Small 1D CNN trained in NumPy, and its 8-bit version where every multiplication goes through an EvoApproxLib lookup table, with bias compensation |
| `ecg_experiment.py` | The experiments, CSVs and the summary figure |

## Run it

From the project folder (the one with `luts/`):

```bash
python build_luts.py --lib evoapproxlib --out luts     # once, if luts/ is missing

# 1. Test bench: synthetic heartbeats, no download (about 5 minutes)
python ecg/ecg_experiment.py

# 2. Real data
pip install wfdb
python ecg/mitbih.py                     # downloads MIT-BIH (~100 MB) and prints class counts
python ecg/ecg_experiment.py --data mitbih
```

Options: `--acc-budget 0.5` (allowed drop in overall accuracy, points),
`--se-budget 1.0` (allowed drop in S and V sensitivity, points), `--seed`, `--epochs`.

## What the experiments measure

1. **Uniform sweep.** Each of the 36 multipliers in every layer, with and without bias
   compensation. Reports accuracy and sensitivity (recall) for S, V and F beats.
2. **Two layer-wise searches, chosen on validation patients and scored on unseen test patients.**
   - *Accuracy constraint*: overall accuracy within 0.5 points of exact. This is what most
     approximate-computing papers optimize.
   - *Clinical constraint*: the same, and S and V sensitivity within 1 point of exact.
3. **Confidence-gated exact fallback.** Run the cheap model on every beat. Re-run a beat on the
   exact multiplier when the cheap model calls it abnormal or its confidence is below τ.
   This traces energy against abnormal beats called normal.
4. **Battery illustration.** Battery gain = 1 / (1 − f·(1 − E)), where E is relative inference
   energy and f is the share of the device's power spent on inference. f is an assumption;
   on a real wearable, the sensor front-end and radio usually dominate.

## Testing on synthetic heartbeats: what it can and cannot tell you

The synthetic bench checks that the pipeline works end to end: the model trains, the 8-bit
model matches the float one, compensation and search run, and figures come out. Its classes are
cleaner than real ECG, so its accuracy is much higher than MIT-BIH inter-patient results.
**Never report synthetic numbers as findings.** Use them to debug, then run on MIT-BIH.

Things to check when you move to MIT-BIH:
- `python ecg/mitbih.py` should print about 45.8k N, 0.9k S, 3.8k V and 0.4k F beats in DS1.
  If not, look at the lead name and the annotation symbols first.
- Expect S sensitivity to be much lower than on synthetic data. Inter-patient S detection is
  hard, and that is where approximation effects will be easiest to see.
- Report the standard metrics for S and V: sensitivity (Se) and positive predictive value
  (+P), both written to the CSVs as `se_*` and `ppv_*`.
