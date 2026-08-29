# All-channel SVM, AlexNet and Transformer benchmark for s09 and s12

## What “all channels” means

The acquisition driver supports at most **100 channels per patient**. This is a
hardware/driver limit, not the number of channels that must be present in every
recording.

The channels actually connected and eligible for analysis are always defined by
`ch_to_keep[patient]` in the external patient `config.py`. Empty driver inputs,
service signals and channels excluded from `ch_to_keep` are not model features.

For this run the configured sets are:

- `s09`: `Fp1, Fpz, Fp2, F8, F4` (5 channels);
- `s12`: `F7, F3, Fz, F4, F8, Fp1, Fpz, FC4, Fp2, Ft8, T4` (11 channels).

The runner resolves spelling/case against the EDF channel names and requires a
consistent channel order across the selected sessions.

## Models

Every condition is evaluated with all three existing model families:

- TFR SVM;
- AlexNet-TFR;
- TFR Transformer.

The model-specific hyperparameters are loaded from the existing patient Optuna
studies. They are not optimized again during this benchmark. All three models
use the same accepted epochs and the same outer five-fold splits.

## Experimental conditions

Only three time windows are evaluated:

- `100–2000 ms`;
- `100–1000 ms`;
- `0–1000 ms`.

Only the `0–1000 ms` window receives the classical time-bin slice `100:-400`:
the first 100 and last 400 **time bins** are removed. These are bins of the `T`
axis, not frequency bins. The `100–1000 ms` and `100–2000 ms` windows are used
without this additional trim. The launcher uses `tfr_decim=2` so the slice is
valid.

Only one frequency range is evaluated: the existing `F0_current_0p1_59p4`
range, corresponding to approximately `0.1–59.4 Hz`. Therefore each patient has
exactly three conditions. The `0–1000 ms` condition is used as the named
baseline in the generated report.

## Processing and evaluation

For each patient the runner:

1. loads accepted epochs from the frozen `open_hand_vs_all_other_gestures`
   manifest;
2. applies notch filtering at 50/100/150 Hz, a 0.1–120 Hz band-pass and CAR over
   the complete `ch_to_keep` set;
3. calculates and saves one full `C×F×T` Morlet TFR per patient so that the same
   source tensor is used by all three models;
4. applies the selected window and, only for the `0–1000 ms` window, the
   `100:-400` time-bin slice while preserving the remaining time axis for the
   models;
5. keeps the `0.1–59.4 Hz` frequency range;
6. fits SVM, AlexNet and Transformer using the existing confirmatory pipeline,
   including train-fold-only robust normalization and identical stratified
   five-fold splits;
7. writes fold metrics, out-of-fold predictions, model metadata and completion
   markers. Existing TFR files and completed folds make the run resumable.

## Server launch

```bash
cd /beegfs/home/t.samsonov/notebooks/Pirogov/NeuronDeCo
sbatch scripts/slurm_all_channel_models_s09_s12.sh
```

The SLURM array creates one GPU task for `s09` and one for `s12`. Prepared TFR
files are written under `PreprocessedData/all_channels_tfr/`; model results are
written under
`PreprocessedData/all_channel_model_benchmark/<patient>/<condition>/`. Separate
patient roots prevent the two SLURM array tasks from overwriting shared progress
and summary files.
