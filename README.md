# surrogate_model

Python package for the PZK surrogate model.

## Install

From this repository root:

```bash
python -m pip install -e .
```

## Recommended Commands

```bash
python -m surrogate_model.cli.train
python -m surrogate_model.cli.train_incremental
python -m surrogate_model.test.Predict_user
```

## Data And Model Contract

- Input txt rows keep the base format: `H, M, exc_id, ch_id, Re, Im, rho, theta ...`
- Every FDR component model keeps the forward signature `model(Hn, Mn, exc_t, ch_t)`.
- `Normalizer_Input.pth` is for two input features: `[H, M]`.
- Prediction output directories include `PR_XGD`, `RESULT_PR_XGD`, `HF_PR_XGD`, `HG_PR_XGD`, `HA_PR_XGD`, `VF_PR_XGD`, `VG_PR_XGD`, and `VA_PR_XGD`.
- Stage 2 trains `low + lin + adaptation` together.
- Validation is enabled by default. `val_split_strategy="height_max_mach"` keeps the existing validation rule; `val_split_strategy="height_mach_kfold"` splits complete `(H, M)` operating-point groups.
- FDR training requires `pole_target="freq_damping"` and `residue_target="polar"`; the input data file format is unchanged.
- Output normalization is fitted on the training split only and stored independently for each component. The residue-magnitude network directly learns the physical magnitude converted from the dataset residue.

## Excitation/channel FDR model layout

This branch trains four independent networks for each excitation/channel combination
that is present in the input dataset: frequency, damping, residue magnitude,
and residue phase (`sin_theta` plus `cos_theta`). The model-version directory is
organized as:

```text
<model_version>/
  dataset_snapshot/
  dataset_manifest.json
  models_manifest.json
  training_failures.json
  FXD_FDC/LVT1/
    frequency/
      model.pth
      Normalizer_Input.pth
      Normalizer_Output.pth
      training_config.json
      train_log.csv
    damping/
      ...
    residue_magnitude/
      ...
    residue_phase/
      ...
  PW_FDC/LHT1/
    ...
```

`cli.train` scans the base dataset and trains only combinations present in that
dataset.  A combination without a usable train/validation split is skipped and
recorded in `training_failures.json`; other combinations continue training.

Use `python -m surrogate_model.cli.train --resume` to keep channel models whose
checkpoint, normalizers, training config, and log are complete, and retrain only
missing or incomplete combinations.

`cli.train_incremental` requires a parent version containing
`models_manifest.json`.  It copies every parent channel model into the new version,
then retrains only the excitation/channel combinations present in the correction
files, using all merged historical rows for each affected combination.  A failed
correction retains the inherited parent model and records the failure.

`test.Predict_user` keeps the existing task-file and output-file interface. It loads
all four component checkpoints for the matching excitation/channel combination,
inverse-transforms their outputs, reconstructs pole real/imaginary parts and complex
residues, and then runs the existing response pipeline. Combinations absent from the
inventory intentionally produce zero/empty output without a warning.
