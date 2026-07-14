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
- Model forward remains `model(Hn, Mn, exc_t, ch_t)`.
- `Normalizer_Input.pth` is for two input features: `[H, M]`.
- Prediction output directories include `PR_XGD`, `RESULT_PR_XGD`, `HF_PR_XGD`, `HG_PR_XGD`, `HA_PR_XGD`, `VF_PR_XGD`, `VG_PR_XGD`, and `VA_PR_XGD`.
- Stage 2 trains `low + lin + adaptation` together.
- Validation is enabled by default. `val_split_strategy="height_max_mach"` keeps the existing validation rule; `val_split_strategy="height_mach_kfold"` splits complete `(H, M)` operating-point groups.
- Residue targets are configurable with `residue_target="polar"` or `residue_target="cartesian"`. The input data file format is unchanged.
- Output normalization is global masked normalization over valid modes only. `polar` uses `[alpha, omega, rho, sin_theta, cos_theta]`; `cartesian` uses `[alpha, omega, residue_re, residue_im]`.

# Excitation/channel model layout

This branch trains one independent model for each excitation/channel combination
that is present in the input dataset.  The model-version directory is organized as:

```text
<model_version>/
  dataset_snapshot/
  dataset_manifest.json
  models_manifest.json
  training_failures.json
  FXD_FDC/LVT1/
    model.pth
    Normalizer_Input.pth
    Normalizer_Output.pth
    training_config.json
    train_log.csv
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

`test.Predict_user` keeps the existing task-file and output-file interface.  It
loads `models_manifest.json` and dispatches every prediction row to the matching
excitation/channel checkpoint.  Missing combinations produce zero/empty output
with a warning.
