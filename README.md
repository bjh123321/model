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
