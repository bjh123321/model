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
- `Normalizer_Input_Sim.pth` and `Normalizer_Input_Real.pth` contain the two
  input features `[H, M]`. They store the same shared simulation-coordinate
  transform; simulation and real output transforms are fitted separately.
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
      Normalizer_Input_Sim.pth
      Normalizer_Output_Sim.pth
      Normalizer_Input_Real.pth
      Normalizer_Output_Real.pth
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

Use component-level parallel training on one or more GPUs with, for example:

```bash
python -m surrogate_model.cli.train --parallel_workers 4
```

`parallel_workers` accepts any positive integer. On a single GPU, benchmark `2`,
`3`, and `4` and use the value with the best completed-networks/hour throughput.
Set DataLoader workers conservatively when several training processes share one
machine, for example `--parallel_workers 4 --num_workers 0`.

Use `python -m surrogate_model.cli.train --resume --parallel_workers 4` after an
interruption. Every component whose checkpoint, normalizers, training config, and
log are complete is retained; only missing or incomplete components are queued.
An interrupted component restarts from its first epoch because optimizer/scheduler
state is not persisted. The parent process refreshes `models_manifest.json` whenever
a complete four-component excitation/channel bundle becomes available, so completed
bundles can be used for prediction before the full queue finishes.

Do not start multiple identical training commands against the same output directory.
The built-in component queue is the supported parallel path; `torchrun`/DDP is
rejected for this independent-model workflow.

Train only selected excitations and/or response channels by name or numeric ID:

```bash
python -m surrogate_model.cli.train --excitation FXD_DC --channel LVT1,LVT2 --parallel_workers 4
python -m surrogate_model.cli.train --excitation 0,1 --channels 8,9,10 --parallel_workers 3
```

Select exact FDR component networks with `--component`. Explicitly selected
components are retrained in place in their existing excitation/channel/component
directories, even when `--resume` is also present; unselected components are not
queued or modified:

```bash
python -m surrogate_model.cli.train --component damping --excitation FXD_DC --channel LVT1
python -m surrogate_model.cli.train --components frequency,damping --excitation 0,1 --channels 8,9
```

Available component names are `frequency`, `damping`, `residue_magnitude`, and
`residue_phase`. Omitting `--component` retains the original all-component behavior.

Checkpoint/early-stop selection for both stages, plus stage-2 real/simulation loss
weights, can be overridden independently for each component:

```powershell
python -m surrogate_model.cli.train `
  --component_checkpoint_selection frequency=training,damping=training,residue_magnitude=training,residue_phase=training `
  --component_real_loss_weight frequency=1,damping=1,residue_magnitude=1,residue_phase=1 `
  --component_sim_loss_weight frequency=0.2,damping=0.2,residue_magnitude=0.2,residue_phase=0.2
```

The same training options are accepted by `train_incremental`. Multifidelity
training remains one global network with fixed component-level real/simulation
loss weights; it does not divide the model or change its loss weights by Mach.
Without an explicit epoch override, the damping component uses at most 200 epochs
in each multifidelity stage. All four FDR components use an early-stop patience
of 50 in both stages. Other component epoch defaults remain unchanged. All four component networks
select checkpoints and apply early stopping from training loss by default; pass
`--checkpoint_selection validation` or a per-component override to use validation loss.

Prediction defaults to `--extrapolation_policy allow`, so frequency, damping,
residue magnitude, and residue phase all use their direct network predictions.
Passing `--extrapolation_policy smooth_linear` explicitly keeps the network damping
inside each excitation/channel/slot's flight-test training Mach envelope, joins the
boundary value and one-sided slope with a cubic Hermite transition, and then uses a
straight line fitted to the in-support `damping * 200` curve. Both value and first
derivative are continuous at the boundary and linear-tail join. The transition and
in-support line-fit widths default to `0.05` Mach and can be changed with
`--damping_extrapolation_transition_mach` and
`--damping_extrapolation_fit_mach`.

Selectors are case-insensitive, repeatable, and may be comma- or semicolon-separated.
Omit one selector to include every value on that axis. Complete unselected bundles
are retained in `models_manifest.json`. To retrain selected models using validation
loss for checkpoint selection, omit `--resume` and pass
`--checkpoint_selection validation`; adding `--resume` retains already-complete
components even if they were originally selected by training loss.

`cli.train_incremental` requires a parent version containing
`models_manifest.json`.  It copies every parent channel model into the new version,
then retrains only the excitation/channel combinations present in the correction
files, using all merged historical rows for each affected combination.  A failed
correction retains the inherited parent model and records the failure.

For direct correction mode, `user_input.dat` line 2 is the parent model version,
line 3 is the new model version, and every non-empty line from line 4 onward is an
added/replacement data file. These files determine the candidate excitation/channel
networks. `--excitation`, `--channel`, and `--component` further restrict that set:

When line 4 and later are empty, correction reads `selected_CG_path.txt` and
`极点频段配置.xlsx` beside `train_incremental.py` (or beside the packaged
executable), runs the App-aligned OE extractor with `prom=0`, and creates the
correction split files in a temporary directory. The temporary files are removed
after merging. The only persistent copy is the merged data in the new version's
`dataset_snapshot/`.

```bash
python -m surrogate_model.cli.train_incremental --component damping
python -m surrogate_model.cli.train_incremental --component damping --excitation PW_FDC --channel LHT1
python -m surrogate_model.cli.train_incremental --excitation FXD_DC,FXD_FDC --parallel_workers 8 --num_workers 0
```

Correction always performs a full retrain from the merged historical and correction
dataset; it does not fine-tune parent weights. Selected component directories replace
their inherited counterparts in the new version after successful training. Unselected
component directories remain unchanged, and a failed retrain keeps the inherited
network. `--parallel_workers` runs independent excitation/channel/component networks
concurrently; the parent process commits a combination only after all selected
components succeed. On one GPU, keep `--num_workers 0` and benchmark the worker count
against available GPU memory.

`test.Predict_user` keeps the existing task-file and output-file interface. It loads
all four component checkpoints for the matching excitation/channel combination,
inverse-transforms their outputs, reconstructs pole real/imaginary parts and complex
residues, and then runs the existing response pipeline. Combinations absent from the
inventory intentionally produce zero/empty output without a warning.

Prediction checks bounds against the displayed `damping * 200` value. In the
`HF/HG/HA` and `VF/VG/VA` scan files, a value below `--prediction_damping_min`
(default `2`) or above `--prediction_damping_max` (default: no upper limit) blanks
only the matching `HG`/`VG` damping cell; frequency and response-amplitude values
for that slot are retained. In `PR_*` response prediction and `RESULT_PR_*`, the
same failed damping check continues to reject the entire modal slot from pole/residue
reconstruction and downstream response calculations. Other slots remain unchanged.
Bounds are inclusive, so values equal to either configured bound are retained. Use
`none` to disable either bound, for example:

```bash
python -m surrogate_model.test.Predict_user --prediction_damping_min 2 --prediction_damping_max 60
python -m surrogate_model.test.Predict_user --prediction_damping_min none
```
