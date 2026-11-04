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

This branch keeps four component bundles for each excitation/channel combination
that is present in the input dataset: frequency, damping, residue magnitude,
and residue phase (`sin_theta` plus `cos_theta`). Each bundle now contains three
independent slot networks: 12 parameter/slot units per excitation/channel. Each
unit has its own low MLP, high-fidelity affine map and nonlinear correction MLP.
Correction inputs only include that unit's low prediction. The existing frozen
excitation/channel embeddings remain fixed conditioning features. Slot positions,
masks, normalization and public three-slot outputs are unchanged; phase keeps
two coupled outputs per slot (15 numerical outputs across the four bundles).

The four component training jobs, optimizers, early stopping and checkpoints
remain grouped by component. This is parameter independence, not twelve separate
training schedules. New checkpoints record `independent_slots`; old checkpoints
without this field still load with the original joint architecture. Resume with
profile 10 retrains old joint components instead of silently reusing them.

The model-version directory is
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

For model generation, `GENERATION/user_input.dat` has four lines: the global
INI path, model version, real-dataset list, and the simulation source **folder
name** (for example, `SAMPLE_FOR_XGD` or `SAMPLE_FOR_XGD_20261025`), without a
path. The source path is `[SensorMap]/path_sensormap` from that INI joined with
the fourth-line folder name.

The generation data root is always `[EXE_XGD]/path_exe_xgd` from that INI plus
`GENERATION`, independently of the current directory, script/executable location,
or task-file location. The real-data selection list contains one filename per
line (for example, `SF_XGD_FXD_DC_H0300_M068.txt`); relative entries are resolved
under that fixed root's `real_dataset` directory. The third line still selects
the list file, with a relative list path resolved beside `user_input.dat`.
Simulation dataset directories are also resolved under the fixed generation
root. Use `--user_input <task-file>` when the task file is stored elsewhere.

`SAMPLE_FOR_XGD` uses `GENERATION/sim_dataset`; `SAMPLE_FOR_XGD_20261025` uses
`GENERATION/sim_dataset_20261025`. Existing dataset directories are used as-is:
source/output filenames, hashes, channel layouts, and generation manifests are
not compared, and mismatches do not trigger OE extraction. Empty or malformed
datasets fail during normal loading rather than being rebuilt automatically.
Only a missing dataset directory triggers OE generation from the selected source,
in a temporary directory published after every source file succeeds.

Simulation files such as `FZ_FXD_DC_H0300_M029.dat` contain time followed by any
number of response columns. Only the first 25,600 samples (50 seconds at 512 Hz)
are used. OE uses a constructed 0.3-amplitude sine sweep starting at zero, sweeping
linearly from 0 to 50 Hz in 50 seconds, with F order 30 and B order 8. One pole is
matched in each of 10–20, 20–40, and 40–50 Hz, within 2 Hz of the detected peak.
An unmatched band remains a missing slot. Output files use the existing
`FZ_XGD_<excitation>_Hhhhh_Mmmm.txt` format.

Simulation channel IDs follow response-column order from zero. Names follow the
current channel workbook's rows whose `相关激励位置` contains FXD, PW, or JFY.
Additional response columns retain sequential IDs and receive `SIM_EXTRA_<id>`
names. Generated datasets save their channel mapping and a completeness manifest.

Correction keeps its existing four-line task file: global INI, parent version,
new version, and selected-CG file list. It obtains the current simulation selection
from `[EXE_XGD]/path_exe_xgd` plus `GENERATION/user_input.dat`, and selects/reuses
that dataset using the same rules as generation. Parent simulation snapshots do
not override this current selection. CG extraction still uses
`极点频段配置.xlsx` beside the correction script/executable; the simulation
parameters above do not change that workbook or CG extraction.

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

