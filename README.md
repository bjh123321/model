# mfdnn_pzk

Modularized refactor of the original monolithic training script for PZK‑parameter MFDNN.

## Layout
```
surrogate_model/
  ├── __init__.py
  ├── config.py
  ├── utils/
  │   └── dist_utils.py
  ├── data/
  │   ├── io.py
  │   ├── normalizer.py
  │   └── datasets.py
  ├── models/
  │   └── blocks.py
  ├── losses/
  │   └── masked_mse.py
  ├── train/
  │   ├── dataloaders.py
  │   └── loops.py
  └── cli/
      ├── train_pzk.py
  │   └── model_test.py
```

## Run (single node, multi‑GPU)
```bash
torchrun --nproc_per_node=2 -m mfdnn_pzk.cli.train_pzk \
  --sim_txt /path/to/sim_dataset_new.txt \  --real_txt /path/to/real_dataset_new.txt \  --out_dir ./outputs_pzk \  --kfold 5 --K_max 8 --embed_dim 16 \  --epochs_s1 2000 --epochs_s2 2000 \  --batch_sim 1024 --batch_real 256
```

## Notes
- Functionality mirrors the original script (two‑stage training, masked loss, K‑fold on REAL only, DDP, checkpoints, early stop, validation).
- Normalizers are saved/loaded from `out_dir`, same filenames as before.
- Keep your data format unchanged — parsing & normalization logic is identical.
