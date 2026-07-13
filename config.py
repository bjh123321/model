from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Fixed-SF model shape
    K_max: int = 3
    embed_excitation_num: int = 6
    embed_channel_num: int = 34
    embed_excitation_dim: int = 4
    embed_channel_dim: int = 12

    low_hidden: List[int] = field(default_factory=lambda: [64, 128, 64])
    dropout_prob: float = 0
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    use_modulation: bool = True
    modulation_mode: str = "legacy"
    mod_gamma: float = 0.2
    debug_direct_output: bool = False

    # Fixed-SF training
    batch_real: int = 64
    num_workers: int = 8
    prefetch: int = 4

    # Train/val split
    use_val_split: bool = True
    val_ratio: float = 0.2
    val_seed: int = 42
    val_split_strategy: str = "height_mach_points" # val_height_mach_points中高度、马赫数点作验证集
    # val_split_strategy: str = "height_max_mach"   # val_heights中高度的最高马赫数作验证集
    val_height_tol: float = 1e-6
    val_mach_tol: float = 1e-6
    val_heights: List[float] = field(default_factory=lambda: [3.0, 8.23])
    # Used when val_split_strategy="height_mach_points", e.g. [(3.0, 0.7), (8.23, 0.85)].
    val_height_mach_points: List[Tuple[float, float]] = field(
        default_factory=lambda:[
            (3.0, 0.98),
            (5.5, 1.05),
            (5.5, 1.17),
            (5.5, 1.3),
            (8.23, 1.7)
        ]
    )

    # Target representation
    pole_target: str = "freq_damping"
    residue_target: str = "polar"

    epochs_stage1: int = 3000
    early_stop_patience_s1: int = 500
    early_stop_min_delta: float = 1e-6
    checkpoint_selection: str = "validation"

    # Optim
    lr_low_adam: float = 1e-3
    weight_decay: float = 5e-5

    # LoRA incremental fine-tuning
    incremental_epochs: int = 2000
    incremental_batch_size: int = 16
    incremental_lr: float = 3e-4
    incremental_weight_decay: float = 1e-4
    incremental_beta_kd: float = 0.2
    incremental_gamma_replay: float = 0.1
    incremental_warmup_epochs: int = 10
    incremental_old_loss_tolerance: float = 0.05
    incremental_lambda_guard: float = 1.0
    incremental_early_stop_patience: int = 50
    incremental_early_stop_min_delta: float = 1e-5
    incremental_guard_max_groups: int = 48
    incremental_guard_min_groups_per_excitation: int = 2
    incremental_replay_primary_h_thr: float = 1.5
    incremental_replay_primary_m_thr: float = 0.10
    incremental_replay_fallback_h_thr: float = 2.5
    incremental_replay_fallback_m_thr: float = 0.15
    incremental_replay_max_groups_per_selected: int = 6
    incremental_replay_min_groups_per_selected: int = 3
    incremental_selection_seed: int = 42
    incremental_initial_guard_tolerance: float = 1e-6
    # Complete historical split dataset used as the replay pool in CG+OE mode.
    incremental_replay_dataset_dir: str = r"C:\Users\leo\Desktop\split_real_dataset_opt"
