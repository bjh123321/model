from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Fixed-SF model shape
    K_max: int = 3
    embed_excitation_num: int = 6
    embed_channel_num: int = 34
    embed_excitation_dim: int = 4
    embed_channel_dim: int = 6

    low_hidden: List[int] = field(default_factory=lambda: [32, 64, 32])
    mod_hidden: List[int] = field(default_factory=lambda: [16, 32])
    dropout_prob: float = 0.0
    use_lora: bool = True
    lora_rank: int = 4
    use_modulation: bool = True
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
    val_split_strategy: str = "height_max_mach"
    val_height_tol: float = 1e-6
    val_mach_tol: float = 1e-6
    val_heights: List[float] = field(default_factory=lambda: [3.0, 8.23])

    # Target representation
    pole_target: str = "alpha_omega"
    residue_target: str = "polar"

    epochs_stage1: int = 3000
    early_stop_patience_s1: int = 500
    early_stop_min_delta: float = 1e-6

    # Optim
    lr_low_adam: float = 1e-3
    weight_decay: float = 5e-5
