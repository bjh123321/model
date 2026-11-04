from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Config:
    # Fixed-SF model shape
    K_max: int = 3
    independent_slots: bool = True
    embed_excitation_num: int = 6
    embed_channel_num: int = 34
    embed_excitation_dim: int = 4
    embed_channel_dim: int = 12

    low_hidden: List[int] = field(default_factory=lambda: [64, 128, 64])
    adapt_hidden: List[int] = field(default_factory=lambda: [32, 32])
    dropout_prob: float = 0
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    lora_target_hidden_indices: Tuple[int, ...] = (2,)
    use_modulation: bool = False
    modulation_mode: str = "legacy"
    mod_gamma: float = 0.2
    debug_direct_output: bool = False

    # Fixed-SF training
    batch_real: int = 8
    batch_sim: int = 16
    num_workers: int = 8
    prefetch: int = 4

    # Train/val split
    use_val_split: bool = False
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

    epochs_stage1: int = 500
    epochs_stage2: int = 500
    early_stop_patience_s1: int = 50
    early_stop_patience_s2: int = 50
    early_stop_min_delta: float = 1e-6
    checkpoint_selection: str = "training"

    # Optim
    lr_low_adam: float = 1e-3
    lr_high_adam: float = 1e-3
    weight_decay: float = 5e-5
    multifidelity_real_loss_weight: float = 1.0
    multifidelity_sim_loss_weight: float = 0.2
    freeze_low_stage2: bool = False

    # Independent FDR-component defaults.  Keep the split models large enough to
    # interpolate the sparse H/M samples; damping smoothness is handled by a
    # separate soft penalty below instead of shrinking every component network.
    fdr_physical_parameterization: bool = True
    fdr_training_profile_version: int = 10
    fdr_component_training_profile_version: Dict[str, int] = field(default_factory=lambda: {
        "frequency": 10,
        "damping": 10,
        "residue_magnitude": 10,
        "residue_phase": 10,
    })
    fdr_low_hidden: List[int] = field(default_factory=lambda: [64, 128, 64])
    fdr_dropout_prob: float = 0.0
    fdr_weight_decay: float = 5e-5
    fdr_early_stop_patience: int = 50
    fdr_component_low_hidden: Dict[str, List[int]] = field(default_factory=lambda: {
        "frequency": [64, 128, 64],
        "damping": [64, 128, 64],
        "residue_magnitude": [64, 128, 64],
        "residue_phase": [64, 128, 64],
    })
    fdr_component_dropout_prob: Dict[str, float] = field(default_factory=lambda: {
        "frequency": 0.0,
        "damping": 0.0,
        "residue_magnitude": 0.0,
        "residue_phase": 0.0,
    })
    fdr_component_weight_decay: Dict[str, float] = field(default_factory=lambda: {
        "frequency": 5e-5,
        "damping": 5e-5,
        "residue_magnitude": 5e-5,
        "residue_phase": 5e-5,
    })
    fdr_component_early_stop_patience: Dict[str, int] = field(default_factory=lambda: {
        "frequency": 50,
        "damping": 50,
        "residue_magnitude": 50,
        "residue_phase": 50,
    })
    fdr_component_epochs_stage1: Dict[str, int] = field(default_factory=lambda: {
        "frequency": 500,
        "damping": 200,
        "residue_magnitude": 500,
        "residue_phase": 500,
    })
    fdr_component_epochs_stage2: Dict[str, int] = field(default_factory=lambda: {
        "frequency": 500,
        "damping": 200,
        "residue_magnitude": 500,
        "residue_phase": 500,
    })
    fdr_component_checkpoint_selection: Dict[str, str] = field(default_factory=lambda: {
        "frequency": "training",
        "damping": "training",
        "residue_magnitude": "training",
        "residue_phase": "training",
    })
    fdr_component_real_loss_weight: Dict[str, float] = field(default_factory=lambda: {
        "frequency": 1.0,
        "damping": 1.0,
        "residue_magnitude": 1.0,
        "residue_phase": 1.0,
    })
    fdr_component_sim_loss_weight: Dict[str, float] = field(default_factory=lambda: {
        "frequency": 0.2,
        "damping": 0.2,
        "residue_magnitude": 0.2,
        "residue_phase": 0.2,
    })
    fdr_checkpoint_train_loss_ceiling: float = 0.05
    fdr_component_checkpoint_train_loss_ceiling: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "frequency": 0.05,
        "damping": None,
        "residue_magnitude": 0.02,
        "residue_phase": 0.02,
    })
    min_slot_train_samples: int = 8
    damping_slope_weight: float = 0.0
    damping_curvature_weight: float = 0.0
    damping_slope_limit_multiplier: float = 1.5
    damping_slope_limit_floor: float = 0.03
    # Robust sim-only pole-slot screening.  Frequency and damping are fitted
    # jointly per excitation/channel/slot from the simulation training split.
    sim_slot_outlier_filter_enabled: bool = True
    sim_slot_outlier_min_samples: int = 8
    sim_slot_outlier_mad_threshold: float = 12.0
    sim_slot_outlier_scale_floor_ratio: float = 0.02

    # Real-data damping screening.  A value above the configured limit masks
    # the whole modal slot for every FDR component.  The channel lists encode
    # the fixed excitation/channel inventory used by this branch.
    real_damping_upper_filter_enabled: bool = True
    real_damping_upper_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 0.15,
        1: 0.15,
        2: 0.30,
        3: 0.30,
        4: 0.30,
        5: 0.30,
    })
    real_damping_upper_channel_ids: Dict[int, Tuple[int, ...]] = field(default_factory=lambda: {
        0: tuple(range(8, 20)),
        1: tuple(range(8, 20)),
        2: tuple(range(20, 28)),
        3: tuple(range(20, 28)),
        4: tuple(range(0, 8)),
        5: tuple(range(0, 8)),
    })

    # One-time migration source for parent versions created before dataset snapshots.
    incremental_history_dataset_dir: str = r"C:\Users\leo\Desktop\split_real_dataset_opt"
