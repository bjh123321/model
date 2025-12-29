from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    K_max: int = 6
    embed_excitation_num: int = 6
    embed_channel_num: int = 34
    embed_excitation_dim: int = 4
    embed_channel_dim: int = 12

    low_hidden: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 256, 128, 64])
    mod_hidden: List[int] = field(default_factory=lambda: [32, 64])            # 调制头
    adapt_hidden: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 128, 64])
    dropout_prob: float = 0
    mod_gamma: float = 0.2
    use_lora: bool = True
    lora_rank: int = 4
    
    # Training
    batch_sim: int = 512
    batch_real: int = 64
    num_workers: int = 8
    prefetch: int = 4

    epochs_stage1: int = 10000
    epochs_stage2: int = 10000
    early_stop_patience_s1: int = 1000
    early_stop_patience_s2: int = 1000
    early_stop_min_delta: float = 1e-6

    # Optim
    lr_low_adam: float = 1e-3
    lr_high_adam: float = 1e-3
    weight_decay: float = 5e-5

    # 网络与损失
    use_modulation: bool = True
    use_modulation_in_high: bool = False
    use_hungarian_loss: bool = False

    # 存在头
    presence_loss_weight: float = 0.2
    cardinality_loss_weight: float = 0.05
    presence_threshold: float = 0.5
    
