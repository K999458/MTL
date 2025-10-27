from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class TrainingConfig:
    # Data locations
    data_root: str = "/storu/zkyang/AAA_MIL/data_generated"
    cache_root: str = "/storu/zkyang/AAA_MIL/data_generated/cache"
    label_root: str = "/storu/zkyang/AAA_MIL/data_generated/labels"
    manifest_root: str = "/storu/zkyang/AAA_MIL/data_generated/manifests"

    loop_manifest: str = "loop_5kb.jsonl"
    stripe_manifest: str = "stripe_10kb.jsonl"
    tad_manifest: str = "tad_10kb_1d.jsonl"
    loop_label: str = "loop_5kb.jsonl"
    stripe_label: str = "stripe_10kb.jsonl"
    tad_label: str = "tad_boundaries_10kb.bed"

    # Patch geometry
    loop_patch: int = 256
    loop_center: int = 224
    loop_stride: int = 112
    stripe_patch: int = 320
    stripe_center: int = 256
    stripe_stride: int = 128
    tad_length: int = 1024
    tad_stride: int = 512
    tad_band_width: int = 64
    tad_ignore_bins: int = 32

    # Data loader
    batch_size_loop: int = 4
    batch_size_stripe: int = 4
    batch_size_tad: int = 8
    num_workers: int = 8
    pin_memory: bool = True

    # Training
    epochs: int = 120
    steps_per_epoch: int = 400
    grad_clip: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    cosine_final_lr: float = 1e-6

    # Loss weights (initial)
    lambda_loop_heat: float = 1.0
    lambda_loop_offset: float = 1.0
    lambda_stripe: float = 1.0
    lambda_tad: float = 1.0
    lambda_consistency: float = 0.1
    loop_pos_weight: float = 20000.0
    stripe_pos_weight: float = 25.0
    stripe_area_weight: float = 0.05
    tad_pos_weight: float = 8.0

    # Multi-task balancing
    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5
    use_pcgrad: bool = True

    # Logging / checkpoint
    log_interval: int = 20
    eval_interval: int = 200
    save_interval: int = 1
    output_dir: str = "/storu/zkyang/AAA_MIL/outputs"

    # Seed & precision
    seed: int = 1337
    amp: bool = True
    base_channels: int = 64
    use_axial_attention: bool = True
    use_tqdm: bool = True


def build_default_config() -> TrainingConfig:
    return TrainingConfig()
