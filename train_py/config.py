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
    tad_manifest: str = "tad_10kb_2d.jsonl"
    tad_binsize: int = 10000
    loop_label: str = "loop_5kb.jsonl"
    stripe_label: str = "stripe_10kb.jsonl"

    # Patch geometry
    loop_patch: int = 256
    loop_center: int = 224
    loop_stride: int = 112
    stripe_patch: int = 320
    stripe_center: int = 256
    stripe_stride: int = 128
    tad_band_width: int = 64

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
    stripe_orientation_weight: float = 0.05
    tad_domain_threshold: float = 0.4
    tad_domain_map_weight: float = 1.0
    tad_min_bins: int = 2
    tad_boundary_weight: float = 1.0
    tad_boundary_pos_weight: float = 6.0
    tad_boundary_threshold: float = 0.4
    tad_boundary_smooth: int = 3
    tad_domain_label: str = "tad_domains_10kb.bed"
    use_tad_detector: bool = False
    tad_detector_backbone: str = "resnet34"
    tad_detector_pretrained: bool = False
    lambda_tad_detector: float = 1.0
    tad_detection_min_bins: int = 4
    tad_detection_max_instances: int = 16
    tad_detection_score_thresh: float = 0.5
    tad_detection_iou_thresh: float = 0.3
    backbone_type: str = "unet"
    gcn_kernel_size: int = 9
    gcn_dilation: int = 1
    gcn_drop_path: float = 0.1
    gcn_stage_blocks: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    gcn_run_on_cpu: bool = False

    # Trident 专用超参
    trident_stripe_kernel: int = 17
    trident_tad_downstages: int = 2
    trident_use_axial: bool = True

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
    input_channels: int = 3


def build_default_config() -> TrainingConfig:
    return TrainingConfig()
