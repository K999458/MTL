import math
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def _init_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        copy_ch = min(in_channels, conv.in_channels)
        new_conv.weight[:, :copy_ch] = conv.weight[:, :copy_ch]
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def build_tad_detector(backbone_name: str = "resnet34",
                       pretrained: bool = False,
                       input_channels: int = 3,
                       num_classes: int = 2) -> MaskRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name,
        pretrained=pretrained,
        trainable_layers=5,
        norm_layer=nn.BatchNorm2d,
    )
    backbone.body.conv1 = _init_first_conv(backbone.body.conv1, input_channels)

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        box_detections_per_img=32,
        image_mean=[0.0] * input_channels,
        image_std=[1.0] * input_channels,
    )

    # 针对小尺寸 Hi-C patch 做简化：减少 Proposal 数量，便于稳定训练
    model.rpn.pre_nms_top_n_train = 256
    model.rpn.pre_nms_top_n_test = 128
    model.rpn.post_nms_top_n_train = 256
    model.rpn.post_nms_top_n_test = 128
    model.rpn.anchor_generator.sizes = ((32, 64, 128, 192),)
    aspect = (0.5, 1.0, 2.0)
    model.rpn.anchor_generator.aspect_ratios = (aspect,)

    model.roi_heads.score_thresh = 0.05
    model.roi_heads.detections_per_img = 32
    return model
