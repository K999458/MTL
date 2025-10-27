from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_c: int, out_c: int, kernel: int = 3, stride: int = 1, padding: Optional[int] = None) -> nn.Sequential:
    if padding is None:
        padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU()
    )


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = conv_bn_act(c, c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c)
        )
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


class AxialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out = x.view(b, c, h * w).transpose(1, 2)
        out = self.norm1(out)
        out,_ = self.row_attn(out, out, out)
        out = out.transpose(1, 2).view(b, c, h, w)

        out2 = x.permute(0, 1, 3, 2).contiguous().view(b, c, w * h).transpose(1, 2)
        out2 = self.norm2(out2)
        out2,_ = self.col_attn(out2, out2, out2)
        out2 = out2.transpose(1, 2).view(b, c, w, h).permute(0,1,3,2)
        return x + out + out2


class DiagonalBandAttention(nn.Module):
    def __init__(self, channels: int, band_width: int = 21):
        super().__init__()
        self.band_width = band_width
        self.conv = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.point = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        b, c, h, w = x.shape
        band = []
        half = self.band_width // 2
        for dk in range(-half, half+1):
            diag = torch.diagonal(x, offset=dk, dim1=2, dim2=3)  # (B,C,L)
            if dk < 0:
                diag = F.pad(diag, (0, -dk))
            elif dk > 0:
                diag = F.pad(diag, (dk, 0))
            band.append(diag)
        band = torch.stack(band, dim=2)  # (B,C,Band,L)
        band = band.mean(dim=2)  # 简化：平均带
        attn = self.conv(band)
        attn = self.point(attn)
        attn = attn.softmax(dim=-1)
        # broadcast back to spatial diag
        out = x.clone()
        diag_indices = torch.arange(min(h, w), device=x.device)
        for i in range(min(h, w)):
            out[:, :, i, i] = out[:, :, i, i] * attn[:, :, i]
        return out


class UNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 64, use_axial_attention: bool = True):
        super().__init__()
        self.use_axial_attention = use_axial_attention
        self.enc1 = nn.Sequential(conv_bn_act(in_channels, base_channels), ResBlock(base_channels))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), conv_bn_act(base_channels, base_channels*2), ResBlock(base_channels*2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), conv_bn_act(base_channels*2, base_channels*4), ResBlock(base_channels*4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), conv_bn_act(base_channels*4, base_channels*8), ResBlock(base_channels*8))

        if use_axial_attention:
            self.axial3 = AxialAttention(base_channels*4)
            self.axial4 = AxialAttention(base_channels*8)
            self.diag_attn = DiagonalBandAttention(base_channels*8)
        else:
            self.axial3 = nn.Identity()
            self.axial4 = nn.Identity()
            self.diag_attn = nn.Identity()

        self.up3 = nn.Sequential(nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2), ResBlock(base_channels*4))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2), ResBlock(base_channels*2))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2), ResBlock(base_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c3 = self.axial3(c3)
        c4 = self.enc4(c3)
        c4 = self.axial4(c4)
        c4 = self.diag_attn(c4)

        u3 = self.up3(c4) + c3
        u2 = self.up2(u3) + c2
        u1 = self.up1(u2) + c1
        return u1


class MMoELayer(nn.Module):
    def __init__(self, channels: int, num_experts: int = 4, tasks: Optional[List[str]] = None):
        super().__init__()
        self.num_experts = num_experts
        self.tasks = tasks or ['loop', 'stripe']
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
                nn.GELU()
            ) for _ in range(num_experts)
        ])
        self.gates = nn.ModuleDict({task: nn.Sequential(nn.Conv2d(channels, num_experts, 1), nn.Softmax(dim=1)) for task in self.tasks})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)  # (E,B,C,H,W)
        outputs: Dict[str, torch.Tensor] = {}
        for task, gate in self.gates.items():
            weights = gate(x)  # (B,E,H,W)
            weights = weights.permute(1,0,2,3).unsqueeze(-1)  # (E,B,H,W,1)
            out = (weights * expert_outputs.permute(0,1,3,4,2)).sum(dim=0)  # (B,H,W,C)
            out = out.permute(0,3,1,2)
            outputs[task] = out
        return outputs


class CrossStitchUnit(nn.Module):
    def __init__(self, tasks: List[str]):
        super().__init__()
        self.tasks = tasks
        init = torch.eye(len(tasks))
        self.alpha = nn.Parameter(init)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for i, task in enumerate(self.tasks):
            acc = 0
            for j, src in enumerate(self.tasks):
                acc = acc + self.alpha[i, j] * feats[src]
            out[task] = acc
        return out


class LoopHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.block = nn.Sequential(conv_bn_act(in_channels, hidden), conv_bn_act(hidden, hidden))
        self.heatmap = nn.Conv2d(hidden, 1, 1)
        self.offset = nn.Conv2d(hidden, 2, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.block(x)
        return {
            'heatmap': self.heatmap(h),
            'offset': self.offset(h)
        }


class StripeHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.block = nn.Sequential(conv_bn_act(in_channels, hidden), conv_bn_act(hidden, hidden))
        self.mask = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.block(x)
        return {'mask': self.mask(h)}


class TAD1DNet(nn.Module):
    def __init__(self, band_width: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(band_width, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, band: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(band)
        logits = self.head(h).squeeze(1)
        return {'boundary': logits}


class MultitaskHiCNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 64,
                 tad_band_width: int = 64, use_axial_attention: bool = True):
        super().__init__()
        self.backbone = UNetBackbone(in_channels=in_channels, base_channels=base_channels,
                                     use_axial_attention=use_axial_attention)
        channels = base_channels
        self.shared_tasks = ['loop', 'stripe']
        self.mmoe = MMoELayer(channels, num_experts=4, tasks=self.shared_tasks)
        self.cross_stitch = CrossStitchUnit(tasks=self.shared_tasks)
        self.loop_head = LoopHead(channels)
        self.stripe_head = StripeHead(channels)
        self.tad_net = TAD1DNet(tad_band_width, hidden=base_channels)

    def forward(self, x: torch.Tensor, task: str, band: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        if task == 'tad':
            assert band is not None, 'TAD task requires band input'
            return self.tad_net(band)

        feat = self.backbone(x)
        task_feats = self.mmoe(feat)
        mixed = self.cross_stitch(task_feats)
        if task == 'loop':
            return self.loop_head(mixed['loop'])
        if task == 'stripe':
            return self.stripe_head(mixed['stripe'])
        raise ValueError(f'Unknown task: {task}')
