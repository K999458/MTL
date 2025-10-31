import math
from typing import Dict, Optional, List, Tuple

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
        features = {
            'c1': c1.contiguous(),
            'c2': c2.contiguous(),
            'c3': c3.contiguous(),
            'c4': c4.contiguous(),
            'u3': u3.contiguous(),
            'u2': u2.contiguous(),
            'u1': u1.contiguous()
        }
        return u1.contiguous(), features


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


def _gcn_act_layer(act: str, inplace: bool = True, neg_slope: float = 0.2) -> nn.Module:
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if act == 'gelu':
        return nn.GELU()
    if act == 'prelu':
        return nn.PReLU()
    if act == 'hswish':
        return nn.Hardswish(inplace)
    raise NotImplementedError(f'Unknown activation: {act}')


def _gcn_norm_layer(norm: str, channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == 'batch':
        return nn.BatchNorm2d(channels, affine=True)
    if norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=False)
    raise NotImplementedError(f'Unknown norm: {norm}')


class _GCNBasicConv(nn.Sequential):
    def __init__(self, channels: List[int], act: str = 'gelu', norm: Optional[str] = 'batch',
                 bias: bool = True, drop: float = 0.0):
        modules: List[nn.Module] = []
        for i in range(1, len(channels)):
            modules.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=bias))
            if norm and norm.lower() != 'none':
                modules.append(_gcn_norm_layer(norm, channels[i]))
            if act and act.lower() != 'none':
                modules.append(_gcn_act_layer(act))
            if drop > 0:
                modules.append(nn.Dropout2d(drop))
        super().__init__(*modules)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def _gcn_batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


def _gcn_pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(x * x, dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def _gcn_part_pairwise_distance(x: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(x_part * x_part, dim=-1, keepdim=True)
        x_inner = -2 * torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(x * x, dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def _gcn_xy_pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(x * x, dim=-1, keepdim=True)
        y_square = torch.sum(y * y, dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def _gcn_dense_knn_matrix(x: torch.Tensor, k: int = 16) -> torch.Tensor:
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, _ = x.shape
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = _gcn_part_pairwise_distance(x.detach(), start_idx, end_idx)
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list.append(nn_idx_part)
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = _gcn_pairwise_distance(x.detach())
            _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def _gcn_xy_dense_knn_matrix(x: torch.Tensor, y: torch.Tensor, k: int = 16) -> torch.Tensor:
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        dist = _gcn_xy_pairwise_distance(x.detach(), y.detach())
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, x.shape[1], device=x.device).repeat(x.shape[0], k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class _GCNDenseDilated(nn.Module):
    def __init__(self, k: int = 9, dilation: int = 1, stochastic: bool = False, epsilon: float = 0.0):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        if self.stochastic and self.training and torch.rand(1) < self.epsilon:
            num = self.k * self.dilation
            randnum = torch.randperm(num, device=edge_index.device)[:self.k]
            return edge_index[:, :, :, randnum]
        return edge_index[:, :, :, ::self.dilation]


class _GCNDenseDilatedKnnGraph(nn.Module):
    def __init__(self, k: int = 9, dilation: int = 1, stochastic: bool = False, epsilon: float = 0.0):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self._dilated = _GCNDenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = _gcn_xy_dense_knn_matrix(x, y, k=self.k * self.dilation)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = _gcn_dense_knn_matrix(x, k=self.k * self.dilation)
        return self._dilated(edge_index)


class _GCNMRConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = 'gelu', norm: Optional[str] = 'batch',
                 bias: bool = True):
        super().__init__()
        self.nn = _GCNBasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_i = _gcn_batched_index_select(x, edge_index[1])
        x_j = _gcn_batched_index_select(y if y is not None else x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        cat = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(cat)


class _GCNEdgeConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = 'gelu', norm: Optional[str] = 'batch',
                 bias: bool = True):
        super().__init__()
        self.nn = _GCNBasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_i = _gcn_batched_index_select(x, edge_index[1])
        x_j = _gcn_batched_index_select(y if y is not None else x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class _GCNGraphConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv: str = 'edge', act: str = 'gelu',
                 norm: Optional[str] = 'batch', bias: bool = True):
        super().__init__()
        if conv == 'edge':
            self.gconv = _GCNEdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = _GCNMRConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError(f'Unsupported conv type: {conv}')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gconv(x, edge_index, y)


class _GCNDyGraphConv2d(_GCNGraphConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, dilation: int = 1,
                 conv: str = 'edge', act: str = 'gelu', norm: Optional[str] = 'batch', bias: bool = True,
                 stochastic: bool = False, epsilon: float = 0.0, r: int = 1):
        super().__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = _GCNDenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x: torch.Tensor, relative_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(b, c, -1, 1).contiguous()
        x_flat = x.reshape(b, c, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x_flat, y)
        out = super().forward(x_flat, edge_index, y)
        return out.reshape(b, -1, h, w).contiguous()


class _GCNGrapher(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 9, dilation: int = 1, conv: str = 'edge',
                 act: str = 'gelu', norm: Optional[str] = 'batch', bias: bool = True,
                 stochastic: bool = False, epsilon: float = 0.0, r: int = 1,
                 n: int = 196, drop_path: float = 0.0):
        super().__init__()
        self.channels = in_channels
        self.r = r
        self.n = n
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = _GCNDyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                                            act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return self.drop_path(x) + residual


class _GCNFeedForward(nn.Module):
    def __init__(self, channels: int, hidden_mul: int = 4, act: str = 'gelu', drop_path: float = 0.0):
        super().__init__()
        hidden = channels * hidden_mul
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = _gcn_act_layer(act)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return self.drop_path(x) + residual


class _GCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, dilation: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.graph = _GCNGrapher(channels, kernel_size=kernel_size, dilation=dilation, drop_path=drop_path)
        self.ffn = _GCNFeedForward(channels, drop_path=drop_path)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.graph(x)
        x = self.ffn(x)
        return self.act(x)


class GCNBackbone(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 64,
                 kernel_size: int = 9, dilation: int = 1,
                 drop_path: float = 0.0, stage_blocks: Optional[List[int]] = None):
        super().__init__()
        if stage_blocks is None:
            stage_blocks = [1, 1, 1, 1]
        assert len(stage_blocks) == 4, "stage_blocks must specify four stages"
        self.stem = nn.Sequential(
            conv_bn_act(in_channels, base_channels),
            conv_bn_act(base_channels, base_channels)
        )
        self.stage1 = nn.Sequential(*[_GCNBlock(base_channels, kernel_size, dilation, drop_path) for _ in range(stage_blocks[0])])
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU()
        )
        self.stage2 = nn.Sequential(*[_GCNBlock(base_channels * 2, kernel_size, dilation, drop_path) for _ in range(stage_blocks[1])])
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU()
        )
        self.stage3 = nn.Sequential(*[_GCNBlock(base_channels * 4, kernel_size, dilation, drop_path) for _ in range(stage_blocks[2])])
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU()
        )
        self.stage4 = nn.Sequential(*[_GCNBlock(base_channels * 8, kernel_size, dilation, drop_path) for _ in range(stage_blocks[3])])

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU()
        )
        self.stage_up3 = _GCNBlock(base_channels * 4, kernel_size, dilation, drop_path)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU()
        )
        self.stage_up2 = _GCNBlock(base_channels * 2, kernel_size, dilation, drop_path)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )
        self.stage_up1 = _GCNBlock(base_channels, kernel_size, dilation, drop_path)

    @staticmethod
    def _ensure_finite(t: torch.Tensor, name: str):
        if not torch.isfinite(t).all():
            nan = torch.isnan(t).any().item()
            inf = torch.isinf(t).any().item()
            raise RuntimeError(f"GCNBackbone produced invalid values after {name}: nan={nan} inf={inf}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stem = self.stem(x)
        self._ensure_finite(stem, 'stem')
        c1 = self.stage1(stem)
        self._ensure_finite(c1, 'stage1')

        d1 = self.down1(c1)
        c2 = self.stage2(d1)
        self._ensure_finite(c2, 'stage2')

        d2 = self.down2(c2)
        c3 = self.stage3(d2)
        self._ensure_finite(c3, 'stage3')

        d3 = self.down3(c3)
        c4 = self.stage4(d3)
        self._ensure_finite(c4, 'stage4')

        u3 = self.up3(c4) + c3
        u3 = self.stage_up3(u3)
        self._ensure_finite(u3, 'stage_up3')

        u2 = self.up2(u3) + c2
        u2 = self.stage_up2(u2)
        self._ensure_finite(u2, 'stage_up2')

        u1 = self.up1(u2) + c1
        u1 = self.stage_up1(u1)
        self._ensure_finite(u1, 'stage_up1')

        features = {
            'c1': c1,
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'u3': u3,
            'u2': u2,
            'u1': u1
        }
        return u1, features


class GCNLightBackbone(nn.Module):
    """轻量级单尺度 GCN 主干：避免下采样与复杂上采样，仅保留少量图卷积块。
    可选在 CPU 上运行以绕开 GPU 后端问题。
    返回单尺度特征 u1 以及最小 multi 字典（仅含 u1）。
    """
    def __init__(self, in_channels: int = 3, channels: int = 64,
                 kernel_size: int = 9, dilation: int = 1,
                 drop_path: float = 0.0, use_cpu: bool = False):
        super().__init__()
        self.use_cpu = use_cpu
        self.cpu_device = torch.device('cpu') if use_cpu else None
        self.stem = nn.Sequential(
            conv_bn_act(in_channels, channels),
            conv_bn_act(channels, channels)
        )
        self.block1 = _GCNBlock(channels, kernel_size, dilation, drop_path)
        self.block2 = _GCNBlock(channels, kernel_size, dilation, drop_path)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        orig_device = x.device
        if self.use_cpu:
            x = x.to(self.cpu_device, non_blocking=False)
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        if self.use_cpu:
            out = out.to(orig_device, non_blocking=False)
        features = {
            'u1': out
        }
        return out, features


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
        x = x.contiguous()
        with torch.backends.cudnn.flags(enabled=False):
            expert_outputs_list = [expert(x).contiguous() for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs_list, dim=0).contiguous()  # (E,B,C,H,W)
        outputs: Dict[str, torch.Tensor] = {}
        for task, gate in self.gates.items():
            weights = gate(x).contiguous()  # (B,E,H,W)
            weights = weights.permute(1,0,2,3).contiguous().unsqueeze(-1)  # (E,B,H,W,1)
            out = (weights * expert_outputs.permute(0,1,3,4,2).contiguous()).sum(dim=0)  # (B,H,W,C)
            out = out.permute(0,3,1,2).contiguous()
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


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduction = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(self.pool(x))
        return x * weight


class DiagonalRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=max(1, channels // 32)),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.blur = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        nn.init.dirac_(self.blur.weight)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        length = min(h, w)
        diag = torch.diagonal(feat, dim1=2, dim2=3)  # (B,C,L)
        diag_refined = self.conv(diag)
        diag_map = torch.diag_embed(diag_refined, dim1=-2, dim2=-1)  # (B,C,H,W) but zeros elsewhere
        if diag_map.shape[-2] != h or diag_map.shape[-1] != w:
            diag_map = F.interpolate(diag_map, size=(h, w), mode='bilinear', align_corners=False)
        diag_map = self.blur(diag_map)
        return diag_map


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


class TAD2DHead(nn.Module):
    def __init__(self, base_channels: int, hidden: int = 128):
        super().__init__()
        self.fpn_dim = max(128, base_channels * 4)

        self.lateral_c4 = nn.Conv2d(base_channels * 8, self.fpn_dim, kernel_size=1)
        self.lateral_u3 = nn.Conv2d(base_channels * 4, self.fpn_dim, kernel_size=1)
        self.lateral_u2 = nn.Conv2d(base_channels * 2, self.fpn_dim, kernel_size=1)
        self.lateral_u1 = nn.Conv2d(base_channels, self.fpn_dim, kernel_size=1)

        self.refine_c4 = conv_bn_act(self.fpn_dim, self.fpn_dim)
        self.refine_u3 = conv_bn_act(self.fpn_dim, self.fpn_dim)
        self.refine_u2 = conv_bn_act(self.fpn_dim, self.fpn_dim)
        self.refine_u1 = conv_bn_act(self.fpn_dim, self.fpn_dim)

        self.loop_proj = nn.Conv2d(base_channels, self.fpn_dim, kernel_size=1)
        self.stripe_proj = nn.Conv2d(base_channels, self.fpn_dim, kernel_size=1)
        self.cross_fuse = conv_bn_act(self.fpn_dim * 2, self.fpn_dim)

        self.merge = nn.Sequential(
            conv_bn_act(self.fpn_dim * 4, self.fpn_dim * 2),
            SqueezeExcite(self.fpn_dim * 2),
            conv_bn_act(self.fpn_dim * 2, self.fpn_dim)
        )

        self.context_se = SqueezeExcite(self.fpn_dim)
        self.diagonal = DiagonalRefine(self.fpn_dim)

        # Boundary / Region dual-branch inspired by detection head
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(self.fpn_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.region_branch = nn.Sequential(
            nn.Conv2d(self.fpn_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.se_fusion = SqueezeExcite(128)

        self.post = nn.Sequential(
            conv_bn_act(128, 128),
            ResBlock(128)
        )
        self.domain_map_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.boundary_map_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, feat: torch.Tensor, multi: Dict[str, torch.Tensor],
                loop_feat: Optional[torch.Tensor] = None,
                stripe_feat: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        p4 = self.refine_c4(self.lateral_c4(multi['c4']))
        p3 = self.refine_u3(self.lateral_u3(multi['u3']) + F.interpolate(p4, size=multi['u3'].shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.refine_u2(self.lateral_u2(multi['u2']) + F.interpolate(p3, size=multi['u2'].shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.refine_u1(self.lateral_u1(multi['u1']))

        size = p1.shape[-2:]
        up2 = F.interpolate(p2, size=size, mode='bilinear', align_corners=False)
        up3 = F.interpolate(p3, size=size, mode='bilinear', align_corners=False)
        up4 = F.interpolate(p4, size=size, mode='bilinear', align_corners=False)

        pyramid = torch.cat([p1, up2, up3, up4], dim=1)
        fused = self.merge(pyramid)

        if loop_feat is not None and stripe_feat is not None:
            loop_ctx = self.loop_proj(loop_feat)
            stripe_ctx = self.stripe_proj(stripe_feat)
            cross = self.cross_fuse(torch.cat([loop_ctx, stripe_ctx], dim=1))
            fused = fused + cross

        fused = self.context_se(fused)
        fused = fused + self.diagonal(fused)

        boundary_feat = self.boundary_branch(fused)
        region_feat = self.region_branch(fused)
        combined = self.fusion_conv(torch.cat([boundary_feat, region_feat], dim=1))
        combined = self.se_fusion(combined)

        refined = self.post(combined)

        domain_map = self.domain_map_head(refined)
        boundary_map = self.boundary_map_head(boundary_feat)
        return {
            'domain_map': domain_map,
            'boundary_map': boundary_map
        }


class TADSimpleHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            conv_bn_act(in_channels, hidden),
            ResBlock(hidden)
        )
        self.diag = DiagonalRefine(hidden)
        self.domain_map = nn.Sequential(
            nn.Conv2d(hidden, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.boundary_map = nn.Sequential(
            nn.Conv2d(hidden, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        h = h + self.diag(h)
        return {
            'domain_map': self.domain_map(h),
            'boundary_map': self.boundary_map(h)
        }


# -------- Trident 编码器与头：面向 Hi-C 三任务的定制路径 --------
class DirectionalDWConv(nn.Module):
    def __init__(self, channels: int, k: int = 17):
        super().__init__()
        pad = k // 2
        self.v = nn.Conv2d(channels, channels, kernel_size=(k,1), padding=(pad,0), groups=channels, bias=False)
        self.h = nn.Conv2d(channels, channels, kernel_size=(1,k), padding=(0,pad), groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.v(x) + self.h(x)
        out = self.pw(out)
        return self.act(self.bn(out))


class TridentEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64, stripe_k: int = 17, use_axial: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_act(in_channels, base_channels),
            conv_bn_act(base_channels, base_channels)
        )
        # Loop：高分辨率，小感受野
        self.loop_path = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        # Stripe：方向分离卷积 + 轻下采样再上采样
        self.stripe_down = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False)
        self.stripe_block = DirectionalDWConv(base_channels, k=stripe_k)
        self.stripe_up = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2, bias=False)
        # TAD：更大感受野 + 对角增强
        self.tad_down1 = nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1, bias=False)
        self.tad_down2 = nn.Conv2d(base_channels*2, base_channels*2, 3, stride=2, padding=1, bias=False)
        self.tad_core = nn.Sequential(
            AxialAttention(base_channels*2) if use_axial else nn.Identity(),
            ResBlock(base_channels*2)
        )
        self.tad_up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2, bias=False)
        self.tad_up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2, bias=False)
        self.tad_diag = DiagonalRefine(base_channels)
        # 融合共享
        self.fuse = nn.Sequential(
            conv_bn_act(base_channels*3, base_channels*2),
            SqueezeExcite(base_channels*2),
            conv_bn_act(base_channels*2, base_channels)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s = self.stem(x)
        loop_feat = self.loop_path(s)
        # Stripe 分支
        sd = self.stripe_down(s)
        sb = self.stripe_block(sd)
        stripe_feat = self.stripe_up(sb)
        if stripe_feat.shape[-2:] != s.shape[-2:]:
            stripe_feat = F.interpolate(stripe_feat, size=s.shape[-2:], mode='bilinear', align_corners=False)
        # TAD 分支
        t1 = self.tad_down1(s)
        t2 = self.tad_down2(t1)
        tc = self.tad_core(t2)
        u2 = self.tad_up2(tc)
        u1 = self.tad_up1(u2)
        tad_feat = self.tad_diag(u1)
        if tad_feat.shape[-2:] != s.shape[-2:]:
            tad_feat = F.interpolate(tad_feat, size=s.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([loop_feat, stripe_feat, tad_feat], dim=1))
        return {'loop': loop_feat, 'stripe': stripe_feat, 'tad': tad_feat, 'shared': fused}


class TADTridentHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.pre = nn.Sequential(conv_bn_act(in_channels, in_channels), ResBlock(in_channels))
        self.domain = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))
        self.boundary = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.pre(x)
        return {'domain_map': self.domain(h), 'boundary_map': self.boundary(h)}


class MultitaskHiCNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 64,
                 tad_band_width: int = 64, use_axial_attention: bool = True,
                 backbone_type: str = 'unet', gcn_kernel_size: int = 9, gcn_dilation: int = 1,
                 gcn_drop_path: float = 0.1, gcn_stage_blocks: Optional[List[int]] = None,
                 gcn_run_on_cpu: bool = False):
        super().__init__()
        backbone_key = (backbone_type or 'unet').lower()
        if backbone_key == 'gcn':
            self.backbone = GCNBackbone(in_channels=in_channels, base_channels=base_channels,
                                        kernel_size=gcn_kernel_size, dilation=gcn_dilation,
                                        drop_path=gcn_drop_path, stage_blocks=gcn_stage_blocks)
        elif backbone_key == 'gcn_simple':
            self.backbone = GCNLightBackbone(in_channels=in_channels, channels=base_channels,
                                             kernel_size=gcn_kernel_size, dilation=gcn_dilation,
                                             drop_path=gcn_drop_path, use_cpu=gcn_run_on_cpu)
        elif backbone_key == 'trident':
            self.encoder = TridentEncoder(in_channels=in_channels, base_channels=base_channels,
                                          stripe_k=gcn_kernel_size, use_axial=use_axial_attention)
            self.loop_head = LoopHead(base_channels)
            self.stripe_head = StripeHead(base_channels)
            self.tad_trident_head = TADTridentHead(base_channels)
            self._use_trident = True
        else:
            self.backbone = UNetBackbone(in_channels=in_channels, base_channels=base_channels,
                                         use_axial_attention=use_axial_attention)
        channels = base_channels
        if getattr(self, '_use_trident', False) is not True:
            self.shared_tasks = ['loop', 'stripe']
            self.mmoe = MMoELayer(channels, num_experts=4, tasks=self.shared_tasks)
            self.cross_stitch = CrossStitchUnit(tasks=self.shared_tasks)
            self.loop_head = LoopHead(channels)
            self.stripe_head = StripeHead(channels)
            self.tad_head = TAD2DHead(base_channels, hidden=base_channels) if backbone_key != 'gcn_simple' else TADSimpleHead(channels)

    def forward(self, x: torch.Tensor, task: str) -> Dict[str, torch.Tensor]:
        if getattr(self, '_use_trident', False) is True:
            feats = self.encoder(x)
            if task == 'loop':
                return self.loop_head(feats['loop'] + feats['shared'])
            if task == 'stripe':
                return self.stripe_head(feats['stripe'] + feats['shared'])
            if task == 'tad':
                return self.tad_trident_head(feats['tad'] + feats['shared'])
            raise ValueError(f'Unknown task: {task}')
        feat, multi = self.backbone(x)
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            raise RuntimeError(f"Backbone produced invalid features for task {task}: "
                               f"nan={torch.isnan(feat).any().item()} inf={torch.isinf(feat).any().item()}")
        if task == 'tad':
            task_feats = self.mmoe(feat)
            mixed = self.cross_stitch(task_feats)
            return self.tad_head(feat, multi, loop_feat=mixed['loop'], stripe_feat=mixed['stripe'])
        task_feats = self.mmoe(feat)
        mixed = self.cross_stitch(task_feats)
        if task == 'loop':
            return self.loop_head(mixed['loop'])
        if task == 'stripe':
            return self.stripe_head(mixed['stripe'])
        raise ValueError(f'Unknown task: {task}')
