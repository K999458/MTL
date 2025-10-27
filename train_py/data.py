import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_loop_labels(path: str) -> Dict[str, List[Dict[str, Any]]]:
    items = load_jsonl(path)
    per_chrom: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        chrom = str(it['chrom'])
        per_chrom.setdefault(chrom, []).append(it)
    return per_chrom


def load_stripe_labels(path: str) -> Dict[str, List[Dict[str, Any]]]:
    items = load_jsonl(path)
    per_chrom: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        chrom = str(it['chrom'])
        per_chrom.setdefault(chrom, []).append(it)
    return per_chrom


def load_tad_boundaries(path: str, binsize: int) -> Dict[str, List[int]]:
    df = np.loadtxt(path, dtype=str)
    if df.ndim == 1:
        df = df[None, :]
    per_chrom: Dict[str, List[int]] = {}
    for row in df:
        chrom, start, _ = row
        try:
            bp = int(float(start))
        except Exception:
            continue
        b = bp // binsize
        per_chrom.setdefault(str(chrom), []).append(b)
    for chrom in per_chrom:
        per_chrom[chrom] = sorted(set(per_chrom[chrom]))
    return per_chrom


def distance_channel(size: int) -> torch.Tensor:
    idx = torch.arange(size, dtype=torch.float32)
    dist = torch.abs(idx[None, :] - idx[:, None])
    dist = torch.log1p(dist)
    dist /= dist.max().clamp(min=1.0)
    return dist


def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> torch.Tensor:
    y = torch.arange(h, dtype=torch.float32)
    x = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    heat = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2.0 * sigma**2))
    return heat


def score_to_sigma(soft: Optional[float], n_tools: Optional[int], base: float=1.6) -> float:
    sigma = base
    if soft is not None:
        sigma *= (1.6 - 0.6 * max(0.0, min(1.0, soft)))
    if n_tools is not None and n_tools > 0:
        sigma *= (1.2 / max(1.0, min(4.0, float(n_tools))))
    return float(np.clip(sigma, 0.8, 2.5))


def ensure_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(data.copy()) if isinstance(data, np.ndarray) else torch.tensor(data)


def center_crop_pad2d(x: torch.Tensor, target: int) -> torch.Tensor:
    """Return a center crop of size (target,target). If x is smaller, pad with zeros."""
    H, W = x.shape[-2], x.shape[-1]
    # crop
    sh = max(0, (H - target) // 2)
    sw = max(0, (W - target) // 2)
    eh = min(H, sh + target)
    ew = min(W, sw + target)
    cropped = x[..., sh:eh, sw:ew]
    # pad if needed
    ch, cw = cropped.shape[-2], cropped.shape[-1]
    if ch == target and cw == target:
        return cropped
    out = torch.zeros((*x.shape[:-2], target, target), dtype=x.dtype, device=x.device)
    oh = (target - ch) // 2
    ow = (target - cw) // 2
    out[..., oh:oh+ch, ow:ow+cw] = cropped
    return out


@dataclass
class DatasetPaths:
    cache_root: str
    label_root: str
    manifest_root: str
    use_hdf5: bool = True


class H5Array:
    def __init__(self, path: str, dataset_name: str):
        self.path = path
        self.dataset_name = dataset_name
        self._h5: Optional[h5py.File] = None
        self._ds: Optional[h5py.Dataset] = None

    def _ensure(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, 'r')
            self._ds = self._h5[self.dataset_name]

    def __getitem__(self, idx):
        self._ensure()
        assert self._ds is not None
        return self._ds[idx]

    def attrs(self):
        self._ensure()
        assert self._h5 is not None
        return self._h5.attrs

    def close(self):
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._ds = None


class LoopDataset(Dataset):
    def __init__(self, paths: DatasetPaths, manifest_name: str, label_name: str,
                 center: int, patch: int, stride: int, binsize: int=5000,
                 include_tad_prior: bool=True):
        self.center = center
        self.patch = patch
        self.stride = stride
        self.binsize = binsize
        self.include_tad_prior = include_tad_prior

        manifest_path = os.path.join(paths.manifest_root, manifest_name)
        label_path = os.path.join(paths.label_root, label_name)
        tad_path = os.path.join(paths.label_root, 'tad_boundaries_10kb.bed')

        with open(manifest_path, 'r') as f:
            self.entries = [json.loads(line) for line in f if line.strip()]

        loop_labels = load_loop_labels(label_path)
        self.loop_labels = loop_labels

        self.tad_boundaries = load_tad_boundaries(tad_path, binsize=10000) if include_tad_prior else {}

        # storage
        pack_path = os.path.join(paths.cache_root, 'loop_5kb.h5')
        use_h5_env = os.environ.get('AAA_MIL_USE_H5', '1').lower() in ('1','true','yes','y','on')
        if paths.use_hdf5 and use_h5_env and os.path.exists(pack_path):
            self.storage_h5 = {
                'X': H5Array(pack_path, 'X'),
                'chrom': H5Array(pack_path, 'chrom'),
                'x0': H5Array(pack_path, 'x0'),
                'patch': H5Array(pack_path, 'patch')
            }
            self.use_h5 = True
        else:
            self.use_h5 = False
            self.cache_dir = os.path.join(paths.cache_root, 'loop_5kb')

        self.distance_cache = distance_channel(center)

    def __len__(self):
        return len(self.entries)

    def _load_patch(self, idx: int) -> Tuple[np.ndarray, str, int]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        x0 = int(entry['x0'])
        patch = int(entry['patch'])
        if self.use_h5:
            X = self.storage_h5['X'][idx]
        else:
            path = os.path.join(self.cache_dir, f"loop_{chrom}_x{x0}_p{patch}.npz")
            with np.load(path, allow_pickle=True) as npz:
                X = npz['X']
        return X, chrom, x0

    def _tad_corner_prior(self, chrom: str, c0_bin: int, center_bins: int) -> torch.Tensor:
        # Build corner priors from TAD boundaries (10kb). Map to 5kb bins by scaling.
        corners = torch.zeros((center_bins, center_bins), dtype=torch.float32)
        tad_bounds = self.tad_boundaries.get(chrom, [])
        if not tad_bounds:
            return corners
        # map to 5kb bins
        scale = 10000 // self.binsize
        local_bounds = []
        for b in tad_bounds:
            b5 = b * scale
            if c0_bin <= b5 < c0_bin + center_bins:
                local_bounds.append(b5 - c0_bin)
        if len(local_bounds) < 2:
            return corners
        local_bounds = sorted(set(local_bounds))
        for i in range(len(local_bounds)-1):
            for j in range(i+1, len(local_bounds)):
                cx = local_bounds[i]
                cy = local_bounds[j]
                if 0 <= cx < center_bins and 0 <= cy < center_bins:
                    corners[min(cx, cy), max(cx, cy)] = 1.0
                    corners[max(cx, cy), min(cx, cy)] = 1.0
        return corners

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        x0 = int(entry['x0'])
        patch = int(entry['patch'])
        center = int(entry['center'])
        X, chrom, x0 = self._load_patch(idx)
        X = torch.from_numpy(X).float()

        # central crop for supervision
        c0_bin = x0 + (patch - center) // 2
        cx0_bp = c0_bin * self.binsize
        cx1_bp = cx0_bp + center * self.binsize

        heat = torch.zeros((center, center), dtype=torch.float32)
        offset = torch.zeros((2, center, center), dtype=torch.float32)
        valid = torch.zeros((center, center), dtype=torch.float32)

        loops = self.loop_labels.get(chrom, [])
        for lp in loops:
            p1 = lp['p1']; p2 = lp['p2']
            if not (cx0_bp <= p1 < cx1_bp and cx0_bp <= p2 < cx1_bp):
                continue
            ix = (p1 // self.binsize) - c0_bin
            iy = (p2 // self.binsize) - c0_bin
            if not (0 <= ix < center and 0 <= iy < center):
                continue
            # 按 labelfix.md 要求：Loop 在 5kb 网格上应为单个像素的点标签（对称标注）
            heat[iy, ix] = 1.0
            offset[:, iy, ix] = 0.0  # loops 定义在 bin 中心
            valid[iy, ix] = 1.0

            if 0 <= iy < center and 0 <= ix < center:
                heat[ix, iy] = 1.0
                offset[:, ix, iy] = 0.0
                valid[ix, iy] = 1.0

        dist = self.distance_cache
        # robust center crop independent of stored patch size
        central_patch = center_crop_pad2d(X, center)
        input_tensor = torch.stack([central_patch, dist], dim=0)

        item = {
            'task': 'loop',
            'inputs': input_tensor,
            'target': {
                'heatmap': heat.unsqueeze(0),
                'offset': offset,
                'valid': valid
            },
            'metadata': {
                'chrom': chrom,
                'bin_start': int(c0_bin)
            }
        }
        if self.include_tad_prior:
            prior = self._tad_corner_prior(chrom, c0_bin, center)
            item['target']['corner_prior'] = prior.unsqueeze(0)
        return item


class StripeDataset(Dataset):
    def __init__(self, paths: DatasetPaths, manifest_name: str, label_name: str,
                 center: int, patch: int, stride: int, binsize: int=10000,
                 add_tad_prior: bool=True):
        self.center = center
        self.patch = patch
        self.stride = stride
        self.binsize = binsize
        self.add_tad_prior = add_tad_prior

        manifest_path = os.path.join(paths.manifest_root, manifest_name)
        label_path = os.path.join(paths.label_root, label_name)
        tad_path = os.path.join(paths.label_root, 'tad_boundaries_10kb.bed')

        with open(manifest_path, 'r') as f:
            self.entries = [json.loads(line) for line in f if line.strip()]

        self.stripe_labels = load_stripe_labels(label_path)
        self.tad_boundaries = load_tad_boundaries(tad_path, binsize=10000) if add_tad_prior else {}

        pack_path = os.path.join(paths.cache_root, 'stripe_10kb.h5')
        use_h5_env = os.environ.get('AAA_MIL_USE_H5', '1').lower() in ('1','true','yes','y','on')
        if paths.use_hdf5 and use_h5_env and os.path.exists(pack_path):
            self.use_h5 = True
            self.storage_h5 = {
                'X': H5Array(pack_path, 'X'),
                'chrom': H5Array(pack_path, 'chrom'),
                'x0': H5Array(pack_path, 'x0'),
                'patch': H5Array(pack_path, 'patch')
            }
        else:
            self.use_h5 = False
            self.cache_dir = os.path.join(paths.cache_root, 'stripe_10kb')

        self.distance_cache = distance_channel(center)

    def __len__(self):
        return len(self.entries)

    def _load_patch(self, idx: int) -> Tuple[np.ndarray, str, int]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        x0 = int(entry['x0'])
        patch = int(entry['patch'])
        if self.use_h5:
            X = self.storage_h5['X'][idx]
        else:
            path = os.path.join(self.cache_dir, f"stripe_{chrom}_x{x0}_p{patch}.npz")
            with np.load(path, allow_pickle=True) as npz:
                X = npz['X']
        return X, chrom, x0

    def _tad_band_prior(self, chrom: str, c0_bin: int, center_bins: int) -> torch.Tensor:
        boundaries = self.tad_boundaries.get(chrom, [])
        prior = torch.zeros((center_bins, center_bins), dtype=torch.float32)
        if not boundaries:
            return prior
        local = [b - c0_bin for b in boundaries if c0_bin <= b < c0_bin + center_bins]
        for b in local:
            if 0 <= b < center_bins:
                prior[:, b] = 1.0
                prior[b, :] = 1.0
        return prior

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        x0 = int(entry['x0'])
        patch = int(entry['patch'])
        center = int(entry['center'])

        X, chrom, x0 = self._load_patch(idx)
        X = torch.from_numpy(X).float()
        c0_bin = x0 + (patch - center)//2
        cx0_bp = c0_bin * self.binsize
        cx1_bp = cx0_bp + center * self.binsize

        mask = torch.zeros((center, center), dtype=torch.float32)
        stripes = self.stripe_labels.get(chrom, [])
        for st in stripes:
            x1 = st['x1']; x2 = st['x2']; y1 = st['y1']; y2 = st['y2']
            if x2 <= cx0_bp or x1 >= cx1_bp or y2 <= cx0_bp or y1 >= cx1_bp:
                continue
            bx1 = max(0, int((x1 - cx0_bp)//self.binsize))
            bx2 = min(center, int(math.ceil((x2 - cx0_bp)/self.binsize)))
            by1 = max(0, int((y1 - cx0_bp)//self.binsize))
            by2 = min(center, int(math.ceil((y2 - cx0_bp)/self.binsize)))
            mask[by1:by2, bx1:bx2] = 1.0

        central_patch = center_crop_pad2d(X, center)
        inputs = torch.stack([central_patch, self.distance_cache], dim=0)

        item = {
            'task': 'stripe',
            'inputs': inputs,
            'target': {
                'mask': mask.unsqueeze(0)
            },
            'metadata': {
                'chrom': chrom,
                'bin_start': int(c0_bin)
            }
        }
        if self.add_tad_prior:
            item['target']['boundary_prior'] = self._tad_band_prior(chrom, c0_bin, center).unsqueeze(0)
        return item


class TADDataset(Dataset):
    def __init__(self, paths: DatasetPaths, manifest_name: str, label_name: str,
                 band_width: int, ignore_bins: int=32, binsize: int=10000):
        self.band_width = band_width
        self.ignore_bins = ignore_bins
        self.binsize = binsize

        manifest_path = os.path.join(paths.manifest_root, manifest_name)
        with open(manifest_path, 'r') as f:
            self.entries = [json.loads(line) for line in f if line.strip()]

        tad_path = os.path.join(paths.label_root, label_name)
        self.boundaries = load_tad_boundaries(tad_path, binsize=binsize)

        pack_path = os.path.join(paths.cache_root, 'tad_10kb_1d.h5')
        use_h5_env = os.environ.get('AAA_MIL_USE_H5', '1').lower() in ('1','true','yes','y','on')
        if paths.use_hdf5 and use_h5_env and os.path.exists(pack_path):
            self.use_h5 = True
            self.storage_h5 = {
                'band': H5Array(pack_path, 'band'),
                'chrom': H5Array(pack_path, 'chrom'),
                'start_bin': H5Array(pack_path, 'start_bin')
            }
            self.binsize_attr = int(self.storage_h5['band'].attrs()['binsize'])
        else:
            self.use_h5 = False
            self.cache_dir = os.path.join(paths.cache_root, 'tad_10kb_1d')

    def __len__(self):
        return len(self.entries)

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        start_bin = int(entry['start_bin'])
        L = int(entry.get('length', 1024))
        if self.use_h5:
            band = self.storage_h5['band'][idx]
            chrom = self.storage_h5['chrom'][idx].astype(str)
            start_bin = int(self.storage_h5['start_bin'][idx])
            # load entire patch? store? to keep compatibility, we require dataset to fetch from cooler? For now band only.
            X = None
        else:
            path = os.path.join(self.cache_dir, f"tad1d_{chrom}_s{start_bin}_L{L}_B{self.band_width}.npz")
            with np.load(path, allow_pickle=True) as npz:
                band = npz['band']
                X = npz['X'] if 'X' in npz.files else None
        return band, X, chrom, start_bin, L

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        start_bin = int(entry['start_bin'])
        L = int(entry.get('length', 1024))
        band, X_patch, chrom_loaded, start_bin_loaded, L_loaded = self._load(idx)
        if chrom_loaded is not None:
            chrom = str(chrom_loaded)
        start_bin = start_bin_loaded
        L = L_loaded
        band = torch.from_numpy(band).float()
        if X_patch is not None:
            X_tensor = torch.from_numpy(X_patch).float()
        else:
            diag_vec = band.mean(dim=0)
            X_tensor = torch.zeros((L, L), dtype=torch.float32)
            idx = torch.arange(L)
            X_tensor[idx, idx] = diag_vec
        # unify size to (L,L)
        if X_tensor.shape[-2] != L or X_tensor.shape[-1] != L:
            H, W = X_tensor.shape[-2], X_tensor.shape[-1]
            # crop or pad bottom-right
            X_fix = torch.zeros((L, L), dtype=torch.float32)
            h = min(L, H); w = min(L, W)
            X_fix[:h, :w] = X_tensor[:h, :w]
            X_tensor = X_fix

        boundary_bins = self.boundaries.get(chrom, [])
        target = torch.zeros((L,), dtype=torch.float32)
        for b in boundary_bins:
            j = b - start_bin
            if 0 <= j < L:
                target[j] = 1.0
        if self.ignore_bins > 0:
            mask = torch.ones((L,), dtype=torch.float32)
            mask[:self.ignore_bins] = 0.0
            mask[-self.ignore_bins:] = 0.0
        else:
            mask = torch.ones((L,), dtype=torch.float32)

        # ensure two channels (image + distance)
        dist = distance_channel(L)
        inputs = torch.stack([X_tensor, dist], dim=0)

        item = {
            'task': 'tad',
            'inputs': inputs,
            'inputs_band': band,
            'target': {
                'boundary': target,
                'ignore': mask
            },
            'metadata': {
                'chrom': chrom,
                'start_bin': int(start_bin)
            }
        }
        return item


def make_dataloaders(paths: DatasetPaths,
                     loop_center: int, loop_patch: int, loop_stride: int,
                     stripe_center: int, stripe_patch: int, stripe_stride: int,
                     tad_length: int, tad_stride: int, tad_band_width: int, tad_ignore: int,
                     batch_sizes: Dict[str, int], num_workers: int, pin_memory: bool) -> Dict[str, DataLoader]:
    loop_ds = LoopDataset(paths, 'loop_5kb.jsonl', 'loop_5kb.jsonl', center=loop_center, patch=loop_patch, stride=loop_stride)
    stripe_ds = StripeDataset(paths, 'stripe_10kb.jsonl', 'stripe_10kb.jsonl', center=stripe_center, patch=stripe_patch, stride=stripe_stride)
    tad_ds = TADDataset(paths, 'tad_10kb_1d.jsonl', 'tad_boundaries_10kb.bed', band_width=tad_band_width, ignore_bins=tad_ignore)

    common = dict(num_workers=num_workers,
                  pin_memory=pin_memory,
                  drop_last=True,
                  persistent_workers=(num_workers > 0),
                  timeout=180,
                  prefetch_factor=(2 if num_workers > 0 else None))

    loaders = {
        'loop': DataLoader(loop_ds, batch_size=batch_sizes.get('loop', 4), shuffle=True, **common),
        'stripe': DataLoader(stripe_ds, batch_size=batch_sizes.get('stripe', 4), shuffle=True, **common),
        'tad': DataLoader(tad_ds, batch_size=batch_sizes.get('tad', 8), shuffle=True, **common)
    }
    return loaders
