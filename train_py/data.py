import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
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


def load_tad_negatives(path: str, binsize: int) -> Dict[str, List[int]]:
    if not path or not os.path.exists(path):
        return {}
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


def load_tad_domains(path: str, binsize: int) -> Dict[str, List[Tuple[int, int]]]:
    if not path or not os.path.exists(path):
        return {}
    df = np.loadtxt(path, dtype=str)
    if df.ndim == 1:
        df = df[None, :]
    per_chrom: Dict[str, List[Tuple[int, int]]] = {}
    for row in df:
        chrom, start, end = row[:3]
        try:
            s_bp = int(float(start))
            e_bp = int(float(end))
        except Exception:
            continue
        if e_bp <= s_bp:
            continue
        s_bin = s_bp // binsize
        e_bin = int(np.ceil(e_bp / binsize))
        per_chrom.setdefault(str(chrom), []).append((s_bin, e_bin))
    for chrom in per_chrom:
        per_chrom[chrom] = sorted(per_chrom[chrom])
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

        self.binsize = binsize
        manifest_path = os.path.join(paths.manifest_root, manifest_name)
        label_path = os.path.join(paths.label_root, label_name)
        tad_path = os.path.join(paths.label_root, 'tad_boundaries_10kb.bed')

        with open(manifest_path, 'r') as f:
            self.entries = [json.loads(line) for line in f if line.strip()]

        loop_labels = load_loop_labels(label_path)
        self.loop_labels = loop_labels

        self.tad_boundaries = load_tad_boundaries(tad_path, binsize=10000) if include_tad_prior else {}

        # storage
        self.use_h5 = False
        self.cache_dir = os.path.join(paths.cache_root, 'loop_5kb')

        self.distance_cache = distance_channel(center)

    def __len__(self):
        return len(self.entries)

    def _load_patch(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, int]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        x0 = int(entry['x0'])
        patch = int(entry['patch'])
        path_raw = os.path.join(self.cache_dir, f"loop_{chrom}_x{x0}_p{patch}.npz")
        with np.load(path_raw, allow_pickle=True) as npz:
            raw = npz['X']
        path_kr = os.path.join(self.cache_dir, f"loop_{chrom}_x{x0}_p{patch}_kr.npz")
        if os.path.exists(path_kr):
            with np.load(path_kr, allow_pickle=True) as npz:
                kr = npz['X']
        else:
            kr = raw
        return raw, kr, chrom, x0

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
        raw_np, kr_np, chrom, x0 = self._load_patch(idx)
        raw_tensor = torch.from_numpy(raw_np).float()
        kr_tensor = torch.from_numpy(kr_np).float()

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
        central_raw = center_crop_pad2d(raw_tensor, center)
        central_kr = center_crop_pad2d(kr_tensor, center)
        input_tensor = torch.stack([central_raw, central_kr, dist], dim=0)

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

        self.use_h5 = False
        self.cache_dir = os.path.join(paths.cache_root, 'stripe_10kb')

        self.distance_cache = distance_channel(center)

    def __len__(self):
        return len(self.entries)

    def _load_patch(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, int]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        x0 = int(entry['x0'])
        patch = int(entry['patch'])
        path_raw = os.path.join(self.cache_dir, f"stripe_{chrom}_x{x0}_p{patch}.npz")
        with np.load(path_raw, allow_pickle=True) as npz:
            raw = npz['X']
        path_kr = os.path.join(self.cache_dir, f"stripe_{chrom}_x{x0}_p{patch}_kr.npz")
        if os.path.exists(path_kr):
            with np.load(path_kr, allow_pickle=True) as npz:
                kr = npz['X']
        else:
            kr = raw
        return raw, kr, chrom, x0

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

        raw_np, kr_np, chrom, x0 = self._load_patch(idx)
        raw_tensor = torch.from_numpy(raw_np).float()
        kr_tensor = torch.from_numpy(kr_np).float()
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

        central_raw = center_crop_pad2d(raw_tensor, center)
        central_kr = center_crop_pad2d(kr_tensor, center)
        inputs = torch.stack([central_raw, central_kr, self.distance_cache], dim=0)

        vert = torch.clamp(central_raw - central_raw.mean(dim=0, keepdim=True), min=0.0)
        hor = torch.clamp(central_raw - central_raw.mean(dim=1, keepdim=True), min=0.0)
        orientation = torch.max(vert, hor)
        if orientation.max() > 0:
            orientation = orientation / orientation.max()
        orientation_prior = orientation.unsqueeze(0)

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
        item['target']['orientation_prior'] = orientation_prior
        return item


class TADDataset(Dataset):
    def __init__(self, paths: DatasetPaths, manifest_name: str, binsize: int = 10000,
                 domain_label_name: str = "tad_domains_10kb.bed",
                 use_detection: bool = False,
                 min_bins: int = 4,
                 max_instances: int = 16):
        self.binsize = binsize
        manifest_path = os.path.join(paths.manifest_root, manifest_name)
        with open(manifest_path, 'r') as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        self.cache_dir = os.path.join(paths.cache_root, 'tad_10kb_2d')
        self.distance_cache: Dict[int, torch.Tensor] = {}
        self.use_detection = use_detection
        self.min_bins = min_bins
        self.max_instances = max_instances
        label_path = os.path.join(paths.label_root, domain_label_name) if domain_label_name else None
        self.domain_intervals = load_tad_domains(label_path, binsize) if (self.use_detection and label_path) else {}

    def __len__(self):
        return len(self.entries)

    def _distance(self, size: int) -> torch.Tensor:
        if size not in self.distance_cache:
            self.distance_cache[size] = distance_channel(size)
        return self.distance_cache[size]

    def _load_npz(self, chrom: str, x0: int, patch: int) -> Dict[str, Any]:
        path = os.path.join(self.cache_dir, f"tad2d_{chrom}_x{x0}_p{patch}.npz")
        with np.load(path, allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        chrom = str(entry['chrom'])
        x0 = int(entry['x0'])
        patch = int(entry['patch'])
        center = int(entry.get('center', patch))
        binsize = int(entry.get('binsize', self.binsize))

        data_npz = self._load_npz(chrom, x0, patch)
        raw_full = torch.from_numpy(data_npz['raw']).float()
        kr_full = torch.from_numpy(data_npz['kr']).float()
        domain_full = torch.from_numpy(data_npz['domain']).float()
        boundary_thin_full = torch.from_numpy(data_npz.get('boundary', np.zeros_like(data_npz['domain']))).float()
        boundary_wide_full = torch.from_numpy(data_npz.get('boundary_wide', np.zeros_like(data_npz['domain']))).float()

        offset = max(0, (patch - center) // 2)
        raw_center = center_crop_pad2d(raw_full, center)
        kr_center = center_crop_pad2d(kr_full, center)
        domain_center = center_crop_pad2d(domain_full, center)
        boundary_thin_center = center_crop_pad2d(boundary_thin_full, center)
        boundary_wide_center = center_crop_pad2d(boundary_wide_full, center)

        inputs = torch.stack([raw_center, kr_center, self._distance(center)], dim=0)
        domain_mask = domain_center.unsqueeze(0)
        boundary_mask = boundary_thin_center.unsqueeze(0)
        boundary_wide = boundary_wide_center.unsqueeze(0)

        item = {
            'task': 'tad',
            'inputs': inputs,
            'target': {
                'domain_mask': domain_mask,
                'boundary_mask': boundary_mask,
                'boundary_wide': boundary_wide
            },
            'metadata': {
                'chrom': chrom,
                'bin_start': int(x0 + offset),
                'patch_start': int(x0),
                'center': int(center),
                'binsize': binsize
            }
        }
        if self.use_detection:
            detection = self._build_detection_targets(chrom, x0 + offset, center)
            item['detection'] = detection
        return item

    def _build_detection_targets(self, chrom: str, window_start: int, size: int) -> Dict[str, torch.Tensor]:
        intervals = self.domain_intervals.get(chrom, [])
        boxes: List[List[float]] = []
        masks: List[torch.Tensor] = []
        labels: List[int] = []
        for start, end in intervals:
            if end <= window_start or start >= window_start + size:
                continue
            s = max(start, window_start)
            e = min(end, window_start + size)
            if (e - s) < self.min_bins:
                continue
            y1 = int(s - window_start)
            y2 = int(e - window_start)
            y1 = max(0, min(size, y1))
            y2 = max(0, min(size, y2))
            if y2 - y1 < 1:
                continue
            mask = torch.zeros((size, size), dtype=torch.float32)
            mask[y1:y2, y1:y2] = 1.0
            boxes.append([float(y1), float(y1), float(y2), float(y2)])
            masks.append(mask)
            labels.append(1)
            if len(boxes) >= self.max_instances:
                break

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            masks_tensor = torch.stack(masks)
            areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, size, size), dtype=torch.float32)
            areas = torch.zeros((0,), dtype=torch.float32)

        detection = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'masks': masks_tensor,
            'areas': areas,
            'iscrowd': torch.zeros((labels_tensor.numel(),), dtype=torch.int64)
        }
        return detection



def collate_tad_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    inputs = torch.stack([item['inputs'] for item in batch], dim=0)
    targets = {
        'domain_mask': torch.stack([item['target']['domain_mask'] for item in batch], dim=0),
        'boundary_mask': torch.stack([item['target']['boundary_mask'] for item in batch], dim=0),
        'boundary_wide': torch.stack([item['target']['boundary_wide'] for item in batch], dim=0),
    }
    metadata = [item.get('metadata', {}) for item in batch]
    detection = [item.get('detection') for item in batch]
    return {
        'task': 'tad',
        'inputs': inputs,
        'target': targets,
        'metadata': metadata,
        'detection': detection
    }


def make_dataloaders(paths: DatasetPaths,
                     loop_center: int, loop_patch: int, loop_stride: int,
                     stripe_center: int, stripe_patch: int, stripe_stride: int,
                     batch_sizes: Dict[str, int], num_workers: int, pin_memory: bool,
                     loop_manifest_name: str = 'loop_5kb.jsonl',
                     loop_label_name: str = 'loop_5kb.jsonl',
                     stripe_manifest_name: str = 'stripe_10kb.jsonl',
                     stripe_label_name: str = 'stripe_10kb.jsonl',
                     tad_manifest_name: Optional[str] = None,
                     tad_binsize: int = 10000,
                     tad_domain_label_name: str = 'tad_domains_10kb.bed',
                     tad_use_detection: bool = False,
                     tad_detection_min_bins: int = 4,
                     tad_detection_max_instances: int = 16) -> Dict[str, DataLoader]:
    loop_ds = LoopDataset(paths, loop_manifest_name, loop_label_name, center=loop_center, patch=loop_patch, stride=loop_stride)
    stripe_ds = StripeDataset(paths, stripe_manifest_name, stripe_label_name, center=stripe_center, patch=stripe_patch, stride=stripe_stride)
    tad_manifest = tad_manifest_name or 'tad_10kb_2d.jsonl'
    tad_ds = TADDataset(paths, tad_manifest, binsize=tad_binsize,
                        domain_label_name=tad_domain_label_name,
                        use_detection=tad_use_detection,
                        min_bins=tad_detection_min_bins,
                        max_instances=tad_detection_max_instances)

    common = dict(num_workers=num_workers,
                  pin_memory=pin_memory,
                  drop_last=True,
                  persistent_workers=(num_workers > 0),
                  timeout=180,
                  prefetch_factor=(2 if num_workers > 0 else None))

    loaders = {
        'loop': DataLoader(loop_ds, batch_size=batch_sizes.get('loop', 4), shuffle=True, **common),
        'stripe': DataLoader(stripe_ds, batch_size=batch_sizes.get('stripe', 4), shuffle=True, **common),
        'tad': DataLoader(tad_ds, batch_size=batch_sizes.get('tad', 8), shuffle=True,
                          collate_fn=collate_tad_batch, **common)
    }
    return loaders
