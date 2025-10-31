#!/usr/bin/env python3
"""
Fill missing 2D cache (.npz) entries for loop@5kb / stripe@10kb / tad@10kb 2D based on manifest.

Usage examples:
  # Stripe@10kb
  python data_generate/fill_missing_2d.py \
    --mcool data/12878.mcool \
    --manifest data_generated/manifests/stripe_10kb.jsonl \
    --cache-dir data_generated/cache/stripe_10kb \
    --task stripe --percentile 98

  # TAD@10kb 2D
  python data_generate/fill_missing_2d.py \
    --mcool data/12878.mcool \
    --manifest data_generated/manifests/tad_10kb_2d.jsonl \
    --cache-dir data_generated/cache/tad_10kb_2d \
    --task tad2d --domain-bed data_generated/labels/tad_domains_10kb.bed \
    --boundary-bed data_generated/labels/tad_boundaries_10kb.bed \
    --percentile 98

Normalization: balanced -> O/E (within patch) -> log1p -> percentile clip到 [0,1]
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import cooler
import numpy as np


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def oe_log1p_clip(mat: np.ndarray, clip_percentile: float = 98.0) -> np.ndarray:
    m = mat.astype(np.float32, copy=True)
    h, w = m.shape
    assert h == w, 'expect square'
    exp = np.zeros(h, dtype=np.float32)
    for d in range(h):
        diag = np.diag(m, k=0) if d == 0 else np.diag(m, k=d)
        if d > 0:
            diag = np.concatenate([diag, np.diag(m, k=-d)])
        exp[d] = np.nanmean(diag) if np.isfinite(diag).any() else np.nan
    idx = np.abs(np.subtract.outer(np.arange(h), np.arange(w)))
    expected = exp[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        m = m / expected
        m[~np.isfinite(m)] = 0.0
        m = np.log1p(m)
    vmax = float(np.nanpercentile(m, clip_percentile)) if clip_percentile is not None else None
    if vmax is None or vmax <= 0:
        vmax = 1.0
    m = np.clip(m, 0.0, vmax) / vmax
    return m


def _log1p_positive(mat: np.ndarray) -> np.ndarray:
    m = np.clip(mat.astype(np.float32), a_min=0.0, a_max=None)
    return np.log1p(m)


def _load_domain_cache(path: str, binsize: int) -> Dict[str, List[Tuple[int, int]]]:
    if not path or not os.path.exists(path):
        return {}
    arr = np.loadtxt(path, dtype=str)
    if arr.ndim == 1:
        arr = arr[None, :]
    cache: Dict[str, List[Tuple[int, int]]] = {}
    for chrom, start, end in arr[:, :3]:
        try:
            s_bp = int(float(start))
            e_bp = int(float(end))
        except Exception:
            continue
        if e_bp <= s_bp:
            continue
        s_bin = s_bp // binsize
        e_bin = int(np.ceil(e_bp / binsize))
        cache.setdefault(str(chrom), []).append((s_bin, e_bin))
    for chrom in cache:
        cache[chrom] = sorted(cache[chrom])
    return cache


def _load_boundary_cache(path: str, binsize: int) -> Dict[str, List[int]]:
    if not path or not os.path.exists(path):
        return {}
    arr = np.loadtxt(path, dtype=str)
    if arr.ndim == 1:
        arr = arr[None, :]
    cache: Dict[str, List[int]] = {}
    for chrom, start, *_ in arr:
        try:
            s_bp = int(float(start))
        except Exception:
            continue
        cache.setdefault(str(chrom), []).append(s_bp // binsize)
    for chrom in cache:
        cache[chrom] = sorted(set(cache[chrom]))
    return cache


def _tad_masks(chrom: str, x0: int, patch: int, binsize: int,
               domain_cache: Dict[str, List[Tuple[int, int]]],
               boundary_cache: Dict[str, List[int]]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[int]]:
    domain_mask = np.zeros((patch, patch), dtype=np.float32)
    boundary_mask = np.zeros((patch, patch), dtype=np.float32)
    domain_pairs: List[Tuple[int, int]] = []
    for s_bin, e_bin in domain_cache.get(chrom, []):
        ls = s_bin - x0
        le = e_bin - x0
        if le <= 0 or ls >= patch:
            continue
        ls = max(0, ls)
        le = min(patch, le)
        if le <= ls:
            continue
        domain_mask[ls:le, ls:le] = 1.0
        domain_pairs.append((ls, le))
    boundary_bins: List[int] = []
    for b in boundary_cache.get(chrom, []):
        idx = b - x0
        if 0 <= idx < patch:
            boundary_mask[idx, :] = 1.0
            boundary_mask[:, idx] = 1.0
            boundary_bins.append(idx)
    return domain_mask, boundary_mask, domain_pairs, boundary_bins


def main():
    ap = argparse.ArgumentParser(description='Fill missing 2D cache (loop/stripe/tad2d) based on manifest')
    ap.add_argument('--mcool', required=True, help='.mcool (will infer resolution from binsize in manifest)')
    ap.add_argument('--manifest', required=True, help='loop_5kb.jsonl / stripe_10kb.jsonl / tad_10kb_2d.jsonl')
    ap.add_argument('--cache-dir', required=True, help='cache output directory')
    ap.add_argument('--task', choices=['loop', 'stripe', 'tad2d'], required=True, help='which task to fill')
    ap.add_argument('--domain-bed', help='task=tad2d 时必需，TAD 域 BED 路径')
    ap.add_argument('--boundary-bed', help='task=tad2d 时可选，若缺省与 domain-bed 相同')
    ap.add_argument('--binsize', type=int, default=10000, help='task=tad2d 时使用的 bin 大小 [默认 10000]')
    ap.add_argument('--percentile', type=float, default=98.0)
    args = ap.parse_args()

    ensure_dir(args.cache_dir)

    items = []
    with open(args.manifest, 'r') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    missing = []
    if args.task in ('loop', 'stripe'):
        for it in items:
            chrom = it['chrom']; x0 = int(it['x0']); patch = int(it['patch'])
            base = f"{args.task}_{chrom}_x{x0}_p{patch}"
            raw_path = os.path.join(args.cache_dir, base + '.npz')
            kr_path = os.path.join(args.cache_dir, base + '_kr.npz')
            if (not os.path.exists(raw_path)) or (not os.path.exists(kr_path)):
                missing.append(it)
    else:
        for it in items:
            chrom = it['chrom']; x0 = int(it['x0']); patch = int(it['patch'])
            base = os.path.join(args.cache_dir, f"tad2d_{chrom}_x{x0}_p{patch}.npz")
            if not os.path.exists(base):
                missing.append(it)

    if not missing:
        print('[fill2d] nothing missing')
        return

    if args.task == 'tad2d':
        if not args.domain_bed:
            raise ValueError('--domain-bed is required for task=tad2d')
        boundary_bed = args.boundary_bed or args.domain_bed
        domain_cache = _load_domain_cache(args.domain_bed, args.binsize)
        boundary_cache = _load_boundary_cache(boundary_bed, args.binsize)

    for it in missing:
        chrom = str(it['chrom'])
        x0 = int(it['x0'])
        patch = int(it['patch'])
        binsize = int(it.get('binsize', args.binsize if args.task == 'tad2d' else it['binsize']))
        uri = args.mcool if '::' in args.mcool else f"{args.mcool}::/resolutions/{binsize}"
        clr = cooler.Cooler(uri)
        start_bp = x0 * binsize
        end_bp = (x0 + patch) * binsize
        chrom_len = int(clr.chromsizes.loc[chrom])
        if end_bp > chrom_len:
            end_bp = chrom_len
            start_bp = max(0, end_bp - patch * binsize)
        mat_raw = clr.matrix(balance=False).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
        mat_bal = clr.matrix(balance=True).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
        mat_raw = _log1p_positive(np.asarray(mat_raw))
        mat_bal = oe_log1p_clip(np.asarray(mat_bal), clip_percentile=args.percentile)

        if args.task in ('loop', 'stripe'):
            base = os.path.join(args.cache_dir, f"{args.task}_{chrom}_x{x0}_p{patch}")
            np.savez_compressed(base + '.npz', X=mat_raw.astype(np.float32), chrom=chrom, x0=x0, patch=patch, binsize=binsize)
            np.savez_compressed(base + '_kr.npz', X=mat_bal.astype(np.float32), chrom=chrom, x0=x0, patch=patch, binsize=binsize)
            print('[fill2d] wrote', base + '.npz', '&', base + '_kr.npz')
        else:
            domain_mask, boundary_mask, domain_pairs, boundary_bins = _tad_masks(
                chrom, x0, patch, binsize, domain_cache, boundary_cache
            )
            base = os.path.join(args.cache_dir, f"tad2d_{chrom}_x{x0}_p{patch}.npz")
            np.savez_compressed(
                base,
                raw=mat_raw.astype(np.float32),
                kr=mat_bal.astype(np.float32),
                domain=domain_mask.astype(np.float32),
                boundary=boundary_mask.astype(np.float32),
                domain_pairs=np.asarray(domain_pairs, dtype=np.int32) if domain_pairs else np.zeros((0, 2), dtype=np.int32),
                boundary_bins=np.asarray(boundary_bins, dtype=np.int32) if boundary_bins else np.zeros((0,), dtype=np.int32),
                chrom=chrom,
                x0=x0,
                patch=patch,
                binsize=binsize
            )
            print('[fill2d] wrote', base)

    print('[fill2d] done, filled', len(missing))


if __name__ == '__main__':
    main()
