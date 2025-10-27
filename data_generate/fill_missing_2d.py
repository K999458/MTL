#!/usr/bin/env python3
"""
Fill missing 2D cache (.npz) entries for loop@5kb or stripe@10kb based on manifest.

Usage examples:
  # Stripe@10kb
  conda run -n vipg python AAA_MIL/data_generate/fill_missing_2d.py \
    --mcool /storu/zkyang/AAA_MIL/data/12878.mcool \
    --manifest /storu/zkyang/AAA_MIL/data_generated/manifests/stripe_10kb.jsonl \
    --cache-dir /storu/zkyang/AAA_MIL/data_generated/cache/stripe_10kb \
    --task stripe --percentile 98

  # Loop@5kb
  conda run -n vipg python AAA_MIL/data_generate/fill_missing_2d.py \
    --mcool /storu/zkyang/AAA_MIL/data/12878.mcool \
    --manifest /storu/zkyang/AAA_MIL/data_generated/manifests/loop_5kb.jsonl \
    --cache-dir /storu/zkyang/AAA_MIL/data_generated/cache/loop_5kb \
    --task loop --percentile 98

Normalization: balanced -> O/E (within patch) -> log1p -> percentile clip to [0,1]
"""

import argparse, os, json
import numpy as np
import cooler


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def oe_log1p_clip(mat: np.ndarray, clip_percentile: float=98.0) -> np.ndarray:
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


def main():
    ap = argparse.ArgumentParser(description='Fill missing 2D cache (loop/stripe) based on manifest')
    ap.add_argument('--mcool', required=True, help='.mcool (will infer resolution from binsize in manifest)')
    ap.add_argument('--manifest', required=True, help='loop_5kb.jsonl or stripe_10kb.jsonl')
    ap.add_argument('--cache-dir', required=True, help='cache output directory (loop_5kb or stripe_10kb)')
    ap.add_argument('--task', choices=['loop','stripe'], required=True, help='which task to fill')
    ap.add_argument('--percentile', type=float, default=98.0)
    args = ap.parse_args()

    # Will open cooler per item to be robust in multiprocess/FD limits
    ensure_dir(args.cache_dir)

    # Load manifest
    items = []
    with open(args.manifest, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    # Detect missing
    missing = []
    for it in items:
        chrom = it['chrom']; x0 = int(it['x0']); patch = int(it['patch'])
        fn = f"{args.task}_{chrom}_x{x0}_p{patch}.npz"
        if not os.path.exists(os.path.join(args.cache_dir, fn)):
            missing.append(it)

    if not missing:
        print('[fill2d] nothing missing')
        return

    # Fill missing
    for it in missing:
        chrom = it['chrom']; x0 = int(it['x0']); patch = int(it['patch']); binsize = int(it['binsize'])
        uri = args.mcool if '::' in args.mcool else f"{args.mcool}::/resolutions/{binsize}"
        clr = cooler.Cooler(uri)
        start_bp = x0 * binsize
        end_bp = (x0 + patch) * binsize
        chrom_len = int(clr.chromsizes.loc[chrom])
        if end_bp > chrom_len:
            end_bp = chrom_len
            start_bp = max(0, end_bp - patch * binsize)
        mat = clr.matrix(balance=True).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
        mat = np.asarray(mat, dtype=np.float32)
        mat = oe_log1p_clip(mat, clip_percentile=args.percentile)
        fn = f"{args.task}_{chrom}_x{x0}_p{patch}.npz"
        np.savez_compressed(os.path.join(args.cache_dir, fn), X=mat, chrom=chrom, x0=x0, patch=patch, binsize=binsize)
        print('[fill2d] wrote', fn)

    print('[fill2d] done, filled', len(missing))


if __name__ == '__main__':
    main()

