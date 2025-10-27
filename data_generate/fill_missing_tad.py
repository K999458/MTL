#!/usr/bin/env python3
"""
Fill missing TAD 1D cache (.npz) entries based on manifest.

Reads:  data_generated/manifests/tad_10kb_1d.jsonl
Writes: data_generated/cache/tad_10kb_1d/tad1d_{chrom}_s{start}_L{L}_B{B}.npz

Normalization: balanced-> O/E (within patch)-> log1p -> percentile clip to [0,1]
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


def build_path(cache_dir: str, chrom: str, start_bin: int, L: int, B: int) -> str:
    return os.path.join(cache_dir, f"tad1d_{chrom}_s{start_bin}_L{L}_B{B}.npz")


def main():
    ap = argparse.ArgumentParser(description='Fill missing TAD 1D .npz based on manifest')
    ap.add_argument('--mcool', required=True, help='.mcool or mcool::/resolutions/10000')
    ap.add_argument('--manifest', required=True, help='tad_10kb_1d.jsonl')
    ap.add_argument('--cache-dir', required=True, help='cache/tad_10kb_1d directory')
    ap.add_argument('--band-width', type=int, default=64)
    ap.add_argument('--percentile', type=float, default=98.0)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    uri = args.mcool if '::' in args.mcool else f"{args.mcool}::/resolutions/10000"
    clr = cooler.Cooler(uri)
    ensure_dir(args.cache_dir)

    # Load manifest
    items = []
    with open(args.manifest, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            items.append(it)

    missing = []
    for it in items:
        chrom = it['chrom']
        sbin = int(it['start_bin'])
        L = int(it.get('length', 1024))
        out = build_path(args.cache_dir, chrom, sbin, L, args.band_width)
        if not os.path.exists(out):
            missing.append((chrom, sbin, L, out))

    if not missing:
        print('[fill] nothing missing')
        return

    # Process (serial to avoid cooldown limits; you can parallelize if needed)
    B = args.band_width
    for chrom, sbin, L, out in missing:
        binsize = int(clr.binsize)
        s_bp = sbin * binsize
        e_bp = (sbin + L) * binsize
        chrom_len = int(clr.chromsizes.loc[chrom])
        if e_bp > chrom_len:
            e_bp = chrom_len
            s_bp = max(0, e_bp - L * binsize)
        mat = clr.matrix(balance=True).fetch(f"{chrom}:{s_bp}-{e_bp}", f"{chrom}:{s_bp}-{e_bp}")
        mat = np.asarray(mat, dtype=np.float32)
        mat = oe_log1p_clip(mat, clip_percentile=args.percentile)
        # band
        b = B // 2
        band = []
        for dk in range(-b, b):
            diag = np.diag(mat, k=dk)
            # shift padding to align offsets
            if dk < 0:
                diag = np.pad(diag, (0, -dk))
            elif dk > 0:
                diag = np.pad(diag, (dk, 0))
            # enforce fixed length L
            if diag.shape[0] < L:
                diag = np.pad(diag, (0, L - diag.shape[0]))
            elif diag.shape[0] > L:
                diag = diag[:L]
            band.append(diag)
        band = np.stack(band, axis=0).astype(np.float32)
        np.savez_compressed(out, X=mat, band=band, chrom=chrom, start_bin=sbin, L=L, B=B, binsize=binsize)
        print('[fill] wrote', out)

    print('[fill] done, filled', len(missing))


if __name__ == '__main__':
    main()
