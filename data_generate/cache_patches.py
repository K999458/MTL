#!/usr/bin/env python3
"""
将 Hi-C .mcool 中的窗口切片缓存为 .npz：
- Loop@5kb：2D patch（按 manifests/loop_5kb.jsonl）
- Stripe@10kb：2D patch（按 manifests/stripe_10kb.jsonl）
- TAD@10kb：对角带 1D（按 manifests/tad_10kb_1d.jsonl）

规范化：balanced（KR/ICE）→ O/E（补丁内逐对角线期望）→ log1p → 分位裁剪到 [0,1]

仅读写 /storu/zkyang/AAA_MIL 下的路径。
"""

import argparse, os, json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import cooler
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def oe_log1p_clip(mat: np.ndarray, clip_percentile: float=98.0) -> np.ndarray:
    m = mat.astype(np.float32, copy=True)
    # expected by distance within patch
    h, w = m.shape
    assert h == w, 'expect square'
    exp = np.zeros(h, dtype=np.float32)
    for d in range(h):
        diag = np.diag(m, k=0) if d == 0 else np.diag(m, k=d)
        if d > 0:
            diag = np.concatenate([diag, np.diag(m, k=-d)])
        if np.isfinite(diag).any():
            exp[d] = np.nanmean(diag)
        else:
            exp[d] = np.nan
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


def _save_npz(path: str, X: np.ndarray, chrom: str, x0: int, patch: int, binsize: int) -> None:
    np.savez_compressed(path, X=X.astype(np.float32), chrom=chrom, x0=x0, patch=patch, binsize=binsize)


def _worker_loop(item, uri: str, outdir: str, clip_pct: float) -> int:
    chrom = item['chrom']; x0 = int(item['x0']); patch = int(item['patch']); binsize = int(item['binsize'])
    start_bp = x0 * binsize; end_bp = (x0 + patch) * binsize
    clr = cooler.Cooler(uri)
    mat_raw = clr.matrix(balance=False).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
    mat_bal = clr.matrix(balance=True).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
    mat_raw = _log1p_positive(np.asarray(mat_raw))
    mat_bal = oe_log1p_clip(np.asarray(mat_bal), clip_percentile=clip_pct)
    base = os.path.join(outdir, f"loop_{chrom}_x{x0}_p{patch}")
    _save_npz(base + ".npz", mat_raw, chrom, x0, patch, binsize)
    _save_npz(base + "_kr.npz", mat_bal, chrom, x0, patch, binsize)
    return 1

def cache_loop(manifest_path: str, uri: str, outdir: str, limit: Optional[int], clip_pct: float, workers: int):
    ensure_dir(outdir)
    if limit is not None and limit <= 0:
        limit = None
    items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            it = json.loads(line)
            items.append(it)
            if limit is not None and len(items) >= limit:
                break
    if workers <= 1:
        cnt = 0
        for it in items:
            cnt += _worker_loop(it, uri, outdir, clip_pct)
        return cnt
    cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker_loop, it, uri, outdir, clip_pct) for it in items]
        for _ in as_completed(futs):
            cnt += 1
    return cnt


def _worker_stripe(item, uri: str, outdir: str, clip_pct: float) -> int:
    chrom = item['chrom']; x0 = int(item['x0']); patch = int(item['patch']); binsize = int(item['binsize'])
    start_bp = x0 * binsize; end_bp = (x0 + patch) * binsize
    clr = cooler.Cooler(uri)
    mat_raw = clr.matrix(balance=False).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
    mat_bal = clr.matrix(balance=True).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
    mat_raw = _log1p_positive(np.asarray(mat_raw))
    mat_bal = oe_log1p_clip(np.asarray(mat_bal), clip_percentile=clip_pct)
    base = os.path.join(outdir, f"stripe_{chrom}_x{x0}_p{patch}")
    _save_npz(base + ".npz", mat_raw, chrom, x0, patch, binsize)
    _save_npz(base + "_kr.npz", mat_bal, chrom, x0, patch, binsize)
    return 1

def cache_stripe(manifest_path: str, uri: str, outdir: str, limit: Optional[int], clip_pct: float, workers: int):
    ensure_dir(outdir)
    if limit is not None and limit <= 0:
        limit = None
    items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            it = json.loads(line)
            items.append(it)
            if limit is not None and len(items) >= limit:
                break
    if workers <= 1:
        cnt = 0
        for it in items:
            cnt += _worker_stripe(it, uri, outdir, clip_pct)
        return cnt
    cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker_stripe, it, uri, outdir, clip_pct) for it in items]
        for _ in as_completed(futs):
            cnt += 1
    return cnt


def _worker_tad1d(item, uri: str, outdir: str, band_width: int, clip_pct: float) -> int:
    chrom = item['chrom']; sbin = int(item['start_bin']); L = int(item.get('length', 1024))
    clr = cooler.Cooler(uri)
    binsize = int(clr.binsize)
    # Use bin indices to ensure exactly L bins
    nbins = clr.bins().fetch(chrom).shape[0]
    ebin = min(sbin + L, nbins)
    sbin_adj = max(0, ebin - L)
    s_bp = sbin_adj * binsize
    e_bp = (sbin_adj + L) * binsize
    mat = clr.matrix(balance=True).fetch(f"{chrom}:{s_bp}-{e_bp}", f"{chrom}:{s_bp}-{e_bp}")
    mat = np.asarray(mat, dtype=np.float32)
    mat = oe_log1p_clip(mat, clip_percentile=clip_pct)
    B = int(band_width); b = B // 2
    band = []
    for dk in range(-b, b):
        diag = np.diag(mat, k=dk)
        if dk < 0:
            diag = np.pad(diag, (0, -dk))
        elif dk > 0:
            diag = np.pad(diag, (dk, 0))
        if diag.shape[0] < L:
            diag = np.pad(diag, (0, L - diag.shape[0]))
        elif diag.shape[0] > L:
            diag = diag[:L]
        band.append(diag)
    band = np.stack(band, axis=0).astype(np.float32)
    fn = f"tad1d_{chrom}_s{sbin}_L{L}_B{B}.npz"
    np.savez_compressed(os.path.join(outdir, fn), X=mat, band=band, chrom=chrom, start_bin=sbin, L=L, B=B, binsize=binsize)
    return 1

def cache_tad1d(manifest_path: str, uri: str, outdir: str, band_width: int, limit: Optional[int], clip_pct: float, workers: int):
    ensure_dir(outdir)
    if limit is not None and limit <= 0:
        limit = None
    items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            it = json.loads(line)
            items.append(it)
            if limit is not None and len(items) >= limit:
                break
    if workers <= 1:
        cnt = 0
        for it in items:
            cnt += _worker_tad1d(it, uri, outdir, band_width, clip_pct)
        return cnt
    cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker_tad1d, it, uri, outdir, band_width, clip_pct) for it in items]
        for _ in as_completed(futs):
            cnt += 1
    return cnt


# ---------- TAD 2D 相关 ----------
DOMAIN_CACHE: Dict[str, List[Tuple[int, int]]] = {}
TAD_BINSIZE: int = 10000
BOUNDARY_THIN: int = 1
BOUNDARY_WIDE: int = 3


def _load_domain_cache(path: str, binsize: int) -> Dict[str, List[Tuple[int, int]]]:
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep='\t', header=None, names=['chrom','start','end'])
    df = df.dropna().copy()
    df['start_bin'] = (df['start'].astype(int) // binsize)
    df['end_bin'] = (df['end'].astype(int) + binsize - 1) // binsize
    cache: Dict[str, List[Tuple[int,int]]] = {}
    for chrom, g in df.groupby('chrom'):
        cache[str(chrom)] = [(int(r['start_bin']), int(r['end_bin'])) for _, r in g.iterrows() if int(r['end_bin']) > int(r['start_bin'])]
    return cache


def _init_tad2d_context(domain_path: str, binsize: int):
    global DOMAIN_CACHE, TAD_BINSIZE
    DOMAIN_CACHE = _load_domain_cache(domain_path, binsize)
    TAD_BINSIZE = binsize


def _tad2d_masks(chrom: str, x0: int, patch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    binsize = TAD_BINSIZE
    interior = np.zeros((patch, patch), dtype=np.float32)
    boundary_thin = np.zeros((patch, patch), dtype=np.float32)
    boundary_wide = np.zeros((patch, patch), dtype=np.float32)
    domains = DOMAIN_CACHE.get(chrom, [])
    for s_bin, e_bin in domains:
        local_s = s_bin - x0
        local_e = e_bin - x0
        if local_e <= 0 or local_s >= patch:
            continue
        ls = max(0, local_s)
        le = min(patch, local_e)
        if le <= ls:
            continue
        interior[ls:le, ls:le] = 1.0

        # thin boundary (width = 1)
        boundary_thin[ls, ls:le] = 1.0
        boundary_thin[le - 1, ls:le] = 1.0
        boundary_thin[ls:le, ls] = 1.0
        boundary_thin[ls:le, le - 1] = 1.0

        # wide boundary ring
        for bw in range(BOUNDARY_WIDE):
            top = max(0, ls - bw)
            bottom = min(patch - 1, le - 1 + bw)
            left = max(0, ls - bw)
            right = min(patch - 1, le - 1 + bw)
            boundary_wide[top, left:right + 1] = 1.0
            boundary_wide[bottom, left:right + 1] = 1.0
            boundary_wide[top:bottom + 1, left] = 1.0
            boundary_wide[top:bottom + 1, right] = 1.0

    boundary_wide = np.clip(boundary_wide, 0.0, 1.0)
    boundary_thin = np.clip(boundary_thin, 0.0, 1.0)
    return interior, boundary_thin, boundary_wide


def _worker_tad2d(item, uri: str, outdir: str, clip_pct: float) -> int:
    chrom = str(item['chrom']); x0 = int(item['x0']); patch = int(item['patch']); binsize = int(item['binsize'])
    start_bp = x0 * binsize; end_bp = (x0 + patch) * binsize
    clr = cooler.Cooler(uri)
    mat_raw = clr.matrix(balance=False).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
    mat_bal = clr.matrix(balance=True).fetch(f"{chrom}:{start_bp}-{end_bp}", f"{chrom}:{start_bp}-{end_bp}")
    mat_raw = _log1p_positive(np.asarray(mat_raw))
    mat_bal = oe_log1p_clip(np.asarray(mat_bal), clip_percentile=clip_pct)
    interior_mask, boundary_thin, boundary_wide = _tad2d_masks(chrom, x0, patch)
    base = os.path.join(outdir, f"tad2d_{chrom}_x{x0}_p{patch}")
    np.savez_compressed(
        base + ".npz",
        raw=mat_raw.astype(np.float32),
        kr=mat_bal.astype(np.float32),
        domain=interior_mask,
        boundary=boundary_thin,
        boundary_wide=boundary_wide,
        chrom=chrom,
        x0=x0,
        patch=patch,
        binsize=binsize
    )
    return 1


def cache_tad2d(manifest_path: str, uri: str, outdir: str, domain_path: str,
                binsize: int, limit: Optional[int], clip_pct: float, workers: int):
    ensure_dir(outdir)
    if limit is not None and limit <= 0:
        limit = None
    items = []
    with open(manifest_path, 'r') as f:
        for line in f:
            it = json.loads(line)
            items.append(it)
            if limit is not None and len(items) >= limit:
                break
    if not items:
        return 0
    if workers is None:
        workers = 1
    if workers <= 1:
        _init_tad2d_context(domain_path, binsize)
        cnt = 0
        for it in items:
            cnt += _worker_tad2d(it, uri, outdir, clip_pct)
        return cnt
    else:
        init_args = (domain_path, binsize)
        cnt = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_tad2d_context, initargs=init_args) as ex:
            futs = [ex.submit(_worker_tad2d, it, uri, outdir, clip_pct) for it in items]
            for _ in as_completed(futs):
                cnt += 1
        return cnt


def main():
    ap = argparse.ArgumentParser(description='缓存 Hi-C 矩阵切片为 .npz (loop/stripe/tad1d)')
    ap.add_argument('--mcool', required=True, help='.mcool 或 mcool::/resolutions/N 基路径')
    ap.add_argument('--manifests', required=True, help='manifests 目录（至少包含 loop_5kb.jsonl / stripe_10kb.jsonl）')
    ap.add_argument('--outdir', required=True, help='输出根目录 e.g. AAA_MIL/data_generated/cache')
    ap.add_argument('--band-width', type=int, default=64, help='TAD 1D 带宽 B [默认 64]')
    ap.add_argument('--percentile', type=float, default=98.0, help='分位裁剪上限 [默认 98]')
    ap.add_argument('--limit-loop', type=int, default=None, help='限制 Loop 切片数量（便于测试）')
    ap.add_argument('--limit-stripe', type=int, default=None, help='限制 Stripe 切片数量')
    ap.add_argument('--tad1d-manifest', default=None, help='可选，TAD 1D manifest 路径；缺省则跳过 1D 缓存')
    ap.add_argument('--limit-tad', type=int, default=None, help='限制 TAD 1D 切片数量')
    ap.add_argument('--tad2d-manifest', default=None, help='可选，TAD 2D manifest 路径')
    ap.add_argument('--domain-bed', default=None, help='TAD 域 BED（用于 2D 标签）')
    ap.add_argument('--tad2d-binsize', type=int, default=10000, help='TAD 2D bin 大小 [默认 10000]')
    ap.add_argument('--limit-tad2d', type=int, default=None, help='限制 TAD 2D 切片数量')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count()//2), help='并行进程数 [默认: CPU/2]')
    args = ap.parse_args()

    uri5  = args.mcool if '::' in args.mcool else f"{args.mcool}::/resolutions/5000"
    uri10 = args.mcool if '::' in args.mcool else f"{args.mcool}::/resolutions/10000"
    clr5  = cooler.Cooler(uri5)
    clr10 = cooler.Cooler(uri10)

    loop_manifest   = os.path.join(args.manifests, 'loop_5kb.jsonl')
    stripe_manifest = os.path.join(args.manifests, 'stripe_10kb.jsonl')
    out_loop   = os.path.join(args.outdir, 'loop_5kb')
    out_stripe = os.path.join(args.outdir, 'stripe_10kb')
    out_tad    = os.path.join(args.outdir, 'tad_10kb_1d')

    n1 = cache_loop(loop_manifest, uri5, out_loop, args.limit_loop, args.percentile, args.workers)
    n2 = cache_stripe(stripe_manifest, uri10, out_stripe, args.limit_stripe, args.percentile, args.workers)
    n3 = 0
    if args.tad1d_manifest:
        tad_manifest = args.tad1d_manifest if os.path.isabs(args.tad1d_manifest) else os.path.join(args.manifests, args.tad1d_manifest)
        n3 = cache_tad1d(tad_manifest, uri10, out_tad, args.band_width, args.limit_tad, args.percentile, args.workers)

    tad2d_manifest = args.tad2d_manifest
    n4 = 0
    if tad2d_manifest:
        if not args.domain_bed:
            raise ValueError("--tad2d-manifest 需要配合 --domain-bed")
        if not os.path.isabs(tad2d_manifest):
            tad2d_manifest = os.path.join(args.manifests, tad2d_manifest)
        out_tad2d = os.path.join(args.outdir, 'tad_10kb_2d')
        n4 = cache_tad2d(tad2d_manifest, uri10, out_tad2d, args.domain_bed,
                         args.tad2d_binsize, args.limit_tad2d, args.percentile, args.workers)

    print(f"[cache] loop={n1}, stripe={n2}, tad1d={n3}, tad2d={n4}")


if __name__ == '__main__':
    main()
