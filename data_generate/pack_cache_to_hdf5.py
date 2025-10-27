#!/usr/bin/env python3
"""
把已缓存的小 .npz 切片（loop/stripe/tad1d）合并为单一 HDF5（按任务分别一个文件）。

优势：
- 单文件、可分块（chunk）读写，适合顺序/随机批量读取；
- 压缩（gzip/lzf） + 可附带字符串/元数据；
- 避免 npz 大文件一次性解压的内存开销问题。

输入：AAA_MIL/data_generated/cache/{loop_5kb,stripe_10kb,tad_10kb_1d}
输出：AAA_MIL/data_generated/packs/{loop_5kb.h5,stripe_10kb.h5,tad_10kb_1d.h5}
"""

import argparse, os, glob
import numpy as np
import h5py


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pack_loop(cache_dir: str, out_h5: str, compression: str='gzip'):
    files = sorted(glob.glob(os.path.join(cache_dir, 'loop_5kb', 'loop_*.npz')))
    if not files:
        print('[loop] no files found, skip')
        return
    # probe
    z = np.load(files[0])
    X0 = z['X']
    H, W = X0.shape
    binsize = int(z['binsize'])
    z.close()
    N = len(files)
    with h5py.File(out_h5, 'w') as f:
        dX = f.create_dataset('X', shape=(N, H, W), dtype='f4', chunks=(1, H, W), compression=compression)
        chrom = f.create_dataset('chrom', shape=(N,), dtype=h5py.string_dtype())
        x0 = f.create_dataset('x0', shape=(N,), dtype='i8')
        patch = f.create_dataset('patch', shape=(N,), dtype='i4')
        f.attrs['binsize'] = binsize
        for i, p in enumerate(files):
            z = np.load(p)
            dX[i] = z['X']
            chrom[i] = str(z['chrom'])
            x0[i] = int(z['x0'])
            patch[i] = int(z['patch'])
            z.close()
    print(f'[loop] packed {N} -> {out_h5}')


def pack_stripe(cache_dir: str, out_h5: str, compression: str='gzip'):
    files = sorted(glob.glob(os.path.join(cache_dir, 'stripe_10kb', 'stripe_*.npz')))
    if not files:
        print('[stripe] no files found, skip')
        return
    z = np.load(files[0])
    X0 = z['X']
    H, W = X0.shape
    binsize = int(z['binsize'])
    z.close()
    N = len(files)
    with h5py.File(out_h5, 'w') as f:
        dX = f.create_dataset('X', shape=(N, H, W), dtype='f4', chunks=(1, H, W), compression=compression)
        chrom = f.create_dataset('chrom', shape=(N,), dtype=h5py.string_dtype())
        x0 = f.create_dataset('x0', shape=(N,), dtype='i8')
        patch = f.create_dataset('patch', shape=(N,), dtype='i4')
        f.attrs['binsize'] = binsize
        for i, p in enumerate(files):
            z = np.load(p)
            dX[i] = z['X']
            chrom[i] = str(z['chrom'])
            x0[i] = int(z['x0'])
            patch[i] = int(z['patch'])
            z.close()
    print(f'[stripe] packed {N} -> {out_h5}')


def pack_tad(cache_dir: str, out_h5: str, compression: str='gzip'):
    files = sorted(glob.glob(os.path.join(cache_dir, 'tad_10kb_1d', 'tad1d_*.npz')))
    if not files:
        print('[tad1d] no files found, skip')
        return
    z = np.load(files[0])
    band0 = z['band']
    B, L = band0.shape
    binsize = int(z['binsize'])
    z.close()
    N = len(files)
    with h5py.File(out_h5, 'w') as f:
        dB = f.create_dataset('band', shape=(N, B, L), dtype='f4', chunks=(1, B, L), compression=compression)
        chrom = f.create_dataset('chrom', shape=(N,), dtype=h5py.string_dtype())
        start_bin = f.create_dataset('start_bin', shape=(N,), dtype='i8')
        meta_B = f.create_dataset('band_width', data=np.full((N,), B, dtype='i4'))
        meta_L = f.create_dataset('length', data=np.full((N,), L, dtype='i4'))
        f.attrs['binsize'] = binsize
        for i, p in enumerate(files):
            z = np.load(p)
            dB[i] = z['band']
            chrom[i] = str(z['chrom'])
            start_bin[i] = int(z['start_bin'])
            z.close()
    print(f'[tad1d] packed {N} -> {out_h5}')


def main():
    ap = argparse.ArgumentParser(description='合并 cache .npz 为单一 HDF5（按任务）')
    ap.add_argument('--cache', required=True, help='缓存根目录 e.g. AAA_MIL/data_generated/cache')
    ap.add_argument('--outdir', required=True, help='输出目录 e.g. AAA_MIL/data_generated/packs')
    ap.add_argument('--compression', default='gzip', choices=['gzip','lzf','None'], help='压缩算法')
    args = ap.parse_args()
    ensure_dir(args.outdir)
    comp = None if args.compression=='None' else args.compression
    pack_loop(args.cache, os.path.join(args.outdir, 'loop_5kb.h5'), comp)
    pack_stripe(args.cache, os.path.join(args.outdir, 'stripe_10kb.h5'), comp)
    pack_tad(args.cache, os.path.join(args.outdir, 'tad_10kb_1d.h5'), comp)


if __name__ == '__main__':
    main()

