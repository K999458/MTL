#!/usr/bin/env python3
"""
生成多任务训练数据的 manifests（按 ChatGPT-方案与 MTL.md）：

三路任务：
- Loop@5kb：2D patch（沿主对角），patch=256, center=224, stride=112
- Stripe@10kb：2D patch（沿主对角），patch=320, center=256, stride=128
- TAD@10kb（1D）：对角带展开，B=64, L=1024, stride=512

可选：融合 IS 与 DI 的 TAD 边界（近邻±merge_bp 合并）。

仅在 /storu/zkyang/AAA_MIL 下读写（符合项目约束）。

依赖：cooler, numpy, pandas
"""

import argparse
import os
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import cooler


def resolve_mcool_uri(mcool: str, binsize: int) -> str:
    if '::' in mcool:
        return mcool
    assert mcool.endswith('.mcool'), 'mcool 路径应为 .mcool 文件或 mcool::/resolutions/N'
    return f"{mcool}::/resolutions/{int(binsize)}"


def chrom_names(clr: cooler.Cooler) -> List[str]:
    return [str(c) for c in clr.chromnames]


def chrom_len_bp(clr: cooler.Cooler, chrom: str) -> int:
    return int(clr.chromsizes.loc[chrom])


def diagonal_windows(nbins: int, patch: int, stride: int) -> List[int]:
    """返回沿对角线的起始 bin 列表（x0==y0）。"""
    out = []
    i = 0
    while i + patch <= nbins:
        out.append(i)
        i += stride
    # 确保尾端覆盖
    if not out or out[-1] + patch < nbins:
        out.append(max(0, nbins - patch))
    # 去重
    out = sorted(set(out))
    return out


def generate_loop_manifest(clr5: cooler.Cooler, patch: int, center: int, stride: int) -> List[Dict]:
    mani = []
    binsize = int(clr5.binsize)
    for chrom in chrom_names(clr5):
        nbins = clr5.bins().fetch(chrom).shape[0]
        for x0 in diagonal_windows(nbins, patch, stride):
            mani.append({
                'chrom': chrom,
                'x0': x0,
                'y0': x0,
                'binsize': binsize,
                'patch': patch,
                'center': center,
                'stride': stride
            })
    return mani


def _generate_diag_manifest(clr: cooler.Cooler, patch: int, center: int, stride: int) -> List[Dict]:
    mani = []
    binsize = int(clr.binsize)
    for chrom in chrom_names(clr):
        nbins = clr.bins().fetch(chrom).shape[0]
        for x0 in diagonal_windows(nbins, patch, stride):
            mani.append({
                'chrom': chrom,
                'x0': x0,
                'y0': x0,
                'binsize': binsize,
                'patch': patch,
                'center': center,
                'stride': stride
            })
    return mani


def generate_stripe_manifest(clr10: cooler.Cooler, patch: int, center: int, stride: int) -> List[Dict]:
    return _generate_diag_manifest(clr10, patch, center, stride)


def generate_tad2d_manifest(clr10: cooler.Cooler, patch: int, center: int, stride: int) -> List[Dict]:
    return _generate_diag_manifest(clr10, patch, center, stride)


def generate_tad1d_manifest(clr10: cooler.Cooler, L: int, stride: int) -> List[Dict]:
    mani = []
    for chrom in chrom_names(clr10):
        nbins = clr10.bins().fetch(chrom).shape[0]
        s = 0
        while s + L <= nbins:
            mani.append({'chrom': chrom, 'start_bin': s, 'length': L})
            s += stride
        if not mani or mani[-1]['chrom'] != chrom or mani[-1]['start_bin'] + L < nbins:
            mani.append({'chrom': chrom, 'start_bin': max(0, nbins - L), 'length': L})
    return mani


def read_bed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None, usecols=[0,1,2], names=['chrom','start','end'])
    df['chrom'] = df['chrom'].astype(str)
    df['start'] = pd.to_numeric(df['start'], errors='coerce').astype('Int64')
    df['end']   = pd.to_numeric(df['end'],   errors='coerce').astype('Int64')
    df = df.dropna().reset_index(drop=True)
    df['start'] = df['start'].astype(int)
    df['end']   = df['end'].astype(int)
    return df


def fuse_tad_boundaries(is_bed: str, di_bed: Optional[str], merge_bp: int) -> pd.DataFrame:
    a = read_bed(is_bed)
    if di_bed and os.path.exists(di_bed):
        b = read_bed(di_bed)
        frames = []
        for chrom, ga in a.groupby('chrom', sort=False):
            gb = b[b['chrom']==chrom]
            cand = pd.concat([ga, gb], axis=0, ignore_index=True)
            cand = cand.sort_values(['start','end']).reset_index(drop=True)
            # 近邻合并
            merged = []
            cur = None
            for _, r in cand.iterrows():
                if cur is None:
                    cur = r
                else:
                    if int(r['start']) - int(cur['end']) <= merge_bp:
                        # 合并为覆盖区间
                        cur['end'] = max(int(cur['end']), int(r['end']))
                    else:
                        merged.append(cur)
                        cur = r
            if cur is not None:
                merged.append(cur)
            frames.append(pd.DataFrame(merged))
        out = pd.concat(frames, axis=0, ignore_index=True)
    else:
        out = a
    return out[['chrom','start','end']]


def loop_bedpe_to_jsonl(loop_bedpe: str, out_jsonl: str):
    """把 BEDPE 转为统一 JSONL：{chrom, p1, p2, (soft?), (n_tools?)}"""
    # 读取（兼容带头/不带头）
    try:
        df = pd.read_csv(loop_bedpe, sep='\t', dtype=str)
        if not {'chrom1','start1','end1','chrom2','start2','end2'}.issubset(df.columns):
            raise ValueError('missing columns')
    except Exception:
        cols = ['chrom1','start1','end1','chrom2','start2','end2']
        df = pd.read_csv(loop_bedpe, sep='\t', header=None, usecols=[0,1,2,3,4,5], names=cols, dtype=str)
    # 只保留 cis
    df = df[df['chrom1'] == df['chrom2']].copy()
    # 中心点
    for c in ['start1','end1','start2','end2']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['start1','end1','start2','end2'])
    df['p1'] = ((df['start1'] + df['end1']) // 2).astype(int)
    df['p2'] = ((df['start2'] + df['end2']) // 2).astype(int)
    # 软分数
    soft = None
    ntools = None
    if 'soft_score' in df.columns:
        soft = pd.to_numeric(df['soft_score'], errors='coerce')
    if 'n_tools' in df.columns:
        ntools = pd.to_numeric(df['n_tools'], errors='coerce')
    # 写 JSONL
    with open(out_jsonl, 'w') as f:
        for _, r in df.iterrows():
            item = {'chrom': str(r['chrom1']), 'p1': int(r['p1']), 'p2': int(r['p2'])}
            if soft is not None and pd.notna(r.get('soft_score', np.nan)):
                try:
                    item['soft'] = float(r['soft_score'])
                except Exception:
                    pass
            if ntools is not None and pd.notna(r.get('n_tools', np.nan)):
                try:
                    item['n_tools'] = int(r['n_tools'])
                except Exception:
                    pass
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def stripe_tsv_to_jsonl(tsv: str, out_jsonl: str, one_based_closed: bool=True):
    """把 Stripenn TSV 转为 JSONL：{chrom, x1,x2,y1,y2}（0-based 半开）"""
    cols = ["chr","pos1","pos2","chr2","pos3","pos4","length","width","Mean","maxpixel","pvalue","Stripiness"]
    try:
        df = pd.read_csv(tsv, sep=r"\s+", engine='python')
        if 'chr' not in df.columns:
            df = pd.read_csv(tsv, sep=r"\s+", engine='python', names=cols)
    except Exception:
        df = pd.read_csv(tsv, sep=r"\s+", engine='python', names=cols)
    df = df[df['chr'] == df.get('chr2', df['chr'])].copy()
    # 坐标制转换
    def s0(x):
        v = int(pd.to_numeric(x, errors='coerce'))
        return v-1 if one_based_closed else v
    def e0(x):
        v = int(pd.to_numeric(x, errors='coerce'))
        return v if one_based_closed else v
    df['x1'] = df['pos1'].map(s0); df['x2'] = df['pos2'].map(e0)
    df['y1'] = df['pos3'].map(s0); df['y2'] = df['pos4'].map(e0)
    df = df.dropna(subset=['x1','x2','y1','y2'])
    with open(out_jsonl, 'w') as f:
        for _, r in df.iterrows():
            item = {'chrom': str(r['chr']), 'x1': int(r['x1']), 'x2': int(r['x2']), 'y1': int(r['y1']), 'y2': int(r['y2'])}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict]):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description='生成多任务训练数据的 manifests')
    ap.add_argument('--mcool', required=True, help='.mcool 路径或 mcool::/resolutions/N')
    ap.add_argument('--loop-bedpe', required=True, help='Loop 标注（5kb）BEDPE 路径，用于存在性检查')
    ap.add_argument('--is-bed', required=True, help='IS 边界（10kb）BED 路径')
    ap.add_argument('--di-bed', default=None, help='DI 边界（10kb）BED 路径（可选）')
    ap.add_argument('--stripe-tsv', required=True, help='Stripenn TSV 路径（10kb）')
    ap.add_argument('--outdir', required=True, help='输出目录，如 /storu/zkyang/AAA_MIL/data_generated')
    ap.add_argument('--fuse-tad', action='store_true', help='融合 IS 与 DI 边界')
    ap.add_argument('--tad-merge-bp', type=int, default=20000, help='TAD 边界近邻合并阈值（bp）[默认 20000]')

    # 超参数（可按方案调整）
    ap.add_argument('--loop-patch', type=int, default=256)
    ap.add_argument('--loop-center', type=int, default=224)
    ap.add_argument('--loop-stride', type=int, default=112)
    ap.add_argument('--stripe-patch', type=int, default=320)
    ap.add_argument('--stripe-center', type=int, default=256)
    ap.add_argument('--stripe-stride', type=int, default=128)
    ap.add_argument('--tad-L', type=int, default=1024)
    ap.add_argument('--tad-stride', type=int, default=512)
    ap.add_argument('--tad2d-patch', type=int, default=320, help='TAD 2D patch 大小 [默认 320]')
    ap.add_argument('--tad2d-center', type=int, default=256, help='TAD 2D 中心区域 [默认 256]')
    ap.add_argument('--tad2d-stride', type=int, default=128, help='TAD 2D 滑窗步长 [默认 128]')
    args = ap.parse_args()

    # 解析 cooler
    uri5  = resolve_mcool_uri(args.mcool, 5000)
    uri10 = resolve_mcool_uri(args.mcool, 10000)
    clr5  = cooler.Cooler(uri5)
    clr10 = cooler.Cooler(uri10)
    assert int(clr5.binsize)==5000 and int(clr10.binsize)==10000, '需要 5kb/10kb 两层'

    # 输出目录
    out_labels = os.path.join(args.outdir, 'labels')
    out_manis  = os.path.join(args.outdir, 'manifests')
    ensure_dir(out_labels)
    ensure_dir(out_manis)

    # 1) TAD 边界（融合或直用 IS）
    if args.fuse_tad:
        tad_df = fuse_tad_boundaries(args.is_bed, args.di_bed, args.tad_merge_bp)
    else:
        tad_df = read_bed(args.is_bed)
    tad_bed_out = os.path.join(out_labels, 'tad_boundaries_10kb.bed')
    tad_df.to_csv(tad_bed_out, sep='\t', header=False, index=False)

    # 1.1) Loop/Stripe 标签标准化输出（JSONL）
    loop_jsonl_out = os.path.join(out_labels, 'loop_5kb.jsonl')
    stripe_jsonl_out = os.path.join(out_labels, 'stripe_10kb.jsonl')
    loop_bedpe_to_jsonl(args.loop_bedpe, loop_jsonl_out)
    stripe_tsv_to_jsonl(args.stripe_tsv, stripe_jsonl_out)

    # 2) manifests
    loop_mani   = generate_loop_manifest(clr5,  args.loop_patch,   args.loop_center,   args.loop_stride)
    stripe_mani = generate_stripe_manifest(clr10, args.stripe_patch, args.stripe_center, args.stripe_stride)
    tad_mani    = generate_tad1d_manifest(clr10, args.tad_L, args.tad_stride)
    tad2d_mani  = generate_tad2d_manifest(clr10, args.tad2d_patch, args.tad2d_center, args.tad2d_stride)

    write_jsonl(os.path.join(out_manis, 'loop_5kb.jsonl'),   loop_mani)
    write_jsonl(os.path.join(out_manis, 'stripe_10kb.jsonl'), stripe_mani)
    write_jsonl(os.path.join(out_manis, 'tad_10kb_1d.jsonl'), tad_mani)
    write_jsonl(os.path.join(out_manis, 'tad_10kb_2d.jsonl'), tad2d_mani)

    print('[done] 输出：')
    print('  ', tad_bed_out)
    print('  ', loop_jsonl_out)
    print('  ', stripe_jsonl_out)
    print('  ', os.path.join(out_manis, 'loop_5kb.jsonl'))
    print('  ', os.path.join(out_manis, 'stripe_10kb.jsonl'))
    print('  ', os.path.join(out_manis, 'tad_10kb_1d.jsonl'))
    print('  ', os.path.join(out_manis, 'tad_10kb_2d.jsonl'))


if __name__ == '__main__':
    main()
