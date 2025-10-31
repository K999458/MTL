#!/usr/bin/env python3
"""
根据 10kb 边界文件与 Hi-C mcool 生成高置信度 TAD 域。

核心思路：
- 对每条染色体的相邻边界进行配对；
- 计算配对区间内 vs. 外的平衡矩阵平均强度与对角均值比值；
- 依据长度与信噪比阈值筛选出可信 TAD；
- 输出 BED（chrom start end score length_bins）。

示例：
python data_generate/infer_tad_domains.py \
  --cool /storu/zkyang/AAA_MIL/data/12878.mcool::/resolutions/10000 \
  --boundaries /storu/zkyang/AAA_MIL/data_generated/labels/tad_boundaries_10kb.bed \
  --out /storu/zkyang/AAA_MIL/data_generated/labels/tad_domains_inferred.bed
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class DomainCandidate:
    chrom: str
    start_bp: int
    end_bp: int
    mean_in: float
    mean_out: float
    diag_in: float
    diag_out: float

    @property
    def bins(self) -> int:
        return (self.end_bp - self.start_bp) // 10000

    def score(self, eps: float = 1e-6) -> float:
        ratio = (self.mean_in + eps) / (self.mean_out + eps)
        diag_ratio = (self.diag_in + eps) / (self.diag_out + eps)
        return float(ratio * diag_ratio)


def load_boundaries(path: str, binsize: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"])
    df["start"] = (df["start"] // binsize) * binsize
    df["end"] = df["start"] + binsize
    df = df.sort_values(["chrom", "start"]).drop_duplicates()
    return df


def fetch_matrix(clr: cooler.Cooler, chrom: str, start_bp: int, end_bp: int) -> np.ndarray:
    region = f"{chrom}:{max(0, start_bp)}-{end_bp}"
    mat = clr.matrix(balance=True, sparse=False).fetch(region)
    return np.asarray(mat, dtype=np.float32)


def compute_candidate(
    clr: cooler.Cooler,
    chrom: str,
    left_bp: int,
    right_bp: int,
    binsize: int,
    flank_bins: int,
) -> DomainCandidate:
    left_bin = left_bp // binsize
    right_bin = right_bp // binsize
    domain_bins = right_bin - left_bin

    fetch_left = max(0, (left_bin - flank_bins) * binsize)
    fetch_right = (right_bin + flank_bins) * binsize
    mat = fetch_matrix(clr, chrom, fetch_left, fetch_right)
    if mat.size == 0:
        raise ValueError("empty matrix")

    s_idx = (left_bp - fetch_left) // binsize
    e_idx = s_idx + domain_bins

    domain_block = mat[s_idx:e_idx, s_idx:e_idx]
    if domain_block.size == 0:
        raise ValueError("empty domain block")

    domain_vals = domain_block[np.isfinite(domain_block)]
    if domain_vals.size == 0:
        raise ValueError("no finite values in domain block")

    # outside区域：取上下左右各 width 条带
    width = max(2, min(flank_bins, mat.shape[0] // 2))
    outside_parts: List[np.ndarray] = []
    # 左侧竖带
    if s_idx > 0:
        left_strip = mat[s_idx:e_idx, max(0, s_idx - width):s_idx]
        outside_parts.append(left_strip)
    # 右侧竖带
    if e_idx < mat.shape[1]:
        right_strip = mat[s_idx:e_idx, e_idx:min(mat.shape[1], e_idx + width)]
        outside_parts.append(right_strip)
    # 上侧横带
    if s_idx > 0:
        top_strip = mat[max(0, s_idx - width):s_idx, s_idx:e_idx]
        outside_parts.append(top_strip)
    # 下侧横带
    if e_idx < mat.shape[0]:
        bottom_strip = mat[e_idx:min(mat.shape[0], e_idx + width), s_idx:e_idx]
        outside_parts.append(bottom_strip)

    if not outside_parts:
        raise ValueError("no outside strips")

    outside_vals = np.concatenate([block.flatten() for block in outside_parts])
    outside_vals = outside_vals[np.isfinite(outside_vals)]
    if outside_vals.size == 0:
        raise ValueError("no finite outside values")

    # 对角统计
    diag_in = np.diag(domain_block)
    diag_in = diag_in[np.isfinite(diag_in)]
    if diag_in.size == 0:
        raise ValueError("no finite diag in domain")
    diag_in_mean = float(np.mean(diag_in))

    diag_out_vals: List[float] = []
    for block in outside_parts:
        diag = np.diag(block)
        diag = diag[np.isfinite(diag)]
        if diag.size:
            diag_out_vals.extend(diag.tolist())
    if not diag_out_vals:
        diag_out_vals = outside_vals.tolist()

    return DomainCandidate(
        chrom=chrom,
        start_bp=left_bp,
        end_bp=right_bp,
        mean_in=float(np.mean(domain_vals)),
        mean_out=float(np.mean(outside_vals)),
        diag_in=diag_in_mean,
        diag_out=float(np.mean(diag_out_vals)),
    )


def infer_domains(
    clr: cooler.Cooler,
    boundaries: pd.DataFrame,
    binsize: int,
    min_bins: int,
    max_bins: int,
    flank_bins: int,
    score_threshold: float,
    diag_threshold: float,
) -> List[Tuple[str, int, int, float, int]]:
    results: List[Tuple[str, int, int, float, int]] = []
    for chrom, group in tqdm(boundaries.groupby("chrom"), desc="chromosomes"):
        starts = group["start"].to_numpy(dtype=np.int64)
        if starts.size < 2:
            continue
        for i in range(starts.size - 1):
            left = int(starts[i])
            right = int(starts[i + 1])
            bins = (right - left) // binsize
            if bins < min_bins or bins > max_bins:
                continue
            try:
                cand = compute_candidate(clr, chrom, left, right, binsize, flank_bins)
            except ValueError:
                continue
            score = cand.score()
            if cand.diag_in <= 0 or cand.diag_out <= 0:
                continue
            diag_ratio = cand.diag_in / cand.diag_out
            if score >= score_threshold and diag_ratio >= diag_threshold:
                results.append((cand.chrom, cand.start_bp, cand.end_bp, score, cand.bins))
    return results


def main():
    ap = argparse.ArgumentParser(description="Infer TAD domains from boundary BED + mcool")
    ap.add_argument("--cool", required=True, help=".cool 或 .mcool::/resolutions/10000 路径")
    ap.add_argument("--boundaries", required=True, help="10kb 边界 BED")
    ap.add_argument("--binsize", type=int, default=10000, help="边界对齐的 bin 大小 [默认: 10000]")
    ap.add_argument("--min-bins", type=int, default=4, help="最小域宽度 (bin)")
    ap.add_argument("--max-bins", type=int, default=400, help="最大域宽度 (bin)")
    ap.add_argument("--flank-bins", type=int, default=20, help="计算外部均值用的条带宽度 (bin)")
    ap.add_argument("--score-threshold", type=float, default=1.2, help="域内/域外+对角得分阈值")
    ap.add_argument("--diag-threshold", type=float, default=1.05, help="对角均值比最小值")
    ap.add_argument("--out", required=True, help="输出 BED 路径")
    ap.add_argument("--score-out", help="可选，输出包含评分的 TSV 路径")
    args = ap.parse_args()

    clr = cooler.Cooler(args.cool)
    boundaries = load_boundaries(args.boundaries, args.binsize)

    domains = infer_domains(
        clr=clr,
        boundaries=boundaries,
        binsize=args.binsize,
        min_bins=args.min_bins,
        max_bins=args.max_bins,
        flank_bins=args.flank_bins,
        score_threshold=args.score_threshold,
        diag_threshold=args.diag_threshold,
    )

    if not domains:
        print("No domains inferred;请调低阈值或检查数据。")
        return

    df = pd.DataFrame(domains, columns=["chrom", "start", "end", "score", "bins"])
    df_sorted = df.sort_values(["chrom", "start"])
    df_sorted[["chrom", "start", "end"]].to_csv(args.out, sep="\t", header=False, index=False)
    print(f"[infer] saved {len(df_sorted)} domains to {args.out}")

    if args.score_out:
        df_sorted.to_csv(args.score_out, sep="\t", header=True, index=False)
        print(f"[infer] score detail saved to {args.score_out}")


if __name__ == "__main__":
    main()
