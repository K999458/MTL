#!/usr/bin/env python3
"""
Compute Directionality Index (DI) per bin for a cooler/mcool at a given resolution.

Formula (Dixon et al., 2012):
  For each bin i, let A be upstream contacts (i with [i-w, i-1]),
  B be downstream contacts (i with [i+1, i+w]).
  E = (A + B) / 2
  DI = sign(B - A) * ( (A - E)^2 / E + (B - E)^2 / E ), with DI=0 when E==0.

Notes
- Works directly on balanced contact matrix from cooler (e.g. KR/ICE if available).
- Uses sparse CSR per chromosome; sums only near-diagonal windows.
- Outputs bedGraph: chrom  start  end  di

Usage
  python compute_di.py \
    --cool "/storu/zkyang/AAA_MIL/data/12878.mcool::/resolutions/10000" \
    --window-bp 2000000 \
    --out "/storu/zkyang/AAA_MIL/tad/di_10kb_2Mb.bedgraph" \
    --include 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X

Only create/modify files under /storu/zkyang/AAA_MIL per project rules.
"""

import argparse
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import cooler


def parse_chroms(clr: cooler.Cooler, include: Optional[str], exclude: Optional[str]) -> List[str]:
    names = [str(c) for c in clr.chromnames]
    inc = None
    exc = set()
    if include:
        inc = [x.strip() for x in include.split(',') if x.strip()]
    if exclude:
        exc = {x.strip() for x in exclude.split(',') if x.strip()}

    if inc is None:
        chosen = [c for c in names if c not in exc]
    else:
        # Map like 'chr1' to '1' or vice versa if needed
        def map_name(want: str) -> str:
            if want in names:
                return want
            if want.startswith('chr') and want[3:] in names:
                return want[3:]
            if (not want.startswith('chr')) and ('chr' + want) in names:
                return 'chr' + want
            raise KeyError(f"Chrom '{want}' not found in cooler names: {names[:12]} ...")
        chosen = [map_name(c) for c in inc if map_name(c) not in exc]
    return chosen


def compute_di_for_chrom(clr: cooler.Cooler, chrom: str, window_bins: int, balanced: bool=True) -> pd.DataFrame:
    """Compute DI for one chromosome, return bedGraph-like DataFrame."""
    mat = clr.matrix(balance=balanced, sparse=True).fetch(chrom).tocsr()
    # bin-level coordinates
    cstart, cend = clr.extent(chrom)
    # may not be exactly bin counts due to masked bins; use binsize and chromsizes for output bp coords
    binsize = int(clr.binsize)
    n = mat.shape[0]

    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)

    # Iterate rows; sum within left/right windows near diagonal
    for i in range(n):
        row = mat.getrow(i)
        idx = row.indices
        dat = row.data
        # upstream [i-w, i)
        l0 = max(0, i - window_bins)
        l1 = i
        if l0 < l1:
            msk = (idx >= l0) & (idx < l1)
            if msk.any():
                A[i] = float(dat[msk].sum())
        # downstream (i, i+w]
        r0 = i + 1
        r1 = min(n, i + 1 + window_bins)
        if r0 < r1:
            msk = (idx >= r0) & (idx < r1)
            if msk.any():
                B[i] = float(dat[msk].sum())

    E = 0.5 * (A + B)
    di = np.zeros_like(E)
    nz = E > 0
    # Avoid division by zero; apply formula only where E>0
    di[nz] = np.sign(B[nz] - A[nz]) * ( ((A[nz] - E[nz])**2) / E[nz] + ((B[nz] - E[nz])**2) / E[nz] )

    starts = np.arange(0, n, dtype=np.int64) * binsize
    ends = starts + binsize

    df = pd.DataFrame({
        'chrom': chrom,
        'start': starts,
        'end': ends,
        'di': di.astype(np.float32)
    })
    return df


def main():
    ap = argparse.ArgumentParser(description="Compute Directionality Index (DI) per bin from a cooler/mcool.")
    ap.add_argument('--cool', required=True, help='Path to .cool or .mcool::/resolutions/N')
    ap.add_argument('--window-bp', type=int, default=2_000_000, help='Window size around each bin in basepairs (default: 2Mb)')
    ap.add_argument('--include', type=str, default=None, help='Comma-separated chroms to include (e.g., 1,2,...,22,X). If omitted, use all.')
    ap.add_argument('--exclude', type=str, default='Y,MT', help='Comma-separated chroms to exclude (default: Y,MT)')
    ap.add_argument('--balanced', action='store_true', help='Use balanced matrix (KR/ICE) if available')
    ap.add_argument('--out', required=True, help='Output bedGraph path')
    args = ap.parse_args()

    clr = cooler.Cooler(args.cool)
    binsize = int(clr.binsize)
    window_bins = max(1, int(round(args.window_bp / binsize)))

    chroms = parse_chroms(clr, args.include, args.exclude)
    print(f"[DI] cooler={args.cool} | binsize={binsize} | window_bp={args.window_bp} -> {window_bins} bins")
    print(f"[DI] chroms: {chroms}")

    outdfs = []
    for chrom in chroms:
        print(f"[DI] computing {chrom} ...", file=sys.stderr)
        df = compute_di_for_chrom(clr, chrom, window_bins, balanced=args.balanced)
        outdfs.append(df)

    out = pd.concat(outdfs, axis=0, ignore_index=True)
    # bedGraph header is optional; we just write TSV without header for broad compatibility
    out.to_csv(args.out, sep='\t', header=False, index=False)
    print(f"[DI] wrote: {args.out} ({len(out)} rows)")


if __name__ == '__main__':
    main()

