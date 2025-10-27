#!/usr/bin/env python3
"""
Convert Directionality Index (DI) bedGraph to TAD boundary BED.

Input DI bedGraph format (per bin):
  chrom  start  end  di

Method
  1) Per chromosome, smooth DI with a rolling window (odd number of bins).
  2) Find sign-change points of the smoothed DI (zero-crossings).
  3) For each sign-change at (i-1, i), pick the boundary bin near the crossing
     as the one with smaller |DI_smooth| within a small search radius.
  4) Filter weak crossings by amplitude threshold: max(|DI[i-1]|, |DI[i]|)
     should exceed a threshold defined by --amp-threshold or by the global
     quantile (--amp-quantile) over all candidates.
  5) Enforce a minimal spacing between boundaries per chromosome and keep the
     stronger boundary when conflicts occur.

Usage example
  python di_to_boundaries.py \
    --di /storu/zkyang/AAA_MIL/tad/di_10kb_2Mb.bedgraph \
    --smooth-win-bins 11 \
    --amp-quantile 0.7 \
    --min-distance-bp 20000 \
    --out /storu/zkyang/AAA_MIL/tad/tad_boundaries_di_10kb.bed

Notes
  - Only creates files within /storu/zkyang/AAA_MIL per project rules.
  - Dependencies: numpy, pandas
"""

import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def read_di_bedgraph(path: str) -> pd.DataFrame:
    names = ['chrom', 'start', 'end', 'di']
    try:
        df = pd.read_csv(path, sep='\t', low_memory=False)
        if not set(['chrom', 'start', 'end']).issubset(df.columns):
            # likely no header
            df = pd.read_csv(path, sep='\t', header=None, names=names, low_memory=False)
        else:
            # align column naming if 'di' has a different name (e.g., 'score')
            if 'di' not in df.columns:
                # pick the last column as di if not found
                last = df.columns[-1]
                df = df.rename(columns={last: 'di'})
    except Exception:
        df = pd.read_csv(path, sep='\t', header=None, names=names, low_memory=False)

    # enforce dtypes
    df = df[['chrom', 'start', 'end', 'di']].copy()
    df['chrom'] = df['chrom'].astype(str)
    df['start'] = pd.to_numeric(df['start'], errors='coerce').astype('Int64')
    df['end'] = pd.to_numeric(df['end'], errors='coerce').astype('Int64')
    df['di'] = pd.to_numeric(df['di'], errors='coerce')
    df = df.dropna(subset=['chrom', 'start', 'end', 'di']).reset_index(drop=True)
    # cast back to int
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    return df


def rolling_smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or win % 2 == 0:
        return x.copy()
    # centered rolling mean using convolution
    k = np.ones(win, dtype=float)
    y = np.convolve(x, k, mode='same') / win
    return y


def find_zero_crossings(di_s: np.ndarray) -> List[int]:
    # sign: -1, 0, +1; treat zeros by forward-fill of sign to avoid spurious toggles
    s = np.sign(di_s)
    # replace zeros with previous non-zero sign where possible, else next
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i-1]
    for i in range(len(s)-2, -1, -1):
        if s[i] == 0:
            s[i] = s[i+1]
    s[s == 0] = 1  # default remain
    # zero-crossings where sign flips between consecutive bins
    idx = np.where(s[1:] * s[:-1] < 0)[0] + 1  # positions i where (i-1)->i flips
    return idx.tolist()


def pick_boundary_index(di_s: np.ndarray, i: int, search_radius: int) -> int:
    a = max(0, i - search_radius)
    b = min(len(di_s), i + search_radius + 1)
    # choose index with minimal absolute DI within window
    local = np.abs(di_s[a:b])
    j = int(np.argmin(local))
    return a + j


def di_to_boundaries(df: pd.DataFrame,
                     smooth_win_bins: int,
                     search_radius_bins: int,
                     amp_threshold: Optional[float],
                     amp_quantile: Optional[float],
                     min_distance_bp: int) -> pd.DataFrame:
    out_rows = []
    amp_all = []  # collect amplitudes to compute quantile if needed
    # First pass to collect candidates and amplitudes
    per_chrom_candidates = {}
    for chrom, g in df.groupby('chrom', sort=False):
        g = g.sort_values('start').reset_index(drop=True)
        di = g['di'].to_numpy(dtype=float)
        di_s = rolling_smooth(di, smooth_win_bins)
        flips = find_zero_crossings(di_s)
        if not flips:
            continue
        cands = []
        for i in flips:
            # amplitude around crossing = max(|di_s[i-1]|, |di_s[i]|)
            if i <= 0 or i >= len(di_s):
                continue
            amp = float(max(abs(di_s[i-1]), abs(di_s[i])))
            j = pick_boundary_index(di_s, i, search_radius_bins)
            # record candidate
            cands.append((j, amp))
            amp_all.append(amp)
        per_chrom_candidates[chrom] = (g, di_s, cands)

    # Decide threshold
    thr = None
    if amp_threshold is not None:
        thr = float(amp_threshold)
    elif amp_quantile is not None and len(amp_all) > 0:
        q = max(0.0, min(1.0, float(amp_quantile)))
        thr = float(np.nanquantile(np.asarray(amp_all, dtype=float), q))
    else:
        thr = 0.0

    # Second pass: filter by threshold and enforce minimal distance
    for chrom, (g, di_s, cands) in per_chrom_candidates.items():
        if not cands:
            continue
        # Build candidate table with coordinates
        binsize = int(g['end'].iloc[0] - g['start'].iloc[0])
        rows = []
        for j, amp in cands:
            if thr is not None and amp < thr:
                continue
            start = int(g['start'].iloc[j])
            end = int(g['end'].iloc[j])
            rows.append((chrom, start, end, float(amp)))
        if not rows:
            continue
        cand_df = pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'amp'])
        # Enforce minimal spacing per chrom
        cand_df = cand_df.sort_values(['start', 'end']).reset_index(drop=True)
        kept = []
        last_start = None
        last_amp = None
        for _, r in cand_df.iterrows():
            if last_start is None:
                kept.append(r)
                last_start = int(r['start'])
                last_amp = float(r['amp'])
            else:
                if int(r['start']) - last_start < int(min_distance_bp):
                    # keep stronger amplitude
                    if float(r['amp']) > last_amp:
                        kept[-1] = r
                        last_start = int(r['start'])
                        last_amp = float(r['amp'])
                else:
                    kept.append(r)
                    last_start = int(r['start'])
                    last_amp = float(r['amp'])
        out_rows.extend(kept)

    if not out_rows:
        return pd.DataFrame(columns=['chrom', 'start', 'end'])
    out = pd.DataFrame(out_rows)
    # Return as BED (without score), but keep amp for optional downstream use
    return out[['chrom', 'start', 'end', 'amp']]


def main():
    ap = argparse.ArgumentParser(description='Convert DI bedGraph to boundary BED by zero-crossings with smoothing')
    ap.add_argument('--di', required=True, help='Input DI bedGraph (chrom start end di)')
    ap.add_argument('--smooth-win-bins', type=int, default=11, help='Centered smoothing window in bins (odd number, default: 11)')
    ap.add_argument('--search-radius-bins', type=int, default=3, help='Radius in bins to pick minimal |DI| around crossing (default: 3)')
    ap.add_argument('--amp-threshold', type=float, default=None, help='Absolute amplitude threshold on max(|DI_left|,|DI_right|)')
    ap.add_argument('--amp-quantile', type=float, default=0.7, help='Global quantile of amplitude to keep (0..1). Ignored if --amp-threshold given [default: 0.7]')
    ap.add_argument('--min-distance-bp', type=int, default=20000, help='Minimal spacing between boundaries per chromosome in bp [default: 20000]')
    ap.add_argument('--include', type=str, default=None, help='Comma-separated chroms to include (e.g., 1,2,...,22,X)')
    ap.add_argument('--exclude', type=str, default='Y,MT', help='Comma-separated chroms to exclude [default: Y,MT]')
    ap.add_argument('--out', required=True, help='Output BED path')
    args = ap.parse_args()

    df = read_di_bedgraph(args.di)
    # filter chromosomes
    if args.include:
        inc = {c.strip() for c in args.include.split(',') if c.strip()}
        df = df[df['chrom'].isin(inc)]
    if args.exclude:
        exc = {c.strip() for c in args.exclude.split(',') if c.strip()}
        df = df[~df['chrom'].isin(exc)]

    out = di_to_boundaries(df,
                           smooth_win_bins=int(args.smooth_win_bins),
                           search_radius_bins=int(args.search_radius_bins),
                           amp_threshold=args.amp_threshold,
                           amp_quantile=args.amp_quantile,
                           min_distance_bp=int(args.min_distance_bp))

    # Write BED (3 cols). If you want a 4th score, change columns below.
    out[['chrom', 'start', 'end']].to_csv(args.out, sep='\t', header=False, index=False)
    print(f"[di->bed] wrote {args.out} (n={len(out)})")


if __name__ == '__main__':
    main()
