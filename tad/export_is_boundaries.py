#!/usr/bin/env python3
"""
Export TAD boundaries from a cooltools insulation TSV to BED.

Typical use (single window 200kb at 10kb resolution):

  python export_is_boundaries.py \
    --tsv /storu/zkyang/AAA_MIL/tad/insul_10kb_w200k.tsv \
    --window-bp 200000 \
    --out /storu/zkyang/AAA_MIL/tad/tad_boundaries_is_10kb.bed

Notes
- If the TSV has an 'is_boundary' column, it is preferred and will be used.
- Otherwise, if 'boundary_strength' exists, you can threshold via
  --quantile 95 (default) or --threshold <float>.
- Excludes Y/MT by default; configurable via --include/--exclude.
- Optionally enforces a minimum spacing between boundaries per chromosome.

Only creates files under /storu/zkyang/AAA_MIL per project rules.
"""

import argparse
from typing import List, Optional
import numpy as np
import pandas as pd


def parse_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_integer_dtype(s):
        return s.astype(int) != 0
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


def select_chroms(df: pd.DataFrame, include: Optional[str], exclude: Optional[str]) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    ccol = cols.get('chrom', None)
    if ccol is None:
        raise ValueError("Input TSV must contain a 'chrom' column")
    names = df[ccol].astype(str)
    if include:
        inc = [x.strip() for x in include.split(',') if x.strip()]
        df = df[names.isin(inc)]
    if exclude:
        exc = [x.strip() for x in exclude.split(',') if x.strip()]
        df = df[~names.isin(exc)]
    return df


def resolve_windowed_columns(df: pd.DataFrame, window_bp: Optional[int]):
    """
    Cooltools may encode window as separate columns with suffixes, e.g.:
      boundary_strength_200000, is_boundary_200000
    or as plain columns plus a 'window' column.
    This function returns (is_boundary_col, boundary_strength_col) names if present.
    """
    cols = set(df.columns)
    # Plain names
    ib_plain = 'is_boundary' if 'is_boundary' in cols else None
    bs_plain = 'boundary_strength' if 'boundary_strength' in cols else None

    # Suffix-based
    suffix = f"_{int(window_bp)}" if window_bp is not None else None
    ib_suf = None
    bs_suf = None
    if suffix is not None:
        cand_ib = f"is_boundary{suffix}"
        cand_bs = f"boundary_strength{suffix}"
        if cand_ib in cols:
            ib_suf = cand_ib
        if cand_bs in cols:
            bs_suf = cand_bs

    # Fallback: pick the only matching suffix columns if window not provided
    if window_bp is None and ib_plain is None and bs_plain is None:
        ib_like = [c for c in cols if c.startswith('is_boundary_')]
        bs_like = [c for c in cols if c.startswith('boundary_strength_')]
        if len(ib_like) == 1:
            ib_suf = ib_like[0]
        if len(bs_like) == 1:
            bs_suf = bs_like[0]

    # Choose priority: suffix match > plain
    ib_col = ib_suf or ib_plain
    bs_col = bs_suf or bs_plain
    return ib_col, bs_col


def pick_boundaries(df: pd.DataFrame,
                    ib_col: Optional[str],
                    bs_col: Optional[str],
                    threshold: Optional[float],
                    quantile: Optional[float]) -> pd.DataFrame:
    # Prefer explicit is_boundary if present
    if ib_col is not None and ib_col in df.columns:
        mask = parse_bool_series(df[ib_col]).astype(bool)
        return df.loc[mask]

    # Else fallback to boundary_strength
    if bs_col is None or bs_col not in df.columns:
        raise ValueError("No 'is_boundary' or 'boundary_strength' column found (including window-suffixed variants)."
                         " Re-run cooltools insulation with boundary calling, or pass the correct --window-bp.")

    bs = pd.to_numeric(df[bs_col], errors='coerce')
    if threshold is not None:
        thr = float(threshold)
    else:
        q = 95.0 if quantile is None else float(quantile)
        thr = float(np.nanpercentile(bs.values, q))
    return df.loc[bs >= thr]


def dedupe_min_distance(bed: pd.DataFrame, min_dist_bp: int) -> pd.DataFrame:
    if min_dist_bp <= 0:
        return bed
    out = []
    for chrom, g in bed.groupby('chrom', sort=False):
        gg = g.sort_values(['start', 'end']).copy()
        kept = []
        last_start = None
        last_end = None
        last_strength = None
        strength_col = 'boundary_strength' if 'boundary_strength' in gg.columns else None
        for _, r in gg.iterrows():
            if last_start is None:
                kept.append(r)
                last_start, last_end = int(r['start']), int(r['end'])
                last_strength = float(r[strength_col]) if strength_col else None
            else:
                if int(r['start']) - last_start < min_dist_bp:
                    # keep stronger if available
                    if strength_col is not None and float(r[strength_col]) > (last_strength or -np.inf):
                        kept[-1] = r
                        last_start, last_end = int(r['start']), int(r['end'])
                        last_strength = float(r[strength_col])
                else:
                    kept.append(r)
                    last_start, last_end = int(r['start']), int(r['end'])
                    last_strength = float(r[strength_col]) if strength_col else None
        out.append(pd.DataFrame(kept))
    return pd.concat(out, axis=0, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Export TAD boundaries from cooltools insulation TSV to BED")
    ap.add_argument('--tsv', required=True, help='cooltools insulation output TSV')
    ap.add_argument('--window-bp', type=int, default=None, help='Select rows for a specific window size in bp (if TSV has a window column)')
    ap.add_argument('--threshold', type=float, default=None, help='Numeric threshold for boundary_strength (used if is_boundary not present)')
    ap.add_argument('--quantile', type=float, default=95.0, help='Percentile on boundary_strength to select boundaries if threshold not given [default: 95]')
    ap.add_argument('--include', type=str, default=None, help='Comma-separated chroms to include (e.g., 1,2,...,22,X)')
    ap.add_argument('--exclude', type=str, default='Y,MT', help='Comma-separated chroms to exclude [default: Y,MT]')
    ap.add_argument('--min-distance-bp', type=int, default=0, help='Minimum spacing between boundaries per chrom, in bp (deduplicate)')
    ap.add_argument('--out', required=True, help='Output BED path')
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep='\t', low_memory=False)
    ib_col, bs_col = resolve_windowed_columns(df, args.window_bp)
    df = select_chroms(df, args.include, args.exclude)
    sel = pick_boundaries(df, ib_col, bs_col, args.threshold, args.quantile)

    # Build a BED-like frame; keep strength if available for optional dedupe
    bed_cols = ['chrom', 'start', 'end']
    if bs_col is not None and bs_col in sel.columns:
        # keep strength for optional dedupe
        bed_cols.append(bs_col)
    bed = sel[bed_cols].copy()

    if args.min_distance_bp > 0:
        bed = dedupe_min_distance(bed, args.min_distance_bp)

    bed[['chrom', 'start', 'end']].to_csv(args.out, sep='\t', header=False, index=False)
    print(f"[export] wrote {args.out} (n={len(bed)})")


if __name__ == '__main__':
    main()
