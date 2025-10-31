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
from typing import List, Optional, Tuple
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
    elif window_bp is not None and bs_suf is None and bs_plain is None:
        bs_like = [c for c in cols if c.startswith('boundary_strength_')]
        if bs_like:
            bs_suf = sorted(bs_like)[0]
        ib_like = [c for c in cols if c.startswith('is_boundary_')]
        if ib_suf is None and ib_plain is None and ib_like:
            ib_suf = sorted(ib_like)[0]

    # Choose priority: suffix match > plain
    ib_col = ib_suf or ib_plain
    bs_col = bs_suf or bs_plain
    return ib_col, bs_col


def pick_boundaries(df: pd.DataFrame,
                    ib_col: Optional[str],
                    bs_col: Optional[str],
                    threshold: Optional[float],
                    quantile: Optional[float],
                    force_strength: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, float, Optional[str]]:
    if bs_col is None or bs_col not in df.columns:
        bs_like = [c for c in df.columns if c.startswith('boundary_strength')]
        if bs_like:
            bs_col = sorted(bs_like)[0]
        elif ib_col is not None and ib_col in df.columns:
            mask = parse_bool_series(df[ib_col]).astype(bool)
            return df.loc[mask], df.loc[~mask], float('nan'), bs_col
        else:
            raise ValueError("No 'is_boundary' or 'boundary_strength' column found (including window-suffixed variants)."
                             " Re-run cooltools insulation with boundary calling, or pass the correct --window-bp.")

    bs = pd.to_numeric(df[bs_col], errors='coerce')
    if threshold is not None:
        thr = float(threshold)
    else:
        q = 95.0 if quantile is None else float(quantile)
        thr = float(np.nanpercentile(bs.values, q))

    pos_mask = bs >= thr
    pos_df = df.loc[pos_mask]
    neg_df = df.loc[~pos_mask]
    return pos_df, neg_df, thr, bs_col


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
    ap.add_argument('--out', required=True, help='Output BED path (positive boundaries)')
    ap.add_argument('--neg-out', default='', help='Optional BED path for negative boundaries (below threshold)')
    ap.add_argument('--seed', type=int, default=1337, help='Random seed for negative sampling balance')
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep='\t', low_memory=False)
    ib_col, bs_col = resolve_windowed_columns(df, args.window_bp)
    df = select_chroms(df, args.include, args.exclude)
    force_strength = args.threshold is not None
    pos_df, neg_df, thr, bs_col = pick_boundaries(df, ib_col, bs_col, args.threshold, args.quantile, force_strength=force_strength)

    bed_cols = ['chrom', 'start', 'end']
    if bs_col is not None:
        bed_cols.append(bs_col)

    pos_bed = pos_df[bed_cols].copy()
    if args.min_distance_bp > 0:
        pos_bed = dedupe_min_distance(pos_bed, args.min_distance_bp)

    rng = np.random.default_rng(args.seed)
    if args.neg_out:
        if len(pos_bed) > 0 and bs_col is not None and len(neg_df) > 0 and np.isfinite(thr):
            total_neg = len(pos_bed)
            thr_low = float(thr) / 2.0
            neg_bs_full = pd.to_numeric(neg_df[bs_col], errors='coerce')

            def sample_df(dataframe: pd.DataFrame, count: int) -> pd.DataFrame:
                if len(dataframe) <= count:
                    return dataframe
                idx = rng.choice(dataframe.index, size=count, replace=False)
                return dataframe.loc[idx]

            mid_df = neg_df[(neg_bs_full >= thr_low) & (neg_bs_full < thr)]
            low_df = neg_df[neg_bs_full < thr_low]
            remaining_df = neg_df.drop(mid_df.index.union(low_df.index))

            target_mid = int(round(total_neg * 0.8))
            mid_pick = sample_df(mid_df, min(target_mid, len(mid_df)))

            remaining_target = total_neg - len(mid_pick)
            target_low = max(int(round(total_neg * 0.2)), remaining_target)
            low_pick = sample_df(low_df, min(target_low, len(low_df)))

            neg_selected = pd.concat([mid_pick, low_pick], axis=0)
            remaining_needed = total_neg - len(neg_selected)
            if remaining_needed > 0:
                pool = neg_df.drop(neg_selected.index)
                extra = sample_df(pool, min(remaining_needed, len(pool)))
                neg_selected = pd.concat([neg_selected, extra], axis=0)

            neg_selected = neg_selected.head(total_neg)
        else:
            neg_selected = neg_df.head(len(pos_bed)) if len(pos_bed) > 0 else neg_df.iloc[0:0]

        neg_bed = neg_selected[bed_cols].copy()
    else:
        neg_bed = neg_df[bed_cols].copy()

    pos_bed[['chrom', 'start', 'end']].to_csv(args.out, sep='\t', header=False, index=False)
    print(f"[export] positives -> {args.out} (n={len(pos_bed)})")

    if args.neg_out:
        neg_bed[['chrom', 'start', 'end']].to_csv(args.neg_out, sep='\t', header=False, index=False)
        print(f"[export] negatives -> {args.neg_out} (n={len(neg_bed)})")


if __name__ == '__main__':
    main()
