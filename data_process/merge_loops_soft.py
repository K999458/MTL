#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

BEDPE_COLS = ["chrom1","start1","end1","chrom2","start2","end2"]

# ---------- 染色体命名与列清洗 ----------
def normalize_chrom_name(s, style="ucsc"):
    s = str(s)
    if style == "ucsc":
        if s.startswith("chr"):
            out = s
        else:
            core = s.upper()
            if core == "MT": core = "M"
            out = "chr" + core
        if out in ("chrMT", "chrMt"): out = "chrM"
        return out
    else:
        if s.startswith("chr"):
            core = s[3:]
            if core.upper() == "MT": core = "M"
            return core
        return s

def normalize_chroms_inplace(df, style="ucsc"):
    df["chrom1"] = df["chrom1"].map(lambda x: normalize_chrom_name(x, style))
    df["chrom2"] = df["chrom2"].map(lambda x: normalize_chrom_name(x, style))
    # 规范 anchor 顺序：先按 chr 成对排序，再在同 chr 内保证 start1<=start2
    swap_pair = df["chrom1"] > df["chrom2"]
    for a,b in [("chrom1","chrom2"),("start1","start2"),("end1","end2")]:
        tmp = df.loc[swap_pair, a].copy()
        df.loc[swap_pair, a] = df.loc[swap_pair, b].values
        df.loc[swap_pair, b] = tmp.values
    swap_same = (df["chrom1"]==df["chrom2"]) & (df["start1"]>df["start2"])
    for a,b in [("start1","start2"),("end1","end2")]:
        tmp = df.loc[swap_same, a].copy()
        df.loc[swap_same, a] = df.loc[swap_same, b].values
        df.loc[swap_same, b] = tmp.values

def _to_bedpe_df(df_like):
    df = df_like.copy()
    df = df.iloc[:, :6].copy()
    df.columns = BEDPE_COLS
    # 强制坐标为整数
    for col in ["start1","end1","start2","end2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    # 确保 cis 情况下 start1<=start2
    swap = (df["chrom1"]==df["chrom2"]) & (df["start1"]>df["start2"])
    for a,b in [("chrom1","chrom2"),("start1","start2"),("end1","end2")]:
        tmp = df.loc[swap, a].copy()
        df.loc[swap, a] = df.loc[swap, b].values
        df.loc[swap, b] = tmp.values
    return df

# ---------- 各工具读取 ----------
def read_chromosight(path):
    try:
        df = pd.read_csv(path, sep="\t", comment="#", header=0)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)
    df6 = _to_bedpe_df(df)
    score = None
    for cand in ["score","pearson","corr"]:
        if isinstance(df.columns, pd.Index) and cand in df.columns:
            score = pd.to_numeric(df[cand], errors="coerce").clip(0,1).fillna(0.6).values
            break
    if score is None: score = np.full(len(df6), 0.6)
    df6["score_chromosight"] = score
    df6["src"] = "chromosight"
    normalize_chroms_inplace(df6, style="ucsc")
    return df6

def read_peakachu(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df6 = _to_bedpe_df(df)
    if df.shape[1] >= 7:
        prob = pd.to_numeric(df.iloc[:,6], errors="coerce").clip(0,1).fillna(0.7).values
    else:
        prob = np.full(len(df6), 0.7)
    df6["score_peakachu"] = prob
    df6["src"] = "peakachu"
    normalize_chroms_inplace(df6, style="ucsc")
    return df6

def read_mustache(path):
    try:
        df = pd.read_csv(path, sep="\t", comment="#", header=0)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)
    df6 = _to_bedpe_df(df)
    sc = None
    for cand in ["qvalue","q_value","qval","pvalue","p_value","score"]:
        if isinstance(df.columns, pd.Index) and cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce")
            if "q" in cand:
                sc = (1 - s.clip(0,1)).fillna(0.7).values
            else:
                s = s.replace([np.inf, -np.inf], np.nan)
                sc = ((s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-9)).fillna(0.7).values
            break
    if sc is None: sc = np.full(len(df6), 0.75)
    df6["score_mustache"] = sc
    df6["src"] = "mustache"
    normalize_chroms_inplace(df6, style="ucsc")
    return df6

def read_hiccups(path):
    try:
        df = pd.read_csv(path, sep="\t", comment="#", header=0)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)
    df6 = _to_bedpe_df(df)
    cand_fdr = []
    if isinstance(df.columns, pd.Index):
        cand_fdr = [c for c in df.columns if isinstance(c,str) and c.lower().startswith("fdr")]
    if cand_fdr:
        f = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in cand_fdr], axis=1)
        sc = (1 - f.min(axis=1).clip(0,1)).fillna(0.9).values
    else:
        sc = np.full(len(df6), 0.95)
    df6["score_hiccups"] = sc
    df6["src"] = "hiccups"
    normalize_chroms_inplace(df6, style="ucsc")
    return df6

# ---------- 融合 ----------
def bins_and_feats(df, binsize):
    a = ((df["start1"] + df["end1"]) * 0.5 / binsize).round().astype(int)
    b = ((df["start2"] + df["end2"]) * 0.5 / binsize).round().astype(int)
    return a, b

def soft_merge(inputs, binsize=5000, eps_bins=1.0, weights=None, min_tools=1):
    if weights is None:
        weights = {"hiccups":1.00, "mustache":0.90, "peakachu":0.85, "chromosight":0.75}
    dfs = []
    if inputs.get("hiccups"):     dfs.append(read_hiccups(inputs["hiccups"]))
    if inputs.get("mustache"):    dfs.append(read_mustache(inputs["mustache"]))
    if inputs.get("peakachu"):    dfs.append(read_peakachu(inputs["peakachu"]))
    if inputs.get("chromosight"): dfs.append(read_chromosight(inputs["chromosight"]))
    if not dfs:
        raise ValueError("No input files provided.")
    all_df = pd.concat(dfs, ignore_index=True)

    out_rows = []
    # 重要：按 (chrom1, chrom2) 配对分组（cis/trans 都支持）
    for (chromA, chromB), sub in all_df.groupby(["chrom1","chrom2"], sort=True):
        if sub.empty: 
            continue
        a, b = bins_and_feats(sub, binsize)
        X = np.vstack([a.values, b.values]).T
        db = DBSCAN(eps=float(eps_bins), min_samples=1).fit(X)
        sub = sub.copy()
        sub["cluster"] = db.labels_

        for clu, g in sub.groupby("cluster"):
            sources = list(g["src"])
            uniq_sources = sorted(set(sources))
            n_tools = len(uniq_sources)
            n_calls = len(g)
            if n_tools < min_tools:
                continue

            # 软分数：1 - Π(1 - w*s)
            acc = 1.0
            wsum = 0.0; ax=0.0; bx=0.0
            for _, row in g.iterrows():
                src = row["src"]
                s = float(row.get(f"score_{src}", 0.7))
                w = float(weights.get(src, 0.8))
                acc *= (1.0 - w*s)
                ai = 0.5*(row["start1"]+row["end1"]) / binsize
                bi = 0.5*(row["start2"]+row["end2"]) / binsize
                wsum += (w*s); ax += (w*s)*ai; bx += (w*s)*bi
            soft = 1.0 - acc
            if wsum == 0:
                a_bar = a[g.index].median(); b_bar = b[g.index].median()
            else:
                a_bar = ax/wsum; b_bar = bx/wsum
            a_bp = int(round(a_bar)*binsize); b_bp = int(round(b_bar)*binsize)

            out_rows.append({
                "chrom1": chromA, "start1": a_bp, "end1": a_bp+binsize,
                "chrom2": chromB, "start2": b_bp, "end2": b_bp+binsize,
                "soft_score": round(float(soft), 6),
                "n_calls": n_calls,
                "n_tools": n_tools,
                "supporters": ",".join(uniq_sources),
            })

    out = pd.DataFrame(out_rows, columns=BEDPE_COLS+["soft_score","n_calls","n_tools","supporters"])
    if not out.empty:
        out = out.sort_values(["chrom1","start1","start2","soft_score"], ascending=[True,True,True,False])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hiccups", required=False)
    ap.add_argument("--chromosight", required=False)
    ap.add_argument("--mustache", required=False)
    ap.add_argument("--peakachu", required=False)
    ap.add_argument("--binsize", type=int, required=True)
    ap.add_argument("--eps_bins", type=float, default=1.0, help="DBSCAN eps (in bins)")
    ap.add_argument("--min_tools", type=int, default=1, help="≥K 不同 caller 命中才保留")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    inputs = {"hiccups":args.hiccups, "chromosight":args.chromosight,
              "mustache":args.mustache, "peakachu":args.peakachu}
    df = soft_merge(inputs, binsize=args.binsize, eps_bins=args.eps_bins, min_tools=args.min_tools)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"[ok] wrote {args.out} with {len(df)} loops (min_tools={args.min_tools})")

if __name__ == "__main__":
    main()
