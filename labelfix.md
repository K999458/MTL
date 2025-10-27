首先你之前生成的标签文件应该是有问题的  并没有把 我们需要训练的loop tad 和stripe的位置勾选出来
在此处我会给你我绘制 可视化的代码 你根据代码可以推断出我们的标签区域应该是可视化区域中 标注出来的区域

### 任务list
你现在需要做的工作是复查代码看看 哪里和我说的不同你需要修改并且注意同步训练代码 和预测代码 
架构上面哪里有问题你也需要同步修改 
修改完成以后告诉我结果并且把修改 的写入 MTL.md 文件中
然后你还要告诉我如何重新生成数据集 和标签

### 任务细节

### loop修改细节
首先是loop 我们用以下代码 可视化 标签文件中的第i个loop
以及其这张图片上的其他loop
# 以下是代码
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cooler
from pathlib import Path

# ===== 需要按你的环境修改的路径 =====
# 方案A：.mcool 基础路径 + 分辨率（推荐）
MCOOL_BASE = "/storu/zkyang/AAA_MIL/data/12878.mcool"
RESOLUTION = 10000  # 可改：5000 / 10000 / 25000 等

# 方案B：直接给 .cool 或 .mcool::/resolutions/5000 的完整路径（留空则使用方案A）
COOL_PATH = ""  # 例如："/path/to/xxx.mcool::/resolutions/10000"

# BEDPE 路径（至少包含 chr1 x1 x2 chr2 y1 y2 这6列）
BEDPE_PATH = "/storu/zkyang/AAA_MIL/data/loops_ground_truth/hiccups_loop_5k_change.bedpe"

# 选择第几条 loop（从0开始）
INDEX = 10

# 可视化窗口大小（单位：bins），与 RESOLUTION 一起决定窗口大小（默认 224 bins）
WINDOW_BINS = 224

# 是否使用平衡矩阵（通常 True）
BALANCE = False
def build_cool_path(mcool_base: str, resolution: int, cool_path_override: str = "") -> str:
    """
    返回 cooler 路径：
    - 若 cool_path_override 非空，直接返回（支持 .cool 或 .mcool::/resolutions/xxx）
    - 否则用 mcool_base + 分辨率 构造 .mcool::/resolutions/xxx
    """
    if cool_path_override and "::" in cool_path_override:
        return cool_path_override
    if cool_path_override and cool_path_override.endswith(".cool"):
        return cool_path_override
    assert mcool_base.endswith(".mcool"), "MCOOL_BASE 必须是 .mcool 文件"
    return f"{mcool_base}::/resolutions/{resolution}"

def list_mcool_resolutions(mcool_base: str):
    """列出 .mcool 中的可用分辨率（整数）"""
    import h5py
    res = []
    with h5py.File(mcool_base, "r") as f:
        if "/resolutions" in f:
            for k in f["/resolutions"].keys():
                try:
                    res.append(int(k))
                except Exception:
                    pass
    return sorted(res)

def read_bedpe(loop_file: str) -> pd.DataFrame:
    """
    读取 BEDPE（带或不带表头都行），最终保证有 chr1 x1 x2 chr2 y1 y2 六列
    """
    try:
        df = pd.read_csv(loop_file, sep="\t", comment="#", dtype=str)
        need = {"chr1","x1","x2","chr2","y1","y2"}
        if not need.issubset(df.columns):
            raise ValueError("no header or missing columns")
        df = df[["chr1","x1","x2","chr2","y1","y2"]]
    except Exception:
        df = pd.read_csv(loop_file, sep="\t", comment="#", header=None,
                         usecols=[0,1,2,3,4,5],
                         names=["chr1","x1","x2","chr2","y1","y2"], dtype=str)
    for col in ["x1","x2","y1","y2"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df

def make_chr_normalizer(cool_obj: cooler.Cooler):
    """
    让 BEDPE 染色体名与 cool 文件一致（有/无 'chr' 前缀）
    """
    has_chr = any(ch.startswith("chr") for ch in cool_obj.chromnames)
    def _norm(s: str) -> str:
        s = str(s).strip()
        if has_chr and not s.startswith("chr"):
            return "chr" + s
        if (not has_chr) and s.startswith("chr"):
            return s[3:]
        return s
    return _norm

def bins_in_chrom(c: cooler.Cooler, chrom: str) -> int:
    """返回该染色体有多少个 bin"""
    return c.bins().fetch(chrom).shape[0]

def clip_window_bins(center_bin: int, half_bins: int, chrom_bins: int):
    """
    以 bin 为单位裁剪窗口，尽量保持固定长度 2*half_bins
    返回 (start_bin, end_bin)，半开区间 [start, end)
    """
    start = max(0, center_bin - half_bins)
    end   = min(chrom_bins, center_bin + half_bins)
    L = 2 * half_bins
    if end - start < L:
        if start == 0:
            end = min(L, chrom_bins)
        else:
            start = max(0, end - L)
    return start, end

def to_bin_span(start_bp: int, end_bp: int, start_bin: int, bin_size: int, win_bins: int):
    """
    把 BEDPE 的 bp 坐标映射为窗口内的 bin 索引区间 [bx1, bx2)
    """
    b1 = start_bp // bin_size
    b2 = math.ceil(end_bp / bin_size)
    bx1 = b1 - start_bin
    bx2 = b2 - start_bin
    bx1 = max(0, min(win_bins, int(bx1)))
    bx2 = max(0, min(win_bins, int(bx2)))
    if bx2 <= bx1:
        bx2 = min(win_bins, bx1 + 1)
    return bx1, bx2
def visualize_loop(
    mcool_base: str,
    bedpe_path: str,
    index: int = 0,
    resolution: int = 5000,
    cool_path_override: str = "",
    window_bins: int = 224,
    balance: bool = True,
    cmap: str = "Reds",
):
    """
    在给定分辨率下，可视化第 index 条（0-based）同染色体 loop 的窗口热图，并叠加同染色体内落在窗口内的所有 loops。
    """
    cool_path = build_cool_path(mcool_base, resolution, cool_path_override)
    c = cooler.Cooler(cool_path)
    bin_size = c.binsize

    loops = read_bedpe(bedpe_path)
    norm_chr = make_chr_normalizer(c)
    loops["chr1"] = loops["chr1"].apply(norm_chr)
    loops["chr2"] = loops["chr2"].apply(norm_chr)

    if not (0 <= index < len(loops)):
        raise IndexError(f"index {index} out of range (0..{len(loops)-1})")

    main = loops.iloc[index]
    chrom1, x1, x2, chrom2, y1, y2 = (
        main["chr1"], int(main["x1"]), int(main["x2"]),
        main["chr2"], int(main["y1"]), int(main["y2"])
    )
    if chrom1 != chrom2:
        raise ValueError(f"第 {index} 条为跨染色体 loop（{chrom1} vs {chrom2}），此函数仅做同染色体可视化。")

    # 以 bin 为单位计算窗口
    center_x_bp = (x1 + x2) // 2
    center_bin  = center_x_bp // bin_size
    half_bins   = window_bins // 2
    chrom_bins  = bins_in_chrom(c, chrom1)
    start_bin, end_bin = clip_window_bins(center_bin, half_bins, chrom_bins)
    # 区间 [start_bin, end_bin) 转为 bp
    start_bp = start_bin * bin_size
    end_bp   = end_bin * bin_size

    region = f"{chrom1}:{start_bp}-{end_bp}"
    mat = c.matrix(balance=balance, sparse=False).fetch(region, region)
    mat = np.asarray(mat, dtype=float)
    mat[~np.isfinite(mat)] = np.nan

    # 百分位裁剪
    vmin = np.nanpercentile(mat, 1)
    vmax = np.nanpercentile(mat, 97)
    if not np.isfinite(vmin): vmin = np.nanmin(mat) if np.isfinite(np.nanmin(mat)) else 0.0
    if not np.isfinite(vmax): vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1.0

    # 画热图
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=120)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.set_title(f"{Path(cool_path).name} | {chrom1}:{start_bp:,}-{end_bp:,} | res={bin_size}bp | index={index}", fontsize=10)
    ax.axis("off")

    # 叠加同染色体内的所有 loops（窗口内才画）
    same_chr = loops[(loops["chr1"] == chrom1) & (loops["chr2"] == chrom1)]
    for _, lp in same_chr.iterrows():
        lx1, lx2, ly1, ly2 = int(lp["x1"]), int(lp["x2"]), int(lp["y1"]), int(lp["y2"])
        # 快速判断是否与窗口有交集
        if lx2 <= start_bp or lx1 >= end_bp or ly2 <= start_bp or ly1 >= end_bp:
            continue
        bx1, bx2 = to_bin_span(lx1, lx2, start_bin, bin_size, end_bin - start_bin)
        by1, by2 = to_bin_span(ly1, ly2, start_bin, bin_size, end_bin - start_bin)

        is_main = lp.equals(main)
        rect = plt.Rectangle(
            (by1, bx1), by2 - by1, bx2 - bx1,
            fill=False,
            edgecolor=("lime" if is_main else "dodgerblue"),
            linewidth=(2.0 if is_main else 0.9)
        )
        ax.add_patch(rect)

    plt.show()
    return fig
# 如果你使用 .mcool + 分辨率（Cell 1 的方案A）：
fig = visualize_loop(
    mcool_base=MCOOL_BASE,
    bedpe_path=BEDPE_PATH,
    index=INDEX,
    resolution=RESOLUTION,
    cool_path_override=COOL_PATH,  # 若留空则用方案A；若填完整 .cool/.mcool::... 则优先生效
    window_bins=WINDOW_BINS,
    balance=BALANCE,
)

所以说 对于5k碱基一个bin 来说 你切割出来的 224*224的矩阵的话 loop就应该是一个点 对应一个值

### tad修改细节
再说tad  10k碱基一个bin的
我们给你的边界是两种方法算出来的tad边界  分别是di 和is 
这里给你的脚本文件是3列 第一列 是染色体号 第二列是 左边界 第三列是右边界 但是hic是对称矩阵 所以这个无所谓可以颠倒
我们讲边界就是也可以说是 一个坐标 (x,x) 也就是一个对角线上的点

这是我绘制边界的代码
# 可视化 TAD 边界（IS 导出的 BED）叠加到 10kb Hi-C 热图
# 依赖：cooler, numpy, pandas, matplotlib
import numpy as np, pandas as pd, cooler
import matplotlib.pyplot as plt

# ========= 路径与参数（按需修改）=========
mcool_path = '/storu/zkyang/AAA_MIL/data/12878.mcool::/resolutions/10000'
bed_path   = '/storu/zkyang/AAA_MIL/tad/di_sample_tads.bed'  # chrom start end
chrom      = '2'         # '1' 或 'chr1' 都可
start_bp   = 230_000_000  # 起点（bp）
end_bp     = 232_000_000  # 终点（bp）
balance    = False        # 使用 KR/ICE 平衡矩阵
vmax_pct   = 98          # 热图上限分位数（增强对比度）
line_color = 'deepskyblue'
line_width = 0.2

# ========= 小工具 =========
def to_str(x):
    import numpy as _np
    return x.decode('utf-8') if isinstance(x, (bytes, _np.bytes_)) else x

def map_chrom_name(c, want: str) -> str:
    names = [to_str(n) for n in c.chromnames]
    if want in names: return want
    if want.startswith('chr') and want[3:] in names: return want[3:]
    if (not want.startswith('chr')) and ('chr'+want) in names: return 'chr'+want
    raise KeyError(f"{want} not in cooler names: {names[:12]} ...")

# ========= 读取矩阵 =========
clr = cooler.Cooler(mcool_path)
chrom = map_chrom_name(clr, str(chrom))
region = f"{chrom}:{start_bp}-{end_bp}"
mat = clr.matrix(balance=balance).fetch(region)
binsize = int(clr.binsize)
H, W = mat.shape

# ========= 绘制热图 =========
vmax = float(np.nanpercentile(mat, vmax_pct)) if np.isfinite(mat).any() else 1.0
plt.figure(figsize=(6.8,6.8), dpi=120)
ax = plt.gca()
ax.imshow(mat, origin='upper', cmap='Reds', vmax=vmax, aspect='auto')
ax.set_title(f"Hi-C heatmap @10kb | {region}\nOverlay: IS boundaries")
ax.axis('off')

# ========= 读取并叠加边界 =========
cols=['chrom','start','end']
bed = pd.read_csv(bed_path, sep='\t', header=None, names=cols, dtype={0:str,1:int,2:int})
bed['chrom'] = bed['chrom'].astype(str)

# 让 BED 染色体命名风格与 cooler 相同（有/无 'chr' 前缀）
if chrom.startswith('chr'):
    bed['chrom'] = bed['chrom'].apply(lambda x: x if x.startswith('chr') else 'chr'+x)
else:
    bed['chrom'] = bed['chrom'].apply(lambda x: x[3:] if x.startswith('chr') else x)

sub = bed[(bed['chrom']==chrom) & (bed['start']<end_bp) & (bed['end']>start_bp)].copy()

# 将边界位置映射到窗口内的 bin 坐标，用 start 作为边界点
for _, r in sub.iterrows():
    b_bp = max(start_bp, int(r['start']))
    b_idx = int((b_bp - start_bp) // binsize)
    if 0 <= b_idx < W:
        ax.axvline(b_idx, color=line_color, lw=line_width, alpha=0.95)
        ax.axhline(b_idx, color=line_color, lw=line_width, alpha=0.95)

plt.show()

这里是直接把边界 那个点 的 水平线和垂直线画出来

但是 我们这里其实想找的是tad   给你tad边界 只是一个软标签 所以说 还需要你加入tad 这个shape的一些先验 这个你可以自己思考一下  然后 我们最终的输出 
是需要能构成tad 的文件
所以我们其实最后要你输出的预测 是 tad的边界 也就是 能够构成tad 的一个左边界和右边界的文件

### stripe修改细节
再说stripe

这个问题最大 
我们给你的文件是
chr	pos1	pos2	chr2	pos3	pos4	length	width	Mean	maxpixel	pvalue	Stripiness
16	22640001	22700000	16	19050001	22700000	3650000	60000	68.85223880597015	95.0%	0.001	48.566972314031425
13	35710001	35760000	13	33050001	35760000	2710000	50000	35.44280442804428	98.0%	0.003	25.100398119342813
16	11030001	11070000	16	11030001	11940000	910000	40000	238.65384615384616	99.0%	0.002	23.33126218371207
X	1210001	1250000	X	190001	1250000	1060000	40000	75.83988764044943	99.0%	0.007	23.129111870090977
18	5020001	5070000	18	3280001	5070000	1790000	50000	86.54078212290503	96.0%	0.004	22.680982744147002
9	128140001	128190000	9	128140001	129350000	1210000	50000	141.1685950413223	99.0%	0.002	22.68047142366759
16	21560001	21610000	16	21560001	23810000	2250000	50000	87.96682926829268	98.0%	0.003	19.474934184777055
类似是这样的  这是tsv文件的部分
chr: chromosome
pos1: x1 position of stripe
pos2: x2 position of stripe
chr2: chromosome (same as chr_1)
pos3: y1 position of stripe
pos4: y2 position of stripe
length: Length of vertical stripe (y2-y1+1)
width: Width of vertical stripe (x2-x1+1)
mean: Average pixel value within stripe
maxpixel: Pixel saturation parameter. See below.
pvalue: P-value of the stripe
Stripiness: Score of the stripe
这是注释的意思

以下是我的画图代码
# %% 可直接运行：按 stripenn_index 可视化指定条纹（竖直矩形条带）
# 依赖：cooler, h5py, numpy, pandas, matplotlib
from math import floor, ceil
import h5py, numpy as np, pandas as pd, cooler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ===================== 路径与参数（按需修改） =====================
mcool_path            = '/storu/zkyang/AAA_MIL/data/12878.mcool'  # .mcool 或 .mcool::/resolutions/N 或 .cool
desired_bin_size_bp   = 5000      # 若给的是纯 .mcool，会在 /resolutions 里选最近的
stripes_tsv_path      = '/storu/zkyang/AAA_MIL/stripenn_out/result_filtered.tsv'

stripenn_index        = 13        # << 写第几条（“第16个”），默认按 1 起始
index_is_one_based    = True      # True=上述索引从1开始；False=从0开始

tsv_one_based_closed  = True      # Stripenn 输出一般是 1-based 闭区间；若是 0-based 半开，改 False
balance_matrix        = False      # 用 cooler 的 balanced 矩阵
draw_symmetric_box    = False      # 是否画对称矩形（看上下三角一致性）
fill_alpha            = 0.22      # 矩形填充透明度（0=只边框）  
edge_color            = 'deepskyblue'
edge_width            = 1.4
cmap_name             = 'Reds'    # 纯 matplotlib colormap
vmax_percentile       = 98        # 热图上限用分位数

# 如果想强制指定染色体（一般不需要，Stripenn 行里自带 chr），可以写成 '1' 或 'chr1'
force_target_chrom    = None      # 例如 '4' 或 'chr4'；None 表示用该条纹自己的 chr

# margin 策略：默认为 max(250kb, 25%*length)，也可以手动覆盖
margin_bp_override    = None      # 例如 400_000；None=使用自动策略

# ===================== 小工具 =====================
def resolve_cool_uri(path_or_uri: str, want_binsize: int):
    """支持 .cool / .mcool / .mcool::/resolutions/N"""
    if '::' in path_or_uri:
        c = cooler.Cooler(path_or_uri)
        return path_or_uri, int(c.binsize)
    if path_or_uri.endswith('.cool'):
        c = cooler.Cooler(path_or_uri)
        return path_or_uri, int(c.binsize)
    if path_or_uri.endswith('.mcool'):
        with h5py.File(path_or_uri, 'r') as f:
            if 'resolutions' not in f:
                raise RuntimeError("该 .mcool 不含 '/resolutions'")
            avail = sorted(int(k) for k in f['/resolutions'].keys())
        if not avail:
            raise RuntimeError("'/resolutions' 下没有整数分辨率")
        chosen = want_binsize if want_binsize in avail else min(avail, key=lambda x: abs(x - want_binsize))
        if chosen != want_binsize:
            print(f"[提示] 未找到 {want_binsize}bp，改用最近的 {chosen}bp")
        return f"{path_or_uri}::/resolutions/{chosen}", int(chosen)
    raise ValueError("仅支持 .cool / .mcool / .mcool::/resolutions/N")

def to_str(x):
    import numpy as _np
    return x.decode('utf-8') if isinstance(x, (bytes, _np.bytes_)) else x

def map_chrom_name(c, want: str) -> str:
    names = [to_str(n) for n in c.chromnames]
    if want in names: return want
    if want.startswith('chr') and want[3:] in names: return want[3:]
    if (not want.startswith('chr')) and ('chr'+want) in names: return 'chr'+want
    raise KeyError(f"{want} not in cooler names: {names[:12]} ...")

def floor_bin(bp, start_bp, bin_size):
    return int(floor((bp - start_bp) / bin_size))

def ceil_bin(bp, start_bp, bin_size):
    return int(ceil((bp - start_bp) / bin_size))

def read_stripenn_tsv(path):
    """尽量兼容：空白分隔、有/无表头；保留必须列。"""
    want = ["chr","pos1","pos2","chr2","pos3","pos4","length","width","Mean","maxpixel","pvalue","Stripiness"]
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        if "chr" not in df.columns or "pos1" not in df.columns:
            df = pd.read_csv(path, sep=r"\s+", engine="python", names=want)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", names=want)
    # 只保留 cis
    if "chr2" in df.columns:
        df = df[df["chr"] == df["chr2"]].copy()
    # 清洗
    for c in ["pos1","pos2","pos3","pos4","length","width","Mean","Stripiness"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "maxpixel" in df.columns:
        df["maxpixel"] = df["maxpixel"].astype(str).str.replace("%","", regex=False)
        df["maxpixel"] = pd.to_numeric(df["maxpixel"], errors="coerce")
    df = df.dropna(subset=["chr","pos1","pos2","pos3","pos4"]).reset_index(drop=True)
    return df

# ===================== 读取数据 =====================
cool_uri, binsize = resolve_cool_uri(mcool_path, desired_bin_size_bp)
print(f"使用分辨率: {binsize} bp | URI: {cool_uri}")
clr = cooler.Cooler(cool_uri)

df = read_stripenn_tsv(stripes_tsv_path)
if len(df) == 0:
    raise RuntimeError("TSV 为空或未解析到必需字段。")

# 索引换算（默认 1 起）
row_idx = (stripenn_index - 1) if index_is_one_based else stripenn_index
if not (0 <= row_idx < len(df)):
    raise IndexError(f"stripenn_index 越界：共有 {len(df)} 条，索引={stripenn_index}（{'1-based' if index_is_one_based else '0-based'}）")

row = df.iloc[row_idx].copy()

# 染色体命名映射（可强制替换）
chrom_raw = row["chr"]
if force_target_chrom is not None:
    chrom_raw = force_target_chrom
chrom = map_chrom_name(clr, str(chrom_raw))
chrom2 = map_chrom_name(clr, str(row.get("chr2", chrom_raw)))
assert chrom == chrom2, "此示例仅演示 cis 条纹"

# 1-based 闭区间 → 0-based 半开
def s0(p): return int(p) - 1 if tsv_one_based_closed else int(p)  # 起点
def e0(p): return int(p)       if tsv_one_based_closed else int(p)  # 终点（半开右端）

X1, X2 = s0(row["pos1"]), e0(row["pos2"])   # x 方向（横轴，宽度）
Y1, Y2 = s0(row["pos3"]), e0(row["pos4"])   # y 方向（纵轴，长度）

# 自动 margin（或使用覆盖值）
length_bp = int(row["pos4"]) - int(row["pos3"]) + (1 if tsv_one_based_closed else 0)
margin_bp = margin_bp_override if margin_bp_override is not None else max(250_000, int(0.25 * length_bp))

# 取窗口（左右取 min/max，再加 margin）
start_bp = max(0, min(X1, Y1) - margin_bp)
end_bp   = max(X2, Y2) + margin_bp
# Clamp 到染色体长度
chrom_len = int(clr.chromsizes.loc[chrom])
end_bp = min(end_bp, chrom_len)

region = f"{chrom}:{start_bp}-{end_bp}"
print(f"Region = {region} | margin = {margin_bp/1e3:.0f} kb | 条纹宽×长（bp）= {(X2-X1):,} × {(Y2-Y1):,}")

# ===================== 取矩阵并作图 =====================
mat = clr.matrix(balance=balance_matrix).fetch(region)
vmax = float(np.nanpercentile(mat, vmax_percentile))

plt.figure(figsize=(6.7,6.7), dpi=110)
ax = plt.gca()
ax.imshow(mat, cmap=cmap_name, vmax=vmax, origin='upper', aspect='auto')
ax.set_title(f"Stripenn #{stripenn_index} @ {region}\n"
             f"x[{row['pos1']}-{row['pos2']}], y[{row['pos3']}-{row['pos4']}] (bp) | res={binsize}")

# 把条纹矩形从 bp 映射到窗口的 bin 坐标
x1b = floor_bin(max(X1, start_bp), start_bp, binsize)
x2b = ceil_bin (min(X2, end_bp),   start_bp, binsize)
y1b = floor_bin(max(Y1, start_bp), start_bp, binsize)
y2b = ceil_bin (min(Y2, end_bp),   start_bp, binsize)

# 裁剪到窗口大小
H, W = mat.shape
x1b = max(0, min(W, x1b)); x2b = max(0, min(W, x2b))
y1b = max(0, min(H, y1b)); y2b = max(0, min(H, y2b))

print(f"映射到 bin：x[{x1b},{x2b}) × y[{y1b},{y2b}) | width={x2b-x1b} bin, length={y2b-y1b} bin")

# 画“竖直矩形条带”（半透明填充）
rect = Rectangle((y1b, x1b), (y2b - y1b), (x2b - x1b),
                 linewidth=edge_width, edgecolor=edge_color,
                 facecolor=edge_color, alpha=fill_alpha)
ax.add_patch(rect)

# （可选）画对称矩形
if draw_symmetric_box:
    rectT = Rectangle((x1b, y1b), (x2b - x1b), (y2b - y1b),
                      linewidth=max(1.0, edge_width-0.2), edgecolor=edge_color,
                      facecolor=edge_color, alpha=fill_alpha*0.85, linestyle='--')
    ax.add_patch(rectT)

ax.axis('off'); ax.set_position([0,0,1,1])
plt.show()
plt.close()

这个我们期望你输出的是和这个相同的 标注文件  所以类似长度宽度 这些你都要有哈


所以说 总的来说 loop你需要的标签训练文件是一个点  stripe是一个 长方形块   tad的话可以说是一个点 也可可以说是一条线
你现在需要做的工作是复查代码看看 哪里和我说的不同你需要修改并且注意同步训练代码 和预测代码 
架构上面哪里有问题你也需要同步修改 
修改完成以后告诉我结果并且把修改 的写入 MTL.md 文件中
然后你还要告诉我如何重新生成数据集 和标签