## 该项目是基于多任务学习，针对三维基因组学中 TAD loop 和stripenn 进行识别的
## 你需要严格按照本项目编写文件和文档


### 项目遵循以下流程 
#### 
1.只能在/storu/zkyang/AAA_MIL 里面新建和修改文件
2.你需要按照步骤一步一步完善执行，需要达到我的要求以后才能进行下一步任务
3.在完成每个生成文件的任务以后你需要查看我的MTL.md 文档 再次阅读一边确保严格按照流程执行
4./storu/zkyang/AAA_MIL/NCFF135MUT_CTCF_untreated 这个文件夹为其他项目文件夹 不用管也不需要使用
5.如果有什么代码或者东西找不到 就在/storu/zkyang/AAA_MIL 文件夹中寻找


##
在使用可打开文本文件作为你写的python脚本的输入时 请你先阅读文本 确定文本格式
##

### 训练日志与指标（新增）
为便于监控训练效果，已在 `train_py/trainer.py` 增加基础评估指标：
- TAD：`tad_map_iou / tad_map_dice / tad_boundary_iou / tad_boundary_dice`（域/边界概率图按 `tad_domain_threshold`、`tad_boundary_threshold` 二值化后统计）
- Stripe：`stripe_iou / stripe_dice`（按 0.5 阈值二值化）
- Loop：`loop_hit`（不计算 IoU；将预测热图阈值化后，与“有效 bin”（`valid`）做±1 bin 膨胀后任意重叠即记为命中；批内取命中率）

打印位置与频率：
- 当 `train_config.json` 中的 `use_tqdm=true` 时，在进度条的 postfix 显示；否则在每个 `log_interval` 步打印一次（默认 20）。

运行示例：
- `python -m train_py.train --config /storu/zkyang/AAA_MIL/train_config.json`

说明：上述指标仅用于快速监控，环路检测更严格的匹配与评估（如基于峰值匹配的 P/R）可在验证阶段扩展。

### 推理脚本（新增）
- 新增 `infer_py/infer_chr2.py`：加载最新检查点，对数据集中染色体2进行 loop/stripe/TAD 推理，并将结果导出：
  - `infer_py/result_filtered_chr2.tsv`（条纹，字段格式参考 `stripenn_out/result_filtered.tsv`）
  - `infer_py/hiccups_loop_5k_pred_chr2.bedpe`（环路，字段格式参考 `data/loops_ground_truth/hiccups_loop_5k_change.bedpe`）
  - `infer_py/tad_boundaries_is_10kb_pred_chr2.bed`（TAD 边界，格式参考 `tad/tad_boundaries_is_10kb.bed`）
  - `infer_py/tads_pred_chr2.bed`（TAD 域，自动把相邻边界配对为 [left,right] 区域，最小长度 40kb）
- 运行示例：
  - `python infer_py/infer_chr2.py --config /storu/zkyang/AAA_MIL/train_config.json`（默认使用 `outputs/` 下最新检查点）

### Label 修正（依据 labelfix.md 已执行）
- Loop（5kb）：将监督热图从高斯改为“单像素点”标签，并对称标注 (i,j) 与 (j,i)，与 Hi-C 对称性一致。
  - 代码变更：`train_py/data.py` 的 `LoopDataset.__getitem__` 中同时对称写入 `heat/offset/valid`，表达式 `heat[iy,ix]=1; heat[ix,iy]=1; offset[:,iy,ix]=0; offset[:,ix,iy]=0; valid[iy,ix]=valid[ix,iy]=1`。
- Stripe（10kb）：维持矩形区域标签，确认坐标采用 0-based 半开（由 Stripenn 1-based 闭区间统一转为 0-based 半开存入 JSONL）。
- TAD（10kb）：以边界为核心的二维标签，来自 `tad/arhead.bed`。缓存阶段仅标注域边界环（内部置 0），并额外生成一次膨胀的边界掩码供训练使用。

### 重新生成数据与标签（流程）
- Loops（BEDPE → JSONL）：读取 `hiccups_loop_5k_change.bedpe`，取每列 `chr1,x1,x2,chr2,y1,y2`，计算每锚点中心（或直接使用 x1/x2 等 5kb 对齐坐标），写入 `data_generated/labels/loop_5kb.jsonl` 字段 `{chrom,p1,p2}`（bp，0-based）。
- Stripes（Stripenn TSV → JSONL）：读取 `stripenn_out/result_filtered.tsv`，仅保留 cis 行，并将 1-based 闭区间转换为 0-based 半开，写入 `data_generated/labels/stripe_10kb.jsonl` 字段 `{chrom,x1,x2,y1,y2}`（bp）。
- TAD 域集合：直接使用 `tad/arhead.bed`（高置信域列表，已对齐 10kb 网格）。该文件可作为 `cache_patches.py --domain-bed` 的输入，生成 `data_generated/cache/tad_10kb_2d/` 下的域/边界缓存。
- TAD 边界：若需 Loop/Stripe 先验，可将 `tad/arhead.bed` 中每个域的起止位置拆分为边界，写入 `data_generated/labels/tad_boundaries_10kb.bed`（三列：chrom start end，10kb 步长）。
- Manifests：核心使用 `tad_10kb_2d.jsonl`（可通过 `data_generate/make_dataset.py` 自动生成；默认 patch=320、center=256、stride=128）。若仍需 1D 训练，可额外维护 `tad_10kb_1d.jsonl`。
- 缓存：`data_generate/cache_patches.py` 通过 `--tad2d-manifest` 在 `data_generated/cache/tad_10kb_2d/` 下生成 `raw/kr/domain/boundary` 等字段；若想额外生成 1D 缓存，需要显式提供 `--tad1d-manifest`。

注意：重新生成标签后，请确保 `train_config.json` 与 `train_py/config.py` 中路径、center/patch/stride 与 labels 坐标体系一致（5kb/10kb）。

### 数据加载健壮性修复（npz 读入）
- 现有缓存 `.npz` 内包含字符串字段（如 chrom），NumPy 会以对象数组存储，默认 `np.load(..., allow_pickle=False)` 会报错。
- 已将 `train_py/data.py` 中所有 `np.load` 改为 `allow_pickle=True`（仅读取本仓库生成的受控缓存，安全可控）。

### 训练调参与 Loader 控制
- Loop 热图损失改为显式的加权 BCE：`loop_pos_weight`（默认 20000）控制正样本权重，标签仍为“单像素点”。必要时可在 `train_config.json` 覆写该值，以匹配实际的正/负样本比例。
- Stripe 掩码损失加入正样本权重 + 面积惩罚：`stripe_pos_weight`（默认 25.0）提升正例梯度，`stripe_area_weight`（默认 0.05）通过 `max(0, mean(pred) - mean(gt))` 抑制大面积假阳性；若条纹仍过粗，可适当增大面积惩罚。
- TAD 分支基于多尺度 FPN + 对角细化：主干聚合 `c4/u3/u2/u1` 四级特征，融合 Loop/Stripe 共享特征后，经 Squeeze-Excite 与对角卷积增强；输出的 `domain_map` 对应细边界环，`boundary_map` 对应膨胀后的边界带。`tad_domain_map_weight` 约束核心边界，`tad_boundary_weight`/`tad_boundary_pos_weight` 专门提升膨胀边界的梯度。
- 推理阶段（`infer_chr2.py`）：Loop 做 ±2 bin 邻域去重；Stripe 采用低阈值连通域 + 均值/峰值/长宽比过滤；TAD 先以 `tad_domain_threshold` 累积 patch 内域概率，再利用 `boundary_map` 沿行/列投影提取边界候选，结合连通域后按 `tad_min_bins` 过滤跨度，最终输出 `tad_boundaries_is_10kb_pred_chr2.bed` 与 `tads_pred_chr2.bed`。
- 数据加载支持按环境变量切换 HDF5/npz：设置 `AAA_MIL_USE_H5=0` 可强制使用 `.npz`，避免在网络盘上因多进程读取 HDF5 造成的假死。
- Loop/Stripe 输入现为三通道 `[raw log1p, KR O/E log1p, distance]`，配置项 `input_channels` 默认 3，可按需要调整；Loop 主要依赖 KR 特征，Stripe 则可利用原始矩阵强化条纹/loop 线索。
- Stripe 样本附带 `orientation_prior`（根据 raw 矩阵纵横残差构建），损失中以 `stripe_orientation_weight`（默认 0.05）约束预测更贴合高强度竖/横条带。
- 数据缓存脚本会同步生成 `*_kr.npz`（KR O/E）与原始 `*.npz`；`fill_missing_2d.py` 已支持自动补齐双版本，重新缓存时注意磁盘占用。
