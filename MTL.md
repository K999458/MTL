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
- TAD：`tad_acc / tad_prec / tad_rec / tad_f1`（对 `ignore`=1 的区域统计，阈值 0.5）
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
- TAD（10kb）：训练标签为 1D 边界向量；推理阶段新增把边界配对输出 TAD 域（`tads_pred_chr2.bed`）。

### 重新生成数据与标签（流程）
- Loops（BEDPE → JSONL）：读取 `hiccups_loop_5k_change.bedpe`，取每列 `chr1,x1,x2,chr2,y1,y2`，计算每锚点中心（或直接使用 x1/x2 等 5kb 对齐坐标），写入 `data_generated/labels/loop_5kb.jsonl` 字段 `{chrom,p1,p2}`（bp，0-based）。
- Stripes（Stripenn TSV → JSONL）：读取 `stripenn_out/result_filtered.tsv`，仅保留 cis 行，并将 1-based 闭区间转换为 0-based 半开，写入 `data_generated/labels/stripe_10kb.jsonl` 字段 `{chrom,x1,x2,y1,y2}`（bp）。
- TAD（BED → BED）：整合 `tad/di_*.bed` 与 `tad/tad_boundaries_is_10kb.bed`（可取并集或高置信子集），生成最终边界 `tad_boundaries_10kb.bed`（三列：chrom start end，10kb 步长）。
- Manifests：根据目标分辨率（5kb/10kb）与裁剪参数，生成/更新 `data_generated/manifests/*.jsonl`（字段包含 `chrom,x0,y0,binsize,patch,center,stride`）。
- 缓存：运行既有数据切块流程，产出 `data_generated/cache/{loop_5kb|stripe_10kb|tad_10kb_1d}/*.npz` 或对应 `.h5`。

注意：重新生成标签后，请确保 `train_config.json` 与 `train_py/config.py` 中路径、center/patch/stride 与 labels 坐标体系一致（5kb/10kb）。

### 数据加载健壮性修复（npz 读入）
- 现有缓存 `.npz` 内包含字符串字段（如 chrom），NumPy 会以对象数组存储，默认 `np.load(..., allow_pickle=False)` 会报错。
- 已将 `train_py/data.py` 中所有 `np.load` 改为 `allow_pickle=True`（仅读取本仓库生成的受控缓存，安全可控）。

### 训练调参与 Loader 控制
- Loop 热图损失改为显式的加权 BCE：`loop_pos_weight`（默认 20000）控制正样本权重，标签仍为“单像素点”。必要时可在 `train_config.json` 覆写该值，以匹配实际的正/负样本比例。
- Stripe 掩码损失加入正样本权重 + 面积惩罚：`stripe_pos_weight`（默认 25.0）提升正例梯度，`stripe_area_weight`（默认 0.05）通过 `max(0, mean(pred) - mean(gt))` 抑制大面积假阳性；若条纹仍过粗，可适当增大面积惩罚。
- TAD 边界损失同样支持 `tad_pos_weight`（默认 8.0），使 1D 边界峰值更突出，配合后续域构建时更易阈值化。
- 数据加载支持按环境变量切换 HDF5/npz：设置 `AAA_MIL_USE_H5=0` 可强制使用 `.npz`，避免在网络盘上因多进程读取 HDF5 造成的假死。
