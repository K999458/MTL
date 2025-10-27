多任务训练数据生成

本目录包含将已有的标注（Loop@5kb、TAD@10kb、Stripe@10kb）与 .mcool 数据
整理为训练用 manifests 的脚本。脚本默认只沿主对角滑窗生成 patch（高效、常规做法）。

主要脚本
- make_dataset.py：
  - 融合 TAD 边界（可选：IS+DI 近邻合并）
  - 生成三路 manifests：
    - Loop@5kb：patch=256, center=224, stride=112
    - Stripe@10kb：patch=320, center=256, stride=128
    - TAD@10kb（1D）：band B=64, L=1024, stride=512（用于对角带→1D 序列）

示例
  conda activate vipg
  python AAA_MIL/data_generate/make_dataset.py \
    --mcool /storu/zkyang/AAA_MIL/data/12878.mcool \
    --loop-bedpe /storu/zkyang/AAA_MIL/data/loops_ground_truth/hiccups_loop_5k_change.bedpe \
    --is-bed /storu/zkyang/AAA_MIL/tad/tad_boundaries_is_10kb.bed \
    --di-bed /storu/zkyang/AAA_MIL/tad/tad_boundaries_di_10kb.bed \
    --stripe-tsv /storu/zkyang/AAA_MIL/stripenn_out/result_filtered.tsv \
    --outdir /storu/zkyang/AAA_MIL/data_generated \
    --fuse-tad --tad-merge-bp 20000

输出
- data_generated/
  - labels/
    - tad_boundaries_10kb.bed  （融合或直用 IS）
  - manifests/
    - loop_5kb.jsonl
    - stripe_10kb.jsonl
    - tad_10kb_1d.jsonl

备注
- 如需预先缓存标签张量（热图/掩码/序列），可后续扩展脚本；当前版本仅生成高效的清单文件。

