#!/usr/bin/env bash
# 幂等的开发环境安装脚本（Cloud Agent install 阶段调用）。
# 目标：在 .venv 中准备好 train_py / data_generate / data_process / infer_py
# 运行所需的全部 Python 依赖。可重复执行，不做交互式操作。
set -euo pipefail

cd "$(dirname "$0")/.."

# 系统依赖：默认镜像的 python3.12 缺少 ensurepip，无法直接创建 venv。
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends python3.12-venv
fi

# 创建 / 复用虚拟环境（对已存在的 .venv 幂等，不会清空已装包）。
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

# Cloud Agent VM 无 GPU，安装 CPU 版 PyTorch。
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 训练 / 推理 / 数据处理与可视化所需的核心科学与基因组学依赖。
.venv/bin/pip install \
  numpy \
  pandas \
  cooler \
  h5py \
  scikit-learn \
  tqdm \
  matplotlib \
  pytest

echo "install.sh: dependencies ready in .venv"
