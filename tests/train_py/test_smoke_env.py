"""环境冒烟测试：用合成张量验证多任务模型可完成一次训练迭代。

该测试不依赖真实 Hi-C 数据/缓存，仅用随机张量走通
`MultitaskHiCNet` 的三条任务分支（loop/stripe/tad）的
前向 -> 损失 -> 反向 -> 优化器 step，用于确认开发环境
（依赖、torch、模型代码）可以端到端运行。
"""

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from train_py.config import build_default_config
from train_py.model import MultitaskHiCNet

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_config():
    cfg = build_default_config()
    cfg_path = REPO_ROOT / "train_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            for key, value in json.load(f).items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
    return cfg


@pytest.fixture(autouse=True)
def _cpu_backend():
    # torch 的 CPU oneDNN/MKLDNN 后端在部分非连续张量的反向传播时会抛出
    # "could not construct a memory descriptor using strides"，与业务代码无关。
    # 在纯 CPU 环境下关闭 mkldnn 即可稳定跑通；GPU 环境不受影响。
    if not torch.cuda.is_available():
        prev = torch.backends.mkldnn.enabled
        torch.backends.mkldnn.enabled = False
        yield
        torch.backends.mkldnn.enabled = prev
    else:
        yield


@pytest.mark.parametrize("task", ["loop", "stripe", "tad"])
def test_forward_backward_step(task):
    torch.manual_seed(0)
    cfg = _load_config()
    net = MultitaskHiCNet(
        in_channels=cfg.input_channels,
        base_channels=cfg.base_channels,
        tad_band_width=cfg.tad_band_width,
        use_axial_attention=cfg.use_axial_attention,
        backbone_type=cfg.backbone_type,
    )
    net.train()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    x = torch.randn(1, cfg.input_channels, 128, 128)
    out = net(x, task=task)

    tensor_outputs = [v for v in out.values() if torch.is_tensor(v)]
    assert tensor_outputs, f"task {task} produced no tensor outputs"
    for value in tensor_outputs:
        assert torch.isfinite(value).all(), f"non-finite output for task {task}"

    loss = sum(v.float().mean() for v in tensor_outputs)
    optim.zero_grad()
    loss.backward()

    grad_total = sum(
        p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None
    )
    assert grad_total > 0, f"no gradients flowed for task {task}"
    optim.step()
