import argparse
import json
import os
import pathlib
import sys

if __package__ is None:
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from train_py.config import TrainingConfig, build_default_config
    from train_py.trainer import Trainer
else:
    from .config import TrainingConfig, build_default_config
    from .trainer import Trainer


def load_config(path: str) -> TrainingConfig:
    cfg = build_default_config()
    if not path:
        return cfg
    with open(path, 'r') as f:
        data = json.load(f)
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def parse_args():
    ap = argparse.ArgumentParser(description='Train multi-task Hi-C network (loop/TAD/stripe)')
    ap.add_argument('--config', type=str, default='', help='JSON 配置文件路径（可选）')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
