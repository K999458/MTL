import os
import time
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import TrainingConfig
from .data import DatasetPaths, make_dataloaders
from .model import MultitaskHiCNet
from . import losses as loss_utils


class GradNormBalancer:
    def __init__(self, tasks: List[str], alpha: float = 1.5, lr: float = 0.025):
        self.tasks = tasks
        self.alpha = alpha
        self.lr = lr
        self.initial_losses = None
        self.weights = torch.nn.Parameter(torch.ones(len(tasks)))

    def parameters(self):
        return [self.weights]

    def normalize(self):
        with torch.no_grad():
            self.weights.data = torch.relu(self.weights.data)
            self.weights.data /= self.weights.data.mean().clamp(min=1e-6)

    def step(self, losses: Dict[str, torch.Tensor], grad_norms: Dict[str, float]):
        if self.initial_losses is None:
            self.initial_losses = {task: loss.detach().item() for task, loss in losses.items()}
        with torch.no_grad():
            losses_det = {k: v.detach().item() for k, v in losses.items()}
            avg_g = sum(grad_norms.values()) / len(grad_norms)
            for i, task in enumerate(self.tasks):
                loss_ratio = losses_det[task] / (self.initial_losses[task] + 1e-6)
                target = avg_g * (loss_ratio ** self.alpha)
                grad = grad_norms[task]
                self.weights.data[i] = self.weights.data[i] - self.lr * (grad - target)
        self.normalize()


def flatten_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    flats = [g.reshape(-1) for g in grads if g is not None]
    if not flats:
        return torch.tensor(0.0)
    return torch.cat(flats)


def pcgrad(projected: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
    tasks = list(projected.keys())
    num_params = len(projected[tasks[0]])
    grads = {task: [g.clone() for g in projected[task]] for task in tasks}
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            if i == j:
                continue
            dot = torch.tensor(0.0, device=grads[ti][0].device)
            norm_sq = torch.tensor(0.0, device=grads[ti][0].device)
            for gi, gj in zip(grads[ti], grads[tj]):
                dot = dot + (gi.reshape(-1) * gj.reshape(-1)).sum()
                norm_sq = norm_sq + (gj.reshape(-1) ** 2).sum()
            if dot < 0:
                proj = dot / (norm_sq + 1e-12)
                for k in range(num_params):
                    grads[ti][k] = grads[ti][k] - proj * grads[tj][k]
    merged = []
    for idx in range(num_params):
        merged.append(sum(grads[task][idx] for task in tasks))
    return merged


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_paths = DatasetPaths(cache_root=config.cache_root, label_root=config.label_root,
                                          manifest_root=config.manifest_root)
        self.model = MultitaskHiCNet(in_channels=2,
                                     base_channels=config.base_channels,
                                     tad_band_width=config.tad_band_width,
                                     use_axial_attention=config.use_axial_attention).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=config.cosine_final_lr)
        self.balancer = GradNormBalancer(['loop','stripe','tad'], alpha=config.gradnorm_alpha) if config.use_gradnorm else None
        if self.balancer is not None:
            self.balancer.normalize()
        self.scaler = None  # not using amp due to manual gradient control

        os.makedirs(config.output_dir, exist_ok=True)

        batch_sizes = {'loop': config.batch_size_loop, 'stripe': config.batch_size_stripe, 'tad': config.batch_size_tad}
        self.loaders = make_dataloaders(self.dataset_paths,
                                        config.loop_center, config.loop_patch, config.loop_stride,
                                        config.stripe_center, config.stripe_patch, config.stripe_stride,
                                        config.tad_length, config.tad_stride, config.tad_band_width, config.tad_ignore_bins,
                                        batch_sizes=batch_sizes, num_workers=config.num_workers, pin_memory=config.pin_memory)

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            iterators = {k: iter(v) for k, v in self.loaders.items()}
            start = time.time()
            if self.cfg.use_tqdm:
                try:
                    from tqdm import trange
                    pbar = trange(1, self.cfg.steps_per_epoch + 1, desc=f"Epoch {epoch}", leave=False)
                    for step in pbar:
                        batches = {}
                        for task in ['loop','stripe','tad']:
                            try:
                                batches[task] = next(iterators[task])
                            except StopIteration:
                                iterators[task] = iter(self.loaders[task])
                                batches[task] = next(iterators[task])
                        loss_dict = self.compute_losses(batches)
                        self.backward(loss_dict)
                        if step % self.cfg.log_interval == 0:
                            metrics = self.compute_metrics(batches)
                            pbar.set_postfix({
                                'loop': f"{loss_dict['loop'].item():.4f}",
                                'stripe': f"{loss_dict['stripe'].item():.4f}",
                                'tad': f"{loss_dict['tad'].item():.4f}",
                                'lhit': f"{metrics.get('loop_hit', float('nan')):.3f}",
                                'siou': f"{metrics.get('stripe_iou', float('nan')):.3f}",
                                'tf1': f"{metrics.get('tad_f1', float('nan')):.3f}",
                            })
                except ImportError:
                    # fallback to plain prints
                    for step in range(1, self.cfg.steps_per_epoch + 1):
                        batches = {}
                        for task in ['loop','stripe','tad']:
                            try:
                                batches[task] = next(iterators[task])
                            except StopIteration:
                                iterators[task] = iter(self.loaders[task])
                                batches[task] = next(iterators[task])
                        loss_dict = self.compute_losses(batches)
                        self.backward(loss_dict)
                        if step % self.cfg.log_interval == 0:
                            metrics = self.compute_metrics(batches)
                            loop_loss = loss_dict['loop'].item()
                            stripe_loss = loss_dict['stripe'].item()
                            tad_loss_val = loss_dict['tad'].item()
                            print(
                                f"Epoch {epoch} Step {step}/{self.cfg.steps_per_epoch}: "
                                f"loop={loop_loss:.4f} stripe={stripe_loss:.4f} tad={tad_loss_val:.4f} "
                                f"| loop_hit={metrics.get('loop_hit', float('nan')):.3f} "
                                f"stripe_iou={metrics.get('stripe_iou', float('nan')):.3f} "
                                f"tad_f1={metrics.get('tad_f1', float('nan')):.3f}"
                            )
            else:
                for step in range(1, self.cfg.steps_per_epoch + 1):
                    batches = {}
                    for task in ['loop','stripe','tad']:
                        try:
                            batches[task] = next(iterators[task])
                        except StopIteration:
                            iterators[task] = iter(self.loaders[task])
                            batches[task] = next(iterators[task])
                    loss_dict = self.compute_losses(batches)
                    self.backward(loss_dict)
                    if step % self.cfg.log_interval == 0:
                        metrics = self.compute_metrics(batches)
                        loop_loss = loss_dict['loop'].item()
                        stripe_loss = loss_dict['stripe'].item()
                        tad_loss_val = loss_dict['tad'].item()
                        print(
                            f"Epoch {epoch} Step {step}/{self.cfg.steps_per_epoch}: "
                            f"loop={loop_loss:.4f} stripe={stripe_loss:.4f} tad={tad_loss_val:.4f} "
                            f"| loop_hit={metrics.get('loop_hit', float('nan')):.3f} "
                            f"stripe_iou={metrics.get('stripe_iou', float('nan')):.3f} "
                            f"tad_f1={metrics.get('tad_f1', float('nan')):.3f}"
                        )

            self.scheduler.step()
            elapsed = time.time() - start
            print(f"Epoch {epoch} completed in {elapsed/60:.2f} min | lr={self.optimizer.param_groups[0]['lr']:.2e}")
            if epoch % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int):
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        if self.balancer is not None:
            ckpt['balancer_weights'] = self.balancer.weights.data.cpu()
        path = os.path.join(self.cfg.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(ckpt, path)
        print(f"[checkpoint] saved {path}")

    def move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                out[k] = self.move_to_device(v)
            else:
                out[k] = v
        return out

    def compute_losses(self, batches: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        loop_batch = self.move_to_device(batches['loop'])
        stripe_batch = self.move_to_device(batches['stripe'])
        tad_batch = self.move_to_device(batches['tad'])

        outputs_loop = self.model(loop_batch['inputs'], task='loop')
        loop_losses = loss_utils.loop_loss(outputs_loop['heatmap'], loop_batch['target']['heatmap'],
                                          outputs_loop['offset'], loop_batch['target']['offset'],
                                          loop_batch['target']['valid'], loop_batch['target'].get('corner_prior'),
                                          positive_weight=self.cfg.loop_pos_weight)
        loop_total = loop_losses['heat'] * self.cfg.lambda_loop_heat + loop_losses['offset'] * self.cfg.lambda_loop_offset

        outputs_stripe = self.model(stripe_batch['inputs'], task='stripe')
        stripe_total = loss_utils.stripe_loss(outputs_stripe['mask'], stripe_batch['target']['mask'],
                                              stripe_batch['target'].get('boundary_prior'),
                                              positive_weight=self.cfg.stripe_pos_weight,
                                              area_weight=self.cfg.stripe_area_weight) * self.cfg.lambda_stripe

        outputs_tad = self.model(tad_batch['inputs'], task='tad', band=tad_batch['inputs_band'])
        tad_total = loss_utils.tad_loss(outputs_tad['boundary'], tad_batch['target']['boundary'],
                                        tad_batch['target']['ignore'],
                                        positive_weight=self.cfg.tad_pos_weight) * self.cfg.lambda_tad

        self.latest_outputs = {
            'loop': outputs_loop,
            'stripe': outputs_stripe,
            'tad': outputs_tad
        }

        return {'loop': loop_total, 'stripe': stripe_total, 'tad': tad_total}

    @torch.no_grad()
    def compute_metrics(self, batches: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        # Loop hit-rate（不计算 IoU）
        try:
            loop_logits = self.latest_outputs['loop']['heatmap']  # (B,1,H,W)
            prob = torch.sigmoid(loop_logits)
            valid = batches['loop']['target']['valid'].to(loop_logits.device).unsqueeze(1)  # (B,1,H,W)
            valid_windows = (valid.sum(dim=(1, 2, 3)) > 0)
            if valid_windows.any():
                valid_dil = F.max_pool2d(valid, kernel_size=3, stride=1, padding=1)
                hits = []
                H, W = prob.shape[-2:]
                flat = prob.view(prob.size(0), -1)
                max_idx = flat.argmax(dim=1)
                for b, idx in enumerate(max_idx):
                    if not valid_windows[b]:
                        continue
                    y = (idx // W).item()
                    x = (idx % W).item()
                    if valid_dil[b, 0, y, x] > 0:
                        hits.append(1.0)
                    else:
                        hits.append(0.0)
                if hits:
                    metrics['loop_hit'] = float(sum(hits) / len(hits))
            else:
                metrics['loop_hit'] = 0.0
        except Exception:
            pass

        # Stripe IoU/Dice
        try:
            stripe_logits = self.latest_outputs['stripe']['mask']  # (B,1,H,W)
            stripe_target = batches['stripe']['target']['mask'].to(stripe_logits.device)
            stripe_prob = torch.sigmoid(stripe_logits)
            stripe_pred = (stripe_prob > 0.5).float()
            stripe_true = (stripe_target > 0.5).float()
            inter = (stripe_pred * stripe_true).sum()
            union = (stripe_pred + stripe_true).clamp(max=1.0).sum() + 1e-6
            dice = (2 * inter) / ((stripe_pred.sum() + stripe_true.sum()) + 1e-6)
            metrics['stripe_iou'] = (inter / union).item()
            metrics['stripe_dice'] = dice.item()
        except Exception:
            pass

        # TAD boundary metrics on valid region
        try:
            tad_logits = self.latest_outputs['tad']['boundary']  # (B,L)
            device = tad_logits.device
            target = batches['tad']['target']['boundary'].to(device)
            ignore = batches['tad']['target']['ignore'].to(device)
            prob = torch.sigmoid(tad_logits)
            pred = (prob > 0.5).float()
            mask = (ignore > 0.5).float()
            tp = (pred * target * mask).sum()
            fp = (pred * (1 - target) * mask).sum()
            fn = ((1 - pred) * target * mask).sum()
            acc = ((pred == target).float() * mask).sum() / (mask.sum() + 1e-6)
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1 = (2 * prec * rec) / (prec + rec + 1e-6)
            metrics['tad_acc'] = acc.item()
            metrics['tad_prec'] = prec.item()
            metrics['tad_rec'] = rec.item()
            metrics['tad_f1'] = f1.item()
        except Exception:
            pass

        return metrics

    def backward(self, loss_dict: Dict[str, torch.Tensor]):
        self.optimizer.zero_grad(set_to_none=True)
        tasks = list(loss_dict.keys())
        losses = [loss_dict[t] for t in tasks]
        weights = torch.ones(len(tasks), device=self.device)
        if self.balancer is not None:
            weights = torch.relu(self.balancer.weights).to(self.device)
            weights = weights / weights.mean().clamp(min=1e-6)

        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        grads_per_task = {}
        grad_norms = {}
        for w, task, loss in zip(weights, tasks, losses):
            grad = torch.autograd.grad(w * loss, shared_params, retain_graph=True, allow_unused=True)
            grads_per_task[task] = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, shared_params)]
            grad_norms[task] = torch.stack([g.norm() for g in grads_per_task[task]]).mean().item()

        if self.cfg.use_pcgrad:
            merged = pcgrad(grads_per_task)
        else:
            merged = [sum(grads) for grads in zip(*grads_per_task.values())]

        for p, g in zip(shared_params, merged):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.balancer is not None:
            self.balancer.step({task: loss.detach() for task, loss in loss_dict.items()}, grad_norms)
