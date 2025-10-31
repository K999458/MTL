import os
import time
import math
from collections import defaultdict
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.ops import box_iou

from .config import TrainingConfig
from .data import DatasetPaths, make_dataloaders
from .detection import build_tad_detector
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
        self._cudnn_prev_state = torch.backends.cudnn.enabled
        self._cudnn_prev_bench = torch.backends.cudnn.benchmark
        if getattr(config, 'backbone_type', 'unet').lower() == 'gcn':
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            print("[trainer] 禁用 cuDNN 以兼容 GCN 主干。")
        self.model = MultitaskHiCNet(in_channels=config.input_channels,
                                     base_channels=config.base_channels,
                                     tad_band_width=config.tad_band_width,
                                     use_axial_attention=config.use_axial_attention,
                                     backbone_type=getattr(config, 'backbone_type', 'unet'),
                                     gcn_kernel_size=getattr(config, 'gcn_kernel_size', 9),
                                     gcn_dilation=getattr(config, 'gcn_dilation', 1),
                                     gcn_drop_path=getattr(config, 'gcn_drop_path', 0.1),
                                     gcn_stage_blocks=getattr(config, 'gcn_stage_blocks', None),
                                     gcn_run_on_cpu=getattr(config, 'gcn_run_on_cpu', False)).to(self.device)
        self.tad_detector = None
        if config.use_tad_detector:
            self.tad_detector = build_tad_detector(backbone_name=config.tad_detector_backbone,
                                                   pretrained=config.tad_detector_pretrained,
                                                   input_channels=config.input_channels,
                                                   num_classes=2).to(self.device)

        params = list(self.model.parameters())
        if self.tad_detector is not None:
            params += list(self.tad_detector.parameters())
        self.optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
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
                                        batch_sizes=batch_sizes,
                                        num_workers=config.num_workers,
                                        pin_memory=config.pin_memory,
                                        loop_manifest_name=config.loop_manifest,
                                        loop_label_name=config.loop_label,
                                        stripe_manifest_name=config.stripe_manifest,
                                        stripe_label_name=config.stripe_label,
                                        tad_manifest_name=getattr(config, 'tad_manifest', None),
                                        tad_binsize=getattr(config, 'tad_binsize', 10000),
                                        tad_domain_label_name=getattr(config, 'tad_domain_label', 'tad_domains_10kb.bed'),
                                        tad_use_detection=config.use_tad_detector,
                                        tad_detection_min_bins=config.tad_detection_min_bins,
                                        tad_detection_max_instances=config.tad_detection_max_instances)
        self.latest_tad_loss_detail: Dict[str, torch.Tensor] = {}
        self.latest_tad_detection_losses: Dict[str, torch.Tensor] = {}

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            if self.tad_detector is not None:
                self.tad_detector.train()
            iterators = {k: iter(v) for k, v in self.loaders.items()}
            start = time.time()
            epoch_loss_sums = defaultdict(float)
            epoch_metrics = defaultdict(lambda: [0.0, 0])
            batches = None
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
                        for task, loss in loss_dict.items():
                            epoch_loss_sums[task] += loss.item()
                        detector_loss = self.latest_tad_loss_detail.get('detector')
                        if detector_loss is not None:
                            epoch_loss_sums['tad_detector'] += float(detector_loss.item())
                        self.backward(loss_dict)
                        if step % self.cfg.log_interval == 0:
                            metrics = self.compute_metrics(batches)
                            self._accumulate_epoch_metrics(epoch_metrics, metrics)
                            pbar.set_postfix({
                                'loop': f"{loss_dict['loop'].item():.4f}",
                                'stripe': f"{loss_dict['stripe'].item():.4f}",
                                'tad': f"{loss_dict['tad'].item():.4f}",
                                'lhit': f"{metrics.get('loop_hit', float('nan')):.3f}",
                                'siou': f"{metrics.get('stripe_iou', float('nan')):.3f}",
                                'tdice': f"{metrics.get('tad_map_dice', float('nan')):.3f}",
                                'tbnd': f"{metrics.get('tad_boundary_dice', float('nan')):.3f}",
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
                        for task, loss in loss_dict.items():
                            epoch_loss_sums[task] += loss.item()
                        detector_loss = self.latest_tad_loss_detail.get('detector')
                        if detector_loss is not None:
                            epoch_loss_sums['tad_detector'] += float(detector_loss.item())
                        self.backward(loss_dict)
                        if step % self.cfg.log_interval == 0:
                            metrics = self.compute_metrics(batches)
                            self._accumulate_epoch_metrics(epoch_metrics, metrics)
                            loop_loss = loss_dict['loop'].item()
                            stripe_loss = loss_dict['stripe'].item()
                            tad_loss_val = loss_dict['tad'].item()
                            print(
                                f"Epoch {epoch} Step {step}/{self.cfg.steps_per_epoch}: "
                                f"loop={loop_loss:.4f} stripe={stripe_loss:.4f} tad={tad_loss_val:.4f} "
                                f"| loop_hit={metrics.get('loop_hit', float('nan')):.3f} "
                                f"stripe_iou={metrics.get('stripe_iou', float('nan')):.3f} "
                                f"tad_dice={metrics.get('tad_map_dice', float('nan')):.3f} "
                                f"tad_bdice={metrics.get('tad_boundary_dice', float('nan')):.3f}"
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
                    for task, loss in loss_dict.items():
                        epoch_loss_sums[task] += loss.item()
                    detector_loss = self.latest_tad_loss_detail.get('detector')
                    if detector_loss is not None:
                        epoch_loss_sums['tad_detector'] += float(detector_loss.item())
                    self.backward(loss_dict)
                    if step % self.cfg.log_interval == 0:
                        metrics = self.compute_metrics(batches)
                        self._accumulate_epoch_metrics(epoch_metrics, metrics)
                        loop_loss = loss_dict['loop'].item()
                        stripe_loss = loss_dict['stripe'].item()
                        tad_loss_val = loss_dict['tad'].item()
                        print(
                            f"Epoch {epoch} Step {step}/{self.cfg.steps_per_epoch}: "
                            f"loop={loop_loss:.4f} stripe={stripe_loss:.4f} tad={tad_loss_val:.4f} "
                            f"| loop_hit={metrics.get('loop_hit', float('nan')):.3f} "
                            f"stripe_iou={metrics.get('stripe_iou', float('nan')):.3f} "
                            f"tad_dice={metrics.get('tad_map_dice', float('nan')):.3f} "
                            f"tad_bdice={metrics.get('tad_boundary_dice', float('nan')):.3f}"
                        )

            if batches is not None:
                if (self.cfg.steps_per_epoch % self.cfg.log_interval) != 0 or not epoch_metrics:
                    final_metrics = self.compute_metrics(batches)
                    self._accumulate_epoch_metrics(epoch_metrics, final_metrics)

            avg_losses = {task: epoch_loss_sums[task] / max(1, self.cfg.steps_per_epoch) for task in epoch_loss_sums}
            avg_metrics = self._finalize_epoch_metrics(epoch_metrics)
            if avg_losses or avg_metrics:
                self._print_epoch_summary(epoch, avg_losses, avg_metrics)

            self.scheduler.step()
            elapsed = time.time() - start
            print(f"Epoch {epoch} completed in {elapsed/60:.2f} min | lr={self.optimizer.param_groups[0]['lr']:.2e}")
            if epoch % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch)
        torch.backends.cudnn.enabled = self._cudnn_prev_state
        torch.backends.cudnn.benchmark = self._cudnn_prev_bench

    def save_checkpoint(self, epoch: int):
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        if self.balancer is not None:
            ckpt['balancer_weights'] = self.balancer.weights.data.cpu()
        if self.tad_detector is not None:
            ckpt['tad_detector'] = self.tad_detector.state_dict()
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
            elif isinstance(v, list):
                converted = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        converted.append(item.to(self.device, non_blocking=True))
                    elif isinstance(item, dict):
                        converted.append(self.move_to_device(item))
                    else:
                        converted.append(item)
                out[k] = converted
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
        orientation_prior = stripe_batch['target'].get('orientation_prior')
        if orientation_prior is not None:
            orientation_prior = orientation_prior.to(self.device)
        stripe_total = loss_utils.stripe_loss(outputs_stripe['mask'], stripe_batch['target']['mask'],
                                              stripe_batch['target'].get('boundary_prior'),
                                              positive_weight=self.cfg.stripe_pos_weight,
                                              area_weight=self.cfg.stripe_area_weight,
                                              orientation_prior=orientation_prior,
                                              orientation_weight=self.cfg.stripe_orientation_weight) * self.cfg.lambda_stripe

        outputs_tad = self.model(tad_batch['inputs'], task='tad')
        tad_loss_dict = loss_utils.tad_loss(
            domain_map_logits=outputs_tad['domain_map'],
            domain_map_target=tad_batch['target']['domain_mask'],
            boundary_map_logits=outputs_tad.get('boundary_map'),
            boundary_map_target=tad_batch['target'].get('boundary_mask'),
            boundary_weight_map=tad_batch['target'].get('boundary_wide'),
            map_weight=self.cfg.tad_domain_map_weight,
            boundary_weight=getattr(self.cfg, 'tad_boundary_weight', 1.0),
            boundary_pos_weight=getattr(self.cfg, 'tad_boundary_pos_weight', 5.0)
        )
        tad_total = tad_loss_dict['total'] * self.cfg.lambda_tad

        det_total = torch.tensor(0.0, device=self.device)
        self.latest_tad_detection_losses = {}
        if self.tad_detector is not None:
            detection_targets = tad_batch.get('detection') or []
            images = [img for img in tad_batch['inputs']]
            prepared_targets = []
            for img, det in zip(images, detection_targets):
                if det is None:
                    H, W = img.shape[-2], img.shape[-1]
                    empty = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                        'labels': torch.zeros((0,), dtype=torch.int64, device=self.device),
                        'masks': torch.zeros((0, H, W), dtype=torch.float32, device=self.device),
                        'iscrowd': torch.zeros((0,), dtype=torch.int64, device=self.device),
                        'areas': torch.zeros((0,), dtype=torch.float32, device=self.device)
                    }
                    prepared_targets.append(empty)
                else:
                    prepared_targets.append(det)
            det_losses = self.tad_detector(images, prepared_targets)
            if isinstance(det_losses, dict):
                det_total = sum(det_losses.values())
                self.latest_tad_detection_losses = {k: v.detach() for k, v in det_losses.items()}
            else:
                det_total = det_losses
                self.latest_tad_detection_losses = {'total': det_losses.detach()}
            tad_total = tad_total + det_total * self.cfg.lambda_tad_detector

        self.latest_outputs = {
            'loop': outputs_loop,
            'stripe': outputs_stripe,
            'tad': outputs_tad
        }

        self.latest_tad_loss_detail = {k: v for k, v in tad_loss_dict.items()}
        if self.tad_detector is not None:
            self.latest_tad_loss_detail['detector'] = det_total.detach()

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

        # TAD 2D domain metrics
        try:
            domain_map = self.latest_outputs['tad']['domain_map']
            device = domain_map.device
            target_map = batches['tad']['target']['domain_mask'].to(device)
            prob_map = torch.sigmoid(domain_map)
            thresh = getattr(self.cfg, 'tad_domain_threshold', 0.5)
            pred_map = (prob_map > thresh).float()
            inter = (pred_map * target_map).sum()
            union = ((pred_map + target_map).clamp(max=1.0)).sum() + 1e-6
            dice = (2 * inter) / (pred_map.sum() + target_map.sum() + 1e-6)
            metrics['tad_map_iou'] = (inter / union).item()
            metrics['tad_map_dice'] = dice.item()

            boundary_map = self.latest_outputs['tad'].get('boundary_map')
            boundary_target = batches['tad']['target'].get('boundary_mask')
            if boundary_map is not None and boundary_target is not None:
                boundary_prob = torch.sigmoid(boundary_map)
                boundary_thresh = getattr(self.cfg, 'tad_boundary_threshold', 0.4)
                boundary_pred = (boundary_prob > boundary_thresh).float()
                boundary_true = boundary_target.to(device)
                inter_b = (boundary_pred * boundary_true).sum()
                union_b = ((boundary_pred + boundary_true).clamp(max=1.0)).sum() + 1e-6
                dice_b = (2 * inter_b) / (boundary_pred.sum() + boundary_true.sum() + 1e-6)
                metrics['tad_boundary_iou'] = (inter_b / union_b).item()
                metrics['tad_boundary_dice'] = dice_b.item()
        except Exception:
            pass

        if self.tad_detector is not None:
            try:
                tad_batch = self.move_to_device(batches['tad'])
                images = [img for img in tad_batch['inputs']]
                det_targets = tad_batch.get('detection') or []
                prev_mode = self.tad_detector.training
                self.tad_detector.eval()
                with torch.no_grad():
                    predictions = self.tad_detector(images)
                if prev_mode:
                    self.tad_detector.train()
                score_thresh = getattr(self.cfg, 'tad_detection_score_thresh', 0.5)
                iou_thresh = getattr(self.cfg, 'tad_detection_iou_thresh', 0.3)
                total_gt = 0
                hit_gt = 0
                total_pred = 0
                hit_pred = 0
                for pred, gt in zip(predictions, det_targets):
                    gt_boxes = gt.get('boxes') if isinstance(gt, dict) else torch.zeros((0, 4), device=self.device)
                    pred_mask = pred['scores'] > score_thresh
                    pred_boxes = pred['boxes'][pred_mask]
                    total_gt += gt_boxes.size(0)
                    total_pred += pred_boxes.size(0)
                    if gt_boxes.size(0) == 0 or pred_boxes.size(0) == 0:
                        continue
                    ious = box_iou(pred_boxes, gt_boxes)
                    hit_gt += int((ious.max(dim=0)[0] >= iou_thresh).sum().item())
                    hit_pred += int((ious.max(dim=1)[0] >= iou_thresh).sum().item())
                if total_gt > 0:
                    metrics['tad_det_recall'] = hit_gt / total_gt
                if total_pred > 0:
                    metrics['tad_det_precision'] = hit_pred / total_pred
            except Exception:
                pass

        return metrics

    def _accumulate_epoch_metrics(self, store: defaultdict, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if value is None:
                continue
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                continue
            entry = store[key]
            entry[0] += val
            entry[1] += 1

    def _finalize_epoch_metrics(self, store: defaultdict) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for key, (total, count) in store.items():
            if count > 0:
                results[key] = total / count
        return results

    def _print_epoch_summary(self, epoch: int, loss_avgs: Dict[str, float], metric_avgs: Dict[str, float]):
        loss_order = ['loop', 'stripe', 'tad']
        loss_parts = [f"{k}={loss_avgs[k]:.4f}" for k in loss_order if k in loss_avgs]
        extra_losses = [k for k in loss_avgs if k not in loss_order]
        loss_parts.extend(f"{k}={loss_avgs[k]:.4f}" for k in sorted(extra_losses))

        metric_order = ['loop_hit', 'stripe_iou', 'stripe_dice', 'tad_map_iou', 'tad_map_dice',
                        'tad_boundary_iou', 'tad_boundary_dice', 'tad_det_recall', 'tad_det_precision']
        metric_parts = [f"{k}={metric_avgs[k]:.3f}" for k in metric_order if k in metric_avgs]
        extra_metrics = [k for k in metric_avgs if k not in metric_order]
        metric_parts.extend(f"{k}={metric_avgs[k]:.3f}" for k in sorted(extra_metrics))

        loss_str = ", ".join(loss_parts) if loss_parts else "n/a"
        metric_str = ", ".join(metric_parts) if metric_parts else "n/a"
        print(f"[epoch {epoch} summary] loss({loss_str}) | metrics({metric_str})")

    def backward(self, loss_dict: Dict[str, torch.Tensor]):
        self.optimizer.zero_grad(set_to_none=True)
        tasks = list(loss_dict.keys())
        losses = [loss_dict[t] for t in tasks]
        weights = torch.ones(len(tasks), device=self.device)
        if self.balancer is not None:
            weights = torch.relu(self.balancer.weights).to(self.device)
            weights = weights / weights.mean().clamp(min=1e-6)

        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.tad_detector is not None:
            shared_params.extend(p for p in self.tad_detector.parameters() if p.requires_grad)
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
            torch.nn.utils.clip_grad_norm_(shared_params, self.cfg.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.balancer is not None:
            self.balancer.step({task: loss.detach() for task, loss in loss_dict.items()}, grad_norms)
