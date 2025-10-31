from typing import Dict, Optional

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = alpha * targets * loss + (1 - alpha) * (1 - targets) * loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    prob = prob.view(prob.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (prob * targets).sum(dim=1)
    union = prob.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return (1 - dice).mean()


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None,
                    positive_weight: float = 1.0) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if mask is not None:
        if positive_weight != 1.0:
            weight = torch.where(targets > 0.5, positive_weight, 1.0)
            weighted = loss * weight * mask
            denom = (weight * mask).sum() + 1e-6
            return weighted.sum() / denom
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-6)
    if positive_weight != 1.0:
        weight = torch.where(targets > 0.5, positive_weight, 1.0)
        return (loss * weight).mean()
    return loss.mean()


def tad_loss(domain_map_logits: torch.Tensor,
             domain_map_target: torch.Tensor,
             boundary_map_logits: Optional[torch.Tensor] = None,
             boundary_map_target: Optional[torch.Tensor] = None,
             boundary_weight_map: Optional[torch.Tensor] = None,
             map_weight: float = 1.0,
             boundary_weight: float = 1.0,
             boundary_pos_weight: float = 5.0) -> Dict[str, torch.Tensor]:
    results: Dict[str, torch.Tensor] = {}
    dom_bce = F.binary_cross_entropy_with_logits(domain_map_logits, domain_map_target)
    dom_dice = dice_loss(domain_map_logits, domain_map_target)
    dom_total = (dom_bce + dom_dice) * map_weight
    total = dom_total
    results['map'] = dom_total

    if boundary_map_logits is not None and boundary_map_target is not None and boundary_weight > 0.0:
        if boundary_weight_map is not None:
            weight = torch.where(boundary_weight_map > 0.5, boundary_pos_weight, 1.0)
        else:
            weight = torch.where(boundary_map_target > 0.5, boundary_pos_weight, 1.0)
        bce = F.binary_cross_entropy_with_logits(boundary_map_logits, boundary_map_target, reduction='none')
        denom = (weight).sum() + 1e-6
        bce = (bce * weight).sum() / denom
        dice = dice_loss(boundary_map_logits, boundary_map_target)
        boundary_total = (bce + dice) * boundary_weight
        total = total + boundary_total
        results['boundary'] = boundary_total

    results['total'] = total
    return results


def stripe_loss(logits: torch.Tensor, target: torch.Tensor, boundary_prior: Optional[torch.Tensor] = None,
                prior_weight: float = 0.05, positive_weight: float = 1.0, area_weight: float = 0.0,
                orientation_prior: Optional[torch.Tensor] = None, orientation_weight: float = 0.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    if positive_weight != 1.0:
        weight = torch.where(target > 0.5, positive_weight, 1.0)
        bce = (bce * weight).mean()
    else:
        bce = bce.mean()
    dice = dice_loss(logits, target)
    area_penalty = 0.0
    if area_weight > 0.0:
        prob = torch.sigmoid(logits)
        pred_mean = prob.mean()
        target_mean = target.mean()
        area_penalty = area_weight * torch.relu(pred_mean - target_mean)
    prior_term = 0.0
    if boundary_prior is not None:
        prior_term = prior_weight * F.binary_cross_entropy_with_logits(logits, boundary_prior, reduction='mean')
    orientation_term = 0.0
    if orientation_prior is not None and orientation_weight > 0.0:
        orientation_term = orientation_weight * F.binary_cross_entropy_with_logits(logits, orientation_prior, reduction='mean')
    return bce + dice + prior_term + area_penalty + orientation_term


def loop_loss(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor,
              pred_offset: torch.Tensor, target_offset: torch.Tensor,
              valid_mask: torch.Tensor, corner_prior: Optional[torch.Tensor] = None, prior_weight: float = 0.05,
              positive_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap, reduction='none')
    weight_map = torch.where(target_heatmap > 0.5, positive_weight, 1.0)
    heat = (bce * weight_map).mean()
    off = (F.l1_loss(pred_offset, target_offset, reduction='none') * valid_mask.unsqueeze(1)).sum() / (valid_mask.sum() + 1e-6)
    if corner_prior is not None:
        prior = F.binary_cross_entropy_with_logits(pred_heatmap, corner_prior, reduction='mean')
        heat = heat + prior_weight * prior
    return {'heat': heat, 'offset': off}


def consistency_losses(loop_heatmap: torch.Tensor, stripe_mask: Optional[torch.Tensor],
                      tad_boundary_logits: Optional[torch.Tensor], metadata: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    if 'corner_prior' in metadata:
        losses['corner'] = F.binary_cross_entropy_with_logits(loop_heatmap, metadata['corner_prior'], reduction='mean')
    if stripe_mask is not None and 'boundary_prior' in metadata:
        losses['edge'] = F.binary_cross_entropy_with_logits(stripe_mask, metadata['boundary_prior'], reduction='mean')
    if tad_boundary_logits is not None and 'loop_prior' in metadata:
        prob = torch.sigmoid(tad_boundary_logits)
        losses['sequence'] = F.mse_loss(prob, metadata['loop_prior'])
    return losses
