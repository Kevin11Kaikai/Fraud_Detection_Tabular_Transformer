"""Bernoulli distillation with the parent's exact MI-weighted Hid-Mix."""
import torch
from torch.nn import functional as F


def mixed_loss(logits, labels, permutation, mix_weight, teacher_probability, available, coefficient):
    z = logits.reshape(-1).float()
    y = labels.float()
    a = mix_weight.float()
    j = torch.as_tensor(permutation, device=z.device, dtype=torch.long)
    q = teacher_probability.detach().float()
    m = available.float()
    hard = (a * F.binary_cross_entropy_with_logits(z, y, reduction='none')
            + (1-a) * F.binary_cross_entropy_with_logits(z, y[j], reduction='none')).mean()
    soft = (a * m * F.binary_cross_entropy_with_logits(z, q, reduction='none')
            + (1-a) * m[j] * F.binary_cross_entropy_with_logits(z, q[j], reduction='none')).mean()
    return hard + coefficient * soft, hard.detach(), soft.detach()
