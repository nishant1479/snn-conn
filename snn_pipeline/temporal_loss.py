from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def temporal_cross_entropy(logits_per_t: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits_per_t.ndim != 3:
        raise ValueError("Expected temporal logits with shape [batch, time, classes].")
    return torch.stack(
        [F.cross_entropy(logits_per_t[:, t], targets) for t in range(logits_per_t.shape[1])],
        dim=0,
    )


class GradNormTemporalLoss(nn.Module):
    def __init__(self, time_steps: int, alpha: float = 1.5, eps: float = 1e-8) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.alpha = alpha
        self.eps = eps
        self.loss_weights = nn.Parameter(torch.ones(time_steps))
        self.register_buffer("initial_losses", torch.zeros(time_steps))
        self.initialized = False

    def forward(
        self,
        logits_per_t: torch.Tensor,
        targets: torch.Tensor,
        shared_parameters: Iterable[torch.nn.Parameter],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        losses = temporal_cross_entropy(logits_per_t, targets)
        if losses.numel() != self.time_steps:
            raise ValueError(f"Expected {self.time_steps} temporal losses, got {losses.numel()}.")

        if not self.initialized:
            self.initial_losses.copy_(losses.detach().clamp_min(self.eps))
            self.initialized = True

        weighted_losses = self.loss_weights * losses
        model_loss = torch.sum(self.loss_weights.detach() * losses)
        gradnorm_loss = self._gradnorm_loss(weighted_losses, losses, tuple(shared_parameters))
        metrics = {
            "temporal_ce": float(losses.detach().mean().cpu()),
            "loss_weight_min": float(self.loss_weights.detach().min().cpu()),
            "loss_weight_max": float(self.loss_weights.detach().max().cpu()),
        }
        return model_loss, gradnorm_loss, losses, metrics

    def _gradnorm_loss(
        self,
        weighted_losses: torch.Tensor,
        losses: torch.Tensor,
        shared_parameters: tuple[torch.nn.Parameter, ...],
    ) -> torch.Tensor:
        if not shared_parameters:
            raise ValueError("GradNorm requires at least one shared parameter tensor.")

        grad_norms = []
        for loss in weighted_losses:
            grads = torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            norm_terms = [grad.norm(2) for grad in grads if grad is not None]
            if not norm_terms:
                grad_norms.append(loss.new_zeros(()))
            else:
                grad_norms.append(torch.norm(torch.stack(norm_terms), 2))
        grad_norms_t = torch.stack(grad_norms)

        with torch.no_grad():
            loss_ratios = losses.detach() / self.initial_losses.clamp_min(self.eps)
            inverse_train_rates = loss_ratios / loss_ratios.mean().clamp_min(self.eps)
            target_norms = grad_norms_t.detach().mean() * inverse_train_rates.pow(self.alpha)

        return F.l1_loss(grad_norms_t, target_norms, reduction="sum")

    @torch.no_grad()
    def renormalize_(self) -> None:
        self.loss_weights.clamp_(min=self.eps)
        self.loss_weights.mul_(self.time_steps / self.loss_weights.sum().clamp_min(self.eps))


def get_named_parameters(model: nn.Module, names: list[str]) -> tuple[torch.nn.Parameter, ...]:
    params = dict(model.named_parameters())
    missing = [name for name in names if name not in params]
    if missing:
        available = ", ".join(params.keys())
        raise ValueError(f"Unknown shared parameter(s) for GradNorm: {missing}. Available: {available}")
    return tuple(params[name] for name in names)
