from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha_kl: float = 0.5, alpha_feature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha_kl = alpha_kl
        self.alpha_feature = alpha_feature
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        student_features: torch.Tensor | None = None,
        teacher_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.ce(student_logits, targets)
        total = ce_loss
        metrics = {"ce": float(ce_loss.detach().cpu())}

        if teacher_logits is not None and self.alpha_kl > 0:
            t = self.temperature
            kl = F.kl_div(
                F.log_softmax(student_logits / t, dim=1),
                F.softmax(teacher_logits.detach() / t, dim=1),
                reduction="batchmean",
            ) * (t * t)
            total = (1.0 - self.alpha_kl) * total + self.alpha_kl * kl
            metrics["kl"] = float(kl.detach().cpu())

        if (
            student_features is not None
            and teacher_features is not None
            and self.alpha_feature > 0
            and student_features.shape == teacher_features.shape
        ):
            feat = F.mse_loss(student_features, teacher_features.detach())
            total = total + self.alpha_feature * feat
            metrics["feature"] = float(feat.detach().cpu())

        metrics["total"] = float(total.detach().cpu())
        return total, metrics
