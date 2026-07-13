from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from .distillation import DistillationLoss
from .metrics import accuracy
from .temporal_loss import GradNormTemporalLoss, temporal_cross_entropy


def save_checkpoint(path: str | Path, model: nn.Module, cfg: dict[str, Any], metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "metrics": metrics}, path)


def load_weights(model: nn.Module, checkpoint: str | Path, strict: bool = False) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"Loaded with missing={len(missing)} unexpected={len(unexpected)}")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    cfg: dict[str, Any],
    teacher: nn.Module | None = None,
    temporal_loss: GradNormTemporalLoss | None = None,
    temporal_optimizer: torch.optim.Optimizer | None = None,
    temporal_shared_parameters: tuple[torch.nn.Parameter, ...] = (),
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    grad_clip = cfg["train"].get("grad_clip_norm")
    loss_method = cfg["train"].get("loss", {}).get("method", "mean_logits")
    if loss_method == "original_tet":
        loss_method = "tet"
    if loss_method not in {"mean_logits", "tet", "gradnorm"}:
        raise ValueError("train.loss.method must be mean_logits, tet/original_tet, or gradnorm.")

    for frames, targets in tqdm(loader, desc="train", leave=False):
        frames = frames.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if loss_method == "gradnorm":
            if teacher is not None:
                raise ValueError("GradNorm temporal loss is not combined with teacher distillation in this pipeline.")
            if temporal_loss is None or temporal_optimizer is None:
                raise ValueError("GradNorm temporal loss requires temporal_loss and temporal_optimizer.")
            logits_per_t = model(frames, return_temporal=True)
            logits = logits_per_t.mean(dim=1)
            loss, gradnorm_loss, _, _ = temporal_loss(logits_per_t, targets, temporal_shared_parameters)
            temporal_optimizer.zero_grad(set_to_none=True)
            weight_grads = torch.autograd.grad(gradnorm_loss, temporal_loss.loss_weights, retain_graph=True)
            temporal_loss.loss_weights.grad = weight_grads[0]
            temporal_optimizer.step()
            temporal_loss.renormalize_()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        elif loss_method == "tet" and teacher is None:
            logits_per_t = model(frames, return_temporal=True)
            logits = logits_per_t.mean(dim=1)
            loss = temporal_cross_entropy(logits_per_t, targets).mean()
            loss.backward()
        elif teacher is not None:
            logits, features = model(frames, return_features=True)
            with torch.no_grad():
                t_logits, t_features = teacher(frames, return_features=True)
            loss, _ = criterion(logits, targets, t_logits, features, t_features)
            loss.backward()
        else:
            logits = model(frames)
            loss = criterion(logits, targets)
            loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch = targets.numel()
        total_seen += batch
        total_loss += float(loss.detach().cpu()) * batch
        total_acc += accuracy(logits.detach(), targets) * batch

    return {"loss": total_loss / total_seen, "acc": total_acc / total_seen}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    for frames, targets in tqdm(loader, desc="eval", leave=False):
        frames = frames.to(device)
        targets = targets.to(device)
        logits = model(frames)
        batch = targets.numel()
        total_seen += batch
        total_loss += float(ce(logits, targets).detach().cpu())
        total_acc += accuracy(logits, targets) * batch
    return {"loss": total_loss / total_seen, "acc": total_acc / total_seen}
