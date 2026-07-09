from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from .distillation import DistillationLoss
from .metrics import accuracy


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
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0
    grad_clip = cfg["train"].get("grad_clip_norm")

    for frames, targets in tqdm(loader, desc="train", leave=False):
        frames = frames.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if teacher is not None:
            logits, features = model(frames, return_features=True)
            with torch.no_grad():
                t_logits, t_features = teacher(frames, return_features=True)
            loss, _ = criterion(logits, targets, t_logits, features, t_features)
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
