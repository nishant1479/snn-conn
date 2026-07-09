from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from snn_pipeline.config import load_config, save_config
from snn_pipeline.data import build_dataloaders
from snn_pipeline.metrics import Timer, append_result, count_parameters
from snn_pipeline.models import build_model
from snn_pipeline.skip_search import run_skip_search
from snn_pipeline.training import evaluate, load_weights, save_checkpoint, train_one_epoch
from snn_pipeline.distillation import DistillationLoss


STAGES = {
    "baseline": {"skip": False, "multi_threshold": False, "teacher": False},
    "skip": {"skip": True, "multi_threshold": False, "teacher": False},
    "threshold": {"skip": True, "multi_threshold": True, "teacher": False},
    "full": {"skip": True, "multi_threshold": True, "teacher": True},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_stage(cfg: dict, stage: str) -> dict:
    out = deepcopy(cfg)
    toggles = STAGES[stage]
    out["model"]["skip"]["enabled"] = toggles["skip"]
    out["model"]["multi_threshold"]["enabled"] = toggles["multi_threshold"]
    out["teacher"]["enabled"] = toggles["teacher"]
    return out


def train_teacher(cfg: dict, train_loader, val_loader, device: torch.device, out_dir: Path) -> Path:
    teacher = build_model(cfg, ann=True).to(device)
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=cfg["teacher"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_path = out_dir / "teacher_best.pt"
    for epoch in range(cfg["teacher"]["epochs"]):
        train_one_epoch(teacher, train_loader, optimizer, device, criterion, cfg)
        val = evaluate(teacher, val_loader, device)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            save_checkpoint(best_path, teacher, cfg, {"epoch": epoch, "val_acc": best_acc})
    return best_path


def run_train(cfg: dict, stage: str, init_checkpoint: str | None = None) -> None:
    cfg = configure_stage(cfg, stage)
    seed_everything(cfg["seed"])
    out_dir = Path(cfg["output_dir"]) / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg)
    cfg["model"]["num_classes"] = num_classes
    device = torch.device(cfg["device"])

    model = build_model(cfg).to(device)
    if init_checkpoint:
        load_weights(model, init_checkpoint, strict=False)

    teacher = None
    if cfg["teacher"]["enabled"]:
        teacher = build_model(cfg, ann=True).to(device)
        teacher_ckpt = cfg["teacher"].get("checkpoint")
        if teacher_ckpt:
            load_weights(teacher, teacher_ckpt, strict=False)
        else:
            teacher_ckpt = train_teacher(cfg, train_loader, val_loader, device, out_dir)
            load_weights(teacher, teacher_ckpt, strict=False)

    criterion = (
        DistillationLoss(
            temperature=cfg["teacher"]["temperature"],
            alpha_kl=cfg["teacher"]["alpha_kl"],
            alpha_feature=cfg["teacher"]["alpha_feature"],
        )
        if teacher is not None
        else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_acc = -1.0
    best_path = out_dir / "best.pt"
    with Timer() as timer:
        for epoch in range(cfg["train"]["epochs"]):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, criterion, cfg, teacher)
            val_metrics = evaluate(model, val_loader, device)
            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                save_checkpoint(
                    best_path,
                    model,
                    cfg,
                    {"epoch": epoch, "train": train_metrics, "val": val_metrics},
                )

    load_weights(model, best_path, strict=False)
    test_metrics = evaluate(model, test_loader, device) if test_loader is not None else {"acc": None, "loss": None}
    append_result(
        Path(cfg["output_dir"]) / "ablation_results.csv",
        {
            "stage": stage,
            "val_acc": best_acc,
            "test_acc": test_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "training_time_sec": timer.elapsed,
            "parameters": count_parameters(model),
            "checkpoint": str(best_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=[*STAGES.keys(), "skip-search"], required=True)
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "skip-search":
        run_skip_search(cfg)
    else:
        run_train(cfg, args.stage, args.init_checkpoint)


if __name__ == "__main__":
    main()
