from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import torch
from torch import nn

from .data import build_dataloaders
from .models import build_model
from .temporal_loss import GradNormTemporalLoss, get_named_parameters
from .training import evaluate, load_weights, train_one_epoch


def suggest_skip_config(trial: optuna.Trial) -> dict[str, Any]:
    count = trial.suggest_int("skip_count", 0, 2)
    if count == 0:
        position = "block_output"
    elif count == 1:
        position = trial.suggest_categorical("skip_position", ["conv1", "block_output"])
    else:
        position = "both"
    return {
        "enabled": count > 0,
        "type": trial.suggest_categorical("skip_type", ["add", "concat"]),
        "count": count,
        "position": position,
    }


def run_skip_search(cfg: dict[str, Any]) -> optuna.Study:
    search_cfg = cfg["optuna"]
    train_loader, val_loader, _, num_classes = build_dataloaders(cfg)
    device = torch.device(cfg["device"])

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = deepcopy(cfg)
        trial_cfg["model"]["num_classes"] = num_classes
        trial_cfg["model"]["skip"] = suggest_skip_config(trial)
        model = build_model(trial_cfg).to(device)
        if search_cfg.get("pretrained_checkpoint"):
            load_weights(model, search_cfg["pretrained_checkpoint"], strict=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial_cfg["train"]["lr"],
            weight_decay=trial_cfg["train"]["weight_decay"],
        )
        temporal_loss = None
        temporal_optimizer = None
        temporal_shared_parameters = ()
        loss_cfg = trial_cfg["train"].get("loss", {})
        if loss_cfg.get("method") == "gradnorm":
            gradnorm_cfg = loss_cfg.get("gradnorm", {})
            temporal_loss = GradNormTemporalLoss(
                time_steps=trial_cfg["dataset"]["time_bins"],
                alpha=gradnorm_cfg.get("alpha", 1.5),
                eps=gradnorm_cfg.get("eps", 1e-8),
            ).to(device)
            temporal_optimizer = torch.optim.Adam(
                temporal_loss.parameters(),
                lr=gradnorm_cfg.get("lr", trial_cfg["train"]["lr"]),
            )
            temporal_shared_parameters = get_named_parameters(
                model,
                gradnorm_cfg.get("shared_parameters", ["fc.weight"]),
            )
        for _ in range(search_cfg["finetune_epochs"]):
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                criterion,
                trial_cfg,
                temporal_loss=temporal_loss,
                temporal_optimizer=temporal_optimizer,
                temporal_shared_parameters=temporal_shared_parameters,
            )
        val = evaluate(model, val_loader, device)
        trial.set_user_attr("skip", trial_cfg["model"]["skip"])
        return val["acc"]

    study = optuna.create_study(
        direction=search_cfg.get("direction", "maximize"),
        study_name=search_cfg.get("study_name", "skip_search"),
        storage=search_cfg.get("storage"),
        load_if_exists=bool(search_cfg.get("storage")),
    )
    study.optimize(objective, n_trials=search_cfg["trials"])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with (Path(cfg["output_dir"]) / "best_skip_config.yaml").open("w", encoding="utf-8") as handle:
        import yaml

        yaml.safe_dump(study.best_trial.user_attrs["skip"], handle, sort_keys=False)
    return study
