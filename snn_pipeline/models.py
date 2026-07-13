from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from .neurons import make_neuron


class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        model_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.skip_cfg = model_cfg.get("skip", {})
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.n1 = make_neuron(model_cfg)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.n2 = make_neuron(model_cfg)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.concat_proj = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
        self.conv1_proj = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)

    def _apply_extra_skip(self, x: torch.Tensor, identity: torch.Tensor, where: str) -> torch.Tensor:
        if not self.skip_cfg.get("enabled", False):
            return x
        count = int(self.skip_cfg.get("count", 0))
        position = self.skip_cfg.get("position", "block_output")
        if count <= 0 or position not in {where, "both"}:
            return x
        skip_type = self.skip_cfg.get("type", "add")
        if skip_type == "add":
            return x + identity
        if skip_type == "concat":
            joined = torch.cat([x, identity], dim=1)
            return self.conv1_proj(joined) if where == "conv1" else self.concat_proj(joined)
        raise ValueError("skip.type must be add or concat.")

    def forward(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor | None],
        prefix: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        identity = self.downsample(x)
        out = self.bn1(self.conv1(x))
        conv1_identity = identity
        if conv1_identity.shape[-2:] != out.shape[-2:]:
            conv1_identity = torch.nn.functional.interpolate(conv1_identity, size=out.shape[-2:], mode="nearest")
        out = self._apply_extra_skip(out, conv1_identity, "conv1")
        out, state[f"{prefix}.n1"] = self.n1(out, state.get(f"{prefix}.n1"))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self._apply_extra_skip(out, identity, "block_output")
        out, state[f"{prefix}.n2"] = self.n2(out, state.get(f"{prefix}.n2"))
        return out, state


class SpikingResNet18(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        c = model_cfg["base_channels"]
        self.num_classes = model_cfg["num_classes"]
        self.stem = nn.Sequential(
            nn.Conv2d(2, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.stem_neuron = make_neuron(model_cfg)
        self.in_channels = c
        self.layers = nn.ModuleList(
            [
                self._make_layer(c, 2, stride=1, model_cfg=model_cfg, name="layer1"),
                self._make_layer(c * 2, 2, stride=2, model_cfg=model_cfg, name="layer2"),
                self._make_layer(c * 4, 2, stride=2, model_cfg=model_cfg, name="layer3"),
                self._make_layer(c * 8, 2, stride=2, model_cfg=model_cfg, name="layer4"),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 8, self.num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int, model_cfg: dict[str, Any], name: str):
        layers = OrderedDict()
        layers[f"{name}_0"] = SpikingBasicBlock(self.in_channels, out_channels, stride, model_cfg)
        self.in_channels = out_channels
        for idx in range(1, blocks):
            layers[f"{name}_{idx}"] = SpikingBasicBlock(self.in_channels, out_channels, 1, model_cfg)
        return nn.ModuleDict(layers)

    def forward(self, frames: torch.Tensor, return_features: bool = False, return_temporal: bool = False):
        if frames.ndim != 5:
            raise ValueError("Expected frames with shape [batch, time, channels, height, width].")
        state: dict[str, torch.Tensor | None] = {}
        logits_per_t = []
        features = []
        for t in range(frames.shape[1]):
            x = self.stem(frames[:, t])
            x, state["stem"] = self.stem_neuron(x, state.get("stem"))
            for layer in self.layers:
                for name, block in layer.items():
                    x, state = block(x, state, name)
            feat = self.pool(x).flatten(1)
            features.append(feat)
            logits_per_t.append(self.fc(feat))
        temporal_logits = torch.stack(logits_per_t, dim=1)
        logits = temporal_logits.mean(dim=1)
        feature = torch.stack(features, dim=1).mean(dim=1)
        if return_features and return_temporal:
            return logits, feature, temporal_logits
        if return_features:
            return logits, feature
        if return_temporal:
            return temporal_logits
        return logits


class ANNResNet18(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        c = model_cfg["base_channels"]
        self.num_classes = model_cfg["num_classes"]
        self.stem = nn.Sequential(nn.Conv2d(2, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.in_channels = c
        self.layers = nn.Sequential(
            self._make_layer(c, 2, stride=1),
            self._make_layer(c * 2, 2, stride=2),
            self._make_layer(c * 4, 2, stride=2),
            self._make_layer(c * 8, 2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 8, self.num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = [ANNBasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ANNBasicBlock(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, frames: torch.Tensor, return_features: bool = False, return_temporal: bool = False):
        x = frames.mean(dim=1) if frames.ndim == 5 else frames
        x = self.stem(x)
        x = self.layers(x)
        feature = self.pool(x).flatten(1)
        logits = self.fc(feature)
        if return_features and return_temporal:
            return logits, feature, logits.unsqueeze(1)
        if return_features:
            return logits, feature
        if return_temporal:
            return logits.unsqueeze(1)
        return logits


class ANNBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def build_model(cfg: dict[str, Any], ann: bool = False) -> nn.Module:
    arch = cfg["model"]["architecture"]
    if arch != "resnet18_snn":
        raise ValueError("Only resnet18_snn is implemented in this scaffold.")
    return ANNResNet18(cfg) if ann else SpikingResNet18(cfg)
