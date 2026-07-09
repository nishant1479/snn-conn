from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def atan_surrogate(x: torch.Tensor) -> torch.Tensor:
    spike = (x > 0).to(x.dtype)
    grad = 1.0 / (1.0 + (torch.pi * x).pow(2))
    return spike.detach() - grad.detach() + grad


def fast_sigmoid_surrogate(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    spike = (x > 0).to(x.dtype)
    grad = x / (1.0 + slope * x.abs())
    return spike.detach() - grad.detach() + grad


def get_surrogate(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "atan":
        return atan_surrogate
    if name == "fast_sigmoid":
        return fast_sigmoid_surrogate
    raise ValueError(f"Unknown surrogate '{name}'.")


class LIFCell(nn.Module):
    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate: str = "atan",
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.detach_reset = detach_reset
        self.spike_fn = get_surrogate(surrogate)

    def forward(self, input_: torch.Tensor, mem: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mem is None:
            mem = torch.zeros_like(input_)
        mem = self.beta * mem + input_
        spike = self.spike_fn(mem - self.threshold)
        reset = spike.detach() if self.detach_reset else spike
        mem = mem - reset * self.threshold
        return spike, mem


class MultiThresholdLIFCell(nn.Module):
    """LIF with multiple firing thresholds.

    Parallel mode emits a normalized sum of spikes across thresholds. Cascade mode
    subtracts threshold mass sequentially, which approximates quantized multi-spike firing.
    """

    def __init__(
        self,
        beta: float = 0.9,
        thresholds: list[float] | tuple[float, ...] = (0.5, 1.0, 1.5),
        surrogate: str = "atan",
        detach_reset: bool = True,
        mode: str = "parallel",
    ) -> None:
        super().__init__()
        if not thresholds:
            raise ValueError("MultiThresholdLIFCell requires at least one threshold.")
        self.beta = beta
        self.detach_reset = detach_reset
        self.mode = mode
        self.spike_fn = get_surrogate(surrogate)
        self.register_buffer("thresholds", torch.tensor(sorted(thresholds), dtype=torch.float32))

    def forward(self, input_: torch.Tensor, mem: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mem is None:
            mem = torch.zeros_like(input_)
        mem = self.beta * mem + input_
        thresholds = self.thresholds.to(device=mem.device, dtype=mem.dtype)

        if self.mode == "parallel":
            spikes = [self.spike_fn(mem - thr) for thr in thresholds]
            spike = torch.stack(spikes, dim=0).mean(dim=0)
            reset_mass = torch.stack(spikes, dim=0).sum(dim=0) * thresholds.mean()
        elif self.mode == "cascade":
            residual = mem
            spikes = []
            reset_mass = torch.zeros_like(mem)
            for thr in thresholds:
                spk = self.spike_fn(residual - thr)
                spikes.append(spk)
                reset = spk.detach() if self.detach_reset else spk
                residual = residual - reset * thr
                reset_mass = reset_mass + reset * thr
            spike = torch.stack(spikes, dim=0).mean(dim=0)
        else:
            raise ValueError("mode must be 'parallel' or 'cascade'.")

        reset = reset_mass.detach() if self.detach_reset else reset_mass
        mem = mem - reset
        return spike, mem


def make_neuron(cfg: dict) -> nn.Module:
    lif = cfg["lif"]
    mt = cfg.get("multi_threshold", {})
    if mt.get("enabled", False):
        return MultiThresholdLIFCell(
            beta=lif["beta"],
            thresholds=mt["thresholds"],
            surrogate=lif["surrogate"],
            detach_reset=lif["detach_reset"],
            mode=mt.get("mode", "parallel"),
        )
    return LIFCell(
        beta=lif["beta"],
        threshold=lif["threshold"],
        surrogate=lif["surrogate"],
        detach_reset=lif["detach_reset"],
    )
