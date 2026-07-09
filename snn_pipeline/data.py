from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split


def _require_tonic():
    try:
        import tonic
        import tonic.transforms as transforms
    except ImportError as exc:
        raise ImportError(
            "Install tonic to use neuromorphic datasets: pip install tonic"
        ) from exc
    return tonic, transforms


def _dataset_spec(name: str) -> tuple[type, tuple[int, int, int], int]:
    tonic, _ = _require_tonic()
    normalized = name.lower()
    if normalized in {"dvs128_gesture", "dvs_gesture", "dvsgesture"}:
        return tonic.datasets.DVSGesture, (2, 128, 128), 11
    if normalized in {"cifar10_dvs", "cifar10dvs"}:
        return tonic.datasets.CIFAR10DVS, (2, 128, 128), 10
    raise ValueError(f"Unsupported dataset '{name}'. Use dvs128_gesture or cifar10_dvs.")


def _make_transform(sensor_size: tuple[int, int, int], time_bins: int, image_size: int):
    _, transforms = _require_tonic()
    frame_transform = transforms.Compose(
        [
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_bins),
            torch.from_numpy,
            lambda x: x.float(),
            lambda x: torch.nn.functional.interpolate(
                x,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ),
        ]
    )
    return frame_transform


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader | None, int]:
    tonic, _ = _require_tonic()
    ds_cfg = cfg["dataset"]
    dataset_cls, sensor_size, num_classes = _dataset_spec(ds_cfg["name"])
    transform = _make_transform(sensor_size, ds_cfg["time_bins"], ds_cfg["image_size"])

    data_dir = Path(ds_cfg["data_dir"])
    cache_dir = Path(ds_cfg["cache_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset_cls.__name__ == "DVSGesture":
        train_set = dataset_cls(save_to=str(data_dir), train=True, transform=transform)
        test_set = dataset_cls(save_to=str(data_dir), train=False, transform=transform)
    else:
        full_set = dataset_cls(save_to=str(data_dir), transform=transform)
        test_len = max(1, int(0.2 * len(full_set)))
        train_len = len(full_set) - test_len
        train_set, test_set = random_split(
            full_set,
            [train_len, test_len],
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )

    val_len = max(1, int(ds_cfg["val_fraction"] * len(train_set)))
    train_len = len(train_set) - val_len
    train_set, val_set = random_split(
        train_set,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    common = {
        "batch_size": ds_cfg["batch_size"],
        "num_workers": ds_cfg["num_workers"],
        "pin_memory": torch.cuda.is_available() and cfg.get("device") != "cpu",
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader, num_classes
