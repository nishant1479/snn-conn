from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


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


class PreprocessedTensorDataset(Dataset):
    def __init__(self, root: Path, time_bins: int, image_size: int) -> None:
        self.files = sorted(root.rglob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {root}.")
        self.time_bins = time_bins
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = torch.load(self.files[index], map_location="cpu")
        frames, target = self._split_sample(sample)
        return self._format_frames(frames), int(target)

    @staticmethod
    def _split_sample(sample: Any) -> tuple[Any, Any]:
        if isinstance(sample, dict):
            frame_keys = ("frames", "frame", "data", "x", "events")
            target_keys = ("target", "targets", "label", "labels", "y")
            frame_key = next((key for key in frame_keys if key in sample), None)
            target_key = next((key for key in target_keys if key in sample), None)
            if frame_key is not None and target_key is not None:
                return sample[frame_key], sample[target_key]
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            return sample[0], sample[1]
        raise ValueError("Expected each .pt sample to contain frames and a target label.")

    def _format_frames(self, frames: Any) -> torch.Tensor:
        frames = torch.as_tensor(frames).float()
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames.squeeze(0)
        if frames.ndim != 4:
            raise ValueError(f"Expected frames with 4 dims [time, channels, height, width], got {tuple(frames.shape)}.")

        if frames.shape[1] != 2 and frames.shape[0] == 2:
            frames = frames.permute(1, 0, 2, 3)
        if frames.shape[1] == 1:
            frames = torch.cat([frames, torch.zeros_like(frames)], dim=1)
        if frames.shape[1] > 2:
            frames = frames[:, :2]
        if frames.shape[1] != 2:
            raise ValueError(f"Expected two polarity channels, got shape {tuple(frames.shape)}.")

        if frames.shape[0] != self.time_bins or frames.shape[-1] != self.image_size or frames.shape[-2] != self.image_size:
            frames = F.interpolate(
                frames.permute(1, 0, 2, 3).unsqueeze(0),
                size=(self.time_bins, self.image_size, self.image_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 0, 2, 3)
        return frames


def _build_preprocessed_tensor_sets(data_dir: Path, time_bins: int, image_size: int):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if train_dir.exists() and test_dir.exists() and list(train_dir.rglob("*.pt")) and list(test_dir.rglob("*.pt")):
        return (
            PreprocessedTensorDataset(train_dir, time_bins, image_size),
            PreprocessedTensorDataset(test_dir, time_bins, image_size),
        )
    return None


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader | None, int]:
    ds_cfg = cfg["dataset"]

    data_dir = Path(ds_cfg["data_dir"])
    cache_dir = Path(ds_cfg["cache_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_sets = _build_preprocessed_tensor_sets(data_dir, ds_cfg["time_bins"], ds_cfg["image_size"])

    if tensor_sets is not None:
        train_set, test_set = tensor_sets
        num_classes = 10 if ds_cfg["name"].lower() in {"cifar10_dvs", "cifar10dvs"} else ds_cfg.get("num_classes", 10)
    else:
        dataset_cls, sensor_size, num_classes = _dataset_spec(ds_cfg["name"])
        transform = _make_transform(sensor_size, ds_cfg["time_bins"], ds_cfg["image_size"])

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
