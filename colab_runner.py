from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


TRAIN_ARCHIVE = "ibmGestureTrain.tar.gz"
TEST_ARCHIVE = "ibmGestureTest.tar.gz"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> None:
    if not in_colab():
        print("Google Drive mounting is only available inside Google Colab.")
        return
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)


def install_requirements() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f"Already copied: {dst}")
        return
    print(f"Copying {src} -> {dst}")
    shutil.copy2(src, dst)


def extract_archive(archive: Path, data_dir: Path) -> None:
    folder_name = archive.name.replace(".tar.gz", "")
    expected_folder = data_dir / folder_name
    if expected_folder.exists():
        print(f"Already extracted: {expected_folder}")
        return
    print(f"Extracting {archive} -> {data_dir}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)


def prepare_dvsgesture_from_drive(drive_dataset_dir: str, data_dir: str = "data") -> None:
    source = Path(drive_dataset_dir)
    target = Path(data_dir)

    for filename in [TRAIN_ARCHIVE, TEST_ARCHIVE]:
        copy_file(source / filename, target / filename)
        extract_archive(target / filename, target)

    print("DVS128 Gesture dataset is ready.")
    print(f"Data directory: {target.resolve()}")


def train(stage: str, config: str, init_checkpoint: str | None = None) -> None:
    cmd = [sys.executable, "train.py", "--config", config, "--stage", stage]
    if init_checkpoint:
        cmd.extend(["--init-checkpoint", init_checkpoint])
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small Colab helper for preparing DVS128 Gesture from Google Drive."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("install", help="Install requirements.txt.")
    subparsers.add_parser("mount-drive", help="Mount Google Drive in Colab.")

    prep = subparsers.add_parser(
        "prepare-dvsgesture",
        help="Copy and extract DVS128 Gesture archives from Google Drive.",
    )
    prep.add_argument(
        "--drive-dataset-dir",
        default="/content/drive/MyDrive/datasets/DVSGesture",
        help="Folder containing ibmGestureTrain.tar.gz and ibmGestureTest.tar.gz.",
    )
    prep.add_argument("--data-dir", default="data", help="Local repo data directory.")

    train_cmd = subparsers.add_parser("train", help="Run one training stage.")
    train_cmd.add_argument(
        "--stage",
        required=True,
        choices=["baseline", "skip-search", "skip", "threshold", "full"],
    )
    train_cmd.add_argument("--config", default="configs/default.yaml")
    train_cmd.add_argument("--init-checkpoint", default=None)

    args = parser.parse_args()
    if args.command == "install":
        install_requirements()
    elif args.command == "mount-drive":
        mount_drive()
    elif args.command == "prepare-dvsgesture":
        prepare_dvsgesture_from_drive(args.drive_dataset_dir, args.data_dir)
    elif args.command == "train":
        train(args.stage, args.config, args.init_checkpoint)


if __name__ == "__main__":
    main()
