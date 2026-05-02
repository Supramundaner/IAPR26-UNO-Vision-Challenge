from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


COMPETITION = "iapr-26-uno-vision-challenge"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_ITEMS = [
    "reference_images",
    "test_images",
    "train_images",
    "sample_submission.csv",
    "train.csv",
]


def copy_item(source: Path, destination: Path) -> None:
    if destination.exists():
        print(f"Already exists, skipping: {destination}")
        return
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    print(f"Copied: {destination}")


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    competition_path = Path(kagglehub.competition_download(COMPETITION))
    print(f"KaggleHub cache: {competition_path}")

    for item in EXPECTED_ITEMS:
        source = competition_path / item
        if not source.exists():
            print(f"Missing expected item in download cache: {source}")
            continue
        copy_item(source, RAW_DATA_DIR / item)

    print(f"Data directory: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
