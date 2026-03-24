from __future__ import annotations

import json
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_video_files(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def find_sample_directories(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    return sorted(path for path in directory.iterdir() if path.is_dir())


def find_phase_image(sample_dir: str | Path, phase_name: str) -> Path | None:
    directory = Path(sample_dir)
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{phase_name}{extension}"
        if candidate.exists():
            return candidate
    return None


def find_image_files(sample_dir: str | Path) -> list[Path]:
    directory = Path(sample_dir)
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def has_existing_output(output_dir: str | Path, video_stem: str) -> bool:
    video_output_dir = Path(output_dir) / video_stem
    summary_path = video_output_dir / f"{video_stem}_summary.json"
    return video_output_dir.is_dir() and summary_path.exists()


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
