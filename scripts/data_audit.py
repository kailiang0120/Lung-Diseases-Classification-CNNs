"""
===============================================================================
DATA AUDIT SCRIPT - X-Ray Classification Project
===============================================================================
Purpose:
    Performs a comprehensive audit of the X-ray image dataset to identify
    potential issues before training. Checks for:
    - Missing class folders
    - Corrupt or unreadable image files
    - Inconsistent color modes within splits
    - Overall dataset statistics

Usage:
    python scripts/data_audit.py [--config PATH] [--data-root PATH]
    
    --config: Path to config file (default: configs/main_config.yaml)
    --data-root: Override dataset root from config

Output:
    Prints a detailed report showing image counts per class and any issues
    found in the dataset.

Example:
    python scripts/data_audit.py
    python scripts/data_audit.py --data-root data_clean
===============================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import yaml
from PIL import Image, UnidentifiedImageError


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file and return as dictionary."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value: str, project_root: Path) -> Path:
    """Convert relative paths to absolute paths using project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def iter_image_files(class_path: Path) -> Iterable[Path]:
    """
    Yield all image files in a class directory.
    Only returns files with image extensions (.jpg, .jpeg, .png).
    """
    for entry in class_path.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        yield entry


def audit_dataset(
    data_root: Path,
    splits: Sequence[str],
    classes: Sequence[str],
) -> None:
    """
    Main audit function that checks each split and class for issues.
    
    Args:
        data_root: Root directory containing all data splits
        splits: List of split names (e.g., ['train', 'val', 'test'])
        classes: List of class names (e.g., ['normal', 'pneumonia', 'tuberculosis'])
    
    Checks performed:
        - Verifies all split folders exist
        - Verifies all class folders exist within each split
        - Counts images per class
        - Detects corrupt/unreadable files
        - Detects color mode inconsistencies
    """
    if not data_root.is_dir():
        print(f"ERROR: Dataset root directory not found at '{data_root}'")
        sys.exit(1)

    print(f"Starting audit of dataset at: {data_root.resolve()}\n")

    for split in splits:
        print(f"--- Auditing split: {split} ---")
        split_path = data_root / split

        if not split_path.is_dir():
            print(f"  Warning: split folder '{split}' not found. Skipping.\n")
            continue

        counts: Dict[str, int] = {cls: 0 for cls in classes}
        issues = {
            "missing_classes": [],
            "corrupt_files": [],
            "mode_mismatches": [],
        }

        for cls in classes:
            class_path = split_path / cls
            if not class_path.is_dir():
                issues["missing_classes"].append(cls)
                continue

            expected_mode = None
            for image_path in iter_image_files(class_path):
                # Check for empty files (0 bytes)
                if image_path.stat().st_size == 0:
                    issues["corrupt_files"].append(str(image_path))
                    continue
                try:
                    with Image.open(image_path) as img:
                        # Track the first image's color mode as the expected mode
                        if expected_mode is None:
                            expected_mode = img.mode
                        # Flag images with different color modes
                        if img.mode != expected_mode:
                            issues["mode_mismatches"].append(
                                f"{image_path} (mode: {img.mode})"
                            )
                        counts[cls] += 1
                except UnidentifiedImageError:
                    issues["corrupt_files"].append(str(image_path))
                except Exception as exc:
                    issues["corrupt_files"].append(f"{image_path} (error: {exc})")

        # Print results for this split
        print("  Image counts per class:")
        for cls, total in counts.items():
            print(f"    - {cls}: {total}")

        if issues["missing_classes"]:
            print(f"  Missing class folders: {issues['missing_classes']}")
        if issues["corrupt_files"]:
            print("  Corrupt or unreadable files:")
            for path in issues["corrupt_files"]:
                print(f"    - {path}")
        if issues["mode_mismatches"]:
            print("  Mode mismatches within split:")
            for entry in issues["mode_mismatches"]:
                print(f"    - {entry}")
        print("-" * 32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit dataset splits for missing classes, corrupt files, and mode inconsistencies."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/main_config.yaml"),
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override dataset root path (defaults to paths.data_root from the config).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the data audit script.
    Loads configuration, resolves paths, and runs the audit.
    """
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent

    # Use command-line override or fall back to config value
    data_root_value = args.data_root or config["paths"]["data_root"]
    data_root = resolve_path(str(data_root_value), project_root)

    dataset_cfg = config["dataset"]
    splits = dataset_cfg.get("splits", ["train", "val", "test"])
    classes = dataset_cfg.get("classes", [])

    audit_dataset(data_root, splits, classes)


if __name__ == "__main__":
    main()
