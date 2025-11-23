"""
===============================================================================
DATA CLEANING SCRIPT - X-Ray Classification Project
===============================================================================
Purpose:
    Cleans and validates the X-ray image dataset by:
    - Detecting and reporting corrupt/unreadable images
    - Finding duplicate images using perceptual hashing
    - Checking image dimensions against minimum thresholds
    - Converting images to a consistent color mode (RGB)
    - Resizing images to a target size
    - Creating a cleaned copy of the dataset (optional)
    - Generating a detailed CSV report of all findings

Usage:
    python scripts/clean.py [OPTIONS]
    
    --config: Path to config file (default: configs/main_config.yaml)
    --data-root: Override input dataset directory
    --report: Override report filename
    --output-root: Override output directory for cleaned dataset
    --copy-clean: Force creation of cleaned dataset copy
    --no-copy-clean: Disable creation of cleaned dataset copy

Output:
    - CSV report: reports/cleaning_report.csv (configurable)
    - Cleaned dataset: data_clean/ (if enabled)

Example:
    python scripts/clean.py
    python scripts/clean.py --copy-clean
    python scripts/clean.py --data-root data --output-root data_cleaned
===============================================================================
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imagehash
import yaml
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

# Allow Pillow to load truncated images so we can flag them rather than crash mid-run.
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1 fallback
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class CleaningConfig:
    """
    Configuration container for all cleaning parameters.
    Consolidates all settings needed for the cleaning process.
    """
    data_root: Path                      # Input dataset directory
    report_path: Path                    # Output CSV report path
    splits: Sequence[str]                # Dataset splits to process
    classes: Sequence[str]               # Expected class labels
    allowed_exts: Sequence[str]          # Valid image extensions
    expected_mode: str                   # Target color mode (e.g., 'RGB')
    target_size: Optional[Tuple[int, int]]  # Target image size (width, height)
    min_width: Optional[int]             # Minimum acceptable width
    min_height: Optional[int]            # Minimum acceptable height
    hash_algorithm: str                  # Hash algorithm for duplicate detection
    duplicate_threshold: int             # Max hash distance for duplicates
    copy_cleaned_dataset: bool           # Whether to create cleaned copy
    cleaned_root: Optional[Path]         # Output directory for cleaned dataset
    convert_to_rgb: bool                 # Whether to convert images to RGB


def load_yaml_config(config_path: Path) -> Dict:
    """Load and parse YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value: str, anchor_dir: Path) -> Path:
    """Convert relative paths to absolute paths anchored to anchor_dir."""
    path = Path(path_value)
    if not path.is_absolute():
        path = (anchor_dir / path).resolve()
    return path


def infer_project_root(config_path: Path) -> Path:
    """
    Infer project root directory from config file location.
    Assumes config is in a 'configs' subdirectory of the project root.
    """
    config_dir = config_path.parent
    if config_dir.name.lower() == "configs":
        return config_dir.parent
    return config_dir


def build_cleaning_config(args: argparse.Namespace) -> CleaningConfig:
    """
    Build a CleaningConfig object from CLI arguments and YAML config.
    Command-line arguments override config file values.
    """
    config_path = args.config.resolve()
    raw_config = load_yaml_config(config_path)

    project_root = infer_project_root(config_path)

    paths_cfg = raw_config.get("paths", {})
    dataset_cfg = raw_config.get("dataset", {})
    cleaning_cfg = raw_config.get("cleaning", {})

    # Resolve data_root (input directory)
    data_root_value = args.data_root or paths_cfg.get("data_root")
    if data_root_value is None:
        print("ERROR: 'data_root' must be set in the CLI or the config file.")
        sys.exit(1)
    data_root = resolve_path(str(data_root_value), project_root)

    # Resolve report path
    report_value = args.report or cleaning_cfg.get("report_filename")
    if report_value is None:
        print("ERROR: Provide --report or set cleaning.report_filename in the config.")
        sys.exit(1)
    reports_dir = resolve_path(str(paths_cfg.get("reports_dir", "reports")), project_root)
    report_name = Path(report_value)
    if not report_name.suffix:
        report_name = report_name.with_suffix(".csv")
    report_path = reports_dir / report_name.name

    # Resolve cleaned dataset output directory
    cleaned_root_value = args.output_root or paths_cfg.get("cleaned_root")
    cleaned_root = (
        resolve_path(str(cleaned_root_value), project_root) if cleaned_root_value else None
    )

    # Parse target image size
    target_size_cfg = dataset_cfg.get("target_image_size")
    if target_size_cfg:
        target_size = (int(target_size_cfg[0]), int(target_size_cfg[1]))
    else:
        target_size = None

    # Determine whether to copy cleaned dataset
    copy_clean_flag = args.copy_clean
    if copy_clean_flag is None:
        copy_clean_flag = cleaning_cfg.get("copy_cleaned_dataset", False)

    return CleaningConfig(
        data_root=data_root,
        report_path=report_path,
        splits=dataset_cfg.get("splits", ["train", "val", "test"]),
        classes=dataset_cfg.get("classes", ["normal", "pneumonia", "tuberculosis"]),
        allowed_exts=[
            ext.lower()
            for ext in dataset_cfg.get("image_exts", [".png", ".jpg", ".jpeg"])
        ],
        expected_mode=dataset_cfg.get("expected_mode", "RGB"),
        target_size=target_size,
        min_width=cleaning_cfg.get("min_width"),
        min_height=cleaning_cfg.get("min_height"),
        hash_algorithm=cleaning_cfg.get("hash_algorithm", "phash"),
        duplicate_threshold=int(cleaning_cfg.get("duplicate_threshold", 0)),
        copy_cleaned_dataset=bool(copy_clean_flag),
        cleaned_root=cleaned_root,
        convert_to_rgb=bool(cleaning_cfg.get("convert_to_rgb", True)),
    )


def walk_image_files(
    data_root: Path,
    splits: Sequence[str],
    allowed_exts: Sequence[str],
) -> Iterable[Path]:
    """
    Recursively find all image files in the dataset.
    
    Args:
        data_root: Root directory of the dataset
        splits: List of split directories to search
        allowed_exts: List of valid file extensions
        
    Yields:
        Path objects for each image file found
    """
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"Warning: split directory '{split_dir}' not found.")
            continue
        for image_path in split_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in allowed_exts:
                yield image_path


def resolve_labels(image_path: Path, data_root: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract split and class labels from image path.
    Assumes directory structure: data_root/split/class/image.jpg
    
    Returns:
        Tuple of (split_name, class_name) or (None, None) if path is invalid
    """
    try:
        relative_path = image_path.relative_to(data_root)
    except ValueError:
        return None, None
    parts = relative_path.parts
    if len(parts) < 3:
        return None, None
    return parts[0], parts[1]  # split, class


def pick_hash_function(name: str):
    """
    Get the hash function from imagehash library by name.
    
    Args:
        name: Hash algorithm name (e.g., 'phash', 'dhash', 'ahash')
        
    Returns:
        Hash function from imagehash library
        
    Raises:
        ValueError if algorithm name is not recognized
    """
    func = getattr(imagehash, name, None)
    if func is None:
        raise ValueError(
            f"Hash algorithm '{name}' is not available in imagehash. "
            "Check the ImageHash documentation for supported names."
        )
    return func


def ensure_clean_destination(clean_cfg: CleaningConfig) -> Optional[Path]:
    """
    Create the output directory for cleaned dataset if needed.
    
    Returns:
        Path to cleaned dataset directory, or None if not copying
    """
    if not clean_cfg.copy_cleaned_dataset:
        return None
    if clean_cfg.cleaned_root is None:
        print("ERROR: copy_cleaned_dataset enabled but no cleaned_root path configured.")
        sys.exit(1)
    clean_cfg.cleaned_root.mkdir(parents=True, exist_ok=True)
    return clean_cfg.cleaned_root


def analyze_images(clean_cfg: CleaningConfig) -> List[Dict[str, str]]:
    """
    Main analysis function that processes all images in the dataset.
    
    For each image, this function:
    1. Validates the image can be opened
    2. Checks dimensions against minimum thresholds
    3. Detects duplicates using perceptual hashing
    4. Optionally creates a cleaned copy (resized and/or converted to RGB)
    5. Records all findings in a report row
    
    Args:
        clean_cfg: CleaningConfig object with all parameters
        
    Returns:
        List of dictionaries, each containing report data for one image
    """
    if not clean_cfg.data_root.is_dir():
        print(f"ERROR: dataset root '{clean_cfg.data_root}' does not exist.")
        sys.exit(1)

    # Initialize hash function for duplicate detection
    hash_func = pick_hash_function(clean_cfg.hash_algorithm)
    
    # Storage for duplicate detection
    approximate_hash_records: List[Tuple[imagehash.ImageHash, str]] = []  # For fuzzy matching
    exact_hash_lookup: Dict[str, str] = {}  # For exact matching
    
    report_rows: List[Dict[str, str]] = []
    split_counts: Counter = Counter()

    # Create output directory if needed
    cleaned_root = ensure_clean_destination(clean_cfg)

    print(f"Scanning dataset at {clean_cfg.data_root}")
    image_paths = list(
        walk_image_files(clean_cfg.data_root, clean_cfg.splits, clean_cfg.allowed_exts)
    )
    if not image_paths:
        print("ERROR: no image files discovered. Check the dataset location.")
        sys.exit(1)

    print(f"Found {len(image_paths)} candidate images.")

    # CSV header for the report
    header = [
        "filepath",
        "split",
        "class",
        "width",
        "height",
        "mode",
        "is_corrupt",
        "duplicate_of",
        "issues",
        "actions",
    ]

    # Process each image with progress bar
    for image_path in tqdm(image_paths, desc="Analyzing", unit="image"):
        notes: List[str] = []      # Issues found with this image
        actions: List[str] = []    # Actions taken (e.g., resized, converted)

        # Extract split and class from directory structure
        split_label, class_label = resolve_labels(image_path, clean_cfg.data_root)
        if split_label is None or class_label is None:
            notes.append("Unrecognized folder layout")

        # Validate split and class names
        if split_label and split_label not in clean_cfg.splits:
            notes.append(f"Unexpected split '{split_label}'")

        if class_label and class_label not in clean_cfg.classes:
            notes.append(f"Unexpected class '{class_label}'")

        # Initialize report row
        row = {key: "" for key in header}
        row["filepath"] = str(image_path)
        row["split"] = split_label or ""
        row["class"] = class_label or ""
        row["is_corrupt"] = "False"

        duplicate_of = ""

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                row["width"] = str(width)
                row["height"] = str(height)
                row["mode"] = img.mode

                # Check minimum dimensions
                if clean_cfg.min_width and width < clean_cfg.min_width:
                    notes.append(f"Width below minimum threshold ({width}px)")
                if clean_cfg.min_height and height < clean_cfg.min_height:
                    notes.append(f"Height below minimum threshold ({height}px)")

                # Check if color mode conversion is needed
                if clean_cfg.convert_to_rgb and img.mode != clean_cfg.expected_mode:
                    notes.append(f"Needs conversion {img.mode} -> {clean_cfg.expected_mode}")

                # Compute perceptual hash for duplicate detection
                image_hash = hash_func(img)
                
                if clean_cfg.duplicate_threshold <= 0:
                    # Exact hash matching (faster)
                    key = str(image_hash)
                    if key in exact_hash_lookup:
                        duplicate_of = exact_hash_lookup[key]
                    else:
                        exact_hash_lookup[key] = str(image_path)
                else:
                    # Approximate hash matching (finds near-duplicates)
                    match_path = None
                    for stored_hash, stored_path in approximate_hash_records:
                        if image_hash - stored_hash <= clean_cfg.duplicate_threshold:
                            match_path = stored_path
                            break
                    if match_path:
                        duplicate_of = match_path
                    else:
                        approximate_hash_records.append((image_hash, str(image_path)))

                # Create cleaned copy if enabled and image is valid
                if cleaned_root and row["is_corrupt"] == "False" and not duplicate_of:
                    processed = img
                    
                    # Convert to RGB if needed
                    if clean_cfg.convert_to_rgb and processed.mode != clean_cfg.expected_mode:
                        processed = processed.convert(clean_cfg.expected_mode)
                        actions.append("converted_to_RGB")
                    
                    # Resize if needed
                    if clean_cfg.target_size and processed.size != clean_cfg.target_size:
                        processed = processed.resize(
                            clean_cfg.target_size, RESAMPLE_LANCZOS
                        )
                        actions.append(
                            f"resized_to_{clean_cfg.target_size[0]}x{clean_cfg.target_size[1]}"
                        )
                    
                    # Save to cleaned dataset directory
                    destination = cleaned_root / (split_label or "unknown") / (class_label or "unknown")
                    destination.mkdir(parents=True, exist_ok=True)
                    processed.save(destination / image_path.name)
                    
        except (UnidentifiedImageError, OSError) as exc:
            # Image is corrupt or unreadable
            row["is_corrupt"] = "True"
            notes.append(f"Unreadable file: {exc}")
        except ValueError as exc:
            # Processing error (e.g., invalid mode conversion)
            row["is_corrupt"] = "True"
            notes.append(f"Processing error: {exc}")

        # Record duplicate information
        if duplicate_of:
            row["duplicate_of"] = duplicate_of
            notes.append("Potential duplicate")
            if cleaned_root and row["is_corrupt"] == "False":
                actions.append("skipped_copy_due_to_duplicate")

        # Combine notes and actions into report columns
        if notes:
            row["issues"] = "; ".join(sorted(notes))
        if actions:
            row["actions"] = "; ".join(actions)

        report_rows.append(row)

        # Count images per split for summary
        if split_label:
            split_counts.update([split_label])

    # Print summary statistics
    print("\nSplit distribution (images counted regardless of issues):")
    for split, count in sorted(split_counts.items()):
        print(f"  - {split}: {count}")

    return report_rows


def write_report(report_rows: List[Dict[str, str]], report_path: Path) -> None:
    """
    Write the cleaning report to a CSV file.
    
    Args:
        report_rows: List of dictionaries containing report data
        report_path: Path where CSV file should be written
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "filepath",
        "split",
        "class",
        "width",
        "height",
        "mode",
        "is_corrupt",
        "duplicate_of",
        "issues",
        "actions",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nCleaning report written to: {report_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an image dataset for issues (corruption, duplicates, size/mode mismatches) "
            "and optionally write a cleaned copy."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/main_config.yaml"),
        help="Path to the project configuration YAML file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override the dataset root directory defined in the config file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Override the report file name (created inside the reports_dir from the config).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional path to write a cleaned dataset copy. Overrides paths.cleaned_root from the config.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--copy-clean",
        dest="copy_clean",
        action="store_true",
        help="Force creation of a cleaned dataset copy.",
    )
    group.add_argument(
        "--no-copy-clean",
        dest="copy_clean",
        action="store_false",
        help="Disable creation of a cleaned dataset copy even if enabled in the config.",
    )
    parser.set_defaults(copy_clean=None)
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the cleaning script.
    Builds configuration, analyzes images, and writes the report.
    """
    args = parse_args()
    clean_cfg = build_cleaning_config(args)
    rows = analyze_images(clean_cfg)
    write_report(rows, clean_cfg.report_path)


if __name__ == "__main__":
    main()
