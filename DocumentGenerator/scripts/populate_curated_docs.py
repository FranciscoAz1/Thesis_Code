"""
Populate curated_docs from qa_dataset.json.

This script:
- Reads qa_dataset.json and extracts values of the key "arquivo".
- Finds those files in the source folder (default: documentos_gerados).
- Copies them to the destination folder (default: curated_docs).
- Optionally adds N random additional files from source that were not in the dataset selection.

Usage (PowerShell examples):
  python populate_curated_docs.py --include-random 10 --seed 42 --verbose
  python populate_curated_docs.py --qa qa_dataset.json --src documentos_gerados --dst curated_docs --dry-run

Notes:
- Matching is done by exact filename (case-insensitive) within the source directory.
- If --overwrite is not provided, existing files in destination are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple


DATASET_FILENAME_KEY = "arquivo"


@dataclass
class CopyResult:
    copied: List[Path]
    skipped_existing: List[Path]
    missing_in_src: List[str]
    random_added: List[Path]


def read_arquivo_values(qa_path: Path) -> List[str]:
    """Read qa_dataset.json and return a list of unique filename strings from key 'arquivo'.

    Supports json structures:
    - List[dict]
    - Dict[str, Any] with iterable values containing dicts
    """
    try:
        with qa_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: File not found: {qa_path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Invalid JSON in {qa_path}: {e}")

    values: Set[str] = set()

    def maybe_add(item: object) -> None:
        if isinstance(item, dict):
            val = item.get(DATASET_FILENAME_KEY)
            if isinstance(val, str) and val.strip():
                values.add(val.strip())

    if isinstance(data, list):
        for item in data:
            maybe_add(item)
    elif isinstance(data, dict):
        # Look for lists of dicts inside the dict
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    maybe_add(item)
            else:
                maybe_add(v)
    else:
        raise SystemExit("ERROR: Unsupported JSON structure; expected list or dict")

    return sorted(values)


def index_source_files(src_dir: Path) -> Tuple[dict, List[Path]]:
    """Index files in source directory by lowercase filename for quick lookup.

    Returns a tuple of (name_index, all_files), where name_index maps lower name -> Path.
    """
    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"ERROR: Source directory not found or not a directory: {src_dir}")

    all_files: List[Path] = [p for p in src_dir.iterdir() if p.is_file()]
    name_index = {p.name.lower(): p for p in all_files}
    return name_index, all_files


def ensure_destination(dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)


def copy_files(
    to_copy_names: Iterable[str],
    name_index: dict,
    dst_dir: Path,
    overwrite: bool,
    verbose: bool,
) -> Tuple[List[Path], List[Path], List[str]]:
    copied: List[Path] = []
    skipped_existing: List[Path] = []
    missing_in_src: List[str] = []

    for name in to_copy_names:
        src_path = name_index.get(name.lower())
        if not src_path:
            missing_in_src.append(name)
            if verbose:
                print(f"WARN: Not found in source: {name}")
            continue

        dst_path = dst_dir / src_path.name
        if dst_path.exists() and not overwrite:
            skipped_existing.append(dst_path)
            if verbose:
                print(f"SKIP: Already exists: {dst_path}")
            continue

        if verbose:
            action = "OVERWRITE" if dst_path.exists() and overwrite else "COPY"
            print(f"{action}: {src_path} -> {dst_path}")
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)

    return copied, skipped_existing, missing_in_src


def pick_random_extra_files(
    all_src_files: List[Path],
    already_selected_names: Set[str],
    count: int,
) -> List[Path]:
    if count <= 0:
        return []

    # Exclude any that are already selected by filename (case-insensitive)
    exclude_lower = {n.lower() for n in already_selected_names}
    candidates = [p for p in all_src_files if p.name.lower() not in exclude_lower]

    if not candidates:
        return []

    if count >= len(candidates):
        # If requesting more than available, just return all candidates
        return candidates

    return random.sample(candidates, count)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Populate curated docs from QA dataset")
    parser.add_argument("--qa", default="datasets/qa_dataset.json", help="Path to qa_dataset.json")
    parser.add_argument("--src", default="documentos_gerados", help="Source folder containing generated documents")
    parser.add_argument("--dst", default="curated_docs", help="Destination folder to copy curated documents")
    parser.add_argument("--include-random", type=int, default=0, help="Number of random extra files to include from source (not already selected)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for selecting extra files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    qa_path = Path(args.qa)
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if args.seed is not None:
        random.seed(args.seed)

    if args.verbose:
        print(f"QA JSON: {qa_path}")
        print(f"Source: {src_dir}")
        print(f"Destination: {dst_dir}")
        if args.include_random:
            print(f"Random extras requested: {args.include_random} (seed={args.seed})")
        print(f"Overwrite: {args.overwrite} | Dry-run: {args.dry_run}")

    arquivo_names = read_arquivo_values(qa_path)
    name_index, all_src_files = index_source_files(src_dir)
    ensure_destination(dst_dir)

    # Copy dataset-listed files
    copied: List[Path] = []
    skipped_existing: List[Path] = []
    missing_in_src: List[str] = []

    if args.dry_run:
        # Simulate copy decisions
        for name in arquivo_names:
            src_path = name_index.get(name.lower())
            if not src_path:
                missing_in_src.append(name)
                if args.verbose:
                    print(f"[DRY] Missing: {name}")
                continue
            dst_path = dst_dir / src_path.name
            if dst_path.exists() and not args.overwrite:
                skipped_existing.append(dst_path)
                if args.verbose:
                    print(f"[DRY] Skip existing: {dst_path}")
            else:
                copied.append(dst_path)
                if args.verbose:
                    print(f"[DRY] Would copy: {src_path} -> {dst_path}")
    else:
        copied, skipped_existing, missing_in_src = copy_files(
            arquivo_names, name_index, dst_dir, args.overwrite, args.verbose
        )

    # Random extras
    random_added: List[Path] = []
    if args.include_random and args.include_random > 0:
        already_selected_set = set(arquivo_names)
        extra_files = pick_random_extra_files(all_src_files, already_selected_set, args.include_random)
        if args.dry_run:
            for p in extra_files:
                dst_path = dst_dir / p.name
                if dst_path.exists() and not args.overwrite:
                    skipped_existing.append(dst_path)
                    if args.verbose:
                        print(f"[DRY] Skip existing (random): {dst_path}")
                else:
                    random_added.append(dst_path)
                    if args.verbose:
                        print(f"[DRY] Would copy (random): {p} -> {dst_path}")
        else:
            # Perform actual copy for extras
            for p in extra_files:
                dst_path = dst_dir / p.name
                if dst_path.exists() and not args.overwrite:
                    skipped_existing.append(dst_path)
                    if args.verbose:
                        print(f"SKIP: Already exists (random): {dst_path}")
                    continue
                if args.verbose:
                    action = "OVERWRITE" if dst_path.exists() and args.overwrite else "COPY"
                    print(f"{action} (random): {p} -> {dst_path}")
                shutil.copy2(p, dst_path)
                random_added.append(dst_path)

    # Report summary
    selected_count = len(arquivo_names)
    copied_count = len(copied)
    skipped_count = len(skipped_existing)
    missing_count = len(missing_in_src)
    random_count = len(random_added)

    print("\nSummary:")
    print(f"- Selected from dataset: {selected_count}")
    print(f"- Copied: {copied_count}")
    print(f"- Skipped existing: {skipped_count}")
    print(f"- Missing in source: {missing_count}")
    if args.include_random:
        print(f"- Random extras requested: {args.include_random}")
        print(f"- Random extras added: {random_count}")

    if missing_count:
        print("\nMissing files (by name from dataset):")
        for name in missing_in_src:
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
