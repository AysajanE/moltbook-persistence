#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET = "SimulaMet/moltbook-observatory-archive"
DEFAULT_SUBSETS = [
    "agents",
    "comments",
    "posts",
    "snapshots",
    "submolts",
    "word_frequency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/export Moltbook Observatory Archive tables from Hugging Face "
            "into a local directory (kept out of git)."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f'Hugging Face dataset name (default: "{DEFAULT_DATASET}")',
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (e.g., data/raw/moltbook-observatory-archive)",
    )
    parser.add_argument(
        "--subset",
        action="append",
        choices=DEFAULT_SUBSETS,
        help=(
            "Subset/table to export (repeatable). "
            f"Default: all ({', '.join(DEFAULT_SUBSETS)})"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Export format (default: parquet)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap per split for quick samples (default: export all rows).",
    )
    return parser.parse_args()


def export_split(split, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        split.to_parquet(str(out_path))
        return

    if fmt == "csv":
        split.to_csv(str(out_path))
        return

    raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    subsets = args.subset or DEFAULT_SUBSETS

    from datasets import __version__ as datasets_version
    from datasets import load_dataset

    export_manifest: dict[str, object] = {
        "dataset": args.dataset,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "max_rows": args.max_rows,
        "subsets": {},
        "datasets_version": datasets_version,
    }

    for subset in subsets:
        ds_dict = load_dataset(args.dataset, subset)
        subset_info: dict[str, object] = {"splits": {}}

        for split_name, split in ds_dict.items():
            if args.max_rows is not None:
                subset_rows = min(args.max_rows, split.num_rows)
                split = split.select(range(subset_rows))

            out_path = out_dir / subset / f"{split_name}.{args.format}"
            export_split(split, out_path, args.format)
            subset_info["splits"][split_name] = {"rows": split.num_rows, "path": str(out_path)}

        export_manifest["subsets"][subset] = subset_info

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "EXPORT_MANIFEST.json"
    manifest_path.write_text(json.dumps(export_manifest, indent=2, sort_keys=True) + "\n")

    print(f"Wrote export manifest: {manifest_path}")


if __name__ == "__main__":
    main()

