#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DEFAULT_SUBSETS = [
    "agents",
    "comments",
    "posts",
    "snapshots",
    "submolts",
    "word_frequency",
]


TIMESTAMP_LIKE_COLUMNS = {"timestamp", "hour"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_schema_mapping(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _is_timestamp_col(name: str) -> bool:
    return name.endswith("_at") or name in TIMESTAMP_LIKE_COLUMNS


def _parse_timestamp_utc(values: pd.Series) -> pd.Series:
    # Coerce invalid or empty values to NaT; always produce UTC tz-aware timestamps.
    return pd.to_datetime(values, errors="coerce", utc=True)


def _timestamp_parse_stats(raw: pd.Series, parsed: pd.Series) -> dict[str, Any]:
    nonnull_mask = raw.notna()
    nonnull_count = int(nonnull_mask.sum())
    parsed_ok = parsed.notna()
    parsed_ok_count = int((nonnull_mask & parsed_ok).sum())
    parsed_fail_count = nonnull_count - parsed_ok_count
    parse_rate = (parsed_ok_count / nonnull_count) if nonnull_count else None

    examples: list[str] = []
    if parsed_fail_count:
        failures = raw[nonnull_mask & ~parsed_ok].astype(str)
        # Keep only a few unique examples; timestamps only (no other content).
        examples = list(dict.fromkeys(failures.tolist()))[:5]

    return {
        "nonnull": nonnull_count,
        "parsed_success": parsed_ok_count,
        "parsed_fail": parsed_fail_count,
        "parse_rate": parse_rate,
        "examples_fail_raw": examples,
    }


def _add_parsed_timestamp_columns(
    table: pa.Table, timestamp_cols: list[str]
) -> tuple[pa.Table, dict[str, Any]]:
    stats: dict[str, Any] = {}
    out = table
    for col in timestamp_cols:
        raw_name = f"{col}_raw"
        utc_name = f"{col}_utc"
        if raw_name not in out.column_names:
            out = out.append_column(raw_name, out[col])

        raw_series = out[col].to_pandas()
        parsed_series = _parse_timestamp_utc(raw_series)
        stats[col] = _timestamp_parse_stats(raw_series, parsed_series)

        utc_array = pa.Array.from_pandas(parsed_series, type=pa.timestamp("ns", tz="UTC"))
        if utc_name in out.column_names:
            idx = out.column_names.index(utc_name)
            out = out.set_column(idx, utc_name, utc_array)
        else:
            out = out.append_column(utc_name, utc_array)

    return out, stats


def _derive_dt(table: pa.Table) -> tuple[pa.Array, str]:
    if "dump_date" in table.column_names:
        return table["dump_date"], "dump_date"

    if "created_at_utc" in table.column_names:
        created = table["created_at_utc"].to_pandas()
        dt = created.dt.strftime("%Y-%m-%d")
        return pa.Array.from_pandas(dt, type=pa.string()), "created_at_utc"

    raise ValueError("Unable to derive dt partition key (missing dump_date and created_at_utc).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate raw Moltbook HF archive Parquet exports into partitioned, "
            "analysis-ready Parquet with UTC-parsed timestamps."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Raw snapshot directory (e.g., data_raw/hf_archive/snapshot_*/)",
    )
    parser.add_argument(
        "--schema-mapping",
        type=Path,
        required=True,
        help="Path to schema mapping JSON (for provenance; not required for parsing).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Curated output directory (e.g., data_curated/hf_archive/snapshot_*/)",
    )
    parser.add_argument(
        "--curation-manifest",
        type=Path,
        required=True,
        help="Output path for curation manifest JSON.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        choices=DEFAULT_SUBSETS,
        help="Subset(s) to curate (repeatable). Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    subsets = args.subset or DEFAULT_SUBSETS

    # Load for provenance only; curation relies on raw schemas.
    schema_mapping = _load_schema_mapping(args.schema_mapping)

    manifest: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "schema_mapping_path": str(args.schema_mapping),
        "tables": {},
    }

    for subset in subsets:
        raw_path = raw_dir / subset / "archive.parquet"
        if not raw_path.exists():
            manifest["tables"][subset] = {"error": f"missing file: {raw_path}"}
            continue

        subset_out_dir = out_dir / subset
        if subset_out_dir.exists() and any(subset_out_dir.rglob("*.parquet")):
            raise FileExistsError(
                f"Refusing to overwrite existing curated data: {subset_out_dir}. "
                "Use a new snapshot directory."
            )

        table = pq.read_table(raw_path)
        timestamp_cols = [c for c in table.column_names if _is_timestamp_col(c)]
        curated_table, ts_stats = _add_parsed_timestamp_columns(table, timestamp_cols)

        dt_array, dt_source = _derive_dt(curated_table)
        if "dt" not in curated_table.column_names:
            curated_table = curated_table.append_column("dt", dt_array)

        ds.write_dataset(
            data=curated_table,
            base_dir=str(subset_out_dir),
            format="parquet",
            partitioning=["dt"],
            partitioning_flavor="hive",
            create_dir=True,
        )

        raw_rows = int(table.num_rows)
        dt_values = pd.Series(dt_array.to_pandas()).dropna().astype(str).unique().tolist()

        manifest["tables"][subset] = {
            "raw_path": str(raw_path),
            "raw_rows": raw_rows,
            "timestamp_columns_parsed": timestamp_cols,
            "timestamp_parse_stats": ts_stats,
            "dt_source_column": dt_source,
            "dt_partitions": sorted(dt_values)[:50],
            "dt_partitions_truncated": len(dt_values) > 50,
            "out_dir": str(subset_out_dir),
            "out_partitioning": "hive(dt=YYYY-MM-DD)",
            "added_columns": (
                [f"{c}_raw" for c in timestamp_cols]
                + [f"{c}_utc" for c in timestamp_cols]
                + ["dt"]
            ),
            "dropped_or_renamed_columns": [],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    args.curation_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.curation_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Include mapping summary for quick inspection (no duplication of full mapping).
    mapping_summary = {
        "generated_at_utc": schema_mapping.get("generated_at_utc"),
        "tables": {
            t: {
                "missing_required_count": schema_mapping.get("tables", {})
                .get(t, {})
                .get("summary", {})
                .get("missing_required_count"),
                "missing_required": schema_mapping.get("tables", {})
                .get(t, {})
                .get("summary", {})
                .get("missing_required"),
            }
            for t in subsets
        },
    }
    (args.curation_manifest.parent / "schema_mapping_summary.json").write_text(
        json.dumps(mapping_summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
