#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _group_key_for_path(raw_root: Path, path: Path) -> tuple[str | None, str | None, str]:
    """
    Expected layout: data_raw/moltbook_api/YYYY-MM-DD/{endpoint}/{timestamp}.json
    Groups are formed by {date}/{endpoint}. If the path is shallower than expected,
    it falls back to "unknown".
    """
    rel = path.relative_to(raw_root)
    parts = rel.parts
    date = parts[0] if len(parts) >= 1 else None
    endpoint = parts[1] if len(parts) >= 2 else None
    key = f"{date}/{endpoint}" if date and endpoint else "unknown/unknown"
    return date, endpoint, key


def _update_object_summary(
    obj: dict[str, Any],
    union_keys: set[str],
    intersection_keys: set[str] | None,
    field_types: dict[str, set[str]],
) -> set[str] | None:
    keys = set(obj.keys())
    union_keys.update(keys)
    intersection_keys = keys if intersection_keys is None else (intersection_keys & keys)
    for k, v in obj.items():
        field_types.setdefault(k, set()).add(_json_type(v))
    return intersection_keys


def _update_array_summary(
    arr: list[Any],
    max_array_elements: int,
    element_type_counts: dict[str, int],
    object_union_keys: set[str],
    object_intersection_keys: set[str] | None,
    object_field_types: dict[str, set[str]],
) -> tuple[int, set[str] | None]:
    sampled = arr[:max_array_elements]
    for el in sampled:
        t = _json_type(el)
        element_type_counts[t] = element_type_counts.get(t, 0) + 1
        if isinstance(el, dict):
            object_intersection_keys = _update_object_summary(
                el,
                union_keys=object_union_keys,
                intersection_keys=object_intersection_keys,
                field_types=object_field_types,
            )
    return len(sampled), object_intersection_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover a minimal schema manifest for locally stored raw Moltbook API JSON "
            "responses under data_raw/moltbook_api/. This script never fabricates fields: "
            "it only summarizes observed JSON structure."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="Root directory to scan.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON manifest path.")
    parser.add_argument(
        "--out-copy",
        type=Path,
        default=None,
        help="Optional second copy of the output JSON (e.g., per-run audit).",
    )
    parser.add_argument(
        "--max-array-elements",
        type=int,
        default=1000,
        help="Max number of elements sampled from any top-level JSON array per file.",
    )
    parser.add_argument(
        "--max-sample-paths",
        type=int,
        default=20,
        help="Max number of example file paths stored per group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root: Path = args.raw_root

    json_paths = sorted(raw_root.rglob("*.json")) if raw_root.exists() else []

    manifest: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "raw_root": str(raw_root),
        "expected_layout": "data_raw/moltbook_api/YYYY-MM-DD/{endpoint}/{timestamp}.json",
        "file_count": len(json_paths),
        "status": "ok",
        "notes": [],
        "groups": {},
        "errors": [],
    }

    if not json_paths:
        manifest["status"] = "no_samples_found"
        manifest["notes"].append(
            "No local raw Moltbook API JSON samples found. Collect at least one raw response "
            "per endpoint under the expected layout, then re-run this script."
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        args.out.write_text(content)
        if args.out_copy is not None:
            args.out_copy.parent.mkdir(parents=True, exist_ok=True)
            args.out_copy.write_text(content)
        return

    groups: dict[str, dict[str, Any]] = {}

    for path in json_paths:
        date, endpoint, key = _group_key_for_path(raw_root, path)
        group = groups.setdefault(
            key,
            {
                "date": date,
                "endpoint": endpoint,
                "group_key": key,
                "file_count": 0,
                "file_paths_sample": [],
                "top_level_type_counts": {},
                "object_keys_union": set(),
                "object_keys_intersection": None,
                "object_field_types": {},
                "array_element_type_counts": {},
                "array_object_keys_union": set(),
                "array_object_keys_intersection": None,
                "array_object_field_types": {},
                "array_sampling": {
                    "max_array_elements": args.max_array_elements,
                    "sampled_total": 0,
                },
                "parse_errors": [],
            },
        )

        group["file_count"] += 1
        if len(group["file_paths_sample"]) < args.max_sample_paths:
            group["file_paths_sample"].append(str(path))

        try:
            payload = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            group["parse_errors"].append({"path": str(path), "error": f"{type(e).__name__}: {e}"})
            continue

        top_t = _json_type(payload)
        tl_counts = group["top_level_type_counts"]
        tl_counts[top_t] = tl_counts.get(top_t, 0) + 1

        if isinstance(payload, dict):
            group["object_keys_intersection"] = _update_object_summary(
                payload,
                union_keys=group["object_keys_union"],
                intersection_keys=group["object_keys_intersection"],
                field_types=group["object_field_types"],
            )
        elif isinstance(payload, list):
            sampled_n, group["array_object_keys_intersection"] = _update_array_summary(
                payload,
                max_array_elements=args.max_array_elements,
                element_type_counts=group["array_element_type_counts"],
                object_union_keys=group["array_object_keys_union"],
                object_intersection_keys=group["array_object_keys_intersection"],
                object_field_types=group["array_object_field_types"],
            )
            group["array_sampling"]["sampled_total"] += int(sampled_n)

    def _finalize_set(s: set[str] | None) -> list[str]:
        return sorted(s) if s is not None else []

    finalized_groups: dict[str, Any] = {}
    parse_error_total = 0
    for key, g in sorted(groups.items()):
        parse_error_total += len(g["parse_errors"])
        finalized_groups[key] = {
            "date": g["date"],
            "endpoint": g["endpoint"],
            "group_key": g["group_key"],
            "file_count": g["file_count"],
            "file_paths_sample": g["file_paths_sample"],
            "top_level_type_counts": g["top_level_type_counts"],
            "object_keys": {
                "union": sorted(g["object_keys_union"]),
                "intersection": _finalize_set(g["object_keys_intersection"]),
                "union_count": len(g["object_keys_union"]),
                "intersection_count": len(g["object_keys_intersection"] or set()),
            },
            "object_field_types": {
                k: sorted(v) for k, v in sorted(g["object_field_types"].items())
            },
            "array": {
                "element_type_counts": g["array_element_type_counts"],
                "object_keys": {
                    "union": sorted(g["array_object_keys_union"]),
                    "intersection": _finalize_set(g["array_object_keys_intersection"]),
                    "union_count": len(g["array_object_keys_union"]),
                    "intersection_count": len(g["array_object_keys_intersection"] or set()),
                },
                "object_field_types": {
                    k: sorted(v) for k, v in sorted(g["array_object_field_types"].items())
                },
                "sampling": g["array_sampling"],
            },
            "parse_errors": g["parse_errors"],
        }

    manifest["groups"] = finalized_groups
    if parse_error_total:
        manifest["status"] = "ok_with_parse_errors"
        manifest["errors"].append(
            {
                "type": "parse_errors",
                "count": parse_error_total,
                "note": "Some files could not be parsed as JSON; see per-group parse_errors.",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.out.write_text(content)
    if args.out_copy is not None:
        args.out_copy.parent.mkdir(parents=True, exist_ok=True)
        args.out_copy.write_text(content)


if __name__ == "__main__":
    main()
