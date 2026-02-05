#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_SUBSETS = [
    "agents",
    "comments",
    "posts",
    "snapshots",
    "submolts",
    "word_frequency",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_parquet_schema(path: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow

    return {
        "path": str(path),
        "rows": parquet_file.metadata.num_rows if parquet_file.metadata else None,
        "columns": [field.name for field in schema],
        "types": {field.name: str(field.type) for field in schema},
    }


def _normalize_columns(columns: list[str]) -> dict[str, str]:
    return {c.lower(): c for c in columns}


def _pick_column(
    normalized: dict[str, str], candidates: list[str]
) -> tuple[str | None, list[str]]:
    matched: list[str] = []
    for cand in candidates:
        found = normalized.get(cand.lower())
        if found is not None:
            matched.append(found)
    if not matched:
        return None, []
    return matched[0], matched


def _required_variable_spec() -> dict[str, dict[str, list[str]]]:
    # Canonical required variables (minimum) for downstream modeling.
    return {
        "posts": {
            "id": ["id"],
            "created_at": ["created_at", "created_utc", "createdAt", "timestamp"],
            "author_id": ["author_id", "authorId", "agent_id", "agentId"],
            "author_name": ["author_name", "author", "username", "authorUsername"],
            "submolt": ["submolt", "submolt_name", "community", "community_name"],
            "title": ["title"],
            "content": ["content", "body", "text", "selftext"],
            "url": ["url", "link", "link_url", "external_url"],
            "score": ["score", "karma", "upvotes", "like_count"],
            "comment_count": ["comment_count", "num_comments", "comments_count"],
            "dump_date": ["dump_date", "dumpDate", "dt"],
        },
        "comments": {
            "id": ["id"],
            "post_id": ["post_id", "postId", "submission_id", "submissionId"],
            "parent_id": ["parent_id", "parentId", "parent_comment_id", "reply_to"],
            "author_id": ["author_id", "authorId", "agent_id", "agentId"],
            "author_name": ["author_name", "author", "username", "authorUsername"],
            "created_at": ["created_at", "created_utc", "createdAt", "timestamp"],
            "content": ["content", "body", "text"],
            "score": ["score", "karma", "upvotes", "like_count"],
            "dump_date": ["dump_date", "dumpDate", "dt"],
        },
        "submolts": {
            "name": ["name", "submolt", "submolt_name", "id"],
            "display_name": ["display_name", "displayName", "title"],
            "description": ["description", "about"],
            "subscriber_count": [
                "subscriber_count",
                "subscriberCount",
                "member_count",
                "memberCount",
                "members",
                "subscribers",
            ],
            "created_at": ["created_at", "createdAt", "timestamp"],
            "dump_date": ["dump_date", "dumpDate", "dt"],
        },
        "snapshots": {
            "captured_at": ["captured_at", "snapshot_at", "created_at", "timestamp"],
            "posts_count": ["posts_count", "post_count", "num_posts"],
            "comments_count": ["comments_count", "comment_count", "num_comments"],
            "agents_count": ["agents_count", "agent_count", "num_agents"],
            "dump_date": ["dump_date", "dumpDate", "dt"],
        },
        "word_frequency": {
            "token": ["token", "word", "term"],
            "count": ["count", "frequency", "n"],
            "dump_date": ["dump_date", "dumpDate", "dt"],
        },
        "agents": {
            "id": ["id"],
            "name": ["name"],
            "first_seen_at": ["first_seen_at", "firstSeenAt"],
            "last_seen_at": ["last_seen_at", "lastSeenAt"],
            "created_at": ["created_at", "createdAt"],
            "dump_date": ["dump_date", "dumpDate", "dt"],
        },
    }


def _build_schema_mapping(columns_by_subset: dict[str, list[str]]) -> dict[str, Any]:
    spec = _required_variable_spec()
    mapping: dict[str, Any] = {"generated_at_utc": _utc_now_iso(), "tables": {}}

    for subset, columns in columns_by_subset.items():
        normalized = _normalize_columns(columns)
        table_spec = spec.get(subset, {})
        table_mapping: dict[str, Any] = {"columns_present": columns, "required": {}}

        missing: list[str] = []
        for var_name, candidates in table_spec.items():
            chosen, matched = _pick_column(normalized, candidates)
            if chosen is None:
                missing.append(var_name)
            table_mapping["required"][var_name] = {
                "candidates": candidates,
                "mapped_to": chosen,
                "matches": matched,
                "status": "found" if chosen is not None else "missing",
            }

        # Special OR requirements (author id vs author name for posts/comments)
        or_groups: list[dict[str, Any]] = []
        if subset in {"posts", "comments"}:
            author_ok = (
                table_mapping["required"]["author_id"]["mapped_to"] is not None
                or table_mapping["required"]["author_name"]["mapped_to"] is not None
            )
            or_groups.append(
                {
                    "name": "author_identifier",
                    "any_of": ["author_id", "author_name"],
                    "status": "satisfied" if author_ok else "missing_both",
                }
            )

        table_mapping["summary"] = {
            "missing_required": missing,
            "missing_required_count": len(missing),
            "or_groups": or_groups,
        }
        mapping["tables"][subset] = table_mapping

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover schema (columns + Arrow types) for the Moltbook HF archive raw "
            "parquet exports, and produce a required-variable mapping report."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Raw snapshot directory (e.g., data_raw/hf_archive/snapshot_*/)",
    )
    parser.add_argument(
        "--out-schema",
        type=Path,
        required=True,
        help="Output path for schema discovery JSON.",
    )
    parser.add_argument(
        "--out-mapping",
        type=Path,
        required=True,
        help="Output path for required-variable mapping JSON.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        choices=DEFAULT_SUBSETS,
        help="Subset(s) to process (repeatable). Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    subsets = args.subset or DEFAULT_SUBSETS

    discovered: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "raw_dir": str(raw_dir),
        "subsets": {},
    }

    columns_by_subset: dict[str, list[str]] = {}
    for subset in subsets:
        parquet_path = raw_dir / subset / "archive.parquet"
        if not parquet_path.exists():
            discovered["subsets"][subset] = {
                "error": f"missing file: {parquet_path}",
                "splits": {},
            }
            continue

        schema_info = _read_parquet_schema(parquet_path)
        discovered["subsets"][subset] = {"splits": {"archive": schema_info}}
        columns_by_subset[subset] = list(schema_info["columns"])

    args.out_schema.parent.mkdir(parents=True, exist_ok=True)
    args.out_schema.write_text(json.dumps(discovered, indent=2, sort_keys=True) + "\n")

    mapping = _build_schema_mapping(columns_by_subset)
    mapping["raw_dir"] = str(raw_dir)
    args.out_mapping.parent.mkdir(parents=True, exist_ok=True)
    args.out_mapping.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
