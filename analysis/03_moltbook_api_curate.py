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

FEED_COLUMNS = [
    "run_id",
    "dt",
    "snapshot_time_raw",
    "sort",
    "limit",
    "rank",
    "post_id",
    "post_created_at_raw",
    "score",
    "comment_count",
    "submolt_name",
    "author_id",
    "author_name",
    "synthetic",
]
POST_COLUMNS = [
    "run_id",
    "dt",
    "retrieved_at_raw",
    "post_id",
    "created_at_raw",
    "title",
    "url",
    "score",
    "comment_count",
    "submolt_name",
    "author_id",
    "author_name",
    "synthetic",
]
COMMENT_COLUMNS = [
    "post_id",
    "comment_id",
    "parent_id",
    "author_id",
    "author_name",
    "created_at_raw",
    "score",
    "depth",
    "run_id",
    "dt",
    "retrieved_at_raw",
    "comment_sort",
    "synthetic",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_timestamp_utc(values: pd.Series) -> pd.Series:
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
        examples = list(dict.fromkeys(failures.tolist()))[:5]

    return {
        "nonnull": nonnull_count,
        "parsed_success": parsed_ok_count,
        "parsed_fail": parsed_fail_count,
        "parse_rate": parse_rate,
        "examples_fail_raw": examples,
    }


def _ensure_no_overwrite(out_dir: Path, run_id: str) -> None:
    existing_partition = out_dir / f"run_id={run_id}"
    if existing_partition.exists() and any(existing_partition.rglob("*.parquet")):
        raise FileExistsError(
            f"Refusing to overwrite existing curated data for run_id={run_id}: {existing_partition}"
        )


def _extract_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ["posts", "data", "items", "results", "comments"]:
            v = payload.get(k)
            if isinstance(v, list):
                return v
    return []


def _get_first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _extract_author_fields(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    author = obj.get("author") or obj.get("agent") or obj.get("user")
    if isinstance(author, dict):
        author_id = _coerce_str(_get_first(author, ["id", "agent_id", "user_id"]))
        author_name = _coerce_str(_get_first(author, ["name", "username", "handle"]))
        return author_id, author_name
    if isinstance(author, str):
        return None, author
    return None, None


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out.loc[:, columns]


def _extract_submolt_name(obj: dict[str, Any]) -> str | None:
    submolt = obj.get("submolt") or obj.get("community")
    if isinstance(submolt, dict):
        return _coerce_str(_get_first(submolt, ["name", "id"]))
    if isinstance(submolt, str):
        return submolt
    return None


def _flatten_comments(payload: Any, post_id: str | None) -> list[dict[str, Any]]:
    roots = _extract_list(payload)
    out: list[dict[str, Any]] = []
    stack: list[tuple[dict[str, Any], str | None, int]] = []
    for item in roots:
        if isinstance(item, dict):
            stack.append((item, None, 0))

    while stack:
        node, inferred_parent_id, depth = stack.pop()
        comment_id = _coerce_str(_get_first(node, ["id", "comment_id"]))
        parent_id = (
            _coerce_str(_get_first(node, ["parent_id", "reply_to_id"])) or inferred_parent_id
        )
        author_id, author_name = _extract_author_fields(node)
        created_at_raw = _coerce_str(
            _get_first(node, ["created_at", "created_at_utc", "timestamp"])
        )
        score = _coerce_int(_get_first(node, ["score", "upvotes"]))

        out.append(
            {
                "post_id": post_id or _coerce_str(_get_first(node, ["post_id"])),
                "comment_id": comment_id,
                "parent_id": parent_id,
                "author_id": author_id,
                "author_name": author_name,
                "created_at_raw": created_at_raw,
                "score": score,
                "depth": int(depth),
            }
        )

        children = node.get("children")
        if isinstance(children, list):
            for child in reversed(children):
                if isinstance(child, dict):
                    stack.append((child, comment_id, depth + 1))

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate raw Moltbook API JSON responses into partitioned Parquet tables under "
            "data_curated/moltbook/ with UTC-parsed timestamps and a curation manifest."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Raw date root (e.g., data_raw/moltbook_api/2026-02-05).",
    )
    parser.add_argument("--attempt-id", required=True, help="Attempt/run identifier.")
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Curated output root (e.g., data_curated/moltbook).",
    )
    parser.add_argument(
        "--curation-manifest",
        type=Path,
        required=True,
        help="Output path for curation manifest JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root: Path = args.raw_root
    run_id: str = args.attempt_id
    out_root: Path = args.out_root

    dt = raw_root.name
    request_log_path = raw_root / "request_log" / f"{run_id}.jsonl"
    entries = _load_jsonl(request_log_path)
    if not entries:
        raise SystemExit(f"Missing or empty request log: {request_log_path}")

    feed_rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    comment_rows: list[dict[str, Any]] = []

    parse_errors: list[dict[str, Any]] = []

    for entry in entries:
        raw_path = entry.get("raw_path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.exists():
            parse_errors.append({"raw_path": str(path), "error": "missing_file"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            parse_errors.append({"raw_path": str(path), "error": f"{type(e).__name__}: {e}"})
            continue

        endpoint = str(entry.get("endpoint") or "")
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        retrieved_at = _coerce_str(entry.get("retrieved_at_utc"))
        synthetic = bool(entry.get("synthetic"))

        if endpoint == "/posts":
            sort = _coerce_str(params.get("sort"))
            limit = _coerce_int(params.get("limit"))
            posts_list = _extract_list(payload)
            for i, post in enumerate(posts_list):
                if not isinstance(post, dict):
                    continue
                post_id = _coerce_str(_get_first(post, ["id", "post_id"]))
                score = _coerce_int(_get_first(post, ["score", "upvotes"]))
                comment_count = _coerce_int(
                    _get_first(post, ["comment_count", "comments_count", "num_comments"])
                )
                author_id, author_name = _extract_author_fields(post)
                submolt_name = _extract_submolt_name(post)
                post_created_at_raw = _coerce_str(
                    _get_first(post, ["created_at", "created_at_utc", "timestamp"])
                )
                feed_rows.append(
                    {
                        "run_id": run_id,
                        "dt": dt,
                        "snapshot_time_raw": retrieved_at,
                        "sort": sort,
                        "limit": limit,
                        "rank": int(i + 1),
                        "post_id": post_id,
                        "post_created_at_raw": post_created_at_raw,
                        "score": score,
                        "comment_count": comment_count,
                        "submolt_name": submolt_name,
                        "author_id": author_id,
                        "author_name": author_name,
                        "synthetic": synthetic,
                    }
                )
        elif endpoint.startswith("/posts/") and endpoint.endswith("/comments"):
            # Endpoint form: /posts/{id}/comments
            parts = endpoint.strip("/").split("/")
            post_id = parts[1] if len(parts) >= 2 else None
            sort = _coerce_str(params.get("sort"))
            flattened = _flatten_comments(payload, post_id=post_id)
            for row in flattened:
                row.update(
                    {
                        "run_id": run_id,
                        "dt": dt,
                        "retrieved_at_raw": retrieved_at,
                        "comment_sort": sort,
                        "synthetic": synthetic,
                    }
                )
                comment_rows.append(row)
        elif endpoint.startswith("/posts/"):
            parts = endpoint.strip("/").split("/")
            post_id = parts[1] if len(parts) >= 2 else None
            post_obj: dict[str, Any] | None = payload if isinstance(payload, dict) else None
            if post_obj is None:
                continue
            # Some APIs wrap: {"post": {...}}
            if "post" in post_obj and isinstance(post_obj.get("post"), dict):
                post_obj = post_obj["post"]
            post_id = _coerce_str(_get_first(post_obj, ["id", "post_id"])) or post_id
            created_at_raw = _coerce_str(
                _get_first(post_obj, ["created_at", "created_at_utc", "timestamp"])
            )
            title = _coerce_str(_get_first(post_obj, ["title"]))
            url = _coerce_str(_get_first(post_obj, ["url", "link"]))
            score = _coerce_int(_get_first(post_obj, ["score", "upvotes"]))
            comment_count = _coerce_int(
                _get_first(post_obj, ["comment_count", "comments_count", "num_comments"])
            )
            author_id, author_name = _extract_author_fields(post_obj)
            submolt_name = _extract_submolt_name(post_obj)

            post_rows.append(
                {
                    "run_id": run_id,
                    "dt": dt,
                    "retrieved_at_raw": retrieved_at,
                    "post_id": post_id,
                    "created_at_raw": created_at_raw,
                    "title": title,
                    "url": url,
                    "score": score,
                    "comment_count": comment_count,
                    "submolt_name": submolt_name,
                    "author_id": author_id,
                    "author_name": author_name,
                    "synthetic": synthetic,
                }
            )

    # Convert to DataFrames and parse timestamps
    feed_df = _ensure_columns(pd.DataFrame(feed_rows), FEED_COLUMNS)
    posts_df = _ensure_columns(pd.DataFrame(post_rows), POST_COLUMNS)
    comments_df = _ensure_columns(pd.DataFrame(comment_rows), COMMENT_COLUMNS)

    ts_stats: dict[str, Any] = {"feed_snapshots": {}, "posts": {}, "comments": {}}

    if not feed_df.empty:
        feed_df["snapshot_time_utc"] = _parse_timestamp_utc(feed_df["snapshot_time_raw"])
        ts_stats["feed_snapshots"]["snapshot_time"] = _timestamp_parse_stats(
            feed_df["snapshot_time_raw"], feed_df["snapshot_time_utc"]
        )
        if "post_created_at_raw" in feed_df.columns:
            feed_df["post_created_at_utc"] = _parse_timestamp_utc(feed_df["post_created_at_raw"])
            ts_stats["feed_snapshots"]["post_created_at"] = _timestamp_parse_stats(
                feed_df["post_created_at_raw"], feed_df["post_created_at_utc"]
            )

    if not posts_df.empty:
        posts_df["retrieved_at_utc"] = _parse_timestamp_utc(posts_df["retrieved_at_raw"])
        ts_stats["posts"]["retrieved_at"] = _timestamp_parse_stats(
            posts_df["retrieved_at_raw"], posts_df["retrieved_at_utc"]
        )
        posts_df["created_at_utc"] = _parse_timestamp_utc(posts_df["created_at_raw"])
        ts_stats["posts"]["created_at"] = _timestamp_parse_stats(
            posts_df["created_at_raw"], posts_df["created_at_utc"]
        )

    if not comments_df.empty:
        comments_df["retrieved_at_utc"] = _parse_timestamp_utc(comments_df["retrieved_at_raw"])
        ts_stats["comments"]["retrieved_at"] = _timestamp_parse_stats(
            comments_df["retrieved_at_raw"], comments_df["retrieved_at_utc"]
        )
        comments_df["created_at_utc"] = _parse_timestamp_utc(comments_df["created_at_raw"])
        ts_stats["comments"]["created_at"] = _timestamp_parse_stats(
            comments_df["created_at_raw"], comments_df["created_at_utc"]
        )

    # Write datasets (hive partitioning by run_id + dt)
    outputs: dict[str, dict[str, Any]] = {}
    for name, df in [("feed_snapshots", feed_df), ("posts", posts_df), ("comments", comments_df)]:
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            outputs[name] = {
                "rows": 0,
                "columns": list(df.columns),
                "out_dir": str(out_dir),
                "partitioning": "hive(run_id=.../dt=YYYY-MM-DD)",
                "written": False,
                "reason": "empty_table",
            }
            continue

        _ensure_no_overwrite(out_dir, run_id=run_id)

        table = pa.Table.from_pandas(df, preserve_index=False)
        ds.write_dataset(
            data=table,
            base_dir=str(out_dir),
            format="parquet",
            partitioning=["run_id", "dt"],
            partitioning_flavor="hive",
            create_dir=True,
            existing_data_behavior="overwrite_or_ignore",
        )
        outputs[name] = {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "out_dir": str(out_dir),
            "partitioning": "hive(run_id=.../dt=YYYY-MM-DD)",
            "written": True,
        }

    manifest: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "raw_root": str(raw_root),
        "request_log_path": str(request_log_path),
        "run_id": run_id,
        "out_root": str(out_root),
        "dt": dt,
        "tables": outputs,
        "timestamp_parse_stats": ts_stats,
        "parse_errors": parse_errors,
    }

    args.curation_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.curation_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"status": "ok", "curation_manifest": str(args.curation_manifest)}, indent=2))


if __name__ == "__main__":
    main()
