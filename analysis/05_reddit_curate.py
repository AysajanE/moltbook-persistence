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


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_no_overwrite(out_dir: Path, run_id: str) -> None:
    existing_partition = out_dir / f"run_id={run_id}"
    if existing_partition.exists() and any(existing_partition.rglob("*.parquet")):
        raise FileExistsError(
            f"Refusing to overwrite existing curated data for run_id={run_id}: {existing_partition}"
        )


def _extract_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        v = payload.get("data")
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
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


def _parse_epoch_seconds_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, unit="s", errors="coerce", utc=True)


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


def _infer_submission_id_from_link_id(link_id: str | None) -> str | None:
    if not link_id:
        return None
    if link_id.startswith("t3_"):
        return link_id.removeprefix("t3_")
    return link_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Curate raw Arctic Shift Reddit JSON responses into partitioned Parquet tables under "
            "data_curated/reddit/ with UTC-normalized timestamps and a curation manifest."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Raw run date root (e.g., data_raw/reddit/2026-02-05).",
    )
    parser.add_argument("--attempt-id", required=True, help="Attempt/run identifier (run_id).")
    parser.add_argument("--dt", required=True, help="Run date (YYYY-MM-DD).")
    parser.add_argument("--out-root", type=Path, required=True, help="Curated output root.")
    parser.add_argument(
        "--curation-manifest",
        type=Path,
        required=True,
        help="Output path for curation manifest JSON.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _curate_submissions(
    raw_root: Path, run_id: str, dt: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted(raw_root.glob(f"*/posts_search/{run_id}__*.json"))
    parse_errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for path in paths:
        try:
            payload = _load_json(path)
        except Exception as e:  # noqa: BLE001
            parse_errors.append({"path": str(path), "error": f"{type(e).__name__}: {e}"})
            continue

        for item in _extract_list(payload):
            submission_id = _coerce_str(item.get("id"))
            created_utc = _coerce_int(item.get("created_utc"))
            created_at_raw = created_utc

            rows.append(
                {
                    "submission_id": submission_id,
                    "created_utc": created_utc,
                    "created_at_raw": created_at_raw,
                    "subreddit": _coerce_str(item.get("subreddit")),
                    "title": _coerce_str(item.get("title")),
                    "selftext": _coerce_str(item.get("selftext")),
                    "url": _coerce_str(item.get("url")),
                    "score": _coerce_int(item.get("score")),
                    "num_comments": _coerce_int(item.get("num_comments")),
                    "author": _coerce_str(item.get("author")),
                    "run_id": run_id,
                    "dt": dt,
                }
            )

    df = pd.DataFrame(rows)
    pre_dedup_rows = int(df.shape[0])
    if not df.empty:
        df = df.dropna(subset=["submission_id"]).drop_duplicates(
            subset=["submission_id"], keep="first"
        )
        df["created_at_utc"] = _parse_epoch_seconds_utc(df["created_utc"])
        invalid_ts = int(df["created_at_utc"].isna().sum())
        if invalid_ts:
            df = df[df["created_at_utc"].notna()].copy()
        else:
            invalid_ts = 0

    stats: dict[str, Any] = {
        "raw_files_scanned": len(paths),
        "rows_extracted_pre_dedup": int(len(rows)),
        "rows_pre_dedup": pre_dedup_rows,
        "rows_dropped_invalid_timestamp": invalid_ts if not df.empty else 0,
        "parse_errors": parse_errors,
    }
    if not df.empty:
        stats["timestamp_parse"] = _timestamp_parse_stats(df["created_utc"], df["created_at_utc"])
    return df, stats


def _curate_comments(
    raw_root: Path, run_id: str, dt: str, submission_ids: set[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = sorted(raw_root.glob(f"*/comments_search/{run_id}__*.json"))
    parse_errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for path in paths:
        try:
            payload = _load_json(path)
        except Exception as e:  # noqa: BLE001
            parse_errors.append({"path": str(path), "error": f"{type(e).__name__}: {e}"})
            continue

        for item in _extract_list(payload):
            comment_id = _coerce_str(item.get("id"))
            link_id = _coerce_str(item.get("link_id"))
            submission_id = _infer_submission_id_from_link_id(link_id)
            created_utc = _coerce_int(item.get("created_utc"))
            created_at_raw = created_utc
            author = _coerce_str(item.get("author"))
            body = _coerce_str(item.get("body"))

            body_is_deleted = body == "[deleted]"
            body_is_removed = body == "[removed]"
            author_is_deleted = (author is None) or (author == "[deleted]")

            rows.append(
                {
                    "comment_id": comment_id,
                    "submission_id": submission_id,
                    "parent_id": _coerce_str(item.get("parent_id")),
                    "author": author,
                    "created_utc": created_utc,
                    "created_at_raw": created_at_raw,
                    "body": body,
                    "score": _coerce_int(item.get("score")),
                    "author_is_deleted": bool(author_is_deleted),
                    "body_is_deleted": bool(body_is_deleted),
                    "body_is_removed": bool(body_is_removed),
                    "body_is_deleted_or_removed": bool(body_is_deleted or body_is_removed),
                    "run_id": run_id,
                    "dt": dt,
                }
            )

    df = pd.DataFrame(rows)
    pre_dedup_rows = int(df.shape[0])

    if not df.empty:
        df = df.dropna(subset=["comment_id"]).drop_duplicates(subset=["comment_id"], keep="first")
        df["created_at_utc"] = _parse_epoch_seconds_utc(df["created_utc"])
        invalid_ts = int(df["created_at_utc"].isna().sum())
        if invalid_ts:
            df = df[df["created_at_utc"].notna()].copy()
        else:
            invalid_ts = 0

        pre_submission_filter_rows = int(df.shape[0])

        # Enforce referential integrity: only keep comments whose submission_id is in submissions.
        df = df[df["submission_id"].isin(submission_ids)].copy()

    stats: dict[str, Any] = {
        "raw_files_scanned": len(paths),
        "rows_extracted_pre_dedup": int(len(rows)),
        "rows_pre_dedup": pre_dedup_rows,
        "rows_dropped_invalid_timestamp": invalid_ts if not df.empty else 0,
        "rows_pre_submission_filter": pre_submission_filter_rows if not df.empty else 0,
        "rows_dropped_missing_submission": (
            int(pre_submission_filter_rows - int(df.shape[0])) if not df.empty else 0
        ),
        "parse_errors": parse_errors,
    }
    if not df.empty:
        stats["timestamp_parse"] = _timestamp_parse_stats(df["created_utc"], df["created_at_utc"])
    return df, stats


def main() -> None:
    args = parse_args()
    raw_root: Path = args.raw_root
    run_id: str = args.attempt_id
    dt: str = args.dt
    out_root: Path = args.out_root

    submissions_df, submissions_stats = _curate_submissions(raw_root, run_id=run_id, dt=dt)
    submission_ids: set[str] = (
        set(submissions_df["submission_id"].astype(str).tolist())
        if not submissions_df.empty
        else set()
    )
    comments_df, comments_stats = _curate_comments(
        raw_root, run_id=run_id, dt=dt, submission_ids=submission_ids
    )

    outputs: dict[str, dict[str, Any]] = {}
    for name, df in [("submissions", submissions_df), ("comments", comments_df)]:
        out_dir = out_root / name
        _ensure_no_overwrite(out_dir, run_id=run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

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
        }

    manifest: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "raw_root": str(raw_root),
        "run_id": run_id,
        "dt": dt,
        "out_root": str(out_root),
        "tables": outputs,
        "stats": {
            "submissions": submissions_stats,
            "comments": comments_stats,
        },
        "notes": [
            "Comments are filtered so submission_id always exists in curated submissions.",
            "Timestamps normalized to UTC via created_utc epoch seconds.",
        ],
    }

    args.curation_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.curation_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"status": "ok", "curation_manifest": str(args.curation_manifest)}, indent=2))


if __name__ == "__main__":
    main()
