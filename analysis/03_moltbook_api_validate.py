#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

ALLOWED_SORTS = {"hot", "new", "top", "rising"}


@dataclass(frozen=True)
class CheckResult:
    status: str  # PASS/WARN/FAIL
    details: dict[str, Any]


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


def _read_dataset(curated_root: Path, subset: str) -> pd.DataFrame:
    dataset = ds.dataset(
        str(curated_root / subset),
        format="parquet",
        partitioning="hive",
    )
    return dataset.to_table().to_pandas()


def _status_from_rate(rate: float | None, pass_min: float = 0.99, warn_min: float = 0.90) -> str:
    if rate is None:
        return "PASS"
    if rate >= pass_min:
        return "PASS"
    if rate >= warn_min:
        return "WARN"
    return "FAIL"


def _timestamp_parse_rate(df: pd.DataFrame, raw_col: str, utc_col: str) -> dict[str, Any]:
    raw = df.get(raw_col)
    utc = df.get(utc_col)
    if raw is None or utc is None:
        return {"error": "missing_columns", "raw_col": raw_col, "utc_col": utc_col}

    nonnull = int(raw.notna().sum())
    parsed_ok = int(utc.notna().sum())
    parsed_fail = int(nonnull - parsed_ok)
    rate = (float(parsed_ok) / float(nonnull)) if nonnull else None
    return {
        "raw_col": raw_col,
        "utc_col": utc_col,
        "nonnull": nonnull,
        "parsed_success": parsed_ok,
        "parsed_fail": parsed_fail,
        "parse_rate": rate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate curated Moltbook API pilot tables for basic invariants."
    )
    parser.add_argument(
        "--curated-root",
        type=Path,
        required=True,
        help="Curated root (e.g., data_curated/moltbook).",
    )
    parser.add_argument(
        "--run-id", required=True, help="run_id / attempt_id partition to validate."
    )
    parser.add_argument(
        "--request-log",
        type=Path,
        required=True,
        help="Raw JSONL request log path for the run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output validation results JSON path.",
    )
    parser.add_argument(
        "--chronology-skew-seconds",
        type=int,
        default=300,
        help="Allowed negative skew (comment earlier than post) in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curated_root: Path = args.curated_root
    run_id: str = args.run_id

    request_entries = _load_jsonl(args.request_log)

    # Infer effective run mode for reporting.
    modes = sorted({str(e.get("mode")) for e in request_entries if e.get("mode")})
    any_live_success = any(
        (e.get("mode") in {"live", "unauth"} and e.get("http_status") == 200)
        for e in request_entries
    )
    any_stub = any(e.get("mode") == "stub" for e in request_entries)
    effective_mode = "live_or_unauth" if any_live_success else ("stub" if any_stub else "unknown")

    # Load curated tables
    feed = _read_dataset(curated_root, "feed_snapshots")
    posts = _read_dataset(curated_root, "posts")
    comments = _read_dataset(curated_root, "comments")

    # Filter to run_id
    for df in [feed, posts, comments]:
        if "run_id" in df.columns:
            df.query("run_id == @run_id", inplace=True)

    checks: dict[str, CheckResult] = {}

    # Check: request log summary and feed 200 presence (when live/unauth)
    feed_req = [
        e
        for e in request_entries
        if e.get("endpoint") == "/posts" and isinstance(e.get("params"), dict)
    ]
    feed_200 = sum(1 for e in feed_req if e.get("http_status") == 200)
    total_reqs = len(request_entries)
    error_reqs = sum(
        1 for e in request_entries if e.get("error") is not None or e.get("http_status") != 200
    )
    error_rate = (error_reqs / total_reqs) if total_reqs else None
    need_feed_200 = effective_mode == "live_or_unauth"
    status = "PASS"
    if need_feed_200 and feed_200 == 0:
        status = "FAIL"
    checks["request_log_summary"] = CheckResult(
        status=status,
        details={
            "effective_mode": effective_mode,
            "modes_observed": modes,
            "request_total": total_reqs,
            "request_errors_or_non200": error_reqs,
            "error_rate": error_rate,
            "feed_requests_total": len(feed_req),
            "feed_requests_200": feed_200,
        },
    )

    # Feed snapshot invariants
    if feed.empty:
        checks["feed_snapshots_nonempty"] = CheckResult(
            status="FAIL", details={"reason": "empty_table"}
        )
    else:
        checks["feed_snapshots_nonempty"] = CheckResult(
            status="PASS", details={"rows": int(feed.shape[0])}
        )

    if not feed.empty:
        bad_sort = (
            sorted(set(feed["sort"].dropna().astype(str)) - ALLOWED_SORTS)
            if "sort" in feed.columns
            else []
        )
        checks["feed_sort_values"] = CheckResult(
            status="PASS" if not bad_sort else "FAIL",
            details={"bad_values": bad_sort, "allowed": sorted(ALLOWED_SORTS)},
        )

        if {"rank", "limit"}.issubset(set(feed.columns)):
            violations = int(((feed["rank"] < 1) | (feed["rank"] > feed["limit"])).sum())
            checks["feed_rank_bounds"] = CheckResult(
                status="PASS" if violations == 0 else "FAIL",
                details={"violations": violations},
            )
        else:
            checks["feed_rank_bounds"] = CheckResult(
                status="FAIL",
                details={"error": "missing_columns", "required": ["rank", "limit"]},
            )

        if "post_id" in feed.columns:
            nulls = int(feed["post_id"].isna().sum())
            checks["feed_post_id_nonnull"] = CheckResult(
                status="PASS" if nulls == 0 else "FAIL",
                details={"null_count": nulls, "rows": int(feed.shape[0])},
            )
        else:
            checks["feed_post_id_nonnull"] = CheckResult(
                status="FAIL", details={"error": "missing_column", "required": "post_id"}
            )

        # Timestamp parse rate for snapshot_time
        if {"snapshot_time_raw", "snapshot_time_utc"}.issubset(set(feed.columns)):
            stats = _timestamp_parse_rate(feed, "snapshot_time_raw", "snapshot_time_utc")
            checks["feed_snapshot_time_parse_rate"] = CheckResult(
                status=_status_from_rate(stats.get("parse_rate")),
                details=stats,
            )
        else:
            checks["feed_snapshot_time_parse_rate"] = CheckResult(
                status="FAIL",
                details={
                    "error": "missing_columns",
                    "required": ["snapshot_time_raw", "snapshot_time_utc"],
                },
            )

    # Posts timestamp parse rate
    if not posts.empty and {"created_at_raw", "created_at_utc"}.issubset(set(posts.columns)):
        stats = _timestamp_parse_rate(posts, "created_at_raw", "created_at_utc")
        checks["posts_created_at_parse_rate"] = CheckResult(
            status=_status_from_rate(stats.get("parse_rate")),
            details=stats,
        )
    else:
        checks["posts_created_at_parse_rate"] = CheckResult(
            status="WARN" if posts.empty else "FAIL",
            details={"reason": "empty_table" if posts.empty else "missing_columns"},
        )

    # Comments timestamp parse rate + uniqueness
    if comments.empty:
        checks["comments_nonempty"] = CheckResult(status="WARN", details={"reason": "empty_table"})
    else:
        checks["comments_nonempty"] = CheckResult(
            status="PASS", details={"rows": int(comments.shape[0])}
        )

    if not comments.empty and {"created_at_raw", "created_at_utc"}.issubset(set(comments.columns)):
        stats = _timestamp_parse_rate(comments, "created_at_raw", "created_at_utc")
        checks["comments_created_at_parse_rate"] = CheckResult(
            status=_status_from_rate(stats.get("parse_rate")),
            details=stats,
        )
    else:
        checks["comments_created_at_parse_rate"] = CheckResult(
            status="WARN" if comments.empty else "FAIL",
            details={"reason": "empty_table" if comments.empty else "missing_columns"},
        )

    if not comments.empty and "comment_id" in comments.columns:
        dup = (
            comments.groupby("comment_id", dropna=False).size().reset_index(name="n").query("n > 1")
        )
        dup_rows = int(dup["n"].sum()) if not dup.empty else 0
        checks["comments_comment_id_unique"] = CheckResult(
            status="PASS" if dup_rows == 0 else "FAIL",
            details={
                "duplicate_rows_total": dup_rows,
                "duplicate_ids": dup["comment_id"].astype(str).tolist()[:20],
            },
        )
    else:
        checks["comments_comment_id_unique"] = CheckResult(
            status="WARN" if comments.empty else "FAIL",
            details={"reason": "empty_table" if comments.empty else "missing_column"},
        )

    # Chronology sanity: comment_created_at_utc >= post_created_at_utc - tolerance
    if (
        not comments.empty
        and not posts.empty
        and {"post_id", "created_at_utc"}.issubset(set(posts.columns))
    ):
        if {"post_id", "created_at_utc"}.issubset(set(comments.columns)):
            merged = comments.merge(
                posts[["post_id", "created_at_utc"]],
                on="post_id",
                how="left",
                suffixes=("_comment", "_post"),
            )
            comment_ts = pd.to_datetime(merged["created_at_utc_comment"], errors="coerce", utc=True)
            post_ts = pd.to_datetime(merged["created_at_utc_post"], errors="coerce", utc=True)
            tol = timedelta(seconds=int(args.chronology_skew_seconds))
            comparable = comment_ts.notna() & post_ts.notna()
            violations = int((comment_ts[comparable] < (post_ts[comparable] - tol)).sum())
            checked = int(comparable.sum())
            status = "PASS" if violations == 0 else "WARN"
            checks["chronology_comment_after_post"] = CheckResult(
                status=status,
                details={
                    "violations": violations,
                    "checked_rows": checked,
                    "tolerance_seconds": int(args.chronology_skew_seconds),
                },
            )
        else:
            checks["chronology_comment_after_post"] = CheckResult(
                status="FAIL",
                details={
                    "error": "missing_columns",
                    "required_posts": ["post_id", "created_at_utc"],
                    "required_comments": ["post_id", "created_at_utc"],
                },
            )
    else:
        checks["chronology_comment_after_post"] = CheckResult(
            status="WARN",
            details={
                "reason": "missing_inputs",
                "comments_empty": comments.empty,
                "posts_empty": posts.empty,
            },
        )

    # Summarize
    fail = [k for k, v in checks.items() if v.status == "FAIL"]
    warn = [k for k, v in checks.items() if v.status == "WARN"]
    summary_status = "FAIL" if fail else ("WARN" if warn else "PASS")

    out_obj: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "curated_root": str(curated_root),
        "run_id": run_id,
        "request_log": str(args.request_log),
        "effective_mode": effective_mode,
        "row_counts": {
            "feed_snapshots": int(feed.shape[0]),
            "posts": int(posts.shape[0]),
            "comments": int(comments.shape[0]),
        },
        "checks": {
            k: {"status": v.status, "details": v.details} for k, v in sorted(checks.items())
        },
        "summary": {
            "status": summary_status,
            "fail_count": len(fail),
            "warn_count": len(warn),
            "fail_checks": fail,
            "warn_checks": warn,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": summary_status, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
