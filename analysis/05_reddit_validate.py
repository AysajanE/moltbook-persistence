#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


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


def _read_partitioned_dataset(curated_root: Path, subset: str, run_id: str) -> pd.DataFrame:
    dataset = ds.dataset(
        str(curated_root / subset),
        format="parquet",
        partitioning="hive",
    )
    table = dataset.to_table(filter=ds.field("run_id") == run_id)
    return table.to_pandas()


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


def _status_from_rate(rate: float | None, pass_min: float = 0.99, warn_min: float = 0.90) -> str:
    if rate is None:
        return "PASS"
    if rate >= pass_min:
        return "PASS"
    if rate >= warn_min:
        return "WARN"
    return "FAIL"


def _dataset_nonempty_check(df: pd.DataFrame) -> CheckResult:
    if df.empty:
        return CheckResult(status="FAIL", details={"reason": "empty_table"})
    return CheckResult(status="PASS", details={"rows": int(df.shape[0])})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate curated Arctic Shift Reddit pilot tables for basic invariants."
    )
    parser.add_argument(
        "--curated-root",
        type=Path,
        required=True,
        help="Curated root (e.g., data_curated/reddit).",
    )
    parser.add_argument(
        "--run-id", required=True, help="run_id / attempt_id partition to validate."
    )
    parser.add_argument(
        "--request-log", type=Path, required=True, help="Raw JSONL request log path."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output validation results JSON path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curated_root: Path = args.curated_root
    run_id: str = args.run_id

    checks: dict[str, CheckResult] = {}

    # Request log checks
    request_entries = _load_jsonl(args.request_log)
    if not args.request_log.exists():
        checks["request_log_exists"] = CheckResult(
            status="FAIL", details={"path": str(args.request_log), "reason": "missing"}
        )
    elif args.request_log.stat().st_size == 0:
        checks["request_log_exists"] = CheckResult(
            status="FAIL", details={"path": str(args.request_log), "reason": "empty"}
        )
    else:
        checks["request_log_exists"] = CheckResult(
            status="PASS",
            details={
                "path": str(args.request_log),
                "entries": int(len(request_entries)),
                "http_200": int(sum(1 for e in request_entries if e.get("http_status") == 200)),
                "non_200_or_error": int(
                    sum(1 for e in request_entries if e.get("http_status") != 200 or e.get("error"))
                ),
            },
        )

    # Load datasets filtered to run_id
    try:
        submissions = _read_partitioned_dataset(curated_root, "submissions", run_id=run_id)
        checks["submissions_nonempty"] = _dataset_nonempty_check(submissions)
    except Exception as e:  # noqa: BLE001
        submissions = pd.DataFrame()
        checks["submissions_nonempty"] = CheckResult(
            status="FAIL", details={"error": f"{type(e).__name__}: {e}"}
        )

    try:
        comments = _read_partitioned_dataset(curated_root, "comments", run_id=run_id)
        checks["comments_nonempty"] = _dataset_nonempty_check(comments)
    except Exception as e:  # noqa: BLE001
        comments = pd.DataFrame()
        checks["comments_nonempty"] = CheckResult(
            status="FAIL", details={"error": f"{type(e).__name__}: {e}"}
        )

    # Required columns
    required_sub_cols = {
        "submission_id",
        "created_utc",
        "subreddit",
        "title",
        "score",
        "num_comments",
        "created_at_utc",
        "created_at_raw",
        "run_id",
        "dt",
    }
    required_com_cols = {
        "comment_id",
        "submission_id",
        "parent_id",
        "author",
        "created_utc",
        "body",
        "score",
        "created_at_utc",
        "created_at_raw",
        "run_id",
        "dt",
        "author_is_deleted",
        "body_is_deleted",
        "body_is_removed",
    }

    if not submissions.empty:
        missing = sorted(required_sub_cols - set(submissions.columns))
        checks["submissions_required_columns"] = CheckResult(
            status="PASS" if not missing else "FAIL",
            details={"missing": missing, "required": sorted(required_sub_cols)},
        )
    else:
        checks["submissions_required_columns"] = CheckResult(
            status="FAIL",
            details={
                "reason": "submissions_empty_or_unreadable",
                "required": sorted(required_sub_cols),
            },
        )

    if not comments.empty:
        missing = sorted(required_com_cols - set(comments.columns))
        checks["comments_required_columns"] = CheckResult(
            status="PASS" if not missing else "FAIL",
            details={"missing": missing, "required": sorted(required_com_cols)},
        )
    else:
        checks["comments_required_columns"] = CheckResult(
            status="FAIL",
            details={
                "reason": "comments_empty_or_unreadable",
                "required": sorted(required_com_cols),
            },
        )

    # Uniqueness
    if not submissions.empty and "submission_id" in submissions.columns:
        dup = int(submissions["submission_id"].duplicated().sum())
        checks["submission_id_unique"] = CheckResult(
            status="PASS" if dup == 0 else "FAIL", details={"duplicate_count": dup}
        )
    else:
        checks["submission_id_unique"] = CheckResult(
            status="FAIL", details={"reason": "missing_submission_id"}
        )

    if not comments.empty and "comment_id" in comments.columns:
        dup = int(comments["comment_id"].duplicated().sum())
        checks["comment_id_unique"] = CheckResult(
            status="PASS" if dup == 0 else "FAIL", details={"duplicate_count": dup}
        )
    else:
        checks["comment_id_unique"] = CheckResult(
            status="FAIL", details={"reason": "missing_comment_id"}
        )

    # Timestamp parse rate
    if not submissions.empty:
        stats = _timestamp_parse_rate(submissions, "created_at_raw", "created_at_utc")
        checks["submissions_timestamp_parse_rate"] = CheckResult(
            status=_status_from_rate(stats.get("parse_rate")), details=stats
        )
    if not comments.empty:
        stats = _timestamp_parse_rate(comments, "created_at_raw", "created_at_utc")
        checks["comments_timestamp_parse_rate"] = CheckResult(
            status=_status_from_rate(stats.get("parse_rate")), details=stats
        )

    # Referential integrity: comments.submission_id must exist in submissions
    if (
        not comments.empty
        and not submissions.empty
        and "submission_id" in comments.columns
        and "submission_id" in submissions.columns
    ):
        sub_ids = set(submissions["submission_id"].astype(str).tolist())
        missing = int((~comments["submission_id"].astype(str).isin(sub_ids)).sum())
        checks["comments_submission_id_fk"] = CheckResult(
            status="PASS" if missing == 0 else "FAIL",
            details={"missing_submission_id_count": missing},
        )
    else:
        checks["comments_submission_id_fk"] = CheckResult(
            status="FAIL",
            details={"reason": "missing_tables_or_columns"},
        )

    # Parent completeness metric (comment parents only). Never FAIL; WARN if incomplete.
    if not comments.empty and {"comment_id", "parent_id"}.issubset(set(comments.columns)):
        comment_ids = set(comments["comment_id"].astype(str).tolist())
        parent_ids = comments["parent_id"].dropna().astype(str)
        comment_parent = parent_ids[parent_ids.str.startswith("t1_")]
        denom = int(comment_parent.shape[0])
        if denom == 0:
            checks["comment_parent_id_completeness"] = CheckResult(
                status="PASS",
                details={"note": "no_t1_parents_observed", "t1_parent_count": 0},
            )
        else:
            parent_comment_ids = comment_parent.str.removeprefix("t1_")
            missing = int((~parent_comment_ids.isin(comment_ids)).sum())
            frac_missing = float(missing) / float(denom) if denom else None
            status = "PASS" if (frac_missing is not None and frac_missing <= 0.10) else "WARN"
            checks["comment_parent_id_completeness"] = CheckResult(
                status=status,
                details={
                    "t1_parent_count": denom,
                    "missing_parent_comment_count": missing,
                    "missing_fraction": frac_missing,
                },
            )
    else:
        checks["comment_parent_id_completeness"] = CheckResult(
            status="WARN",
            details={"reason": "missing_comments_or_columns"},
        )

    statuses = [c.status for c in checks.values()]
    fail_count = int(sum(1 for s in statuses if s == "FAIL"))
    warn_count = int(sum(1 for s in statuses if s == "WARN"))
    overall_status = "FAIL" if fail_count else ("PASS_WITH_WARN" if warn_count else "PASS")

    out: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "inputs": {
            "curated_root": str(curated_root),
            "request_log": str(args.request_log),
        },
        "overall_status": overall_status,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": {
            k: {"status": v.status, "details": v.details} for k, v in sorted(checks.items())
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"status": overall_status, "fail_count": fail_count, "warn_count": warn_count})
    )


if __name__ == "__main__":
    main()
