#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CheckResult:
    status: str  # PASS/WARN/FAIL
    details: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_dataset(curated_dir: Path, subset: str, columns: list[str] | None = None) -> pd.DataFrame:
    dataset = ds.dataset(
        str(curated_dir / subset),
        format="parquet",
        partitioning="hive",
    )
    table = dataset.to_table(columns=columns)
    return table.to_pandas()


def _parse_dt_utc(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(ts) else ts


def _infer_pull_time(out_path: Path, curated_dir: Path) -> dict[str, Any]:
    attempt_dir = out_path.parent
    run_manifest_path = attempt_dir / "run_manifest.json"
    started_at_utc = None
    if run_manifest_path.exists():
        manifest = _load_json(run_manifest_path)
        started_at_utc = manifest.get("started_at_utc")

    snapshot_id = curated_dir.name
    raw_export_manifest = Path("data_raw") / "hf_archive" / snapshot_id / "EXPORT_MANIFEST.json"
    exported_at_utc = None
    if raw_export_manifest.exists():
        raw_manifest = _load_json(raw_export_manifest)
        exported_at_utc = raw_manifest.get("exported_at")

    started_ts = _parse_dt_utc(started_at_utc)
    exported_ts = _parse_dt_utc(exported_at_utc)

    reference = (
        "run_manifest.started_at_utc"
        if started_ts is not None
        else "raw_export_manifest.exported_at"
    )
    pull_time = started_ts or exported_ts or pd.Timestamp.now(tz="UTC")

    return {
        "reference": reference,
        "pull_time_utc": pull_time.isoformat(),
        "started_at_utc": started_at_utc,
        "exported_at_utc": exported_at_utc,
        "run_manifest_path": str(run_manifest_path) if run_manifest_path.exists() else None,
        "raw_export_manifest_path": (
            str(raw_export_manifest) if raw_export_manifest.exists() else None
        ),
    }


def _status_from_rate(rate: float | None, pass_min: float = 0.99, warn_min: float = 0.90) -> str:
    if rate is None:
        return "PASS"
    if rate >= pass_min:
        return "PASS"
    if rate >= warn_min:
        return "WARN"
    return "FAIL"


def _uniqueness_by_day(df: pd.DataFrame, id_col: str, day_col: str) -> CheckResult:
    if df.empty:
        return CheckResult(status="WARN", details={"reason": "empty_table"})

    dup_counts = (
        df.groupby([day_col, id_col], dropna=False)
        .size()
        .reset_index(name="n")
        .query("n > 1")
        .groupby(day_col)["n"]
        .sum()
        .sort_index()
    )
    total_dup_rows = int(dup_counts.sum()) if not dup_counts.empty else 0
    status = "PASS" if total_dup_rows == 0 else "WARN"
    return CheckResult(
        status=status,
        details={
            "id_col": id_col,
            "day_col": day_col,
            "duplicate_rows_total": total_dup_rows,
            "duplicate_rows_by_day": {str(k): int(v) for k, v in dup_counts.items()},
        },
    )


def _timestamp_parse_rate(
    df: pd.DataFrame, raw_col: str, utc_col: str
) -> dict[str, Any]:
    raw = df.get(raw_col)
    utc = df.get(utc_col)
    if raw is None or utc is None:
        return {"error": "missing_columns", "raw_col": raw_col, "utc_col": utc_col}

    nonnull = raw.notna().sum()
    parsed_ok = utc.notna().sum()
    parsed_fail = int(nonnull - parsed_ok)
    rate = (float(parsed_ok) / float(nonnull)) if nonnull else None
    return {
        "raw_col": raw_col,
        "utc_col": utc_col,
        "nonnull": int(nonnull),
        "parsed_success": int(parsed_ok),
        "parsed_fail": parsed_fail,
        "parse_rate": rate,
    }


def _referential_integrity(
    comments: pd.DataFrame,
    posts: pd.DataFrame,
    agents: pd.DataFrame,
) -> CheckResult:
    details: dict[str, Any] = {}

    posts_ids = set(posts["id"].dropna().astype(str)) if "id" in posts.columns else set()
    comments_ids = set(comments["id"].dropna().astype(str)) if "id" in comments.columns else set()
    agent_ids = set(agents["id"].dropna().astype(str)) if "id" in agents.columns else set()

    def _missing_rate(series: pd.Series, universe: set[str]) -> tuple[int, int, float | None]:
        nonnull = series.dropna().astype(str)
        if nonnull.empty:
            return 0, 0, None
        missing = int((~nonnull.isin(universe)).sum())
        total = int(len(nonnull))
        return missing, total, (missing / total) if total else None

    if "post_id" in comments.columns:
        missing, total, rate = _missing_rate(comments["post_id"], posts_ids)
        details["comments_post_id_in_posts_id"] = {
            "missing": missing,
            "total_nonnull": total,
            "missing_rate": rate,
        }
    else:
        details["comments_post_id_in_posts_id"] = {"error": "missing_comments.post_id"}

    if "parent_id" in comments.columns:
        parent = comments["parent_id"]
        missing, total, rate = _missing_rate(parent[parent.notna()], comments_ids)
        details["comments_parent_id_in_comments_id"] = {
            "missing": missing,
            "total_nonnull": total,
            "missing_rate": rate,
        }
    else:
        details["comments_parent_id_in_comments_id"] = {"error": "missing_comments.parent_id"}

    for table_name, df, col in [
        ("posts", posts, "agent_id"),
        ("comments", comments, "agent_id"),
    ]:
        key = f"{table_name}_{col}_in_agents_id"
        if col in df.columns:
            missing, total, rate = _missing_rate(df[col], agent_ids)
            details[key] = {"missing": missing, "total_nonnull": total, "missing_rate": rate}
        else:
            details[key] = {"error": f"missing_{table_name}.{col}"}

    # status thresholds: tolerate small mismatches due to lag/backfill.
    fail = False
    warn = False
    for entry in details.values():
        rate = entry.get("missing_rate")
        if rate is None:
            continue
        if rate > 0.01:
            fail = True
        elif rate > 0:
            warn = True
    status = "FAIL" if fail else ("WARN" if warn else "PASS")
    return CheckResult(status=status, details=details)


def _monotonicity_agents(agents: pd.DataFrame) -> CheckResult:
    required = {"first_seen_at_utc", "last_seen_at_utc"}
    if not required.issubset(set(agents.columns)):
        return CheckResult(
            status="FAIL",
            details={"error": "missing_columns", "required": sorted(required)},
        )

    first = pd.to_datetime(agents["first_seen_at_utc"], errors="coerce", utc=True)
    last = pd.to_datetime(agents["last_seen_at_utc"], errors="coerce", utc=True)
    mask = first.notna() & last.notna()
    violations = int((last[mask] < first[mask]).sum())
    status = "PASS" if violations == 0 else "WARN"
    return CheckResult(
        status=status,
        details={"violations": violations, "checked_rows": int(mask.sum())},
    )


def _future_timestamps(
    tables: dict[str, pd.DataFrame],
    pull_time_utc: pd.Timestamp,
    utc_columns_by_table: dict[str, Iterable[str]],
) -> CheckResult:
    details: dict[str, Any] = {"pull_time_utc": pull_time_utc.isoformat(), "tables": {}}
    total_future = 0
    for table_name, df in tables.items():
        cols = [c for c in utc_columns_by_table.get(table_name, []) if c in df.columns]
        table_details: dict[str, Any] = {}
        for col in cols:
            values = pd.to_datetime(df[col], errors="coerce", utc=True)
            future = int((values > pull_time_utc).sum())
            total_future += future
            table_details[col] = {"future_count": future}
        details["tables"][table_name] = table_details
    status = "PASS" if total_future == 0 else "WARN"
    details["future_total"] = total_future
    return CheckResult(status=status, details=details)


def _dedup_latest_state(
    df: pd.DataFrame, id_col: str, dump_date_col: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return df, {"rows_in": 0, "rows_out": 0, "rows_removed": 0}

    work = df.copy()
    work[dump_date_col] = work[dump_date_col].astype(str)
    # Lexicographic max works for YYYY-MM-DD.
    work = work.sort_values([id_col, dump_date_col])
    latest = work.groupby(id_col, as_index=False).tail(1)
    removed = int(len(work) - len(latest))
    return latest, {
        "rows_in": int(len(work)),
        "rows_out": int(len(latest)),
        "rows_removed": removed,
    }


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A.5 QC checks on curated HF archive data.")
    parser.add_argument("--curated-dir", type=Path, required=True)
    parser.add_argument("--schema-mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output path for qc_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curated_dir: Path = args.curated_dir
    schema_mapping = _load_json(args.schema_mapping)

    pull_info = _infer_pull_time(args.out, curated_dir)
    pull_time = pd.to_datetime(pull_info["pull_time_utc"], utc=True)

    # Load only the columns required for checks.
    agents = _read_dataset(
        curated_dir,
        "agents",
        columns=[
            "id",
            "dump_date",
            "first_seen_at_raw",
            "first_seen_at_utc",
            "last_seen_at_raw",
            "last_seen_at_utc",
            "created_at_raw",
            "created_at_utc",
        ],
    )
    posts = _read_dataset(
        curated_dir,
        "posts",
        columns=[
            "id",
            "dump_date",
            "agent_id",
            "created_at_raw",
            "created_at_utc",
        ],
    )
    comments = _read_dataset(
        curated_dir,
        "comments",
        columns=[
            "id",
            "dump_date",
            "post_id",
            "parent_id",
            "agent_id",
            "created_at_raw",
            "created_at_utc",
        ],
    )

    checks: dict[str, Any] = {}

    uniq_agents = _uniqueness_by_day(agents, id_col="id", day_col="dump_date")
    uniq_posts = _uniqueness_by_day(posts, id_col="id", day_col="dump_date")
    uniq_comments = _uniqueness_by_day(comments, id_col="id", day_col="dump_date")
    checks["uniqueness_by_day"] = {
        "status": (
            "PASS"
            if all(r.status == "PASS" for r in [uniq_agents, uniq_posts, uniq_comments])
            else "WARN"
        ),
        "tables": {
            "agents": uniq_agents.details,
            "posts": uniq_posts.details,
            "comments": uniq_comments.details,
        },
    }

    parse_stats: dict[str, Any] = {"tables": {}}
    for table_name, df, pairs in [
        (
            "agents",
            agents,
            [
                ("first_seen_at_raw", "first_seen_at_utc"),
                ("last_seen_at_raw", "last_seen_at_utc"),
                ("created_at_raw", "created_at_utc"),
            ],
        ),
        ("posts", posts, [("created_at_raw", "created_at_utc")]),
        ("comments", comments, [("created_at_raw", "created_at_utc")]),
    ]:
        table_stats: dict[str, Any] = {}
        table_status = "PASS"
        for raw_col, utc_col in pairs:
            stat = _timestamp_parse_rate(df, raw_col=raw_col, utc_col=utc_col)
            table_stats[utc_col] = stat
            status = _status_from_rate(stat.get("parse_rate"))
            if status == "FAIL":
                table_status = "FAIL"
            elif status == "WARN" and table_status != "FAIL":
                table_status = "WARN"
        parse_stats["tables"][table_name] = {"status": table_status, "columns": table_stats}

    parse_stats["status"] = (
        "FAIL"
        if any(t["status"] == "FAIL" for t in parse_stats["tables"].values())
        else (
            "WARN"
            if any(t["status"] == "WARN" for t in parse_stats["tables"].values())
            else "PASS"
        )
    )
    checks["timestamp_parse_rate"] = parse_stats

    ref = _referential_integrity(comments=comments, posts=posts, agents=agents)
    checks["referential_integrity"] = {"status": ref.status, "details": ref.details}

    mono = _monotonicity_agents(agents)
    checks["monotonicity"] = {"status": mono.status, "details": mono.details}

    future = _future_timestamps(
        tables={"agents": agents, "posts": posts, "comments": comments},
        pull_time_utc=pull_time,
        utc_columns_by_table={
            "agents": ["first_seen_at_utc", "last_seen_at_utc", "created_at_utc"],
            "posts": ["created_at_utc"],
            "comments": ["created_at_utc"],
        },
    )
    checks["future_timestamps"] = {"status": future.status, "details": future.details}

    canonical_dir = curated_dir / "canonical"
    canonical: dict[str, Any] = {"out_dir": str(canonical_dir), "tables": {}}

    for name, df in [("posts", posts), ("comments", comments), ("agents", agents)]:
        latest, stats = _dedup_latest_state(df, id_col="id", dump_date_col="dump_date")
        out_path = canonical_dir / f"{name}_latest.parquet"
        _write_parquet(latest, out_path)
        canonical["tables"][name] = {"latest_path": str(out_path), **stats}

    canonical_status = "PASS"
    checks["backfill_dedup"] = {"status": canonical_status, "details": canonical}

    # Provide mapping provenance summary.
    mapping_missing: dict[str, Any] = {
        t: schema_mapping.get("tables", {}).get(t, {}).get("summary", {}).get("missing_required")
        for t in DEFAULT_SUBSETS
        if t in schema_mapping.get("tables", {})
    }

    summary_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for check in checks.values():
        status = check.get("status", "WARN")
        summary_counts[status] = summary_counts.get(status, 0) + 1

    qc_results: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "curated_dir": str(curated_dir),
        "pull_time": pull_info,
        "schema_mapping_path": str(args.schema_mapping),
        "schema_mapping_missing_required": mapping_missing,
        "checks": checks,
        "summary": summary_counts,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(qc_results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
