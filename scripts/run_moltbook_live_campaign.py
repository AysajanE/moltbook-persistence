#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

COLLECT_SCRIPT = Path("analysis/03_moltbook_api_collect.py")
CURATE_SCRIPT = Path("analysis/03_moltbook_api_curate.py")
VALIDATE_SCRIPT = Path("analysis/03_moltbook_api_validate.py")

ITEM3_RELATIVE = Path("item3")
COLLECTION_PLAN_FILE = "docs/feedback/data_collection_plan.md"
COLLECTION_PLAN_SECTION = "B.5.4 High-frequency clock detection run (72 hours)"
DEFAULT_CHRONOLOGY_SKEW_SECONDS = 300


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    exit_code: int
    stdout_path: str
    stderr_path: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%d-%H%M%SZ")


def _parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return datetime.fromisoformat(value)


def _parse_utc_date(value: str) -> date:
    return date.fromisoformat(value)


def _json_dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _path_as_repo_relative(path: Path, repo_root: Path) -> str:
    candidate = path if path.is_absolute() else (repo_root / path)
    try:
        return str(candidate.resolve().relative_to(repo_root.resolve()))
    except Exception:  # noqa: BLE001
        return str(path)


def _path_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0}
    if path.is_file():
        return {"exists": True, "files": 1, "bytes": int(path.stat().st_size)}
    files = 0
    total_bytes = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            total_bytes += int(p.stat().st_size)
    return {"exists": True, "files": files, "bytes": total_bytes}


def _attempt_storage_stats(raw_day_root: Path, attempt_id: str) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    for name in ["posts_feed", "posts_detail", "posts_comments", "submolts"]:
        stats = _path_stats(raw_day_root / name / attempt_id)
        if stats["exists"]:
            components[name] = stats

    request_stats: dict[str, Any] = {"files": 0, "bytes": 0, "paths": []}
    for suffix in [".jsonl", ".jsonl.gz"]:
        p = raw_day_root / "request_log" / f"{attempt_id}{suffix}"
        if p.exists():
            request_stats["files"] += 1
            request_stats["bytes"] += int(p.stat().st_size)
            request_stats["paths"].append(str(p))
    if request_stats["files"] > 0:
        request_stats["exists"] = True
        components["request_log"] = request_stats

    total_bytes = sum(int(v["bytes"]) for v in components.values())
    total_files = sum(int(v["files"]) for v in components.values())
    return {
        "components": components,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
        "total_files": total_files,
    }


def _run_command(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    started = _utc_now()
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)
    finished = _utc_now()

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    return CommandResult(
        name=name,
        command=command,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_seconds=(finished - started).total_seconds(),
        exit_code=int(proc.returncode),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _command_result_dict(result: CommandResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "command": result.command,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "duration_seconds": result.duration_seconds,
        "exit_code": result.exit_code,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
    }


def _write_completion_manifest(
    *,
    repo_root: Path,
    attempt_dir: Path,
    request_log_path: Path,
    attempt_id: str,
    run_date: str,
    sorts: list[str],
    snapshots: int,
    interval_seconds: int,
) -> Path:
    entries = _load_jsonl(request_log_path)
    if not entries:
        raise FileNotFoundError(
            f"Missing or empty request log for completion manifest: {request_log_path}"
        )

    timestamps = [
        _parse_iso_datetime(str(e["retrieved_at_utc"]))
        for e in entries
        if e.get("retrieved_at_utc")
    ]
    first = min(timestamps)
    last = max(timestamps)
    elapsed_seconds = (last - first).total_seconds()

    endpoint_counts = Counter(str(e.get("endpoint")) for e in entries)
    status_counts = Counter(str(e.get("http_status")) for e in entries)

    observed_sort_counts: dict[str, int] = {}
    for sort in sorts:
        observed_sort_counts[sort] = sum(
            1
            for e in entries
            if e.get("endpoint") == "/posts"
            and isinstance(e.get("params"), dict)
            and e["params"].get("sort") == sort
        )

    expected_posts_calls = int(snapshots) * len(sorts)
    min_expected_duration_seconds = max(int(snapshots) - 1, 0) * int(interval_seconds)

    raw_day_root = request_log_path.parent.parent
    storage = _attempt_storage_stats(raw_day_root, attempt_id)

    completion = {
        "manifest_type": "moltbook_live_collection_completion",
        "manifest_generated_at_utc": _utc_now_iso(),
        "attempt_id": attempt_id,
        "run_date_folder": run_date,
        "data_collection_plan_reference": {
            "file": COLLECTION_PLAN_FILE,
            "section": COLLECTION_PLAN_SECTION,
        },
        "source_artifacts": {
            "request_log_path": _path_as_repo_relative(request_log_path, repo_root),
            "raw_day_root": _path_as_repo_relative(raw_day_root, repo_root),
        },
        "collection_window_utc": {
            "started_at_utc": first.isoformat(),
            "ended_at_utc": last.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hours": elapsed_seconds / 3600.0,
            "elapsed_days": elapsed_seconds / 86400.0,
        },
        "completion_check": {
            "expected_posts_calls": expected_posts_calls,
            "observed_posts_calls": int(endpoint_counts.get("/posts", 0)),
            "expected_sort_counts": {sort: int(snapshots) for sort in sorts},
            "observed_sort_counts": observed_sort_counts,
            "snapshot_target_met": (
                int(endpoint_counts.get("/posts", 0)) == expected_posts_calls
                and all(observed_sort_counts.get(sort, 0) == int(snapshots) for sort in sorts)
            ),
            "min_expected_duration_seconds": min_expected_duration_seconds,
            "duration_target_met": elapsed_seconds >= float(min_expected_duration_seconds),
        },
        "request_summary": {
            "total_requests_logged": len(entries),
            "endpoint_counts": dict(endpoint_counts),
            "http_status_counts": dict(status_counts),
            "non_200_count": sum(v for k, v in status_counts.items() if k != "200"),
            "non_200_rate": (
                sum(v for k, v in status_counts.items() if k != "200") / len(entries)
                if entries
                else None
            ),
            "mode_values": sorted(
                {str(e.get("mode")) for e in entries if e.get("mode") is not None}
            ),
            "synthetic_values": sorted(
                {bool(e.get("synthetic")) for e in entries if e.get("synthetic") is not None}
            ),
        },
        "storage_observed": storage,
    }

    out_path = attempt_dir / f"completion_manifest_{_utc_stamp()}.json"
    _json_dump(out_path, completion)
    return out_path


def _gzip_one(path: Path, compression_level: int) -> tuple[int, int]:
    out_path = Path(f"{path}.gz")
    if out_path.exists():
        return int(path.stat().st_size), int(out_path.stat().st_size)

    tmp_path = Path(f"{out_path}.tmp")
    original_size = int(path.stat().st_size)
    with path.open("rb") as src, tmp_path.open("wb") as tmp_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=int(compression_level),
            fileobj=tmp_file,
            mtime=0,
        ) as dst:
            shutil.copyfileobj(src, dst)
    tmp_path.replace(out_path)
    compressed_size = int(out_path.stat().st_size)
    path.unlink()
    return original_size, compressed_size


def _compress_attempt_raw(
    *,
    raw_day_root: Path,
    attempt_id: str,
    compression_level: int,
    keep_uncompressed_request_log: bool,
) -> dict[str, Any]:
    targets: list[Path] = []
    for name in ["posts_feed", "posts_detail", "posts_comments", "submolts"]:
        base = raw_day_root / name / attempt_id
        if base.exists():
            targets.extend(sorted(base.rglob("*.json")))

    request_log_path = raw_day_root / "request_log" / f"{attempt_id}.jsonl"
    if (not keep_uncompressed_request_log) and request_log_path.exists():
        targets.append(request_log_path)

    converted = 0
    bytes_before = 0
    bytes_after = 0
    failures: list[dict[str, str]] = []
    for path in targets:
        try:
            b_before, b_after = _gzip_one(path, compression_level=compression_level)
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        converted += 1
        bytes_before += b_before
        bytes_after += b_after

    return {
        "compression_level": int(compression_level),
        "converted_files": converted,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "saved_bytes": bytes_before - bytes_after,
        "saved_ratio": ((bytes_before - bytes_after) / bytes_before if bytes_before > 0 else None),
        "failed_files": failures,
    }


def _git_info(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "dirty": None}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode == 0:
        info["commit"] = commit.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode == 0:
        lines = [line for line in status.stdout.splitlines() if line.strip()]
        info["dirty"] = len(lines) > 0
        info["status_porcelain_sample"] = lines[:50]
    return info


def _campaign_projection(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in runs if r.get("status") in {"PASS", "WARN"}]
    if not completed:
        return {
            "completed_runs": 0,
            "avg_uncompressed_bytes_per_day": None,
            "avg_compressed_bytes_per_day": None,
            "projected_30d_uncompressed_gb": None,
            "projected_30d_compressed_gb": None,
        }

    uncompressed = [
        float(r.get("storage", {}).get("before_compression_total_bytes", 0)) for r in completed
    ]
    compressed = [
        float(r.get("storage", {}).get("after_compression_total_bytes", 0)) for r in completed
    ]

    avg_uncompressed = sum(uncompressed) / len(uncompressed)
    avg_compressed = sum(compressed) / len(compressed)
    return {
        "completed_runs": len(completed),
        "avg_uncompressed_bytes_per_day": avg_uncompressed,
        "avg_compressed_bytes_per_day": avg_compressed,
        "projected_30d_uncompressed_gb": (avg_uncompressed * 30.0) / 1e9,
        "projected_30d_compressed_gb": (avg_compressed * 30.0) / 1e9,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Moltbook live collection campaign in daily chunks with automatic curation, "
            "validation, completion manifests, and optional lossless raw compression."
        )
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Campaign identifier. Default: moltbook_live_<UTC timestamp>.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of daily runs to execute (default: 30).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="UTC date for day 1 (YYYY-MM-DD). Default: today's UTC date.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "live", "unauth", "stub"],
        default="live",
        help="Collector mode passed through to analysis/03_moltbook_api_collect.py.",
    )
    parser.add_argument(
        "--base-url",
        default="https://www.moltbook.com/api/v1",
        help="API base URL.",
    )
    parser.add_argument("--sorts", default="hot,new", help="Comma-separated sorts for /posts.")
    parser.add_argument("--limit", type=int, default=100, help="Feed limit per request.")
    parser.add_argument(
        "--snapshots-per-day",
        type=int,
        default=1440,
        help="Snapshot rounds per day (default: 1440, i.e., every 60 seconds for ~24h).",
    )
    parser.add_argument("--interval-seconds", type=int, default=60, help="Sleep interval.")
    parser.add_argument("--max-post-details", type=int, default=0, help="Max /posts/:id calls.")
    parser.add_argument(
        "--max-comment-posts",
        type=int,
        default=0,
        help="Max /posts/:id/comments calls.",
    )
    parser.add_argument(
        "--comment-poll-every-rounds",
        type=int,
        default=0,
        help=(
            "If >0, poll /posts/:id/comments every N snapshot rounds using "
            "--comment-poll-top-k top posts from that round."
        ),
    )
    parser.add_argument(
        "--comment-poll-top-k",
        type=int,
        default=0,
        help=(
            "If >0 with --comment-poll-every-rounds, number of top posts per periodic comment poll."
        ),
    )
    parser.add_argument(
        "--include-submolts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch /submolts once per run.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data_raw/moltbook_api"),
        help="Raw output root.",
    )
    parser.add_argument(
        "--curated-root",
        type=Path,
        default=Path("data_curated/moltbook"),
        help="Curated output root.",
    )
    parser.add_argument(
        "--ops-root",
        type=Path,
        default=Path("outputs/ops"),
        help="Operations output root.",
    )
    parser.add_argument(
        "--compress-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress raw JSON payloads to .json.gz after curation/validation.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="gzip compression level (1-9).",
    )
    parser.add_argument(
        "--keep-uncompressed-request-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep request log as .jsonl (recommended for quick inspection).",
    )
    parser.add_argument(
        "--curate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run curation after collection.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation after curation.",
    )
    parser.add_argument(
        "--chronology-skew-seconds",
        type=int,
        default=DEFAULT_CHRONOLOGY_SKEW_SECONDS,
        help="Validation tolerance for chronology check.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Abort before a day-run if free space is below this threshold (decimal GB).",
    )
    parser.add_argument(
        "--sleep-between-days-seconds",
        type=int,
        default=5,
        help="Cooldown sleep between day-runs.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop campaign immediately when a day-run fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.days < 1:
        raise SystemExit("--days must be >= 1.")
    if args.snapshots_per_day < 1:
        raise SystemExit("--snapshots-per-day must be >= 1.")
    if args.interval_seconds < 0:
        raise SystemExit("--interval-seconds must be >= 0.")
    if args.compression_level < 1 or args.compression_level > 9:
        raise SystemExit("--compression-level must be in [1, 9].")
    if int(args.comment_poll_every_rounds) < 0 or int(args.comment_poll_top_k) < 0:
        raise SystemExit("--comment-poll-every-rounds and --comment-poll-top-k must be >= 0.")
    if (int(args.comment_poll_every_rounds) == 0) != (int(args.comment_poll_top_k) == 0):
        raise SystemExit(
            "--comment-poll-every-rounds and --comment-poll-top-k must both be zero "
            "(disabled) or both >0 (enabled)."
        )

    if args.mode == "live" and not os.getenv("MOLTBOOK_API_KEY"):
        raise SystemExit("MOLTBOOK_API_KEY is not set; cannot run in --mode live.")

    start_date = (
        _parse_utc_date(args.start_date) if args.start_date is not None else _utc_now().date()
    )
    campaign_id = args.campaign_id or f"moltbook_live_{_utc_stamp()}"
    sorts = [s.strip() for s in str(args.sorts).split(",") if s.strip()]
    if not sorts:
        raise SystemExit("No valid sorts provided.")

    campaign_root = (args.ops_root / campaign_id / ITEM3_RELATIVE).resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    campaign_manifest_path = campaign_root / "campaign_manifest.json"

    campaign_manifest: dict[str, Any] = {
        "manifest_type": "moltbook_live_campaign",
        "campaign_id": campaign_id,
        "generated_at_utc": _utc_now_iso(),
        "repo_root": str(repo_root),
        "git": _git_info(repo_root),
        "config": {
            "days": int(args.days),
            "start_date_utc": start_date.isoformat(),
            "mode": args.mode,
            "base_url": args.base_url,
            "sorts": sorts,
            "limit": int(args.limit),
            "snapshots_per_day": int(args.snapshots_per_day),
            "interval_seconds": int(args.interval_seconds),
            "max_post_details": int(args.max_post_details),
            "max_comment_posts": int(args.max_comment_posts),
            "comment_poll_every_rounds": int(args.comment_poll_every_rounds),
            "comment_poll_top_k": int(args.comment_poll_top_k),
            "include_submolts": bool(args.include_submolts),
            "raw_root": str(args.raw_root),
            "curated_root": str(args.curated_root),
            "ops_root": str(args.ops_root),
            "compress_raw": bool(args.compress_raw),
            "compression_level": int(args.compression_level),
            "keep_uncompressed_request_log": bool(args.keep_uncompressed_request_log),
            "curate": bool(args.curate),
            "validate": bool(args.validate),
            "chronology_skew_seconds": int(args.chronology_skew_seconds),
            "min_free_gb": float(args.min_free_gb),
            "sleep_between_days_seconds": int(args.sleep_between_days_seconds),
            "stop_on_failure": bool(args.stop_on_failure),
        },
        "runs": [],
    }

    for day_idx in range(int(args.days)):
        run_date = (start_date + timedelta(days=day_idx)).isoformat()
        run_date_compact = run_date.replace("-", "")
        attempt_id = f"attempt_live_{campaign_id}_d{day_idx + 1:02d}_{run_date_compact}"
        attempt_dir = campaign_root / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)

        disk_before = shutil.disk_usage(repo_root)
        free_before_gb = disk_before.free / 1e9
        if free_before_gb < float(args.min_free_gb):
            attempt_manifest = {
                "manifest_type": "moltbook_live_daily_run",
                "campaign_id": campaign_id,
                "attempt_id": attempt_id,
                "run_date": run_date,
                "status": "FAIL",
                "failure_reason": (
                    f"free space {free_before_gb:.2f} GB below threshold "
                    f"{float(args.min_free_gb):.2f} GB"
                ),
                "disk_before": {
                    "total_bytes": disk_before.total,
                    "used_bytes": disk_before.used,
                    "free_bytes": disk_before.free,
                    "free_gb_decimal": free_before_gb,
                },
                "generated_at_utc": _utc_now_iso(),
            }
            attempt_manifest_path = attempt_dir / "run_manifest.json"
            _json_dump(attempt_manifest_path, attempt_manifest)
            campaign_manifest["runs"].append(
                {
                    "day_index": day_idx + 1,
                    "run_date": run_date,
                    "attempt_id": attempt_id,
                    "status": "FAIL",
                    "run_manifest_path": str(attempt_manifest_path),
                }
            )
            campaign_manifest["projection"] = _campaign_projection(campaign_manifest["runs"])
            campaign_manifest["generated_at_utc"] = _utc_now_iso()
            _json_dump(campaign_manifest_path, campaign_manifest)
            if args.stop_on_failure:
                break
            continue

        collector_cmd = [
            sys.executable,
            str(COLLECT_SCRIPT),
            "--attempt-id",
            attempt_id,
            "--date",
            run_date,
            "--mode",
            str(args.mode),
            "--base-url",
            str(args.base_url),
            "--sorts",
            ",".join(sorts),
            "--limit",
            str(int(args.limit)),
            "--snapshots",
            str(int(args.snapshots_per_day)),
            "--interval-seconds",
            str(int(args.interval_seconds)),
            "--max-post-details",
            str(int(args.max_post_details)),
            "--max-comment-posts",
            str(int(args.max_comment_posts)),
            "--comment-poll-every-rounds",
            str(int(args.comment_poll_every_rounds)),
            "--comment-poll-top-k",
            str(int(args.comment_poll_top_k)),
            "--out-raw-root",
            str(args.raw_root),
        ]
        if bool(args.include_submolts):
            collector_cmd.append("--include-submolts")

        collect_result = _run_command(
            name="collect",
            command=collector_cmd,
            cwd=repo_root,
            stdout_path=attempt_dir / "collect.stdout.log",
            stderr_path=attempt_dir / "collect.stderr.log",
        )

        request_log_path = args.raw_root / run_date / "request_log" / f"{attempt_id}.jsonl"
        completion_manifest_path: Path | None = None
        completion_error: str | None = None
        if collect_result.exit_code == 0:
            try:
                completion_manifest_path = _write_completion_manifest(
                    repo_root=repo_root,
                    attempt_dir=attempt_dir,
                    request_log_path=request_log_path,
                    attempt_id=attempt_id,
                    run_date=run_date,
                    sorts=sorts,
                    snapshots=int(args.snapshots_per_day),
                    interval_seconds=int(args.interval_seconds),
                )
            except Exception as exc:  # noqa: BLE001
                completion_error = f"{type(exc).__name__}: {exc}"

        curation_manifest_path = attempt_dir / "curation_manifest.json"
        curate_result: CommandResult | None = None
        if bool(args.curate) and collect_result.exit_code == 0:
            curate_cmd = [
                sys.executable,
                str(CURATE_SCRIPT),
                "--raw-root",
                str(args.raw_root / run_date),
                "--attempt-id",
                attempt_id,
                "--out-root",
                str(args.curated_root),
                "--curation-manifest",
                str(curation_manifest_path),
            ]
            curate_result = _run_command(
                name="curate",
                command=curate_cmd,
                cwd=repo_root,
                stdout_path=attempt_dir / "curate.stdout.log",
                stderr_path=attempt_dir / "curate.stderr.log",
            )

        validation_path = attempt_dir / "validation_results.json"
        validate_result: CommandResult | None = None
        validation_summary_status: str | None = None
        validation_warn_checks: list[str] = []
        validation_fail_checks: list[str] = []
        if (
            bool(args.validate)
            and collect_result.exit_code == 0
            and (curate_result is None or curate_result.exit_code == 0)
        ):
            validate_cmd = [
                sys.executable,
                str(VALIDATE_SCRIPT),
                "--curated-root",
                str(args.curated_root),
                "--run-id",
                attempt_id,
                "--request-log",
                str(request_log_path),
                "--out",
                str(validation_path),
                "--chronology-skew-seconds",
                str(int(args.chronology_skew_seconds)),
            ]
            validate_result = _run_command(
                name="validate",
                command=validate_cmd,
                cwd=repo_root,
                stdout_path=attempt_dir / "validate.stdout.log",
                stderr_path=attempt_dir / "validate.stderr.log",
            )
            if validation_path.exists():
                try:
                    validation_obj = json.loads(validation_path.read_text(encoding="utf-8"))
                    validation_summary_status = (
                        validation_obj.get("summary", {}).get("status") if validation_obj else None
                    )
                    summary = validation_obj.get("summary", {}) if validation_obj else {}
                    validation_warn_checks = [
                        str(x) for x in summary.get("warn_checks", []) if x is not None
                    ]
                    validation_fail_checks = [
                        str(x) for x in summary.get("fail_checks", []) if x is not None
                    ]
                except Exception:  # noqa: BLE001
                    validation_summary_status = None

        storage_before_compression = _attempt_storage_stats(args.raw_root / run_date, attempt_id)
        compression_stats: dict[str, Any] | None = None
        if bool(args.compress_raw) and collect_result.exit_code == 0:
            compression_stats = _compress_attempt_raw(
                raw_day_root=args.raw_root / run_date,
                attempt_id=attempt_id,
                compression_level=int(args.compression_level),
                keep_uncompressed_request_log=bool(args.keep_uncompressed_request_log),
            )
        storage_after_compression = _attempt_storage_stats(args.raw_root / run_date, attempt_id)

        disk_after = shutil.disk_usage(repo_root)
        status = "PASS"
        failure_reasons: list[str] = []
        warnings: list[str] = []

        if collect_result.exit_code != 0:
            status = "FAIL"
            failure_reasons.append("collector_exit_nonzero")
        if completion_error is not None:
            status = "FAIL"
            failure_reasons.append(f"completion_manifest_error: {completion_error}")
        if curate_result is not None and curate_result.exit_code != 0:
            status = "FAIL"
            failure_reasons.append("curation_exit_nonzero")
        if validate_result is not None and validate_result.exit_code != 0:
            status = "FAIL"
            failure_reasons.append("validation_exit_nonzero")
        if validation_summary_status == "FAIL":
            status = "FAIL"
            failure_reasons.append("validation_summary_fail")
        if validation_summary_status == "WARN" and status != "FAIL":
            ignorable_warns: set[str] = set()
            if int(args.max_post_details) == 0:
                ignorable_warns.update(
                    {
                        "posts_created_at_parse_rate",
                        "chronology_comment_after_post",
                    }
                )
            comments_disabled = (
                int(args.max_comment_posts) == 0 and int(args.comment_poll_top_k) == 0
            )
            if comments_disabled:
                ignorable_warns.update(
                    {
                        "comments_nonempty",
                        "comments_created_at_parse_rate",
                        "comments_comment_id_unique",
                    }
                )
            non_ignorable_warns = [w for w in validation_warn_checks if w not in ignorable_warns]
            if non_ignorable_warns:
                status = "WARN"
                warnings.append("validation_summary_warn")
                warnings.append(
                    f"validation_non_ignorable_warn_checks={','.join(sorted(non_ignorable_warns))}"
                )
            else:
                warnings.append("validation_warns_expected_from_disabled_detail_comment_crawls")
        if compression_stats is not None and compression_stats.get("failed_files"):
            if status == "PASS":
                status = "WARN"
            warnings.append("compression_had_failed_files")

        attempt_manifest = {
            "manifest_type": "moltbook_live_daily_run",
            "generated_at_utc": _utc_now_iso(),
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "run_date": run_date,
            "status": status,
            "failure_reasons": failure_reasons,
            "warnings": warnings,
            "collection_plan_reference": {
                "file": COLLECTION_PLAN_FILE,
                "section": COLLECTION_PLAN_SECTION,
            },
            "request_log_path": str(request_log_path),
            "completion_manifest_path": (
                str(completion_manifest_path) if completion_manifest_path is not None else None
            ),
            "commands": {
                "collect": _command_result_dict(collect_result),
                "curate": _command_result_dict(curate_result)
                if curate_result is not None
                else None,
                "validate": (
                    _command_result_dict(validate_result) if validate_result is not None else None
                ),
            },
            "curation_manifest_path": (
                str(curation_manifest_path) if curation_manifest_path.exists() else None
            ),
            "validation_results_path": str(validation_path) if validation_path.exists() else None,
            "validation_summary_status": validation_summary_status,
            "validation_warn_checks": validation_warn_checks,
            "validation_fail_checks": validation_fail_checks,
            "storage": {
                "before_compression": storage_before_compression,
                "after_compression": storage_after_compression,
                "before_compression_total_bytes": storage_before_compression["total_bytes"],
                "after_compression_total_bytes": storage_after_compression["total_bytes"],
                "compression": compression_stats,
            },
            "disk_before": {
                "total_bytes": disk_before.total,
                "used_bytes": disk_before.used,
                "free_bytes": disk_before.free,
                "free_gb_decimal": disk_before.free / 1e9,
            },
            "disk_after": {
                "total_bytes": disk_after.total,
                "used_bytes": disk_after.used,
                "free_bytes": disk_after.free,
                "free_gb_decimal": disk_after.free / 1e9,
            },
        }

        attempt_manifest_path = attempt_dir / "run_manifest.json"
        _json_dump(attempt_manifest_path, attempt_manifest)

        campaign_manifest["runs"].append(
            {
                "day_index": day_idx + 1,
                "run_date": run_date,
                "attempt_id": attempt_id,
                "status": status,
                "run_manifest_path": str(attempt_manifest_path),
                "completion_manifest_path": (
                    str(completion_manifest_path) if completion_manifest_path is not None else None
                ),
                "storage": {
                    "before_compression_total_bytes": storage_before_compression["total_bytes"],
                    "after_compression_total_bytes": storage_after_compression["total_bytes"],
                },
            }
        )
        campaign_manifest["projection"] = _campaign_projection(campaign_manifest["runs"])
        campaign_manifest["generated_at_utc"] = _utc_now_iso()
        _json_dump(campaign_manifest_path, campaign_manifest)

        print(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "day_index": day_idx + 1,
                    "attempt_id": attempt_id,
                    "status": status,
                    "run_manifest_path": str(attempt_manifest_path),
                    "campaign_manifest_path": str(campaign_manifest_path),
                },
                indent=2,
                sort_keys=True,
            )
        )

        if status == "FAIL" and bool(args.stop_on_failure):
            break

        if day_idx < int(args.days) - 1 and int(args.sleep_between_days_seconds) > 0:
            import time

            time.sleep(int(args.sleep_between_days_seconds))


if __name__ == "__main__":
    main()
