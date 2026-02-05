#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "https://arctic-shift.photon-reddit.com"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; pwj_pipeline/1.0)"
DEFAULT_REFERER = "https://arctic-shift.photon-reddit.com/download-tool"


@dataclass(frozen=True)
class FetchResult:
    http_status: int | None
    body_text: str | None
    latency_ms: int
    bytes: int
    error: str | None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now_filename() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%fZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _parse_utc(dt_str: str) -> datetime:
    s = dt_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    parsed = datetime.fromisoformat(s)
    if parsed.tzinfo is None:
        raise ValueError(f"Expected timezone-aware datetime, got: {dt_str!r}")
    return parsed.astimezone(UTC)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _compute_backoff_seconds(attempt: int, retry_after_header: str | None, base: float) -> float:
    if retry_after_header:
        try:
            retry_after = float(retry_after_header)
            if retry_after >= 0:
                return min(60.0, retry_after)
        except ValueError:
            pass
    return min(60.0, base * (2**attempt))


def _http_get(
    *,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    max_retries: int,
    backoff_base_seconds: float,
) -> FetchResult:
    # Never log headers; they may contain auth in other contexts.
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout_seconds)
            latency_ms = int((time.monotonic() - started) * 1000)
            status = int(resp.status_code)
            body_text = resp.text
            body_bytes = resp.content

            if status in {429} or status >= 500:
                if attempt < max_retries:
                    delay = _compute_backoff_seconds(
                        attempt, resp.headers.get("Retry-After"), backoff_base_seconds
                    )
                    time.sleep(delay)
                    continue
            return FetchResult(
                http_status=status,
                body_text=body_text,
                latency_ms=latency_ms,
                bytes=len(body_bytes),
                error=None,
            )
        except requests.RequestException as e:
            latency_ms = int((time.monotonic() - started) * 1000)
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                delay = _compute_backoff_seconds(attempt, None, backoff_base_seconds)
                time.sleep(delay)
                continue
            return FetchResult(
                http_status=None,
                body_text=None,
                latency_ms=latency_ms,
                bytes=0,
                error=last_error,
            )
    return FetchResult(http_status=None, body_text=None, latency_ms=0, bytes=0, error=last_error)


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        v = payload.get("data")
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _save_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_response_json(
    *,
    out_path: Path,
    fetch: FetchResult,
    url: str,
    params: dict[str, Any],
) -> tuple[Path, int | None, int, int | None]:
    """
    Persist a JSON object to out_path. Prefer saving the server JSON body verbatim.
    If the body is not parseable JSON, persist a small error JSON with hashes only.
    Returns: (path, http_status, bytes, items_count)
    """
    payload: Any
    items_count: int | None = None
    if fetch.body_text is None:
        payload = {
            "_meta": {"saved_at_utc": _utc_now_iso(), "url": url, "params": params},
            "_error": {"type": "no_body", "error": fetch.error},
        }
    else:
        try:
            payload = json.loads(fetch.body_text)
            items_count = len(_extract_items(payload))
        except json.JSONDecodeError as e:
            payload = {
                "_meta": {"saved_at_utc": _utc_now_iso(), "url": url, "params": params},
                "_error": {
                    "type": "json_decode_error",
                    "message": str(e),
                    "body_bytes": fetch.bytes,
                    "body_sha256": _sha256_text(fetch.body_text),
                },
            }
    _save_json(out_path, payload)
    return out_path, fetch.http_status, fetch.bytes, items_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a bounded pilot Reddit corpus via the credential-free Arctic Shift "
            "archive endpoints (posts/search and comments/search). Saves raw JSON responses "
            "under data_raw/reddit/<dt>/<subreddit>/ and a request log JSONL without headers."
        )
    )
    parser.add_argument("--attempt-id", type=str, required=True)
    parser.add_argument("--dt", type=str, required=True, help="Run date (YYYY-MM-DD).")
    parser.add_argument(
        "--subreddits",
        type=str,
        required=True,
        help="Comma-separated subreddit list (e.g., programming,python,MachineLearning).",
    )
    parser.add_argument(
        "--after-utc", type=str, required=True, help="Inclusive window start (UTC)."
    )
    parser.add_argument("--before-utc", type=str, required=True, help="Exclusive window end (UTC).")
    parser.add_argument("--max-posts-per-subreddit", type=int, default=50)
    parser.add_argument("--max-comments-per-subreddit", type=int, default=800)
    parser.add_argument("--out-raw-root", type=Path, required=True)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-base-seconds", type=float, default=1.0)
    parser.add_argument("--page-limit", type=int, default=100, help="Max items requested per call.")
    return parser.parse_args()


def _collect_endpoint(
    *,
    endpoint: str,
    subreddit: str,
    attempt_id: str,
    after_ms: int,
    before_ms: int,
    max_items: int,
    out_dir: Path,
    request_log_path: Path,
    base_url: str,
    headers: dict[str, str],
    timeout_seconds: int,
    max_retries: int,
    backoff_base_seconds: float,
    page_limit: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/{endpoint}/search"

    collected_ids: set[str] = set()
    cursor_after_ms = after_ms
    page = 0
    requests_made = 0
    saved_files: list[str] = []

    while len(collected_ids) < max_items:
        page += 1
        remaining = max_items - len(collected_ids)
        limit = min(page_limit, remaining if remaining > 0 else page_limit)
        params: dict[str, Any] = {
            "subreddit": subreddit,
            "after": cursor_after_ms,
            "before": before_ms,
            "sort": "asc",
            "limit": limit,
        }

        fetch = _http_get(
            url=url,
            params=params,
            headers=headers,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base_seconds,
        )
        requests_made += 1

        out_path = (
            out_dir
            / f"{attempt_id}__{_utc_now_filename()}__after_{cursor_after_ms}__p{page:03d}.json"
        )
        saved_path, status, n_bytes, items_count = _save_response_json(
            out_path=out_path, fetch=fetch, url=url, params=params
        )
        saved_files.append(str(saved_path))

        _write_jsonl(
            request_log_path,
            {
                "ts_utc": _utc_now_iso(),
                "endpoint": endpoint,
                "method": "GET",
                "url": url,
                "params": params,
                "http_status": status,
                "bytes": n_bytes,
                "latency_ms": fetch.latency_ms,
                "saved_path": str(saved_path),
                "items_count": items_count,
                "error": fetch.error,
            },
        )

        # Stop conditions
        if fetch.body_text is None:
            break
        try:
            payload = json.loads(fetch.body_text)
        except json.JSONDecodeError:
            break

        items = _extract_items(payload)
        if not items:
            break

        for it in items:
            _id = it.get("id")
            if _id is not None:
                collected_ids.add(str(_id))

        # Cursor advance (time-based pagination).
        last_created_utc = items[-1].get("created_utc")
        if not isinstance(last_created_utc, int):
            break
        next_after_ms = (last_created_utc * 1000) + 1
        if next_after_ms <= cursor_after_ms:
            break
        cursor_after_ms = next_after_ms
        if cursor_after_ms >= before_ms:
            break

    return {
        "endpoint": endpoint,
        "subreddit": subreddit,
        "after_ms": after_ms,
        "before_ms": before_ms,
        "requests_made": requests_made,
        "unique_ids_collected": len(collected_ids),
        "saved_files_count": len(saved_files),
        "saved_files_sample": saved_files[:5],
    }


def main() -> None:
    args = parse_args()

    after_dt = _parse_utc(args.after_utc)
    before_dt = _parse_utc(args.before_utc)
    after_ms = int(after_dt.timestamp() * 1000)
    before_ms = int(before_dt.timestamp() * 1000)

    out_raw_root: Path = args.out_raw_root
    dt_str: str = args.dt
    attempt_id: str = args.attempt_id
    request_log_path = out_raw_root / dt_str / "request_log" / f"{attempt_id}.jsonl"

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": DEFAULT_REFERER,
    }

    subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    if not subreddits:
        raise SystemExit("No subreddits provided.")

    run_summary: dict[str, Any] = {
        "attempt_id": attempt_id,
        "dt": dt_str,
        "generated_at_utc": _utc_now_iso(),
        "base_url": args.base_url,
        "after_utc": after_dt.isoformat(),
        "before_utc": before_dt.isoformat(),
        "subreddits": subreddits,
        "caps": {
            "max_posts_per_subreddit": int(args.max_posts_per_subreddit),
            "max_comments_per_subreddit": int(args.max_comments_per_subreddit),
        },
        "results": [],
    }

    for subreddit in subreddits:
        sub_root = out_raw_root / dt_str / subreddit
        posts_dir = sub_root / "posts_search"
        comments_dir = sub_root / "comments_search"
        _ensure_dir(posts_dir)
        _ensure_dir(comments_dir)

        posts_summary = _collect_endpoint(
            endpoint="posts",
            subreddit=subreddit,
            attempt_id=attempt_id,
            after_ms=after_ms,
            before_ms=before_ms,
            max_items=int(args.max_posts_per_subreddit),
            out_dir=posts_dir,
            request_log_path=request_log_path,
            base_url=args.base_url,
            headers=headers,
            timeout_seconds=int(args.timeout_seconds),
            max_retries=int(args.max_retries),
            backoff_base_seconds=float(args.backoff_base_seconds),
            page_limit=int(args.page_limit),
        )
        comments_summary = _collect_endpoint(
            endpoint="comments",
            subreddit=subreddit,
            attempt_id=attempt_id,
            after_ms=after_ms,
            before_ms=before_ms,
            max_items=int(args.max_comments_per_subreddit),
            out_dir=comments_dir,
            request_log_path=request_log_path,
            base_url=args.base_url,
            headers=headers,
            timeout_seconds=int(args.timeout_seconds),
            max_retries=int(args.max_retries),
            backoff_base_seconds=float(args.backoff_base_seconds),
            page_limit=int(args.page_limit),
        )

        run_summary["results"].append({"posts": posts_summary, "comments": comments_summary})

        # Minimal progress output (counts only; never print titles/bodies).
        print(
            json.dumps(
                {
                    "subreddit": subreddit,
                    "posts_unique_ids": posts_summary["unique_ids_collected"],
                    "comments_unique_ids": comments_summary["unique_ids_collected"],
                    "requests_made": posts_summary["requests_made"]
                    + comments_summary["requests_made"],
                }
            )
        )

    # Write a small run summary JSON next to the request log (non-sensitive).
    summary_path = out_raw_root / dt_str / "request_log" / f"{attempt_id}__summary.json"
    _ensure_dir(summary_path.parent)
    summary_path.write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
