#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ALLOWED_SORTS = {"hot", "new", "top", "rising"}


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


def _extract_post_id(post: Any) -> str | None:
    if not isinstance(post, dict):
        return None
    for k in ["id", "post_id"]:
        v = post.get(k)
        if v is not None:
            return str(v)
    return None


def _extract_comment_count(post: Any) -> int | None:
    if not isinstance(post, dict):
        return None
    for k in ["comment_count", "comments_count", "num_comments"]:
        if k in post:
            return _coerce_int(post.get(k))
    return None


def _extract_posts_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ["posts", "data", "items", "results"]:
            v = payload.get(k)
            if isinstance(v, list):
                return v
    return []


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
    headers: dict[str, str],
    timeout_seconds: int,
    max_retries: int,
    backoff_base_seconds: float,
) -> FetchResult:
    # Never log headers; they may include auth.
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        req = Request(url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
                body_bytes = resp.read()
                latency_ms = int((time.monotonic() - started) * 1000)
                status = int(getattr(resp, "status", 200))
                body_text = body_bytes.decode("utf-8", errors="replace")

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
        except HTTPError as e:
            latency_ms = int((time.monotonic() - started) * 1000)
            status = int(getattr(e, "code", 0)) or None
            body_bytes = b""
            try:
                body_bytes = e.read()
            except Exception:  # noqa: BLE001
                body_bytes = b""
            body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else None
            last_error = f"HTTPError: {status}"
            if status in {429} or (status is not None and status >= 500):
                if attempt < max_retries:
                    delay = _compute_backoff_seconds(
                        attempt,
                        e.headers.get("Retry-After") if hasattr(e, "headers") else None,
                        backoff_base_seconds,
                    )
                    time.sleep(delay)
                    continue
            return FetchResult(
                http_status=status,
                body_text=body_text,
                latency_ms=latency_ms,
                bytes=len(body_bytes),
                error=last_error,
            )
        except URLError as e:
            latency_ms = int((time.monotonic() - started) * 1000)
            last_error = f"URLError: {getattr(e, 'reason', e)}"
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


def _stub_feed_payload(
    *, sort: str, limit: int, snapshot_index: int, date: str
) -> list[dict[str, Any]]:
    base_time = f"{date}T15:00:00+00:00"
    out: list[dict[str, Any]] = []
    for i in range(limit):
        post_id = f"stub_post_{sort}_{snapshot_index}_{i + 1:03d}"
        out.append(
            {
                "id": post_id,
                "created_at": base_time,
                "title": f"Stub post {i + 1} ({sort})",
                "score": int(limit - i),
                "comment_count": int((i % 5) + 1),
                "submolt": {"name": "stubmolt"},
                "author": {"name": "stub_agent"},
                "_synthetic": True,
            }
        )
    return out


def _stub_post_detail_payload(*, post_id: str, date: str) -> dict[str, Any]:
    return {
        "id": post_id,
        "created_at": f"{date}T15:00:00+00:00",
        "title": f"Stub detail for {post_id}",
        "content": "synthetic",
        "score": 1,
        "comment_count": 3,
        "submolt": {"name": "stubmolt"},
        "author": {"name": "stub_agent"},
        "_synthetic": True,
    }


def _stub_comments_payload(*, post_id: str, date: str) -> list[dict[str, Any]]:
    # Simple 2-level thread; created_at >= post created_at.
    root_id = f"stub_comment_{post_id}_001"
    return [
        {
            "id": root_id,
            "post_id": post_id,
            "parent_id": None,
            "created_at": f"{date}T15:05:00+00:00",
            "content": "synthetic root comment",
            "score": 1,
            "author": {"name": "stub_commenter"},
            "children": [
                {
                    "id": f"stub_comment_{post_id}_002",
                    "post_id": post_id,
                    "parent_id": root_id,
                    "created_at": f"{date}T15:06:00+00:00",
                    "content": "synthetic reply",
                    "score": 1,
                    "author": {"name": "stub_commenter_2"},
                    "_synthetic": True,
                }
            ],
            "_synthetic": True,
        }
    ]


def _stub_submolts_payload() -> list[dict[str, Any]]:
    return [{"name": "stubmolt", "description": "synthetic", "_synthetic": True}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a small, auditable pilot of Moltbook REST API responses into "
            "data_raw/ with a safe JSONL request log (no headers). Supports live/unauth "
            "collection when possible and a deterministic stub fallback."
        )
    )
    parser.add_argument("--attempt-id", required=True, help="Attempt/run identifier.")
    parser.add_argument("--date", required=True, help="Collection date (YYYY-MM-DD) for foldering.")
    parser.add_argument(
        "--mode",
        choices=["auto", "live", "unauth", "stub"],
        default="auto",
        help=(
            "Collection mode: auto chooses live if API key exists, else unauth; "
            "falls back to stub on failure."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="https://www.moltbook.com/api/v1",
        help="API base URL (default: https://www.moltbook.com/api/v1).",
    )
    parser.add_argument(
        "--sorts",
        default="hot,new",
        help="Comma-separated feed sorts to collect (subset of hot,new,top,rising).",
    )
    parser.add_argument("--limit", type=int, default=25, help="Feed limit per request.")
    parser.add_argument(
        "--snapshots", type=int, default=2, help="Number of snapshot rounds per sort."
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Sleep interval between snapshot rounds (seconds).",
    )
    parser.add_argument(
        "--max-post-details",
        type=int,
        default=5,
        help="Max number of post detail fetches (/posts/:id).",
    )
    parser.add_argument(
        "--max-comment-posts",
        type=int,
        default=2,
        help="Max number of posts to fetch comment threads for (/posts/:id/comments).",
    )
    parser.add_argument(
        "--comment-poll-every-rounds",
        type=int,
        default=0,
        help=(
            "If >0, poll /posts/:id/comments periodically every N snapshot rounds using "
            "--comment-poll-top-k top posts from that round."
        ),
    )
    parser.add_argument(
        "--comment-poll-top-k",
        type=int,
        default=0,
        help=(
            "If >0 with --comment-poll-every-rounds, number of top posts per poll "
            "for /posts/:id/comments."
        ),
    )
    parser.add_argument(
        "--include-submolts",
        action="store_true",
        help="If set, also fetch /submolts (GET only).",
    )
    parser.add_argument(
        "--out-raw-root",
        type=Path,
        required=True,
        help="Root output directory (e.g., data_raw/moltbook_api).",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30, help="HTTP timeout.")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries for 429/5xx/URLError.")
    parser.add_argument(
        "--backoff-base-seconds",
        type=float,
        default=1.0,
        help="Base seconds for exponential backoff (2^attempt scaling).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attempt_id: str = args.attempt_id
    date: str = args.date
    base_url: str = str(args.base_url).rstrip("/")
    out_root: Path = args.out_raw_root

    sorts = [s.strip() for s in str(args.sorts).split(",") if s.strip()]
    unknown_sorts = sorted(set(sorts) - ALLOWED_SORTS)
    if unknown_sorts:
        raise SystemExit(f"Unknown sort(s): {unknown_sorts}. Allowed: {sorted(ALLOWED_SORTS)}")

    comment_poll_every_rounds = int(args.comment_poll_every_rounds)
    comment_poll_top_k = int(args.comment_poll_top_k)
    if comment_poll_every_rounds < 0 or comment_poll_top_k < 0:
        raise SystemExit("--comment-poll-every-rounds and --comment-poll-top-k must be >= 0.")
    if (comment_poll_every_rounds == 0) != (comment_poll_top_k == 0):
        raise SystemExit(
            "Both --comment-poll-every-rounds and --comment-poll-top-k must be set together "
            "(both zero to disable, both >0 to enable)."
        )

    request_log_path = out_root / date / "request_log" / f"{attempt_id}.jsonl"

    api_key = os.getenv("MOLTBOOK_API_KEY")
    initial_mode = args.mode
    if initial_mode == "auto":
        initial_mode = "live" if api_key else "unauth"

    if initial_mode == "live" and not api_key and args.mode != "auto":
        raise SystemExit("Mode live requested, but MOLTBOOK_API_KEY is not set.")

    user_agent = "conversation-persistence-research/0.1 (+academic; safe-readonly)"
    base_headers = {"Accept": "application/json", "User-Agent": user_agent}

    def _headers_for_mode(mode: str) -> dict[str, str]:
        if mode == "live":
            # Never log headers; value remains in-memory only.
            return {**base_headers, "Authorization": f"Bearer {api_key}"}
        return dict(base_headers)

    effective_mode = initial_mode
    fallback_to_stub = False

    collected_posts: list[dict[str, Any]] = []

    def _sort_key(p: dict[str, Any]) -> tuple[int, str]:
        cc = p.get("comment_count")
        cc_int = cc if isinstance(cc, int) else (-1 if cc is None else int(cc))
        return (cc_int, str(p.get("post_id")))

    def _log_request(
        *,
        mode: str,
        synthetic: bool,
        endpoint: str,
        url: str,
        params: dict[str, Any],
        retrieved_at_utc: str,
        result: FetchResult,
        raw_path: Path | None,
    ) -> None:
        _write_jsonl(
            request_log_path,
            {
                "attempt_id": attempt_id,
                "mode": mode,
                "synthetic": synthetic,
                "method": "GET",
                "endpoint": endpoint,
                "url": url,
                "params": params,
                "retrieved_at_utc": retrieved_at_utc,
                "http_status": result.http_status,
                "latency_ms": result.latency_ms,
                "bytes": result.bytes,
                "error": result.error,
                "raw_path": str(raw_path) if raw_path is not None else None,
            },
        )

    def _save_raw_text(raw_path: Path, body_text: str) -> None:
        _ensure_dir(raw_path.parent)
        raw_path.write_text(body_text, encoding="utf-8")

    def _request_json(
        *,
        mode: str,
        endpoint: str,
        params: dict[str, Any],
        out_dir_name: str,
        file_suffix: str,
        stub_payload: Any | None,
    ) -> Any | None:
        nonlocal fallback_to_stub

        retrieved_at_utc = _utc_now_iso()
        query = f"?{urlencode(params)}" if params else ""
        url = f"{base_url}{endpoint}{query}"
        raw_dir = out_root / date / out_dir_name / attempt_id
        raw_path = raw_dir / f"{_utc_now_filename()}{file_suffix}.json"

        if mode == "stub":
            payload = stub_payload
            body_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            _save_raw_text(raw_path, body_text)
            _log_request(
                mode=mode,
                synthetic=True,
                endpoint=endpoint,
                url=f"{base_url}{endpoint}",
                params=params,
                retrieved_at_utc=retrieved_at_utc,
                result=FetchResult(
                    http_status=200, body_text=None, latency_ms=0, bytes=len(body_text), error=None
                ),
                raw_path=raw_path,
            )
            return payload

        if fallback_to_stub:
            return _request_json(
                mode="stub",
                endpoint=endpoint,
                params=params,
                out_dir_name=out_dir_name,
                file_suffix=file_suffix,
                stub_payload=stub_payload,
            )

        result = _http_get(
            url=url,
            headers=_headers_for_mode(mode),
            timeout_seconds=int(args.timeout_seconds),
            max_retries=int(args.max_retries),
            backoff_base_seconds=float(args.backoff_base_seconds),
        )

        # Decide fallback for auto mode on connectivity/auth issues.
        if args.mode == "auto":
            if result.http_status is None or (result.http_status in {401, 403}):
                fallback_to_stub = True

        payload: Any | None = None
        if result.body_text is not None:
            _save_raw_text(raw_path, result.body_text)
            try:
                payload = json.loads(result.body_text)
            except Exception:  # noqa: BLE001
                payload = None
            saved_path: Path | None = raw_path
        else:
            saved_path = None

        _log_request(
            mode=mode,
            synthetic=False,
            endpoint=endpoint,
            url=f"{base_url}{endpoint}",
            params=params,
            retrieved_at_utc=retrieved_at_utc,
            result=result,
            raw_path=saved_path,
        )
        return payload

    # 1) Feed snapshots
    for snapshot_index in range(int(args.snapshots)):
        round_posts: list[dict[str, Any]] = []
        for sort in sorts:
            payload = _request_json(
                mode=effective_mode,
                endpoint="/posts",
                params={"sort": sort, "limit": int(args.limit)},
                out_dir_name="posts_feed",
                file_suffix=f"__sort={sort}__limit={int(args.limit)}",
                stub_payload=_stub_feed_payload(
                    sort=sort, limit=int(args.limit), snapshot_index=snapshot_index, date=date
                ),
            )
            for post in _extract_posts_list(payload):
                pid = _extract_post_id(post)
                if pid is None:
                    continue
                post_item = {
                    "post_id": pid,
                    "comment_count": _extract_comment_count(post),
                }
                collected_posts.append(post_item)
                round_posts.append(post_item)

        if comment_poll_every_rounds > 0 and comment_poll_top_k > 0:
            round_number = snapshot_index + 1
            if round_number % comment_poll_every_rounds == 0:
                seen_round: set[str] = set()
                unique_round_posts: list[dict[str, Any]] = []
                for p in round_posts:
                    pid = str(p["post_id"])
                    if pid in seen_round:
                        continue
                    seen_round.add(pid)
                    unique_round_posts.append(p)
                round_top_post_ids = [
                    p["post_id"]
                    for p in sorted(unique_round_posts, key=_sort_key, reverse=True)[
                        :comment_poll_top_k
                    ]
                ]
                for pid in round_top_post_ids:
                    _request_json(
                        mode=effective_mode,
                        endpoint=f"/posts/{pid}/comments",
                        params={"sort": "new"},
                        out_dir_name="posts_comments",
                        file_suffix=f"__post_id={pid}__sort=new__round={round_number:06d}",
                        stub_payload=_stub_comments_payload(post_id=pid, date=date),
                    )

        if snapshot_index < int(args.snapshots) - 1:
            time.sleep(int(args.interval_seconds))

    # Determine post IDs for follow-up calls:
    # - deterministic stable order
    # - prefer higher comment_count when present
    seen: set[str] = set()
    unique_posts: list[dict[str, Any]] = []
    for p in collected_posts:
        pid = p["post_id"]
        if pid in seen:
            continue
        seen.add(pid)
        unique_posts.append(p)

    unique_posts_sorted = sorted(unique_posts, key=_sort_key, reverse=True)
    post_ids_for_details = [p["post_id"] for p in unique_posts_sorted[: int(args.max_post_details)]]
    post_ids_for_comments = [
        p["post_id"] for p in unique_posts_sorted[: int(args.max_comment_posts)]
    ]

    # 2) Post details
    for pid in post_ids_for_details:
        _request_json(
            mode=effective_mode,
            endpoint=f"/posts/{pid}",
            params={},
            out_dir_name="posts_detail",
            file_suffix=f"__post_id={pid}",
            stub_payload=_stub_post_detail_payload(post_id=pid, date=date),
        )

    # 3) Comment threads
    for pid in post_ids_for_comments:
        _request_json(
            mode=effective_mode,
            endpoint=f"/posts/{pid}/comments",
            params={"sort": "new"},
            out_dir_name="posts_comments",
            file_suffix=f"__post_id={pid}__sort=new",
            stub_payload=_stub_comments_payload(post_id=pid, date=date),
        )

    # 4) Submolts (optional)
    if bool(args.include_submolts):
        _request_json(
            mode=effective_mode,
            endpoint="/submolts",
            params={},
            out_dir_name="submolts",
            file_suffix="",
            stub_payload=_stub_submolts_payload(),
        )

    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "date": date,
                "base_url": base_url,
                "mode_requested": args.mode,
                "mode_initial": initial_mode,
                "mode_effective": ("stub" if fallback_to_stub else effective_mode),
                "comment_poll_every_rounds": comment_poll_every_rounds,
                "comment_poll_top_k": comment_poll_top_k,
                "request_log_path": str(request_log_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
