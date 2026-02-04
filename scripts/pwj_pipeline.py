#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = REPO_ROOT / "docs" / "data_collection_plan.md"

SCHEMA_DIR = REPO_ROOT / "pipelines" / "pwj" / "schemas"
PLANNER_SCHEMA = SCHEMA_DIR / "planner.schema.json"
WORKER_SCHEMA = SCHEMA_DIR / "worker.schema.json"
JUDGE_SCHEMA = SCHEMA_DIR / "judge.schema.json"

DEFAULT_RUNS_DIR = REPO_ROOT / "outputs" / "pwj_pipeline"
DEFAULT_STATE_PATH = DEFAULT_RUNS_DIR / "state.json"

SECTION_RE = re.compile(r"^##\s+(?P<id>\d+)\)\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True)
class PlanItem:
    item_id: str
    title: str
    body: str


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_codex_on_path() -> None:
    if shutil.which("codex") is None:
        raise SystemExit(
            "Could not find `codex` on PATH. Install Codex CLI and run `codex login` first."
        )


def parse_items_from_plan(plan_path: Path) -> list[PlanItem]:
    text = load_text(plan_path)
    lines = text.splitlines(keepends=True)

    headers: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        match = SECTION_RE.match(line)
        if match:
            headers.append((idx, match.group("id"), match.group("title")))

    if not headers:
        raise SystemExit(
            f"No sections found in {plan_path}. Expected headings like `## 2) ...` at line start."
        )

    items: list[PlanItem] = []
    for header_idx, (start, item_id, title) in enumerate(headers):
        end = headers[header_idx + 1][0] if header_idx + 1 < len(headers) else len(lines)
        body = "".join(lines[start + 1 : end]).strip()
        items.append(PlanItem(item_id=item_id, title=title.strip(), body=body))

    return items


def parse_id_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    ids: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            ids.append(part)
    return ids or None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def snapshot_files(paths: Iterable[Path]) -> set[str]:
    snapshot: set[str] = set()
    for p in paths:
        abs_path = (REPO_ROOT / p).resolve()
        if abs_path.is_file():
            snapshot.add(str(p))
            continue
        if abs_path.is_dir():
            for file_path in abs_path.rglob("*"):
                if file_path.is_file():
                    rel = file_path.relative_to(REPO_ROOT)
                    snapshot.add(str(rel))
    return snapshot


def make_backup(backup_tar_path: Path, paths: Iterable[Path]) -> None:
    backup_tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(backup_tar_path, "w:gz") as tar:
        for p in paths:
            abs_path = (REPO_ROOT / p).resolve()
            if abs_path.exists():
                tar.add(abs_path, arcname=str(p))


def restore_missing_files(backup_tar_path: Path, missing_relpaths: set[str]) -> None:
    if not missing_relpaths:
        return
    with tarfile.open(backup_tar_path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        for rel in sorted(missing_relpaths):
            member = members.get(rel)
            if member is None:
                continue
            tar.extract(member, path=REPO_ROOT)


def git_status_porcelain() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout or "").strip()


def build_planner_prompt(item: PlanItem, iteration: int, prior_judge: dict[str, Any] | None) -> str:
    judge_feedback = ""
    if prior_judge is not None:
        judge_feedback = (
            "\n\nPrior judge feedback (for iteration retry):\n"
            + json.dumps(prior_judge, indent=2, sort_keys=True)
        )

    return f"""You are the PLANNER agent in a planner→worker→judge pipeline.

Task: Create a detailed, executable plan to complete ONE item from the project's data collection plan.

Hard rules:
- Output MUST be valid JSON matching the provided JSON Schema.
- Do NOT delete files.
- Keep the plan minimal, idempotent, and auditable (manifests/logs).
- Prefer writing data to: data_raw/, data_curated/, data_features/ (these are gitignored).
- Prefer writing run logs/reports to: outputs/
- If network access is needed for shell commands, assume it is allowed (workspace-write network enabled).

Item:
- item_id: {item.item_id}
- item_title: {item.title}

Relevant excerpt from docs/data_collection_plan.md (this is the item scope):
\"\"\"\n{item.body}\n\"\"\"
{judge_feedback}

Return JSON only. No markdown.
"""


def build_worker_prompt(
    item: PlanItem,
    iteration: int,
    planner_output: dict[str, Any],
    prior_judge: dict[str, Any] | None,
) -> str:
    judge_feedback = ""
    if prior_judge is not None:
        judge_feedback = (
            "\n\nPrior judge feedback (fix these issues first):\n"
            + json.dumps(prior_judge, indent=2, sort_keys=True)
        )

    return f"""You are the WORKER agent in a planner→worker→judge pipeline.

Goal: Execute the planner's plan for the given item and produce the required artifacts.

Hard rules:
- Output MUST be valid JSON matching the provided JSON Schema.
- Do NOT delete files (tracked or untracked).
- Make changes only as needed for this item. Avoid refactors.
- Put downloaded/derived data under data_raw/, data_curated/, data_features/.
- Put logs/manifests/reports under outputs/.
- Prefer idempotent snapshots (timestamped folders), do not overwrite prior outputs.

Item:
- item_id: {item.item_id}
- item_title: {item.title}

Planner output JSON:
{json.dumps(planner_output, indent=2, sort_keys=True)}

Relevant excerpt from docs/data_collection_plan.md:
\"\"\"\n{item.body}\n\"\"\"
{judge_feedback}

Return JSON only. No markdown.
"""


def build_judge_prompt(
    item: PlanItem,
    iteration: int,
    planner_output: dict[str, Any],
    worker_output: dict[str, Any],
    guardrails: dict[str, Any],
) -> str:
    return f"""You are the JUDGE agent in a planner→worker→judge pipeline.

You must audit the worker's work for the item. Output PASS only if the worker satisfied the planner's success criteria and produced the required artifacts with reasonable validation.

Hard rules:
- Output MUST be valid JSON matching the provided JSON Schema.
- Be strict and concrete: list exactly what is missing if you FAIL.

Item:
- item_id: {item.item_id}
- item_title: {item.title}
- iteration: {iteration}

Planner output JSON:
{json.dumps(planner_output, indent=2, sort_keys=True)}

Worker output JSON:
{json.dumps(worker_output, indent=2, sort_keys=True)}

Guardrails report (file deletions, git status, etc):
{json.dumps(guardrails, indent=2, sort_keys=True)}

Relevant excerpt from docs/data_collection_plan.md:
\"\"\"\n{item.body}\n\"\"\"

Return JSON only. No markdown.
"""


def run_codex_exec(
    *,
    prompt: str,
    schema_path: Path,
    output_last_message_path: Path,
    logs_dir: Path,
    model: str | None,
    sandbox: str,
    enable_network_in_workspace_write: bool,
    full_auto: bool,
) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "stdout.jsonl"
    stderr_path = logs_dir / "stderr.log"

    cmd: list[str] = [
        "codex",
        "exec",
        "--cd",
        str(REPO_ROOT),
        "--sandbox",
        sandbox,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_last_message_path),
        "--json",
        "-",
    ]

    if model:
        cmd[2:2] = ["--model", model]

    if full_auto:
        cmd.insert(2, "--full-auto")

    if enable_network_in_workspace_write and sandbox == "workspace-write":
        # See Codex security docs: enable network access in workspace-write mode.
        cmd[2:2] = ["--config", "sandbox_workspace_write.network_access=true"]

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_f:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            input=prompt,
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
            check=False,
            env=os.environ.copy(),
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"codex exec failed (exit={proc.returncode}). See logs: {stderr_path}, {stdout_path}"
        )

    if not output_last_message_path.exists():
        raise RuntimeError(f"Codex did not write output file: {output_last_message_path}")

    try:
        return read_json(output_last_message_path)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse JSON from {output_last_message_path}: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a planner→worker→judge Codex exec pipeline.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Path to docs/data_collection_plan.md",
    )
    parser.add_argument(
        "--items",
        type=str,
        default=None,
        help="Comma-separated item IDs to run (e.g., 1,2,3). Default: start..end.",
    )
    parser.add_argument("--start", type=int, default=1, help="Start item id (inclusive).")
    parser.add_argument("--end", type=int, default=5, help="End item id (inclusive).")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max iterations per item.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Run output root.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH, help="State JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="List items and exit.")

    parser.add_argument("--planner-model", type=str, default=None)
    parser.add_argument("--worker-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)

    parser.add_argument(
        "--enable-worker-network",
        action="store_true",
        help="Enable network access inside workspace-write sandbox for worker runs.",
    )

    args = parser.parse_args()

    ensure_codex_on_path()

    plan_path: Path = args.plan
    if not plan_path.exists():
        raise SystemExit(f"Plan not found: {plan_path}")

    items = parse_items_from_plan(plan_path)
    items_by_id = {i.item_id: i for i in items}

    selected_ids = parse_id_list(args.items)
    if selected_ids is None:
        selected_ids = [str(i) for i in range(args.start, args.end + 1)]

    missing = [i for i in selected_ids if i not in items_by_id]
    if missing:
        raise SystemExit(f"Unknown item id(s): {missing}. Available: {[i.item_id for i in items]}")

    if args.dry_run:
        for item_id in selected_ids:
            item = items_by_id[item_id]
            print(f"{item.item_id}) {item.title}")
        return

    runs_dir: Path = args.runs_dir
    state_path: Path = args.state
    runs_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {"version": 1, "plan_path": str(plan_path), "items": {}}
    if state_path.exists():
        try:
            state = read_json(state_path)
        except Exception:
            # If the state file is corrupt, do not overwrite it.
            raise SystemExit(f"Could not read state file: {state_path}")

    protected_paths = [Path("docs"), Path("paper"), Path("scripts"), Path("analysis"), Path("README.md")]

    for item_id in selected_ids:
        item_state = state.setdefault("items", {}).get(item_id, {})
        if item_state.get("status") == "completed":
            continue

        prior_judge: dict[str, Any] | None = None
        if isinstance(item_state.get("last_judge"), dict):
            prior_judge = item_state["last_judge"]

        for attempt in range(1, args.max_attempts + 1):
            attempt_dir = runs_dir / f"item_{item_id}" / f"attempt_{attempt}_{utc_now_compact()}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            # Backup protected paths before worker runs.
            backup_tar = attempt_dir / "backup_protected.tar.gz"
            before_snapshot = snapshot_files(protected_paths)
            make_backup(backup_tar, protected_paths)

            planner_out_path = attempt_dir / "planner.json"
            worker_out_path = attempt_dir / "worker.json"
            judge_out_path = attempt_dir / "judge.json"

            planner_logs = attempt_dir / "planner_logs"
            worker_logs = attempt_dir / "worker_logs"
            judge_logs = attempt_dir / "judge_logs"

            planner_prompt = build_planner_prompt(item, attempt, prior_judge)
            planner = run_codex_exec(
                prompt=planner_prompt,
                schema_path=PLANNER_SCHEMA,
                output_last_message_path=planner_out_path,
                logs_dir=planner_logs,
                model=args.planner_model,
                sandbox="read-only",
                enable_network_in_workspace_write=False,
                full_auto=False,
            )

            worker_prompt = build_worker_prompt(item, attempt, planner, prior_judge)
            worker = run_codex_exec(
                prompt=worker_prompt,
                schema_path=WORKER_SCHEMA,
                output_last_message_path=worker_out_path,
                logs_dir=worker_logs,
                model=args.worker_model,
                sandbox="workspace-write",
                enable_network_in_workspace_write=bool(args.enable_worker_network),
                full_auto=True,
            )

            after_snapshot = snapshot_files(protected_paths)
            missing_files = before_snapshot - after_snapshot
            restored = False
            if missing_files:
                restore_missing_files(backup_tar, missing_files)
                restored = True

            guardrails = {
                "protected_missing_files": sorted(missing_files),
                "protected_missing_files_restored": restored,
                "git_status_porcelain": git_status_porcelain(),
            }
            write_json(attempt_dir / "guardrails.json", guardrails)

            judge_prompt = build_judge_prompt(item, attempt, planner, worker, guardrails)
            judge = run_codex_exec(
                prompt=judge_prompt,
                schema_path=JUDGE_SCHEMA,
                output_last_message_path=judge_out_path,
                logs_dir=judge_logs,
                model=args.judge_model,
                sandbox="read-only",
                enable_network_in_workspace_write=False,
                full_auto=False,
            )

            state.setdefault("items", {})[item_id] = {
                "status": "completed" if judge.get("decision") == "PASS" else "failed",
                "last_attempt_dir": str(attempt_dir),
                "last_judge": judge,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(state_path, state)

            if judge.get("decision") == "PASS":
                break

            prior_judge = judge

        # Stop the whole pipeline if we failed an item after max attempts.
        final_state = state.get("items", {}).get(item_id, {})
        if final_state.get("status") != "completed":
            raise SystemExit(
                f"Item {item_id} failed after {args.max_attempts} attempts. "
                f"See runs under: {runs_dir / f'item_{item_id}'}"
            )


if __name__ == "__main__":
    main()

