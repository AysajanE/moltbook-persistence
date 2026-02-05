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
DEFAULT_ENV_LOCAL_PATH = REPO_ROOT / ".env.local"

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


def load_env_local(path: Path) -> dict[str, str]:
    """Parse a lightweight .env file (dotenv style) without executing it as shell.

    Supports:
    - Blank lines and leading `#` comments
    - Optional `export ` prefix
    - Optional whitespace around `=`
    - Single or double quoted values (quotes stripped)
    """

    if not path.exists():
        return {}

    env: dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            # Ignore malformed lines quietly (do not print; may contain secrets).
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        # Strip surrounding quotes if present.
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]

        env[key] = value

    return env


def apply_env_aliases(dotenv: dict[str, str]) -> None:
    """Populate expected uppercase env vars from common dotenv key variants.

    Never prints values; only sets if missing in current environment.
    """

    aliases: dict[str, list[str]] = {
        "MOLTBOOK_API_KEY": ["MOLTBOOK_API_KEY", "moltbook_api_key", "MOLTBOOK_APIKEY"],
        "REDDIT_CLIENT_ID": ["REDDIT_CLIENT_ID", "reddit_client_id", "Reddit_client_id"],
        "REDDIT_CLIENT_SECRET": [
            "REDDIT_CLIENT_SECRET",
            "reddit_client_secret",
            "Reddit_client_secret",
        ],
        "REDDIT_USER_AGENT": ["REDDIT_USER_AGENT", "reddit_user_agent", "Reddit_user_agent"],
    }

    for target_key, candidates in aliases.items():
        if os.environ.get(target_key):
            continue
        for candidate in candidates:
            candidate_value = dotenv.get(candidate)
            if candidate_value:
                os.environ[target_key] = candidate_value
                break


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


def build_planner_prompt(
    item: PlanItem,
    iteration: int,
    prior_judge: dict[str, Any] | None,
    *,
    attempt_dir: str,
    pilot: bool,
    pilot_max_rows: int,
) -> str:
    judge_feedback = ""
    if prior_judge is not None:
        judge_feedback = (
            "\n\nPrior judge feedback (for iteration retry):\n"
            + json.dumps(prior_judge, indent=2, sort_keys=True)
        )

    pilot_block = ""
    if pilot:
        pilot_block = f"""
PILOT MODE (enabled):
- Keep scope small and fast. The purpose is to validate the PWJ pipeline mechanics (structured outputs, artifact creation, and judge gating).
- Avoid large downloads / long crawls. If sampling a dataset, cap exports to ~{pilot_max_rows} rows per split/table unless the item explicitly requires full coverage.
- Avoid irreversible actions (account registration, posting, mass API crawling) unless explicitly required and clearly safe.
"""

    return f"""You are the PLANNER in a planner→worker→judge pipeline.

You produce a plan for exactly ONE item from `docs/data_collection_plan.md`. Your output MUST be valid JSON that matches the provided JSON Schema exactly (no markdown, no extra keys).

You do not execute the plan. Assume the worker starts “fresh” with only this prompt + the repo contents (and any judge feedback).

Secrets / credentials (critical):
- This repo uses gitignored `.env.local` for local secrets. Some items require API credentials (e.g., Moltbook, Reddit).
- Do NOT print, paste, or write any secret values to disk (including in `outputs/`, logs, manifests, or reports).
- If you need to reference an authenticated request, use placeholders like `Authorization: Bearer $MOLTBOOK_API_KEY` (never the literal token).
- Do NOT require the worker to `cat .env.local`. If needed, the worker may load env vars from `.env.local` without printing values.

Non-negotiable standards (academic integrity):
- Do NOT fabricate results, citations, schemas, or “it worked” claims. If unsure, specify what to check and how.
- Keep the plan deterministic and auditable: every important output should be written to disk with a manifest and basic validation.

Collaboration / safety rules:
- Do NOT delete files (tracked or untracked).
- Do NOT change unrelated code/docs. Plan should minimize code changes.
- Do NOT assume credentials exist. If an item requires secrets/auth, plan a safe “no-auth pilot” sub-scope or explicitly call out the missing prerequisite.

Execution model:
- A WORKER agent will execute your plan in a `workspace-write` sandbox and can run commands + write files.
- A JUDGE agent will audit: it will check that required artifacts exist and validations were run.

Run context:
- Write run-specific reports/manifests under: `{attempt_dir}/`

Item to plan:
- item_id: {item.item_id}
- item_title: {item.title}
- iteration: {iteration}

Item scope excerpt (treat this as the spec):
\"\"\"\n{item.body}\n\"\"\"
{judge_feedback}
{pilot_block}

Plan requirements:
- Define a crisp `objective` that can be judged PASS/FAIL.
- Provide 6–12 concrete `plan_steps` (imperative, testable; avoid vague steps). Each step should mention either:
  - an exact command to run, OR
  - an exact file to create/update, OR
  - a measurable check to perform.
- Include explicit `artifacts` the worker must create. For each artifact:
  - Use repo-relative paths.
  - Mark `must_exist=true` for artifacts required for PASS.
  - Prefer per-run manifests/reports under `outputs/` and data under `data_raw/`, `data_curated/`, `data_features/`.
- Include `suggested_commands` as an array of strings (can be empty); use it to give a minimal, copy-pastable command sequence.
- Include `checks`: exact validations the worker must run (row counts, schema discovery output, referential integrity, timestamp parse rate, etc).
- Include `constraints`: key guardrails (no deletions, no secrets, no huge downloads if doing a pilot, etc). If an item requires auth, state the prerequisite explicitly.
- Include `notes` as a string (can be empty) for any extra nuance not captured above.

Minimum required artifacts (include these with `must_exist=true`):
- `{attempt_dir}/run_manifest.json` (machine-readable: inputs, commands, versions, timestamps, row counts)
- `{attempt_dir}/report.md` (human-readable: what was done, what was verified, and how to scale to full run)

run_manifest.json guidance (for the worker; encode this expectation in your plan):
- Include: item_id, item_title, iteration, started_at_utc, finished_at_utc
- Record: git_commit, python_version (and any package versions you installed), and OS basics if available
- Record: the exact commands run (or scripts invoked) and their key parameters (especially dataset identifiers and row caps)
- Record: artifacts produced with paths + row counts when known
- Record: validation results (pass/fail + notes)

Return JSON only. No markdown. No commentary outside JSON.
"""


def build_worker_prompt(
    item: PlanItem,
    iteration: int,
    planner_output: dict[str, Any],
    prior_judge: dict[str, Any] | None,
    *,
    attempt_dir: str,
    pilot: bool,
    pilot_max_rows: int,
) -> str:
    judge_feedback = ""
    if prior_judge is not None:
        judge_feedback = (
            "\n\nPrior judge feedback (fix these issues first):\n"
            + json.dumps(prior_judge, indent=2, sort_keys=True)
        )

    pilot_block = ""
    if pilot:
        pilot_block = f"""
PILOT MODE (enabled):
- Prefer a small, high-signal slice that proves the pipeline works end-to-end.
- Cap exports to ~{pilot_max_rows} rows per split/table where applicable.
- Focus on: artifact creation, schema discovery output, and at least one meaningful validation check.
"""

    return f"""You are the WORKER in a planner→worker→judge pipeline.

Your job: execute the planner’s plan for exactly ONE item. Your output MUST be valid JSON matching the provided JSON Schema exactly (no markdown, no extra keys).

Assume you start “fresh” with only this prompt + the repository state (and any judge feedback below). Do not rely on unstated prior context.

Secrets / credentials (critical):
- Some items require API credentials (e.g., `MOLTBOOK_API_KEY`, Reddit OAuth). These may be present as environment variables and/or stored locally in gitignored `.env.local`.
- NEVER print secret values (no `echo $MOLTBOOK_API_KEY`, no dumping env, no printing request headers, no printing credential JSON).
- NEVER write secret values to disk (including in `outputs/`, `data_raw/`, logs, manifests, or reports).
- NEVER store auth headers in raw request logs. If you persist request metadata, exclude headers entirely or redact auth.
- If you must verify a secret exists, only do non-printing checks (e.g., `test -n \"$MOLTBOOK_API_KEY\"`).
- Prefer using `MOLTBOOK_API_KEY` if already available; do not call `/agents/register` unless explicitly required (it returns secrets).

Non-negotiable standards (academic integrity):
- Do NOT fabricate results or claim checks passed unless you actually ran them.
- If something fails, report it plainly in `issues` and adjust with the smallest corrective step.

Collaboration / safety rules:
- Do NOT delete files (tracked or untracked).
- Keep changes tightly scoped to this item. Avoid refactors and formatting churn.
- Do NOT commit, push, open PRs, or modify git history.
- Do NOT add secrets to the repo; do not print tokens/keys in logs.

Data handling:
- Put downloaded/derived datasets under: `data_raw/`, `data_curated/`, `data_features/` (gitignored).
- Put run manifests, schema manifests, and validation reports under: `outputs/` (commit is optional, but outputs should be saved to disk for audit).
- Prefer timestamped snapshot folders; do not overwrite prior artifacts.

Run context:
- Write run-specific reports/manifests under: `{attempt_dir}/`

Item:
- item_id: {item.item_id}
- item_title: {item.title}
- iteration: {iteration}

Planner output JSON (follow this exactly unless it is impossible; then explain why):
{json.dumps(planner_output, indent=2, sort_keys=True)}

Item scope excerpt:
\"\"\"\n{item.body}\n\"\"\"
{judge_feedback}
{pilot_block}

Implementation expectations:
- Before writing new scripts, look for existing helpers in `scripts/` and reuse them when possible.
- Produce all `must_exist=true` artifacts listed by the planner.
- Ensure `{attempt_dir}/run_manifest.json` and `{attempt_dir}/report.md` exist and are coherent.
- Run the planner’s `checks` and record what you ran in `checks_run` (include command names and what they validated).
- Populate `artifacts_created` with accurate repo-relative paths and descriptions. For tabular artifacts, include row counts; otherwise set `rows=-1`.
- If the work would take a long time or download large volumes, implement a small, verifiable slice first and document how to scale it up (but do not pretend it is complete).
- Populate `next_steps` (can be empty) with what to do next to scale from this run.

Return JSON only. No markdown. No commentary outside JSON.
"""


def build_judge_prompt(
    item: PlanItem,
    iteration: int,
    planner_output: dict[str, Any],
    worker_output: dict[str, Any],
    guardrails: dict[str, Any],
    *,
    attempt_dir: str,
    pilot: bool,
) -> str:
    pilot_block = ""
    if pilot:
        pilot_block = """
PILOT MODE (enabled):
- Judge against the *pilot objective* and required artifacts. Do not require full-scale collection if the planner scoped it explicitly as a pilot.
- Still require real evidence: artifacts exist, checks ran, and outputs are interpretable/reproducible.
"""

    return f"""You are the JUDGE in a planner→worker→judge pipeline.

You must audit the worker’s work for exactly ONE item and decide PASS/FAIL. Your output MUST be valid JSON matching the provided JSON Schema exactly (no markdown, no extra keys).

Decision rule:
- PASS only if the worker met the planner’s objective, produced all required artifacts, and ran the required checks with credible evidence.
- Otherwise FAIL and provide concrete, minimal required fixes.

Audit standards (be strict):
- Do NOT accept “should be fine” or unverifiable claims.
- Prefer checking the filesystem and git status/diff directly (read-only is fine). If you do not verify, FAIL.

Secrets / credentials:
- If the worker printed or wrote any secret values (API keys, OAuth secrets, auth headers, registration responses containing `api_key`, etc.), this is an automatic FAIL.
- Ensure `.env.local` was not copied into outputs and that reports/manifests use placeholders/redactions for any auth steps.

Safety / collaboration:
- If guardrails indicate protected file deletions, this is an automatic FAIL unless fully restored and explained.
- If the worker changed many unrelated files, FAIL and request a narrower fix.

Run context:
- Attempt directory: `{attempt_dir}/`

Item:
- item_id: {item.item_id}
- item_title: {item.title}
- iteration: {iteration}

Planner output JSON:
{json.dumps(planner_output, indent=2, sort_keys=True)}

Worker output JSON:
{json.dumps(worker_output, indent=2, sort_keys=True)}

Guardrails report:
{json.dumps(guardrails, indent=2, sort_keys=True)}

Item scope excerpt:
\"\"\"\n{item.body}\n\"\"\"
{pilot_block}

What to verify (minimum):
1) All planner artifacts with `must_exist=true` exist at the specified paths and are non-empty where applicable.
2) Worker’s `checks_run` plausibly correspond to the planner’s `checks` (and results are recorded somewhere on disk or inferable from artifacts/logs).
3) Data landed in the intended directories (`data_raw/`, `data_curated/`, `data_features/`) and no raw data was added to git.
4) Changes are scoped and reproducible (manifests, schema logs, timestamps, versions).
5) `{attempt_dir}/run_manifest.json` and `{attempt_dir}/report.md` exist and contain sufficient detail to reproduce the run.

Output requirements:
- `decision`: PASS or FAIL
- `checks`: include multiple checks with clear pass/fail and details (include paths you inspected and commands you ran).
- `required_fixes`: if FAIL, list exact missing artifacts/commands/edits needed.
- `suggested_worker_instructions`: required. If FAIL, give a ready-to-paste instruction for the next worker iteration. If PASS, set to an empty string.

Return JSON only. No markdown. No commentary outside JSON.
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run selected items even if state marks them completed.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Ignore any existing state file and start fresh.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List items and exit.")

    parser.add_argument("--planner-model", type=str, default=None)
    parser.add_argument("--worker-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: instruct agents to do a small, fast slice to validate the pipeline.",
    )
    parser.add_argument(
        "--pilot-max-rows",
        type=int,
        default=1000,
        help="Row cap per split/table when in --pilot mode (default: 1000).",
    )

    parser.add_argument(
        "--enable-worker-network",
        action="store_true",
        help="Enable network access inside workspace-write sandbox for worker runs.",
    )

    args = parser.parse_args()

    # Load dotenv secrets locally (gitignored) without executing it as shell.
    apply_env_aliases(load_env_local(DEFAULT_ENV_LOCAL_PATH))

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
    if state_path.exists() and not args.reset_state:
        try:
            state = read_json(state_path)
        except Exception:
            # If the state file is corrupt, do not overwrite it.
            raise SystemExit(f"Could not read state file: {state_path}")

    protected_paths = [Path("docs"), Path("paper"), Path("scripts"), Path("analysis"), Path("README.md")]

    for item_id in selected_ids:
        item = items_by_id[item_id]
        item_state = state.setdefault("items", {}).get(item_id, {})
        if item_state.get("status") == "completed" and not args.force:
            continue

        prior_judge: dict[str, Any] | None = None
        if isinstance(item_state.get("last_judge"), dict):
            prior_judge = item_state["last_judge"]

        for attempt in range(1, args.max_attempts + 1):
            attempt_dir = runs_dir / f"item_{item_id}" / f"attempt_{attempt}_{utc_now_compact()}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_rel = str(attempt_dir.relative_to(REPO_ROOT))

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

            planner_prompt = build_planner_prompt(
                item,
                attempt,
                prior_judge,
                attempt_dir=attempt_rel,
                pilot=bool(args.pilot),
                pilot_max_rows=int(args.pilot_max_rows),
            )
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

            worker_prompt = build_worker_prompt(
                item,
                attempt,
                planner,
                prior_judge,
                attempt_dir=attempt_rel,
                pilot=bool(args.pilot),
                pilot_max_rows=int(args.pilot_max_rows),
            )
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

            judge_prompt = build_judge_prompt(
                item,
                attempt,
                planner,
                worker,
                guardrails,
                attempt_dir=attempt_rel,
                pilot=bool(args.pilot),
            )
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
