# PWJ data-collection pipeline (Codex CLI)

This project uses a simple **planner → worker → judge** workflow to automate data collection work defined in `docs/data_collection_plan.md`.

## Key Codex CLI notes

- The non-interactive command is `codex exec` (not `codex --exec`).
- `codex exec` runs in a **read-only sandbox by default**. For automation that needs to write files/run commands, use `--full-auto` (or pass `--sandbox workspace-write`).
- In `workspace-write` mode, **network access is off by default**. Enable it with config:

  ```toml
  [sandbox_workspace_write]
  network_access = true
  ```

  The PWJ orchestrator can also enable it per-run with `codex exec --config sandbox_workspace_write.network_access=true ...`.

## Orchestrator script

Run the pipeline from the repo root:

```bash
python scripts/pwj_pipeline.py --dry-run
python scripts/pwj_pipeline.py --enable-worker-network
```

By default it runs items **1–5** (data architecture + the data sources).

Common variations:

```bash
# Run just Hugging Face archive ingestion work (item 2)
python scripts/pwj_pipeline.py --items 2 --enable-worker-network

# Run API + Reddit only
python scripts/pwj_pipeline.py --items 3,5 --enable-worker-network

# Use explicit models per role
python scripts/pwj_pipeline.py --enable-worker-network \
  --planner-model gpt-5.2-codex \
  --worker-model gpt-5.2-codex \
  --judge-model gpt-5.2-codex
```

## Outputs and state

- Runs: `outputs/pwj_pipeline/item_<id>/attempt_*/`
  - `planner.json`, `worker.json`, `judge.json`: structured outputs per role
  - `*_logs/`: JSONL event stream + stderr logs
  - `guardrails.json`: protected file deletion checks + `git status` summary
- Pipeline state (for resume): `outputs/pwj_pipeline/state.json`

## Guardrails (to avoid drift / chaos)

- Worker runs are audited by a judge agent and must PASS before moving on.
- The orchestrator snapshots `docs/`, `paper/`, `scripts/`, `analysis/`, and `README.md` before worker runs; if any of these files are deleted, it restores them from a local backup in the attempt directory.

