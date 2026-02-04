# Planner → Worker → Judge (PWJ) pipeline

This folder contains a simple orchestrator that runs **three Codex agents** per item:

1) **Planner**: creates a concrete, testable plan for a single item from `docs/data_collection_plan.md`.
2) **Worker**: executes the plan (data collection / scripts / artifacts).
3) **Judge**: audits the worker output and returns **PASS/FAIL**.

The orchestrator only moves to the next item when the judge returns **PASS**. It retries up to 3 times per item.

## Why structured outputs

The pipeline uses `codex exec --output-schema` so each agent returns machine-readable JSON that the orchestrator can parse and act on.

Schemas:

- `pipelines/pwj/schemas/planner.schema.json`
- `pipelines/pwj/schemas/worker.schema.json`
- `pipelines/pwj/schemas/judge.schema.json`

## Run

From repo root:

```bash
python scripts/pwj_pipeline.py --help
```

