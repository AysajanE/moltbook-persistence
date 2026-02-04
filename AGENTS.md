# Project instructions for Codex AI agents

This repository is an **academic research** codebase. **Scientific rigor and academic integrity are non-negotiable.**

## Non-fabrication / evidence standard (must follow)

- **Do not fabricate** facts, numbers, citations, datasets, file contents, results, or experiment outcomes.
- **Do not guess**. If you are unsure, explicitly say what is unknown and propose a concrete verification plan (e.g., which script to run, which file to inspect, which dataset field to validate).
- **No “fill-in” results:** do **not** replace `\todo{...}` placeholders in the paper unless you have generated the underlying values from this repo’s code and/or documented data, and you can point to the exact provenance (script + inputs + date + version).
- When the user requests an estimate/forecast, only provide it if you can (a) justify it with explicit assumptions and (b) label it clearly as an estimate; otherwise, decline and propose how to compute it from data.

## Project context (source-of-truth docs)

- Research scope and hypotheses: `docs/moltbook_research_proposal.md`
- Background pointers and related work: `docs/background.md`
- Data sources and norms: `docs/data_sources.md`, `docs/data_collection_plan.md`
- Decision log (update when choices affect the study): `docs/decisions.md`
- Execution checklist: `docs/project_checklist.md`
- Paper source: `paper/main.tex` and `paper/sections/`

## Reproducibility and provenance

- Prefer **small, end-to-end scripts** over frameworks. Keep pipelines deterministic where possible.
- For every derived artifact (tables/figures/metrics), ensure there is:
  - a generating script (prefer `analysis/NN_*.py`),
  - a clear input path (`data/raw|interim|processed`),
  - recorded dataset versioning info (download date + upstream identifier/version/hash when available).
- If you introduce randomness (sampling, bootstraps), set and record seeds.

## Data handling (do not break)

- **Do not commit raw data**. Follow `data/README.md`:
  - `data/raw/` = immutable upstream exports
  - `data/interim/` = cleaned/merged intermediates
  - `data/processed/` = analysis-ready tables
- Keep sensitive/redistributable content out of git. For Reddit, store only IDs and/or anonymized derivatives as permitted.

## Paper-writing rules (LaTeX)

- Do not “polish” claims into certainty without evidence.
- Keep claims aligned with computed results and cited sources in `paper/references.bib`.
- Prefer adding/updating figures/tables via reproducible scripts and placing drafts in `outputs/`; move finalized assets into `paper/figures/` (commit those).
- When changing LaTeX sources, ensure `make paper` succeeds (requires a LaTeX toolchain).

## Coding conventions (Python)

- Target Python 3.11; keep code lint/format compatible with `ruff` (`pyproject.toml`).
- Favor `pathlib`, type hints, and explicit CLI args for scripts.
- After code changes, run `make lint` (and `make format` when appropriate).

## External research and citations

- If you use external sources, prefer primary/authoritative references and record them:
  - Add BibTeX entries to `paper/references.bib`.
  - Add short pointers/notes to `docs/background.md` as needed.
- Do not cite sources you have not actually opened/verified.

## Privacy / ethics

- Do not attempt to de-anonymize humans behind agents or Reddit accounts.
- Avoid quoting or reproducing user-generated content beyond what’s necessary for analysis; prefer aggregate statistics and anonymized examples.

## About AGENTS.md (Codex behavior)

- Codex loads instruction files at the start of a run/session, combining:
  - global guidance from `~/.codex/` (prefers `AGENTS.override.md`, else `AGENTS.md`), then
  - project guidance from the repo root down to the current working directory (prefers `AGENTS.override.md`, else `AGENTS.md`).
- Instruction context is size-capped (configurable via `project_doc_max_bytes` and `project_doc_fallback_filenames` in `~/.codex/config.toml`). Keep this file concise.
- If project instructions grow, add more specific `AGENTS.md` or `AGENTS.override.md` files inside subdirectories (e.g., `paper/`, `analysis/`) rather than bloating the root file.
