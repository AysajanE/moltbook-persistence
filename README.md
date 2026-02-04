# Moltbook Conversation Persistence (Research Repo)

This repository contains a fast, reproducible research workflow for studying **conversation persistence and coordination limits** in **Moltbook** (a social network for AI agents), with a goal of producing an **arXiv preprint**.

**Working title:** *The 4‑Hour Half‑Life: Conversation Persistence and Coordination Limits in an AI‑Agent Social Network*

## Quickstart

1) Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Download the Moltbook Observatory Archive (Hugging Face) tables to `data/raw/`:

```bash
python scripts/download_moltbook_observatory_archive.py --out-dir data/raw/moltbook-observatory-archive
```

3) Build the paper PDF (optional; requires a local LaTeX install):

```bash
make paper
```

## Repo layout

- `moltbook_research_proposal.md`: original research proposal (source of truth for scope).
- `docs/`: background notes, data notes, and project checklists.
- Key docs:
  - `docs/background.md`: web-sourced context + pointers.
  - `docs/data_collection_plan.md`: detailed collection plan (source of truth for data work).
  - `docs/pwj_pipeline.md`: how to run the planner→worker→judge Codex pipeline.
  - `docs/project_checklist.md`: week-by-week execution checklist.
- `paper/`: LaTeX source for the arXiv submission.
- `scripts/`: small CLIs (download/export data, packaging helpers).
- `analysis/`: analysis entrypoints (kept minimal; add as the project evolves).
- `data/`: local data workspace (not committed; see `data/README.md`).
- `outputs/`: generated figures/tables (generally not committed; see `outputs/README.md`).

## Project norms (keep it fast)

- Prefer small, end-to-end scripts over big frameworks.
- Track “decisions” in `docs/decisions.md` so we don’t re-litigate.
- Keep raw data out of git; export clean intermediates for reproducibility.
