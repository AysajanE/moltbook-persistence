# Fast Response or Silence: Conversation Persistence in an AI-Agent Social Network

This repository contains the manuscript and reproducible analysis pipeline for studying conversation persistence and coordination limits on Moltbook (an AI-agent social network), with Reddit as a contextual baseline.

## What Is In This Repo

- `paper/`: LaTeX manuscript sources, plus committed figures/tables used by the manuscript.
- `analysis/`: reproducible Python entrypoints for curation, platform-specific analysis, and cross-platform matched comparison.
- `scripts/`: helper CLIs for data export and submission/source packaging.
- `docs/`: background notes, data-source notes, and decision log.
- `data/`, `data_raw/`, `data_curated/`, `data_features/`: local data workspaces (raw data is not committed).
- `outputs/`: run-scoped derived artifacts, diagnostics, and manuscript-facing tables/figures.

## Environment Setup

Target Python version is 3.11.

```bash
make install
```

Useful developer commands:

```bash
make lint
make format
make clean-paper
make paper
```

## Build The Manuscript

```bash
make clean-paper && make paper
```

This builds `paper/main.pdf` from `paper/main.tex`.
Additional manuscript build targets are available; see `Makefile` and `paper/README.md`.

## Analysis Entrypoints

Primary analysis scripts:

- `analysis/06_moltbook_only_analysis.py`
- `analysis/07_reddit_only_analysis.py`
- `analysis/08_cross_platform_matched_analysis.py`

Example commands:

```bash
python analysis/06_moltbook_only_analysis.py
python analysis/07_reddit_only_analysis.py --run-id attempt_scaled_20260206-142651Z
python analysis/08_cross_platform_matched_analysis.py
```

For script-specific options, run `--help` on each script.

## Data Sources And Handling

Primary upstream source (Moltbook):
- https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive

Data and ethics constraints:
- Do not commit raw data.
- Keep redistributable constraints in mind for third-party data (especially Reddit).
- Preserve run manifests and seed settings for reproducibility/provenance.

See:
- `data/README.md`
- `docs/data_sources.md`
- `docs/decisions.md`

## Repository Guide

- `analysis/README.md`: analysis workflow notes and scope.
- `scripts/README.md`: helper script usage.
- `docs/background.md`: background pointers and related references.
- `docs/attention_dynamics_model.md`: modeling notes.
- `docs/feedback/`: reviewer feedback and revision materials.
- `docs/decisions.md`: project decision log.
