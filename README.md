# Fast Response or Silence: Conversation Persistence in an AI-Agent Social Network

This repository contains the manuscript and reproducible analysis pipeline for studying conversation persistence and coordination limits on Moltbook (an AI-agent social network), with Reddit as a contextual baseline.

## What Is In This Repo

- `paper/`: LaTeX manuscript source (`paper/main.tex`) and committed manuscript PDF (`paper/main.pdf`).
- `analysis/`: reproducible Python entrypoints for curation, platform-specific analysis, and cross-platform matched comparison.
- `scripts/`: helper CLIs for data export and arXiv packaging.
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
make paper-ejor
```

## Build The Manuscript

```bash
make clean-paper && make paper
```

This builds `paper/main.pdf` from `paper/main.tex`.
For the EJOR-oriented build target, use `make paper-ejor` (outputs `paper/main_ejor.pdf`).

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

## arXiv Source Packaging

Preferred arXiv bundle command:

```bash
python scripts/build_arxiv_submission.py \
  --paper-dir paper \
  --bundle-dir arxiv \
  --tar-path arxiv_source.tar.gz \
  --check-compile
```

This creates:
- `arxiv/` (clean source bundle)
- `arxiv_source.tar.gz` (upload-ready tarball)

Legacy timestamped bundling script is also available:
- `scripts/build_arxiv_bundle.py`

## Repository Guide

- `analysis/README.md`: analysis workflow notes and scope.
- `scripts/README.md`: helper script usage.
- `docs/background.md`: background pointers and related references.
- `docs/attention_dynamics_model.md`: modeling notes.
- `docs/feedback/`: reviewer feedback and revision materials.
- `docs/decisions.md`: project decision log.
