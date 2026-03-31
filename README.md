# Fast Response or Silence: Conversation Persistence in an AI-Agent Social Network

Manuscript and reproducible analysis pipeline for studying conversation persistence and coordination limits on Moltbook, an AI-agent social network, with Reddit used as a contextual comparison baseline.

This repo is structured as an academic research project rather than a loose notebook dump: manuscript sources, analysis scripts, data-workspace conventions, and decision records are all kept together so the claims can be traced back to code and inputs.

## Research question

The project asks a simple but nontrivial question:

When do conversations in an AI-agent social network persist, and when do they collapse into silence?

The empirical framing is comparative:

- Moltbook is the primary setting
- Reddit provides a contextual baseline
- matched and platform-specific analyses are both included

## Repository map

- `paper/`: LaTeX manuscript sources and paper-facing assets
- `analysis/`: reproducible Python scripts for collection, curation, validation, and analysis
- `scripts/`: helper CLIs for export and submission/source packaging
- `docs/`: background notes, data-source notes, and decision log
- `data/`, `data_raw/`, `data_curated/`, `data_features/`: local data workspaces
- `outputs/`: derived run artifacts, diagnostics, tables, and figures

## Reproducibility workflow

Target Python version is 3.11.

```bash
make install
make lint
make format
```

To build the manuscript:

```bash
make clean-paper
make paper
```

That builds `paper/main.pdf` from `paper/main.tex`.

## Main analysis entrypoints

Primary scripts:

- `analysis/06_moltbook_only_analysis.py`
- `analysis/07_reddit_only_analysis.py`
- `analysis/08_cross_platform_matched_analysis.py`

Example usage:

```bash
python analysis/06_moltbook_only_analysis.py
python analysis/07_reddit_only_analysis.py --run-id attempt_scaled_20260206-142651Z
python analysis/08_cross_platform_matched_analysis.py
```

Run `--help` on each script for options.

## Data handling and ethics

- Raw data is not committed.
- Redistribution constraints matter, especially for third-party Reddit data.
- Reproducibility depends on preserving run manifests, seeds, and intermediate provenance.

For data norms and project decisions, see:

- `data/README.md`
- `docs/data_sources.md`
- `docs/decisions.md`

## Why this repo matters

This repository signals:

- original empirical research on AI-agent social behavior
- manuscript-first reproducibility discipline
- comfort working across data collection, cleaning, statistical analysis, and scholarly writing
