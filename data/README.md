# Data directory

This repo keeps data **out of git** for size and licensing reasons.

## Layout

- `data/raw/`: immutable exports from upstream sources (Hugging Face, APIs)
- `data/interim/`: cleaned/merged intermediate artifacts
- `data/processed/`: analysis-ready tables used by scripts/notebooks

## What to commit

Commit only small metadata and documentation (e.g., `README.md`, `.gitkeep`).

