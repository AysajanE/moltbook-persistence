# Decisions (living log)

Keep this file short and pragmatic: record decisions that prevent rework.

## 2026-02-04

- **Paper toolchain:** LaTeX in `paper/` (arXiv-friendly).
- **Analysis language:** Python (minimal scripts + notebooks as needed).
- **Data handling:** raw/interim/processed data live under `data/` and are **not committed** to git.
- **Outputs:** generated figures/tables go under `outputs/`; only finalized, paper-ready figures should move into `paper/figures/`.

