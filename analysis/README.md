# Analysis

This folder is intentionally kept minimal at project start.

Suggested flow:

1) Export raw tables from Hugging Face to `data/raw/` (`scripts/download_moltbook_observatory_archive.py`)
2) Build thread/comment-tree tables (depth, parent links, re-entry, reciprocity)
3) Estimate interaction decay / half-life (survival models)
4) Generate paper-ready figures/tables into `outputs/`, then move final assets into `paper/figures/`

Add numbered scripts here as the pipeline stabilizes (e.g., `01_reconstruct_threads.py`).

