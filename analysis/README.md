# Analysis

This folder is intentionally kept minimal at project start.

Suggested flow:

1) Export raw tables from Hugging Face to `data/raw/` (`scripts/download_moltbook_observatory_archive.py`)
2) Build thread/comment-tree tables (depth, parent links, re-entry, reciprocity)
3) Estimate interaction decay / half-life (survival models)
4) Generate paper-ready figures/tables into `outputs/`, then move final assets into `paper/figures/`

Add numbered scripts here as the pipeline stabilizes (e.g., `01_reconstruct_threads.py`).

Current stabilized entrypoint for Moltbook-only analysis:

- `06_moltbook_only_analysis.py`
  - Inputs: `data_curated/hf_archive/snapshot_*/`
  - Outputs: derived features under `data_features/moltbook_only/<run_id>/`
    and manuscript-facing tables/figures under `outputs/moltbook_only/<run_id>/`
  - Covers: thread geometry, first-reply survival estimates, periodicity checks, and
    run manifests for provenance.

- `07_reddit_only_analysis.py`
  - Inputs: `data_curated/reddit/{submissions,comments}` filtered by `--run-id`
  - Outputs: derived features under `data_features/reddit_only/<run_id>/`
    and manuscript-facing tables/figures under `outputs/reddit_only/<run_id>/`
  - Covers: Reddit thread geometry, re-entry, first-reply survival half-life, PSD/Fisher-g
    periodicity checks, and run manifests with explicit upstream warning caveats.
