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

- `08_cross_platform_matched_analysis.py`
  - Inputs: run-scoped Moltbook and Reddit feature tables
    (`thread_metrics.parquet`, `thread_events.parquet`, `survival_units.parquet`)
  - Outputs: run-scoped matched-pair artifacts under
    `outputs/cross_platform_matched/<run_id>/` including sample flow, pre/post balance
    diagnostics, paired outcome estimates, matched-subset half-life summaries, figures,
    `analysis_summary.json`, and `run_manifest.json`.
  - Covers: deterministic coarse topic mapping, UTC posting-hour and first-30-minute
    engagement coarsening, exact matching by stratum, and paired effect estimation.
