# Decisions (living log)

Keep this file short and pragmatic: record decisions that prevent rework.

## 2026-02-04

- **Paper toolchain:** LaTeX in `paper/` (arXiv-friendly).
- **Analysis language:** Python (minimal scripts + notebooks as needed).
- **Data handling:** raw/interim/processed data live under `data/` and are **not committed** to git.
- **Outputs:** generated figures/tables go under `outputs/`; only finalized, paper-ready figures should move into `paper/figures/`.

## 2026-02-06

- **Moltbook-only analysis entrypoint:** `analysis/06_moltbook_only_analysis.py`.
- **Canonical manuscript run:** `run_20260206-145240Z` with manifest at
  `outputs/moltbook_only/run_20260206-145240Z/run_manifest.json`.
- **Event-time policy:** all temporal inference uses `created_at_utc`; `dump_date` is treated as snapshot provenance only.
- **Half-life censoring policy:** primary survival estimates exclude parent comments created within 4 hours of observation end; no-boundary estimate is reported as sensitivity.
- **Periodicity policy:** split timeline on gaps >6 hours and estimate PSD on the longest contiguous segment; significance is calibrated with AR(1) simulations.
- **Submolt categorization policy:** deterministic keyword mapping into
  Builder/Technical, Philosophy/Meta, Social/Casual, Creative, Spam/Low-Signal, Other.

## 2026-02-06

- **Reddit-only analysis entrypoint:** `analysis/07_reddit_only_analysis.py`.
- **Canonical Reddit manuscript run_id:** `attempt_scaled_20260206-142651Z`.
- **Manifest path:** `outputs/reddit_only/attempt_scaled_20260206-142651Z/run_manifest.json`.
- **Input pinning policy:** enforce `run_id` filtering to avoid pilot/scaled overlap.
- **Reddit caveat policy:** report dropped-missing-submission and non-200 request-log counts in manifests/results.

## 2026-02-06

- **arXiv packaging workflow:** build source bundles via
  `scripts/build_arxiv_bundle.py` (`make arxiv-bundle`) to produce timestamped
  tarballs under `outputs/arxiv/` with optional compile checks.
