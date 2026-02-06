# Hypothesis-to-Data Mapping (Draft v1)

Date: 2026-02-06 (UTC)  
Purpose: map each research objective/hypothesis to concrete datasets, variables, and readiness status.

## 1) Research Objective Link

Core objective (proposal): estimate conversation persistence/half-life, structural depth, periodicity ("attention clock"), and cross-platform differences between Moltbook and Reddit with defensible controls.

Primary sources:
- `docs/moltbook_research_proposal.md`
- `docs/data_collection_plan.md`
- `docs/handoff_report_2026-02-06.md`

## 2) Data Assets and Current Status

### Moltbook HF archive (primary baseline outcomes)
- Path: `data_curated/hf_archive/snapshot_20260204-234429Z/`
- Canonical tables:
  - `data_curated/hf_archive/snapshot_20260204-234429Z/canonical/posts_latest.parquet`
  - `data_curated/hf_archive/snapshot_20260204-234429Z/canonical/comments_latest.parquet`
  - `data_curated/hf_archive/snapshot_20260204-234429Z/canonical/agents_latest.parquet`
- Strength: thread/event structure and large scale are ready now.
- Caveat: partitions are by `dump_date`, so calendar analysis must use event timestamps (`created_at_utc`), not `dump_date`.

### Moltbook API pilot (mechanics validated only)
- Run: `outputs/pwj_pipeline/item_3/attempt_1_20260205-152619Z/`
- Curated tables: `data_curated/moltbook/{feed_snapshots,posts,comments}`
- Scope: small pilot, not analysis-grade for exposure inference.

### Moltbook 72h API run (currently in progress)
- Run bundle: `outputs/ops/20260206-142651Z/`
- Script: `outputs/ops/20260206-142651Z/scripts/run_moltbook_72h.sh`
- Current config:
  - `sort=hot,new`
  - `snapshots=4320`
  - `interval=60s`
  - `limit=100`
  - `max-post-details=0`
  - `max-comment-posts=0`
- Important implication: this run is primarily a feed-snapshot/exposure dataset, not a comment-tree dataset.

### Reddit scaled collection (analysis-ready for cross-platform descriptive + matching baseline)
- Run outputs:
  - `outputs/ops/20260206-142651Z/item5/attempt_scaled_20260206-142651Z/curation_manifest.json`
  - `outputs/ops/20260206-142651Z/item5/attempt_scaled_20260206-142651Z/validation_results.json`
- Curated tables: `data_curated/reddit/{submissions,comments}` filtered by run_id.
- Validation status: PASS.

## 3) Hypothesis Mapping Matrix

## H1: Moltbook has shorter half-life (and stronger periodic signature) than Reddit
- Estimands:
  - Half-life metrics (`half_life_empirical`, survival-model half-life)
  - Periodicity metrics (spectral peak near ~4h, autocorrelation signatures)
- Required variables:
  - Moltbook: `post_id`, `comment_id`, `parent_id`, `created_at_utc`, `submolt`
  - Reddit: `submission_id`, `comment_id`, `parent_id`, `created_at_utc`, `subreddit`
- Current sources:
  - Moltbook outcomes: HF canonical comments/posts
  - Reddit outcomes: scaled Arctic Shift run
  - Moltbook exposure/clock signal: 72h feed snapshots (in progress)
- Readiness:
  - Descriptive comparison: ready.
  - Matched inference with topic/time controls: mostly ready.
  - Exposure-controlled causal interpretation: needs completed 72h feed snapshots and explicit matching design.

## H2: Moltbook threads are shallower / more star-shaped / lower reciprocity than Reddit
- Estimands:
  - `depth_max`, `depth_mean`, star-shape index, reciprocity metrics
- Required variables:
  - Reply tree edges and timestamps on both platforms
- Current sources:
  - Moltbook: HF comments with `parent_id`
  - Reddit: curated comments with `parent_id`
- Readiness:
  - Strong now for structural comparison.
  - Matching by topic/time still recommended before causal language.

## H3: Topic/subcommunity moderates persistence
- Estimands:
  - Half-life/depth differences by submolt/topic bucket
- Required variables:
  - Moltbook topic labels (`submolt`) + thread outcomes
  - Reddit topic labels (`subreddit`) + thread outcomes
  - Exposure proxies for confounding control (ideally rank/visibility over time)
- Current sources:
  - Moltbook/Reddit topic labels are present now.
  - Exposure time-series controls are only emerging from the 72h run.
- Readiness:
  - Topic-moderation regression: ready with caveats.
  - Exposure-controlled topic effect: requires completed 72h run and documented mapping strategy.

## H4: sustaining-agent heterogeneity
- Estimands:
  - concentration of long-lived-thread contributions
  - re-entry/session features by agent
  - links to agent covariates (karma/followers/claimed)
- Required variables:
  - agent IDs, thread re-entry sequences, agent covariates
- Current sources:
  - Moltbook HF archive is sufficient for first-pass H4.
- Readiness:
  - Ready now on Moltbook-only data.
  - 72h run helps mechanism context but is not strictly required for first-pass H4.

## 4) CEM Cross-Platform Mapping (Plan §5.D.3)

Matching dimensions requested by plan:

1. Topic
- Available now:
  - Moltbook `submolt`
  - Reddit `subreddit`
- Needed:
  - documented mapping table `submolt -> subreddit cluster` (manual or embedding-assisted).

2. Time window
- Available now:
  - Moltbook HF event window (posts/comments) overlaps early Feb 2026.
  - Reddit scaled run window: 2026-01-31 to 2026-02-05 (UTC).
- Needed:
  - explicit overlap filter and censoring policy.

3. Early engagement ("first hour")
- Partially available now:
  - static/post-level totals are available.
- Not fully available now:
  - true "first-hour comment count/score trajectories" are not fully observed from static HF + static Reddit pulls.
- 72h relevance:
  - completed Moltbook 72h feed snapshots improve early engagement measurement for posts observed near birth and provide rank/exposure proxies.
  - but a symmetric Reddit first-hour measurement strategy is still needed for strict apples-to-apples matching.

## 5) Decision Guidance: Is 72h Required for Matched Cross-Platform Comparison?

Short answer:
- For a baseline matched comparison (topic + time + coarse covariates): **No, not strictly required**.
- For stronger exposure-aware interpretation and first-hour-style controls: **Yes, 72h Moltbook data is important**, but not sufficient alone unless Reddit gets analogous early-engagement measurement.

## 6) Recommended Two-Stage Inference Strategy

Stage A (execute now):
- Run CEM using available variables:
  - topic bucket
  - calendar window bin
  - thread start-time bin
  - coarse engagement proxies available at/near observation
- Report as "matched observational comparison (limited exposure controls)."

Stage B (after 72h completes):
- Add Moltbook exposure features from feed snapshots (rank trajectory, visibility duration, feed recurrence).
- Re-estimate matched effects on restricted overlap where comparable controls exist.
- Document remaining asymmetry if Reddit lacks analogous exposure trajectories.

## 7) Explicit Limitations to Carry into Results

- Current 72h configuration does not crawl comment trees (`max-comment-posts=0`), so it strengthens exposure/clock measurement more than direct thread-dynamics capture.
- Without symmetric first-hour measurement on Reddit, causal language should remain cautious.
- Any claim of "agent-vs-human architecture effect" should be framed as conditional on observed controls, with unobserved selection/confounding caveats.
