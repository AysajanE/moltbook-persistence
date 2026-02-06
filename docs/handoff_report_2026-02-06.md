<handoff_report>

# Team Handoff Report — Moltbook “4‑Hour Half‑Life” Project

**Handoff date:** 2026-02-06 (UTC)  
**Repo root:** `conversation-persistance-coordination-limits-in-AI-agent-social-network/`  
**Goal:** move from “data collection pipeline validated” → “analysis + figures + arXiv paper”

This report is written to help the team continue immediately without re-discovering context.

## 0) Executive summary

### What is already sufficient to start analysis now

The **curated Hugging Face (HF) Moltbook Observatory Archive** is ready and passes QC. It contains the key relational structure needed for the model and core analyses:

- `posts.id` + `comments.post_id` (threads)
- `comments.parent_id` (reply edges → tree depth/shape)
- `comments.created_at_utc` (event times → decay/half-life; periodicity; lags)
- `comments.agent_id` + `agents.*` (heterogeneity; re-entry)
- `posts.submolt` + `submolts.*` (topic moderation tests)

You can begin:
- thread reconstruction + metrics (depth/star-shape/reciprocity/re-entry),
- decay / half-life estimation from reply lags (survival / Hawkes-style),
- periodicity tests on aggregate arrival times (with careful handling of missingness/censoring),
- heterogeneity analysis (heavy-tail, sustaining agents, covariates).

### What is missing to fully support the proposal’s cross-platform + exposure-controlled claims

We validated a simple Planner→Worker→Judge (PWJ) pipeline and completed **pilot** runs for:
- Moltbook API (Item 3 pilot),
- Reddit via Arctic Shift (Item 5 pilot).

But for arXiv-grade results we still need:

1) **Longer Moltbook API feed snapshots** (to approximate exposure/visibility and to strengthen attention-clock inference beyond HF backfill artifacts).  
2) **Scaled Reddit comparison corpus** aligned to the Moltbook window (and ideally more complete comment trees per submission).  
3) A clear policy for **censoring/missingness** and **dump/backfill** (HF exports are by `dump_date`, not event time).

## 1) Current project state (what’s done)

### PWJ pipeline status

The PWJ pipeline is implemented and working. All items 1–5 are marked completed in:

- `outputs/pwj_pipeline/state.json`

The pipeline entrypoint:

- `scripts/pwj_pipeline.py`

Key run artifacts exist locally under:

- `outputs/pwj_pipeline/`

### Data inventories (local; gitignored)

#### A) Moltbook HF archive (curated; primary analysis dataset)

Curated snapshot directory:

- `data_curated/hf_archive/snapshot_20260204-234429Z/`

Tables (partitioned by `dt` = `dump_date`, *not* event time):

- `data_curated/hf_archive/snapshot_20260204-234429Z/agents`
- `data_curated/hf_archive/snapshot_20260204-234429Z/posts`
- `data_curated/hf_archive/snapshot_20260204-234429Z/comments`
- `data_curated/hf_archive/snapshot_20260204-234429Z/submolts`
- `data_curated/hf_archive/snapshot_20260204-234429Z/snapshots`
- `data_curated/hf_archive/snapshot_20260204-234429Z/word_frequency`

Canonical “latest state” (dedup/backfill-resolved by id+max dump_date; **useful for core thread/event analyses**):

- `data_curated/hf_archive/snapshot_20260204-234429Z/canonical/posts_latest.parquet`
- `data_curated/hf_archive/snapshot_20260204-234429Z/canonical/comments_latest.parquet`
- `data_curated/hf_archive/snapshot_20260204-234429Z/canonical/agents_latest.parquet`

QC evidence:

- `outputs/pwj_pipeline/item_2/attempt_1_20260204-234429Z/qc_results.json` (PASS checks)
- `outputs/pwj_pipeline/item_2/attempt_1_20260204-234429Z/curation_manifest.json` (parse stats, raw row counts)

Observed event-time coverage (from `created_at_utc`):
- posts: 2026-01-28 → 2026-02-04
- comments: 2026-01-31 → 2026-02-04

Important note for analysis:
- `dt` partitions are derived from `dump_date` (export date), so **always filter by `created_at_utc` for calendar-time analysis**.

#### B) Moltbook official API pilot (schema + collection mechanics validated)

Curated pilot outputs:

- `data_curated/moltbook/` (feed_snapshots, posts, comments partitioned by run_id/dt)

Run manifests:

- `outputs/pwj_pipeline/item_3/attempt_1_20260205-152619Z/run_manifest.json`
- `outputs/pwj_pipeline/item_3/attempt_1_20260205-152619Z/report.md`

This is a **small pilot** (not the 72h/60s cadence run described in the plan).

#### C) Reddit pilot via Arctic Shift (pipeline validated; not full comparison)

Curated pilot outputs:

- `data_curated/reddit/submissions/run_id=attempt_1_20260205-201543Z/dt=2026-02-05/part-0.parquet`
- `data_curated/reddit/comments/run_id=attempt_1_20260205-201543Z/dt=2026-02-05/part-0.parquet`

Run manifests:

- `outputs/pwj_pipeline/item_5/attempt_1_20260205-201543Z/run_manifest.json`
- `outputs/pwj_pipeline/item_5/attempt_1_20260205-201543Z/report.md`
- `outputs/pwj_pipeline/item_5/attempt_1_20260205-201543Z/validation_results.json` (PASS)

Pilot scope:
- UTC window: 2026-02-01T00:00:00Z → 2026-02-02T00:00:00Z
- Subreddits: programming, python, MachineLearning

## 2) Secrets + collaboration guardrails (must follow)

- `.env.local` is gitignored and must **never** be copied into `outputs/` or `data_*`.
- Never print/write keys or auth headers. Use placeholders only (e.g., `Authorization: Bearer $MOLTBOOK_API_KEY`).
- Do not delete files (tracked or untracked).
- Data (`data_raw/`, `data_curated/`, `data_features/`) and run outputs (`outputs/`) are gitignored by default; they will be visible locally but **will not appear for teammates who freshly clone the repo** unless shared separately.

## 3) Are we ready to start analysis now?

### Yes: Moltbook-only analysis can start immediately

The HF curated tables contain everything needed for:

- **Thread event logs** (plan §6.1): `post_id`, `comment_id`, `parent_id`, `agent_id`, `created_at_utc`
- **Thread metrics** (plan §6.2): lifetime, half-life empirical, depth, star-shape, branching proxies
- **Agent activity time series** (plan §6.3): counts per 5/15 min; autocorrelation; periodogram
- **Interaction graph** (plan §6.4): reply edges i→k; reciprocity; dyads
- **Re-entry features** (plan §6.5): per-agent per-thread sequences and re-entry gaps

### Critical caveats (do not ignore)

1) **Right-censoring near the end of the window**  
   The dataset ends around 2026-02-04 19:51Z; threads active near the end are censored. Survival/half-life estimation must account for censoring (or exclude boundary-near threads).

2) **`dump_date` vs `created_at_utc`**  
   HF exports are incremental with backfill; never treat `dump_date` as event time.

3) **Possible coverage gaps / missingness**  
   There is a large gap in `comments.created_at_utc` between ~2026-01-31 10:38Z and ~2026-02-02 04:21Z in the current canonical slice. This could be real low activity or an artifact; either way it affects periodicity/autocorrelation claims unless handled explicitly (segmenting, missingness model, or restricting windows).

## 4) What data is missing (and why it matters)

### Missing #1 — Longer Moltbook API “exposure/visibility” snapshots (plan §3)

**Why it matters:** H1–H3 talk about periodicity and early-engagement/exposure controls. HF archive is excellent for structure and timestamps, but it is not a high-frequency exposure instrument. A 72h snapshot series enables:

- a better estimate of an availability baseline b(t) (model Eq. 3/17),
- exposure/confounding controls (feed rank, visibility),
- validation of HF completeness and timing resolution.

**What we have:** a small pilot (`data_curated/moltbook/feed_snapshots` etc.).  
**What we need:** a long run (e.g., 72h at 60s interval per sort).

### Missing #2 — Scaled Reddit comparison corpus aligned to Moltbook window (plan §5)

**Why it matters:** The paper’s strongest claims are comparative (Moltbook vs humans). A one-day pilot cannot support:

- stable half-life distribution comparisons,
- matched-sample comparisons (topic + early engagement),
- robust depth/reciprocity comparisons across topics/time.

**What we have:** a validated Arctic Shift pilot (1 day, 3 subreddits).  
**What we need:** larger window aligned to the Moltbook `created_at_utc` window (and better per-submission comment-tree completeness if feasible).

### Missing #3 — Matching + “early engagement” measurement plan

The plan suggests matching on “comment_count in first hour” and “score in first hour”. Neither HF nor the Arctic Shift pilot provides true “first hour” time series. We need one of:

- repeated snapshots (both platforms), or
- official API collection strategies, or
- a revised matching scheme for the quick-turnaround paper (documented explicitly).

## 5) Proposed team split (3 subteams)

### Subteam A — Analysis on HF curated Moltbook data (start now)

**Objective:** produce first-pass paper results/figures from HF archive: depth/shape, decay/half-life, periodicity, heterogeneity.

**Inputs:**
- `data_curated/hf_archive/snapshot_20260204-234429Z/{posts,comments,agents,submolts,snapshots,word_frequency}`
- canonical dedup: `data_curated/hf_archive/snapshot_20260204-234429Z/canonical/*_latest.parquet`

**Deliverables (write to gitignored `data_features/` + paper assets):**
- `data_features/thread_events/*.parquet` (plan §6.1)
- `data_features/thread_metrics/*.parquet` (plan §6.2)
- `data_features/agent_timeseries/*.parquet` (plan §6.3)
- `data_features/interaction_graph/*.parquet` (plan §6.4)
- initial figures/tables into `outputs/figures/` and `outputs/tables/` (then promote final versions into `paper/figures/` and `paper/tables/`)

**Critical decisions to document early:**
- censoring policy (how close to dataset end counts as censored),
- which time window(s) are safe given potential missingness,
- whether to analyze canonical-only vs full tables (canonical is deduped but narrower; full tables have richer columns).

### Subteam B — Longer Moltbook API snapshot run (72h feed snapshots)

**Objective:** collect high-frequency feed snapshots (+ limited post/comment details) for exposure/availability signals.

**Inputs/Prereqs:**
- `MOLTBOOK_API_KEY` in environment or `.env.local` (never print it)
- stable long-running environment (tmux/systemd)

**Recommended approach (keep simple):**
- Start with a 12h shakedown run, then extend to 72h.
- Use `analysis/03_moltbook_api_collect.py` with `--snapshots` and `--interval-seconds 60`.

Example command skeleton (edit caps as needed; do not paste keys):

```bash
set -a; [ -f .env.local ] && source .env.local; set +a
ATTEMPT_ID=attempt_long_$(date -u +%Y%m%d-%H%M%SZ)
python analysis/03_moltbook_api_collect.py \\
  --attempt-id \"$ATTEMPT_ID\" \\
  --date $(date -u +%Y-%m-%d) \\
  --mode auto \\
  --base-url https://www.moltbook.com/api/v1 \\
  --sorts hot,new \\
  --limit 100 \\
  --snapshots 4320 \\
  --interval-seconds 60 \\
  --max-post-details 0 \\
  --max-comment-posts 0 \\
  --include-submolts \\
  --out-raw-root data_raw/moltbook_api
```

Then curate/validate per day using:
- `analysis/03_moltbook_api_curate.py`
- `analysis/03_moltbook_api_validate.py`

**Deliverables:**
- raw logs + JSON: `data_raw/moltbook_api/YYYY-MM-DD/...`
- curated snapshots: `data_curated/moltbook/feed_snapshots` (and optional posts/comments)
- run manifests under `outputs/pwj_pipeline/item_3/...` or a new `outputs/` run folder if run manually

### Subteam C — Scaled Reddit corpus via Arctic Shift (credential-free; non-official)

**Objective:** collect a larger Reddit dataset aligned to the Moltbook window using Arctic Shift without waiting for Reddit OAuth.

**Inputs:**
- Arctic Shift endpoints (headers required to avoid 403):
  - `https://arctic-shift.photon-reddit.com/api/posts/search`
  - `https://arctic-shift.photon-reddit.com/api/comments/search`
  - required headers: `User-Agent` + `Referer: https://arctic-shift.photon-reddit.com/download-tool`

**Current tooling:**
- Collector: `analysis/05_reddit_collect.py` (time-window + subreddit pagination)
- Curator: `analysis/05_reddit_curate.py`
- Validator: `analysis/05_reddit_validate.py`

**What to extend next (simple path):**
- Expand the window to match Moltbook’s active period (e.g., 2026-01-31 → 2026-02-04) and add a small, justified set of subreddits mapped to Moltbook submolts.
- Keep request logs header-free; avoid printing any post/comment text.

**What to consider (if time allows):**
- Improve per-thread completeness by collecting comments *by submission* (if Arctic Shift supports `link_id=` style queries; confirm first). If not feasible, quantify completeness with parent-missing metrics and report limitations.

**Deliverables:**
- raw: `data_raw/reddit/YYYY-MM-DD/...` + request logs
- curated: `data_curated/reddit/{submissions,comments}` (partitioned by run_id/dt)
- validation reports + run manifests in `outputs/pwj_pipeline/item_5/...` or equivalent

## 6) Git / repo housekeeping (do this before teammates build on it)

As of the latest checks, these are uncommitted:

- `analysis/05_reddit_collect.py`
- `analysis/05_reddit_curate.py`
- `analysis/05_reddit_validate.py`
- `schemas/reddit_schema.json` (now populated from real raw samples)

Recommended: review and commit these code/schema updates (do **not** add `data_raw/`, `data_curated/`, or `outputs/`).

## 7) Quick “where to look” index

- Core model writeup: `docs/attention_dynamics_model.md`
- Data collection plan/spec: `docs/data_collection_plan.md`
- PWJ pipeline how-to: `docs/pwj_pipeline.md`
- Item 2 (HF) evidence: `outputs/pwj_pipeline/item_2/attempt_1_20260204-234429Z/`
- Item 3 (API pilot) evidence: `outputs/pwj_pipeline/item_3/attempt_1_20260205-152619Z/`
- Item 5 (Reddit pilot) evidence: `outputs/pwj_pipeline/item_5/attempt_1_20260205-201543Z/`

</handoff_report>

