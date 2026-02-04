<data_collection_plan>

## 0) Working assumptions and “equation map” for model–data linkage

This plan assumes the modeling section defines an **attention‑clock Hawkes/branching** thread model with (i) exponential interaction decay, (ii) branching comment trees, (iii) heterogeneous agents with periodic “check‑in” behavior, and (iv) re‑entry/self‑excitation within threads.

To make the plan executable without ambiguity, we will reference the following **core equations** (these should correspond to the modeling section; if numbering differs, match by the equation content):

* **(Eq. 3) Reply intensity / interaction decay kernel**
  Exponential decay at lag (u=t-t_m>0): (g_i(u)=\alpha_i e^{-\beta_i u}). Used inside the conditional intensity (\lambda(t)).
* **(Eq. 4) Half‑life relation**
  (t_{1/2,i}=\ln 2/\beta_i).
* **(Eq. 7) Branching ratio (offspring mean)**
  (n_i=\int_0^\infty g_i(u),du=\alpha_i/\beta_i) and/or (n=\mathbb{E}[n_i]).
* **(Eq. 12) Expected depth (or depth tail) under branching approximation**
  Example: (\mathbb{E}[D]\approx n/(1-n)) for (n<1), or (\Pr(D\ge d)\approx n^d).
* **(Eq. 17) Periodic baseline / attention clock**
  Baseline activity (b(t)) has periodic component with period (\tau) and amplitude (\kappa) (e.g., harmonic regression).
* **(Eq. 18) Agent re‑entry hazard**
  Re‑entry propensity after last participation (s): (r_{i,j}(t)=\rho_i+\eta_i e^{-\beta^{(r)}_i(t-s)}) (possibly modulated by (b(t))).
* **(Eq. 23) Periodicity signature**
  A measurable statistic capturing mass at frequency (1/\tau) (e.g., spectral peak / autocorrelation).

### Parameters requiring data (minimum set)

* **Decay/attention**: (\beta_i) (Eq. 3), half‑life (t_{1/2,i}) (Eq. 4)
* **Reply propensity / attractiveness**: (\alpha_i) (Eq. 3)
* **Branching**: (n_i, n) (Eq. 7), depth prediction (Eq. 12)
* **Re‑entry**: (\rho_i, \eta_i, \beta_i^{(r)}) (Eq. 18)
* **Clock / periodicity**: (\tau, \kappa) (Eq. 17), periodicity statistic (Eq. 23)

### Hypotheses requiring data (from proposal)

* **H1 (Short half‑life + clock spike)**: Moltbook threads show shorter half‑life and stronger periodicity at (\tau) than Reddit matched threads. (Eq. 4, 17, 23)
* **H2 (More star‑shaped / less reciprocity)**: Moltbook threads have lower depth and lower reciprocity compared to Reddit. (Eq. 7, 12; plus reciprocity metrics)
* **H3 (Topic/submolt moderates persistence)**: “Utility/builder” submolts have longer half‑life and deeper threads. (Eq. 4, 12)
* **H4 (Heterogeneous sustaining agents)**: A small set of agents account for a disproportionate share of long‑lived threads; these agents have higher (\alpha_i), higher (\rho_i), and stronger periodic signatures. (Eq. 3, 18, 23)

---

## 1) Data architecture (execute first)

### 1.1 Storage layout (raw → curated)

* **Raw layer** (immutable): store every API response and HF parquet snapshot exactly as retrieved.

  * `data_raw/moltbook_api/YYYY-MM-DD/{endpoint}/{timestamp}.json`
  * `data_raw/hf_archive/{table}/snapshot_{pull_timestamp}/...`
  * `data_raw/reddit/YYYY-MM-DD/{subreddit or query}/...`
* **Curated layer** (normalized tables): Parquet partitioned by day and table.

  * `data_curated/moltbook/{agents,posts,comments,submolts,feed_snapshots}/dt=YYYY-MM-DD/*.parquet`
  * `data_curated/reddit/{submissions,comments}/dt=YYYY-MM-DD/*.parquet`
* **Analytical layer** (derived features for modeling):

  * `data_features/thread_events/*.parquet` (event logs with parent links + lags)
  * `data_features/agent_timeseries/*.parquet` (agent activity series)
  * `data_features/thread_metrics/*.parquet` (half‑life, depth, branching, reciprocity)

### 1.2 Canonical IDs and time

* **IDs**: preserve platform IDs as strings (UUIDs on Moltbook appear common).
* **Timestamps**: convert all timestamps to **UTC** `datetime64[ns, UTC]` in curated tables.
* Store both:

  * `created_at_utc` (parsed)
  * `created_at_raw` (string) for audit

### 1.3 Schema discovery step (mandatory on day 1)

Even if we list expected fields below, the team should run:

* HF: `dataset.features` / `dataset.column_names`
* API: log JSON keys for each endpoint and generate a schema manifest (`schemas/moltbook_api_schema.json`)
* Reddit: schema manifest (`schemas/reddit_schema.json`)

This prevents breakage if schemas evolve.

---

## 2) Data source A — Moltbook Observatory Archive (Hugging Face)

### A.1 Source identification

* **Dataset name (HF identifier):** `SimulaMet/moltbook-observatory-archive` ([Hugging Face][1])
* **Access:** public download via Hugging Face Datasets / parquet files
* **License:** MIT (per dataset card) ([Hugging Face][1])
* **Update cadence:** incremental exports are being pushed frequently (commit messages show “Incremental export …” within hours). ([Hugging Face][2])

### A.2 What’s available (tables/subsets + scale)

HF viewer lists **6 subsets** and row counts:

* `agents` (~25.6k)
* `comments` (~226k)
* `posts` (~120k)
* `snapshots` (~114)
* `submolts` (~3.68k)
* `word_frequency` (~15.3k) ([Hugging Face][3])

### A.3 Data elements to collect (exact fields where verified)

#### A.3.1 `agents` subset — **verified fields**

Collect all fields:

* `id` (string UUID)
* `name` (string)
* `description` (string)
* `karma` (int64)
* `follower_count` (int64)
* `following_count` (int64)
* `is_claimed` (int64; treat as boolean)
* `owner_x_handle` (string; may be null)
* `first_seen_at` (string timestamp; parse to UTC)
* `last_seen_at` (string timestamp; parse to UTC)
* `created_at` (string timestamp; parse to UTC; may be null)
* `avatar_url` (string; may be null)
* `dump_date` (string `YYYY-MM-DD`) ([Hugging Face][3])

**Model linkage**

* (\alpha_i) (Eq. 3) heterogeneity covariates: `karma`, `follower_count`, `is_claimed`
* (\rho_i) (Eq. 18) proxy covariates: `last_seen_at - first_seen_at`, activity counts joined from comments
* H4: “sustaining agents” correlates with `karma`, `follower_count`

#### A.3.2 `posts`, `comments`, `submolts`, `snapshots`, `word_frequency` — fields to collect

Because we cannot guarantee field lists without running schema discovery, the **actionable rule** is: **ingest all columns** in each subset, then ensure the following **required variables** exist (and if naming differs, map via a schema dictionary).

Minimum required variables by table:

* `posts`: `id`, `created_at` (or equivalent), `author_id` (or `author_name`), `submolt`/community, `title`, `content`/`url`, `score`/`karma`, `comment_count`
* `comments`: `id`, `post_id`, `parent_id` (nullable), `author_id`/`author_name`, `created_at`, `content`, `score`
* `submolts`: `name`, `display_name`, `description`, `subscriber_count`/`member_count`, `created_at`
* `snapshots`: timestamps + aggregate counters (posts/comments/agents) and/or frontpage snapshots
* `word_frequency`: timestamped `(token, count)` (or similar) for topic/time controls

**Model linkage**

* Eq. 3 (kernel): needs `comments.created_at`, `comments.parent_id`, `comments.author_id`, `comments.post_id`
* Eq. 7 / Eq. 12 (branching/depth): needs `comments.parent_id`, `comments.post_id`
* Eq. 17 / Eq. 23 (periodicity): needs `comments.created_at` aggregated platform‑wide, by submolt, and by agent
* H1–H3: needs `submolt` assignment for each post/comment

### A.4 Collection procedure (step‑by‑step)

#### A.4.1 One‑time pull (historical baseline)

1. Create a Python env (>=3.10) with:

   * `datasets`, `pyarrow`, `polars` or `pandas`, `duckdb`
2. Load each subset and write to curated Parquet:

   ```python
   from datasets import load_dataset
   for cfg in ["agents","posts","comments","submolts","snapshots","word_frequency"]:
       ds = load_dataset("SimulaMet/moltbook-observatory-archive", cfg, split="archive")
       ds.to_parquet(f"data_raw/hf_archive/{cfg}/archive.parquet")
   ```
3. Parse timestamps and materialize curated tables partitioned by `dump_date` (and/or `created_at_utc` date).

**Expected volume**

* Dataset card indicates ~143MB parquet + ~390k rows overall. ([Hugging Face][1])

#### A.4.2 Incremental refresh (daily)

Because the dataset is updated incrementally, run a nightly job:

* Pull latest HF repo snapshot (via `datasets` or `huggingface_hub`).
* Detect new `dump_date` partitions and append.

Use the dataset’s own export logic as reference:

* The repo includes logic that creates a `dump_date` and writes parquet partitioned by date, with table‑specific backfill (posts/comments/agents 7 days, submolts 30). ([Hugging Face][4])

### A.5 Quality checks (HF archive)

1. **Row‑level uniqueness**

   * For each table, check primary key uniqueness at least within a `dump_date`.
   * For `agents`, `id` should be unique per `dump_date`.
2. **Timestamp parse rate**

   * ≥99% of `created_at` / `first_seen_at` / `last_seen_at` should parse; log failures.
3. **Referential integrity**

   * Every `comments.post_id` exists in `posts.id` (allow a small mismatch if the archive lags).
   * Every non‑null `comments.parent_id` exists in `comments.id`.
   * Every `posts.author_id` / `comments.author_id` exists in `agents.id` (or map via `name`).
4. **Monotonicity / plausibility**

   * `last_seen_at >= first_seen_at`
   * `created_at` not in the future relative to pull time
5. **Backfill overlap**

   * Because posts/comments can be re‑exported with backfill, deduplicate by (`id`, max(`dump_date`)) when building a canonical “latest state” table.

### A.6 Timeline for stream A

* Day 1–2: one‑time baseline pull + schema discovery + curated build
* Day 3: integrity checks + first derived feature tables
* Ongoing: nightly incremental refresh (1–2 hours dev; then automated)

---

## 3) Data source B — Moltbook Official REST API (live + high‑frequency feed snapshots)

### B.1 Source identification

* **Repository / documentation:** `moltbook/api` on GitHub ([GitHub][5])
* **Base URL (production):**

```text
https://www.moltbook.com/api/v1
```

([GitHub][5])

* **Auth:** Bearer token in header:

```text
Authorization: Bearer YOUR_API_KEY
```

([GitHub][5])

### B.2 Why we need the live API (beyond HF archive)

HF archive gives broad coverage, but live API crawling is needed for:

* **Higher‑frequency timing** to detect periodic spikes at (\tau) (Eq. 17, 23) with better resolution than daily `dump_date`
* **Feed snapshotting** to approximate exposure/visibility (needed to reduce confounding in H1–H3)
* **Real‑time validation** that archive hasn’t silently missed an event class

### B.3 Endpoints to collect (exact paths)

#### B.3.1 Agent registration (for researcher “observer agent”)

```text
POST /agents/register
```

Response includes `api_key`, `claim_url`, `verification_code`. ([GitHub][5])

#### B.3.2 Agents (metadata for heterogeneity)

```text
GET /agents/me
PATCH /agents/me
GET /agents/status
GET /agents/profile?name=AGENT_NAME
```

([GitHub][5])

#### B.3.3 Posts (global feed + post details)

```text
GET /posts?sort=hot&limit=25
GET /posts?sort=new&limit=25
GET /posts?sort=top&limit=25
GET /posts?sort=rising&limit=25
GET /posts/:id
```

([GitHub][5])

#### B.3.4 Comments (thread tree + timing)

```text
GET /posts/:id/comments?sort=top
GET /posts/:id/comments?sort=new
GET /posts/:id/comments?sort=controversial
```

([GitHub][5])

#### B.3.5 Voting (optional, if timestamps not available elsewhere)

```text
POST /posts/:id/upvote
POST /posts/:id/downvote
POST /comments/:id/upvote
```

([GitHub][5])
(Do **not** upvote/downvote in research collection; only record vote counts if returned in GET endpoints.)

#### B.3.6 Submolts (topic/community structure)

```text
GET /submolts
GET /submolts/:name
POST /submolts/:name/subscribe
DELETE /submolts/:name/subscribe
```

([GitHub][5])

#### B.3.7 Following + personalized feed (observer‑agent exposure approximation)

```text
POST /agents/:name/follow
DELETE /agents/:name/follow
GET /feed?sort=hot&limit=25
```

([GitHub][5])

#### B.3.8 Search (topic matching + sampling)

```text
GET /search?q=...&limit=25
```

([GitHub][5])

### B.4 Data elements to collect (API responses)

For each endpoint, store:

* `http_status`, `request_url`, `request_params`, `retrieved_at_utc`
* full JSON payload (raw)
* parsed fields into curated tables (below)

**B.4.1 Feed snapshots table** (critical for Eq. 17/23 + confounding control)
From `GET /posts?sort=hot&limit=25` and `GET /posts?sort=new&limit=25`:

* `snapshot_time_utc` (when fetched)
* For each returned post:

  * `post_id`
  * `rank` (position in list 1..limit)
  * `sort` (`hot`/`new`/`top`/`rising`)
  * `score`, `comment_count` (if present)
  * `submolt`, `author_id/name` (if present)

**Model linkage**

* Eq. 17 & Eq. 23: estimate periodicity by looking at changes in feed composition and subsequent comment bursts
* H1/H3: use rank/exposure controls when comparing half‑life across submolts/topics

**B.4.2 Post details table**
From `GET /posts/:id`:

* `post_id`
* `created_at_utc`
* `title`, `content` or `url`
* `submolt`
* `author_id/name`
* `score`, `comment_count`

**Model linkage**

* Thread start time; necessary for half‑life computation (Eq. 4)
* Controls for topic (H3) and matching to Reddit (H1/H2)

**B.4.3 Comment events table**
From `GET /posts/:id/comments?sort=new` (use `new` to best approximate chronological):

* `comment_id`
* `post_id`
* `parent_id` (nullable)
* `author_id/name`
* `created_at_utc`
* `content`
* `score`

**Model linkage**

* Eq. 3: estimate (\alpha_i,\beta_i) from reply lags conditioned on parent comment time
* Eq. 7/12: estimate (n_i,n) and depth distributions from tree structure
* Eq. 17/23: build time series of comments to detect attention clock spikes
* Eq. 18: infer re‑entry from repeated participation by same agent in same thread

### B.5 Collection procedure (step‑by‑step)

#### B.5.1 Obtain an API key (observer agent)

1. Use:

```bash
curl -X POST "https://www.moltbook.com/api/v1/agents/register" \
  -H "Content-Type: application/json" \
  -d '{"name":"research_observer_01","description":"Academic measurement agent. No posting."}'
```

(Then store `api_key` securely.)
([GitHub][5])

2. Set environment variable in crawler machine:

* `MOLTBOOK_API_KEY=...`

#### B.5.2 Build a “post universe” index (every 5–10 minutes)

Goal: discover new posts quickly and track their exposure trajectory.

1. Every 5 minutes:

   * Fetch `/posts?sort=new&limit=25`
   * Fetch `/posts?sort=hot&limit=25`
2. Parse `post_id`s and insert into `post_universe` table with first‑seen time.
3. Keep a rolling window (e.g., last 7 days of new posts) for intensive comment crawling.

**Rate limiting / error handling**

* Implement retries with exponential backoff for HTTP 429/5xx.
* Persist a `request_log` table: (endpoint, params, status, latency_ms, bytes, error).

#### B.5.3 Crawl comments for active posts (every 10–20 minutes per active post)

Heuristic: active if (a) appeared in hot feed in last 2 hours, or (b) comment_count increases.

1. For each active post:

   * Fetch `/posts/:id/comments?sort=new`
2. Store raw payload + parse into `comment_events` table.
3. Deduplicate comments by `comment_id`.

**Critical detail for Eq. 3 (reply lags)**

* Preserve **parent–child links** exactly as returned (`parent_id`).
* Compute reply lags later in features stage, not during ingestion.

#### B.5.4 High‑frequency clock detection run (2–3 days, then weekly)

To estimate (\tau,\kappa) robustly:

* For 72 hours, sample:

  * `/posts?sort=hot&limit=25` every **60 seconds**
  * `/posts?sort=new&limit=25` every **60 seconds**
* In parallel, track comment arrivals for posts in feed every 2–5 minutes.

**Model linkage**

* Eq. 17: estimate (\tau) by spectral peak on event times
* Eq. 23: compute periodicity statistic on inter‑event times and aggregated counts

### B.6 Expected data volume & formats

* Feed snapshots: 25 posts × 2 sorts × 60/min × 1440 min/day ≈ 72,000 rows/day (small)
* Comment crawls: depends on activity; store as compressed JSON + parsed Parquet
* Storage: expect <5–10GB for a month including raw payloads (manageable)

### B.7 Quality checks (API stream)

1. **Completeness of post universe**

   * Compare count of posts discovered via API crawling vs HF archive for overlapping days (join by `post_id`).
2. **Event duplication**

   * `comment_id` should be unique; enforce dedup.
3. **Chronology sanity**

   * For each post, comment timestamps should be ≥ post created time (allow rare clock skew).
4. **Schema drift**

   * If JSON keys change, log diff and bump schema version; keep raw payload for re‑parse.

### B.8 Timeline for stream B

* Day 1: register observer agent + scaffold crawler + request logs
* Day 2–3: run 72‑hour high‑frequency sampling for (\tau)
* Week 1–4: continuous crawling (lightweight) + nightly ETL

---

## 4) Data source C — Open‑source Moltbook API codebase (for reproducibility + controlled experiments)

### C.1 Source identification

* **Repo:** `https://github.com/moltbook/api` ([GitHub][5])
* **Purpose for us:** verify endpoint behavior, run local instance, instrument for “ground truth” on exposure mechanisms.

### C.2 What to collect from the repo (non‑content)

1. **API spec confirmation** (already in README):

   * Base URL, auth mechanism, endpoint paths ([GitHub][5])
2. **Environment variable defaults** (from `.env.example`) and migration scripts.
3. **Rate limiting logic** (if implemented in middleware; helps set crawler pacing).
4. **Feed ranking logic** (to interpret “hot” vs “new” and connect to exposure confounds).

### C.3 Procedure (controlled local run)

From README quick start:

```bash
git clone https://github.com/moltbook/api.git
cd api
npm install
cp .env.example .env
npm run db:migrate
npm run dev
```

([GitHub][5])

**Local instrumentation goal**

* Add logging around feed ranking and request timing to generate synthetic data where “clock” period is known. This enables a pipeline test:

  * Can we recover (\tau) (Eq. 17) and (\beta) (Eq. 3) from synthetic logs?

**Model linkage**

* Validates parameter recovery methods before applying to real data.

### C.4 Timeline

* Week 1: local run for pipeline smoke tests
* Week 2: optional instrumentation if exposure confounding becomes central

---

## 5) Data source D — Reddit comparison corpus (for H1/H2 external validation)

### D.1 Source identification (recommended)

* **Primary:** Reddit official API via OAuth + PRAW (Python Reddit API Wrapper).
* **Secondary (optional):** public historical archives / BigQuery, only if permitted and stable.

### D.2 Data elements to collect

#### Submissions (match to Moltbook posts)

* `submission_id`
* `created_utc`
* `subreddit`
* `title`
* `selftext` / `url`
* `score`
* `num_comments`

#### Comments (tree + timing)

* `comment_id`
* `submission_id`
* `parent_id`
* `author`
* `created_utc`
* `body`
* `score`

### D.3 Sampling strategy (to make comparisons meaningful)

1. **Topic matching**

   * Option A: map Moltbook submolts to subreddits manually (e.g., “builders” ↔ r/programming, etc.)
   * Option B: embed titles/first 200 tokens of content (Moltbook vs Reddit) and nearest‑neighbor match.
2. **Early engagement matching**

   * Match on:

     * comment_count in first hour
     * score in first hour (if available)
     * post length bucket
3. **Time window matching**

   * Use the same calendar window as the Moltbook dataset for seasonality control.

### D.4 Model linkage

* **H1 / Eq. 4:** compare half‑life distributions for matched threads.
* **Eq. 12 / H2:** compare depth distributions and branching ratio estimates.
* **Eq. 23 / H1:** test whether Reddit lacks a strong 4‑hour periodic spectral peak relative to Moltbook.

### D.5 Quality checks

* Ensure comment trees are complete (Reddit API sometimes returns “more comments”; handle via recursive fetch).
* Normalize timestamps to UTC.
* Exclude deleted/removed authors/bodies or mark with flags.

### D.6 Timeline

* Week 1: build Reddit collector + small pilot sample (e.g., 2–3 subreddits)
* Week 2–3: full matched sample construction

---

## 6) Feature construction required for parameter estimation (ties every field to a parameter)

This section is the **bridge** between raw collection and model estimation. The team should implement these derived datasets as part of ETL.

### 6.1 Thread event log (input to Eq. 3 / Hawkes estimation)

Create `thread_events` with one row per comment:

* `thread_id` (= `post_id`)
* `event_id` (= `comment_id`)
* `parent_event_id` (= `parent_id`)
* `author_id`
* `t_event` (= `created_at_utc`)
* `generation` (computed via parent links)
* `lag_to_parent` = `t_event - t_parent` (seconds)

**Estimates enabled**

* (\beta_i): fit exponential decay of reply lags conditional on parent (Eq. 3)
* (\alpha_i): fit excitation amplitude / expected offspring (Eq. 3, Eq. 7)
* (n): mean offspring count / integral of kernel (Eq. 7)

### 6.2 Thread metrics table (input to H1–H3 and Eq. 4/12)

For each `thread_id`:

* `t0` = post created time
* `t_last_comment`
* `thread_lifetime` = `t_last_comment - t0`
* `half_life_empirical` = time until cumulative comments reaches 50% of final count
* `depth_max`, `depth_mean`
* `branching_ratio_empirical` = mean out‑degree across comments
* `star_shape_index` = fraction of comments that reply to root vs non‑root
* `reciprocity_index` (see 6.4)

**Hypothesis tests enabled**

* H1: compare `half_life_empirical` Moltbook vs Reddit; relate to (\beta) via Eq. 4.
* H2: compare `depth_max`, `star_shape_index` vs Reddit; test Eq. 12 predictions.
* H3: regress `half_life_empirical` and `depth_max` on submolt/topic.

### 6.3 Agent activity time series (input to Eq. 17/18/23)

For each `author_id`:

* `count_per_5min` or `count_per_15min`
* `autocorrelation` at lags 1h…24h
* `spectral_peak_freq`, `spectral_peak_power` near 4 hours

**Estimates enabled**

* (\tau,\kappa) (Eq. 17): harmonic regression / periodogram on aggregated counts.
* (\rho_i) (Eq. 18): baseline activity rate (mean comments per hour).
* H4: identify agents with high periodic power and relate to sustaining behavior.

### 6.4 Interaction graph (agent–agent reciprocity)

Build directed edge list where an edge (i\to k) exists if agent (i) replies to a comment by agent (k):

* `src_agent_id` (replier)
* `dst_agent_id` (parent author)
* `thread_id`
* `t_event`

Compute:

* `reciprocity_index_thread` = fraction of dyads with both directions present within a thread
* `reciprocity_index_global` similarly over a window
* `assortativity` by karma/follower_count

**Model linkage**

* H2: “less reciprocity” claim is directly tested here.
* Eq. 18 re‑entry: also compute “self‑reply” and repeated dyads.

### 6.5 Re‑entry features (Eq. 18)

For each pair (agent i, thread j):

* `t_first` = first comment time of i in j
* `t_last` = last comment time of i in j
* `num_entries` = count of “sessions” of participation (define session break if gap > X minutes)
* `reentry_gaps` = gaps between sessions

**Estimates enabled**

* (\eta_i, \beta_i^{(r)}): fit decay of hazard of returning after leaving thread (Eq. 18)
* H4: sustaining agents show higher (\eta_i) and slower decay (\beta_i^{(r)})

---

## 7) Parameter estimation plan (what data fields → what method)

### 7.1 Estimate (\beta_i) and (\alpha_i) (Eq. 3) from reply lags and offspring

**Required fields**

* From `comments`: `comment_id`, `parent_id`, `created_at_utc`, `author_id`
* From `posts`: `post_id`, `created_at_utc` (thread start)

**Method (short‑timeframe feasible)**

* **Stage 1 (per‑comment survival approximation):**
  For each parent comment (m) by agent (i), treat time‑to‑first‑reply as survival time; fit exponential with rate tied to (\beta_i) (coarse).
* **Stage 2 (Hawkes MLE):**
  Fit Hawkes with exponential kernels per agent group (e.g., top‑K agents + pooled tail) to stabilize.
* **Stage 3 (offspring regression):**
  Empirical offspring count per comment by agent i: (\widehat{n_i}=\mathbb{E}[\text{#replies to i’s comments}]). Combine with (\widehat{\beta_i}) to get (\widehat{\alpha_i}=\widehat{n_i}\widehat{\beta_i}) (Eq. 7).

### 7.2 Estimate (\tau,\kappa) (Eq. 17) and periodicity statistic (Eq. 23)

**Required fields**

* `comments.created_at_utc` (all)
* Optional: `feed_snapshots.snapshot_time_utc` + post ranks (for exposure control)

**Method**

* Aggregate counts per minute / 5‑minute bin.
* Compute periodogram; peak near 4 hours yields (\widehat{\tau}).
* Fit harmonic regression: (y_t = \mu + a\cos(2\pi t/\tau)+b\sin(2\pi t/\tau)); amplitude gives (\kappa).

### 7.3 Estimate (\rho_i,\eta_i,\beta_i^{(r)}) (Eq. 18)

**Required fields**

* Per‑agent per‑thread event sequences from `thread_events`

**Method**

* Define “session” as contiguous commenting with gaps ≤ 30 minutes (sensitivity analysis at 10/60).
* Model re‑entry hazard as exponential decay after session end:

  * Estimate (\beta_i^{(r)}) from distribution of re‑entry gaps.
  * (\eta_i) from re‑entry probability mass (fraction of sessions that have a subsequent session).
  * (\rho_i) from baseline commenting rate outside re‑entry windows.

### 7.4 Branching ratio (n) and depth prediction (Eq. 7, Eq. 12)

**Required fields**

* `comments.parent_id` and per‑thread tree reconstruction

**Method**

* ( \widehat{n} = \frac{#\text{non‑root comments}}{#\text{all comments}} ) (approx) or mean out‑degree.
* Compute empirical depth distribution; compare to (n^d) tail or expected depth formula.

---

## 8) Data completeness, anomaly detection, and validation

### 8.1 Cross‑source reconciliation (HF vs API)

For overlapping time windows:

* Compare:

  * number of posts/day
  * number of comments/day
  * distribution of comment lags
* Sample 100 posts from API universe; verify they exist in HF archive within ±7 days (consistent with backfill for posts/comments in export logic). ([Hugging Face][4])

### 8.2 Detect “bursts” and abnormal agents

* Flag agents with extreme posting rates (top 0.1%) and inspect for bots/spam.
* Flag threads with unusually high depth or size; keep but mark for robustness checks.

### 8.3 Handling missingness

* If `author_id` missing but `author_name` present, build a mapping table:

  * `agents.id ↔ agents.name` from HF `agents` subset
* If `parent_id` missing for some replies, treat as root‑level but add `parent_missing_flag=1`.

### 8.4 Versioning / reproducibility

* Every daily build should output:

  * `schemas/*.json` (schema manifests)
  * `run_metadata.json` with git commit hash of analysis code + pull timestamps
  * hash checksums of curated parquet partitions

---

## 9) Parallelizable work plan and timeline (4–8 weeks)

### Week 1

* **Team A (HF ingest):** pull HF archive, build curated tables, integrity checks
* **Team B (API crawl):** obtain API key, implement feed snapshot + comment crawler, start 72‑hour high‑freq run
* **Team C (Reddit pilot):** implement Reddit collector + pilot matched sample

### Week 2

* Build derived feature tables (thread_events, thread_metrics, interaction_graph, agent_timeseries)
* Run first estimates of (\tau) and global (\beta) (pooled) to validate model signal

### Weeks 3–4

* Full parameter estimation: (\beta_i,\alpha_i) for top agents + pooled tail; branching/depth; re‑entry parameters
* Hypothesis tests H1–H4 with robustness (submolts, topic controls, exposure controls using feed snapshots)

### Weeks 5–8 (optional, depending on paper scope)

* Controlled local run of `moltbook/api` for pipeline validation
* Extend comparison beyond Reddit (optional)

---

## 10) Checklist: every collected element maps to model or hypothesis

* **comments.created_at + parent_id + author_id + post_id** → Eq. 3, 4, 7, 12, 17, 18, 23; H1–H4
* **posts.created_at + submolt + content/title** → H1/H3 matching and controls
* **agents.karma/followers/is_claimed** → heterogeneity covariates for (\alpha_i,\rho_i); H4
* **feed snapshots (rank/time/sort)** → exposure control; strengthens causal interpretation for H1/H3
* **submolts metadata** → community‑level heterogeneity; H3
* **Reddit submission/comment trees + times** → external validation; H1/H2; Eq. 4/12/23 comparisons

</data_collection_plan>

[1]: https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive "https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive"
[2]: https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive/tree/main "https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive/tree/main"
[3]: https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive/viewer/ "https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive/viewer/"
[4]: https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive/blob/main/sqlite_to_hf_parquet.py "https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive/blob/main/sqlite_to_hf_parquet.py"
[5]: https://github.com/moltbook/api "https://github.com/moltbook/api"
