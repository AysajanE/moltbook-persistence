# Scripts

Small command-line helpers used by this project.

## Download Moltbook Observatory Archive

```bash
python scripts/download_moltbook_observatory_archive.py --out-dir data/raw/moltbook-observatory-archive
```

## Build arXiv source bundle

```bash
python scripts/build_arxiv_bundle.py --paper-dir paper --out-root outputs/arxiv --check
```

## Run Moltbook live campaign (daily chunks)

```bash
python scripts/run_moltbook_live_campaign.py \
  --campaign-id campaign_20260209 \
  --days 30 \
  --mode live \
  --sorts hot,new \
  --limit 100 \
  --snapshots-per-day 1440 \
  --interval-seconds 60 \
  --max-post-details 0 \
  --max-comment-posts 0 \
  --comment-poll-every-rounds 0 \
  --comment-poll-top-k 0 \
  --include-submolts \
  --compress-raw \
  --keep-uncompressed-request-log \
  --min-free-gb 20
```

Outputs:

- Per-day run manifests and completion manifests under `outputs/ops/<campaign-id>/item3/<attempt_id>/`
- Campaign-level manifest under `outputs/ops/<campaign-id>/item3/campaign_manifest.json`

## Run Moltbook live campaign (comment pulse profile)

```bash
python scripts/run_moltbook_live_campaign.py \
  --campaign-id campaign_20260209_comment_pulse \
  --days 30 \
  --mode live \
  --sorts hot,new \
  --limit 100 \
  --snapshots-per-day 1440 \
  --interval-seconds 60 \
  --max-post-details 500 \
  --max-comment-posts 0 \
  --comment-poll-every-rounds 5 \
  --comment-poll-top-k 25 \
  --include-submolts \
  --compress-raw \
  --keep-uncompressed-request-log \
  --min-free-gb 20
```

Storage estimate for this profile:

- Estimate is assumption-based and depends on comment payload sizes.
- Using observed pilot mean of about 116 KB per comments endpoint response,
  `comment-poll-every-rounds=5` and `comment-poll-top-k=25` implies about
  7200 comment polls/day.
- Rough monthly footprint: about 35–37 GB uncompressed (before gzip), plus/minus
  variability in payload size and API activity.
