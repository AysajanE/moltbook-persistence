# Project checklist (quick turnaround)

## Week 1: ingestion + thread reconstruction

- [ ] Export Moltbook Observatory Archive tables to `data/raw/`
- [ ] Reconstruct comment trees (parent links) + basic sanity checks
- [ ] Compute descriptive “conversation geometry” (depth, branching, reciprocity, re-entry)

## Weeks 2–3: persistence modeling

- [ ] Fit time-to-next-comment models (exponential / Weibull survival)
- [ ] Report interaction half-life overall + by submolt/topic bucket
- [ ] Check periodicity signals (e.g., ~4-hour cadence) in aggregate arrivals

## Weeks 4–6: cross-platform comparisons

- [ ] Collect matched Reddit threads for comparison baselines
- [ ] Compare decay and depth metrics on matched samples
- [ ] Robustness / sensitivity analyses (spam windows, high-reputation agents, etc.)

## Weeks 7–8: writing + release

- [ ] Draft paper in `paper/` and keep `paper/references.bib` current
- [ ] Freeze a reproducible pipeline snapshot (scripts + configs)
- [ ] Prepare an arXiv upload bundle (sources, figures, bib)

