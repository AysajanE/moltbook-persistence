# Background & pointers (updated 2026-02-04)

This project studies conversation persistence and coordination on **Moltbook**, a social network oriented around **AI agent accounts**.

## Primary starting points

- Simon Willison’s writeup and stats snapshot (2026-01-30): https://simonwillison.net/2026/jan/30/moltbook/
- Scott Alexander / Astral Codex Ten early observations:
  - “Moltbook: After The First Weekend”: https://www.astralcodexten.com/p/moltbook-after-the-first-weekend
  - “Best Of Moltbook”: https://www.astralcodexten.com/p/best-of-moltbook
- Community orientation post (LessWrong): https://www.lesswrong.com/posts/y66jnvmyJ4AFE4Z5h/welcome-to-moltbook

## Data + engineering entrypoints

- Moltbook Observatory Archive dataset (Hugging Face): https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
  - Provides multiple relational tables (including `comments` with parent links), plus time-aware snapshots and word-frequency time series.
- Moltbook API (open-source): https://github.com/moltbook/api
  - Useful for schema validation and (optionally) extending beyond observatory snapshots.

## Related work (already on arXiv)

- “Moltbook Observatory: A Large Scale Dataset of an Agentic Social Network” (arXiv:2602.00931): https://arxiv.org/abs/2602.00931
- “On the Emergence of Agentic Social Networks” (arXiv:2602.01012): https://arxiv.org/abs/2602.01012

## Caveats worth tracking

- **Platform volatility:** norms, moderation, and features may change quickly; analyses should be timestamped.
- **Human influence / account compromise:** some activity may be human-driven; focus on interaction dynamics and run sensitivity analyses by time window, agent reputation/age, and abnormal-activity periods.

