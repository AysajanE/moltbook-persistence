# Data sources

## Moltbook (primary)

### Moltbook Observatory Archive (Hugging Face)

Dataset card: https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive

The dataset is organized as multiple tables/subsets (e.g., `agents`, `posts`, `comments`, `submolts`, `snapshots`, `word_frequency`). Use this as the fast-start source for reconstructing **comment trees** and estimating **interaction decay / half-life**.

Repro note:
- Keep the downloaded tables under `data/raw/` and record the download date + any dataset version/hash you use.

### Moltbook API (schema verification / extensions)

Repo: https://github.com/moltbook/api

Use the API to:
- Validate schema assumptions made from the observatory tables.
- Optionally fetch additional recent posts/comments for targeted checks or time-window extensions.

## Reddit (comparison baseline)

Suggested approach:
- Use official Reddit APIs and respect terms of service.
- Match threads to Moltbook posts on topic keywords and early engagement (e.g., score / comment count in the first X minutes) to avoid trivial confounds.

## Link hygiene

Keep a stable reference list in `paper/references.bib` and log any key assumptions/edge cases in `docs/decisions.md`.

