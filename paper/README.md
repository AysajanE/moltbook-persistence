# Paper (LaTeX)

This folder contains the LaTeX source intended for an arXiv submission.

## Build

From the repo root:

```bash
make paper
```

Or directly:

```bash
cd paper
latexmk -pdf main.tex
```

## Figures

- Draft figures: `outputs/figures/` (not committed)
- Final paper figures: `paper/figures/` (commit)

