# Paper (LaTeX)

This folder contains dual LaTeX entrypoints that share one manuscript content base:

- arXiv target: `main.tex` (compatibility wrapper to `main_arxiv.tex`)
- EJOR target: `main_ejor.tex` (Elsevier `elsarticle`-based layout)

## Build

From the repo root:

```bash
make paper
make paper-arxiv
make paper-ejor
```

Or directly:

```bash
cd paper
latexmk -pdf main.tex
latexmk -pdf main_ejor.tex
```

## Figures

- Draft figures: `outputs/figures/` (not committed)
- Final paper figures: `paper/figures/` (commit)

## Tables

- Working tables: `paper/tables/` (commit; keep small and source-based)

## Sections

Primary entry files:

- `paper/main.tex` (arXiv compatibility wrapper)
- `paper/main_arxiv.tex`
- `paper/main_ejor.tex`

Section files live in `paper/sections/` and are assembled via `\\input{...}`.
