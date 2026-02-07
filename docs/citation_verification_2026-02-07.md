# Citation Verification Audit (2026-02-07)

## Scope
- Verified all in-text citations currently used in `paper/sections/`.
- Checked bibliography metadata in `paper/references.bib`.
- Checked claim-to-source alignment for each cited claim sentence.

## API note
- Semantic Scholar API key was loaded from local env (`SEMANTIC_SCHOLAR_API_KEY` present), but API requests returned `403 Forbidden` in this environment.
- Verification therefore used direct source pages, DOI metadata, and paper PDFs/abstracts.

## Claim-level verification matrix

| Key | Main manuscript claim(s) | Verification status | Evidence / source |
|---|---|---|---|
| `willison2026moltbook` | Moltbook context; ~4-hour heartbeat; qualitative "science-fiction" characterization | Supported for heartbeat + qualitative characterization; intro count claim needed citation fix | https://simonwillison.net/2026/jan/30/moltbook/ |
| `alexander2026afterweekend` | "graveyard of abandoned projects" and short conversation horizons | Supported | https://www.astralcodexten.com/p/moltbook-after-the-first-weekend |
| `simulamet2026observatoryarchive` | Public archived dataset and snapshot basis | Supported for public archive existence | https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive |
| `hawkes1971spectra` | Foundational self-exciting point-process framework | Supported | https://doi.org/10.1093/biomet/58.1.83 |
| `crane2008robust` | Burst + relaxation temporal classes / power-law framing | Supported | https://pmc.ncbi.nlm.nih.gov/articles/PMC3178576/ and DOI metadata: https://doi.org/10.1073/pnas.0803685105 |
| `zhao2015seismic` | SEISMIC predicts tweet popularity with self-exciting dynamics | Supported | https://doi.org/10.1145/2783258.2783401 ; arXiv mirror: https://arxiv.org/abs/1506.02594 |
| `rizoiu2017expecting` | HIP decomposes popularity into endogenous/exogenous components (virality/sensitivity/promotions) | Supported | https://doi.org/10.1145/3038912.3052650 ; arXiv mirror: https://arxiv.org/abs/1602.06033 |
| `gomez2013structure` | Preferential attachment with root bias for discussion cascades | Supported | https://doi.org/10.1145/1995966.1995992 ; arXiv mirror: https://arxiv.org/abs/1011.0673 |
| `aragon2017thread` | Threaded interface increases reciprocity / bidirectional exchange | Supported for reciprocity effect; **no clear source support found for manuscript's former numeric mean-depth claim** | https://doi.org/10.1609/icwsm.v11i1.14880 |
| `meital2024branch` | Branching prediction on Reddit; structural/temporal/linguistic features all contribute | Supported | https://arxiv.org/abs/2404.13613 |
| `harris1963theory` | Branching-process subcriticality and finite expected cascade size | Conceptually standard and consistent; full source book text not directly accessible online in this environment | Book metadata: https://www.springer.com/gp/book/9783540030668 |
| `gleeson2014competition` | Competition-induced criticality and heavy-tailed popularity | Supported | https://doi.org/10.1103/PhysRevLett.112.048701 |
| `park2023generative` | 25 agents; relationships/conversations/coordination (Valentine's party) | Supported | https://doi.org/10.1145/3586183.3606763 ; arXiv mirror: https://arxiv.org/abs/2304.03442 |

## Issues found and corrected

1. Kernel-form mismatch (textual, not analysis-code bug):
- Previous text implied SEISMIC/HIP as examples of **exponential** kernels.
- Source papers use power-law-style memory kernels.
- Fixed in:
  - `paper/sections/background.tex`
  - `paper/sections/model.tex`

2. Unsupported over-specific depth statistic:
- Prior text attributed mean depth (~3--4) to `aragon2017thread` without clear source support.
- Reworded to reciprocity finding only.
- Fixed in:
  - `paper/sections/background.tex`

3. Citation-source alignment for platform counts:
- Intro sentence originally cited Willison while stating exact archive count (`119,000` / now exact `119,677`).
- Exact count is now sourced to the archive citation.
- Fixed in:
  - `paper/sections/introduction.tex`

4. Bibliography metadata hardening:
- Added verified DOI fields for cited academic entries where available.
- Fixed malformed conference ordinal string in SEISMIC entry (`21st`).
- Made `park2023generative` a proceedings entry (UIST) consistent with DOI.
- Updated in:
  - `paper/references.bib`

## Unused bibliography entries
- `cox1972regression`
- `iacus2012causal`

These are currently present in `references.bib` but not cited in manuscript sections.

## Validation run
- `make paper` passes after edits.
- `make lint` currently fails due pre-existing issues in `scripts/` unrelated to this citation audit pass.
