#!/usr/bin/env python3
"""Submolt keyword-mapping sensitivity checks for H3 heterogeneity direction.

This script recomputes the Social/Casual versus Philosophy/Meta two-part readout
under alternative deterministic keyword mappings and exclusion rules, and writes:

- paper/tables/moltbook_submolt_keyword_sensitivity.csv
- paper/tables/moltbook_submolt_keyword_sensitivity_table.tex

Inputs are derived features produced by the Moltbook-only pipeline:
- survival_units.parquet (one row per at-risk non-root parent comment)
- thread_events.parquet  (comment-level events with thread_id -> submolt mapping)

The goal is not to "optimize" topic inference, but to bound the risk that the
direction of the key heterogeneity contrast is an artifact of a particular
keyword-trigger dictionary.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

DEFAULT_SURVIVAL_UNITS = Path(
    "data_features/moltbook_only/run_20260206-145240Z/survival_units.parquet"
)
DEFAULT_THREAD_EVENTS = Path(
    "data_features/moltbook_only/run_20260206-145240Z/thread_events.parquet"
)
DEFAULT_OUT_CSV = Path("paper/tables/moltbook_submolt_keyword_sensitivity.csv")
DEFAULT_OUT_TEX = Path("paper/tables/moltbook_submolt_keyword_sensitivity_table.tex")


MatchMode = Literal["token_only", "token_or_substring"]


@dataclass(frozen=True)
class KeywordSets:
    spam: set[str]
    builder: set[str]
    philosophy: set[str]
    creative: set[str]
    social: set[str]


def tokenize(name: str) -> set[str]:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    return {tok for tok in tokens if tok}


def categorize_submolt(
    name: Any, *, keywords: KeywordSets, match_mode: MatchMode = "token_or_substring"
) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Other"
    lower = name.strip().lower()
    tokens = tokenize(lower)

    def hit(words: set[str]) -> bool:
        if tokens & words:
            return True
        if match_mode == "token_only":
            return False
        return any(w in lower for w in words)

    # Priority order matches the main analysis (Spam first).
    if hit(keywords.spam):
        return "Spam/Low-Signal"
    if hit(keywords.builder):
        return "Builder/Technical"
    if hit(keywords.philosophy):
        return "Philosophy/Meta"
    if hit(keywords.creative):
        return "Creative"
    if hit(keywords.social):
        return "Social/Casual"
    return "Other"


def keyword_sets(version: str) -> KeywordSets:
    """Return deterministic keyword triggers for a given sensitivity variant."""
    base = KeywordSets(
        spam={
            "airdrop",
            "bitcoin",
            "crypto",
            "cryptocurrency",
            "defi",
            "nft",
            "scam",
            "shitpost",
            "solana",
            "token",
            "usdc",
        },
        builder={
            "agentops",
            "agents",
            "agentskills",
            "aiagents",
            "automation",
            "build",
            "builders",
            "buildinpublic",
            "buildlogs",
            "clawtasks",
            "coding",
            "codinghelp",
            "dev",
            "engineering",
            "framework",
            "infrastructure",
            "localllm",
            "mcp",
            "programming",
            "research",
            "security",
            "shipping",
            "skills",
            "standards",
            "tech",
            "technology",
            "tool",
            "tools",
        },
        philosophy={
            "agentphilosophy",
            "agentsouls",
            "aithoughts",
            "consciousness",
            "existential",
            "meta",
            "musings",
            "philosophy",
            "ponderings",
            "souls",
            "thebecoming",
            "thoughts",
        },
        creative={
            "creative",
            "creativeprojects",
            "music",
            "poetry",
            "shakespeare",
            "story",
            "theatre",
            "writing",
        },
        social={
            "askmoltys",
            "blesstheirhearts",
            "casual",
            "general",
            "gaming",
            "humanwatching",
            "introductions",
            "jokes",
            "offmychest",
            "pixelwar",
            "random",
            "showandtell",
            "social",
            "sport",
            "todayilearned",
        },
    )

    if version == "baseline":
        return base

    if version == "expanded":
        # Alternative keyword list: fold a small set of borderline submolts into
        # topic categories rather than "Other".
        return KeywordSets(
            spam=base.spam,
            builder=set(base.builder) | {"ai", "llm"},
            philosophy=set(base.philosophy) | {"emergence"},
            creative=base.creative,
            social=base.social,
        )

    raise ValueError(f"Unknown keyword-set version: {version!r}")


def summarize_two_part(df: pd.DataFrame) -> dict[str, float]:
    """Compute unconditional incidence and conditional t90 (minutes)."""
    n = int(len(df))
    n_events = int(df["event_observed"].sum())
    incidence_pct = float(100.0 * n_events / n) if n else float("nan")

    if n_events:
        durs_min = (
            df.loc[df["event_observed"] == 1, "duration_hours"].astype(float) * 60.0
        ).to_numpy()
        t90_min = float(np.quantile(durs_min, 0.90))
    else:
        t90_min = float("nan")

    return {
        "n_parents": float(n),
        "n_replied": float(n_events),
        "reply_incidence_pct": incidence_pct,
        "conditional_t90_minutes": t90_min,
    }


def build_sensitivity_table(
    survival: pd.DataFrame,
    thread_submolt: pd.Series,
    *,
    seed: int,
    min_category_n: int,
    input_survival_path: Path,
    input_thread_events_path: Path,
) -> pd.DataFrame:
    work = survival.copy()
    work["submolt"] = work["thread_id"].map(thread_submolt)
    work["event_observed"] = (
        pd.to_numeric(work["event_observed"], errors="coerce").fillna(0).astype(int)
    )

    rows: list[dict[str, object]] = []

    variants: list[tuple[str, str, MatchMode]] = [
        ("baseline", "token+substring", "token_or_substring"),
        ("baseline", "token-only", "token_only"),
        ("expanded", "token+substring", "token_or_substring"),
    ]

    filter_rules: list[tuple[str, str]] = [
        ("none", "None"),
        ("drop_other", "Drop Other"),
        ("drop_other_and_small", f"Drop Other + small (<{min_category_n})"),
    ]

    for keyword_version, match_label, match_mode in variants:
        keys = keyword_sets(keyword_version)
        categorized = work.copy()
        categorized["submolt_category_sensitivity"] = categorized["submolt"].map(
            lambda x: categorize_submolt(x, keywords=keys, match_mode=match_mode)
        )

        for filter_id, filter_desc in filter_rules:
            filtered = categorized
            if filter_id in {"drop_other", "drop_other_and_small"}:
                filtered = filtered[filtered["submolt_category_sensitivity"] != "Other"].copy()
            if filter_id == "drop_other_and_small":
                counts = (
                    filtered["submolt_category_sensitivity"].value_counts(dropna=False).to_dict()
                )
                keep = {k for k, v in counts.items() if int(v) >= int(min_category_n)}
                filtered = filtered[filtered["submolt_category_sensitivity"].isin(keep)].copy()

            for cat in ("Social/Casual", "Philosophy/Meta"):
                subset = filtered[filtered["submolt_category_sensitivity"] == cat].copy()
                stats = summarize_two_part(subset)
                rows.append(
                    {
                        "keyword_version": keyword_version,
                        "match_mode": match_label,
                        "filter_id": filter_id,
                        "filter_desc": filter_desc,
                        "category": cat,
                        **stats,
                        "min_category_n": int(min_category_n),
                        "analysis_seed": int(seed),
                        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "input_survival_path": str(input_survival_path),
                        "input_thread_events_path": str(input_thread_events_path),
                    }
                )

    out = pd.DataFrame(rows)

    # Pivot to one row per variant+filter for the LaTeX table and easy reading.
    wide = out.pivot_table(
        index=["keyword_version", "match_mode", "filter_id", "filter_desc", "min_category_n"],
        columns="category",
        values=["n_parents", "reply_incidence_pct", "conditional_t90_minutes"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}__{cat}" for stat, cat in wide.columns.to_flat_index()]
    wide = wide.reset_index()

    wide["ordering_incidence_preserved"] = (
        wide["reply_incidence_pct__Social/Casual"] > wide["reply_incidence_pct__Philosophy/Meta"]
    )
    wide["ordering_t90_preserved"] = (
        wide["conditional_t90_minutes__Social/Casual"]
        < wide["conditional_t90_minutes__Philosophy/Meta"]
    )

    # Attach provenance (same for all rows).
    wide["analysis_seed"] = int(seed)
    wide["generated_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    wide["input_survival_path"] = str(input_survival_path)
    wide["input_thread_events_path"] = str(input_thread_events_path)

    return wide


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    def fmt_int(x: float) -> str:
        if not np.isfinite(x):
            return ""
        return f"{int(x):,}"

    def fmt_pct(x: float) -> str:
        if not np.isfinite(x):
            return ""
        return f"{x:.2f}"

    def fmt_min(x: float) -> str:
        if not np.isfinite(x):
            return ""
        return f"{x:.2f}"

    lines: list[str] = []
    lines.append(r"\begin{tabular}{@{}llrrr rrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Variant} & \textbf{Filter} & "
        r"\multicolumn{3}{c}{\textbf{Social/Casual}} & "
        r"\multicolumn{3}{c}{\textbf{Philosophy/Meta}} \\"
    )
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(
        r"& & \textbf{Parents} & \textbf{Incidence \%} & \textbf{$t_{90}$ (minutes)}"
        r" & \textbf{Parents} & \textbf{Incidence \%} & \textbf{$t_{90}$ (minutes)} \\"
    )
    lines.append(r"\midrule")

    for row in df.to_dict(orient="records"):
        variant = f"{row['keyword_version']} ({row['match_mode']})"
        filter_desc = str(row["filter_desc"])
        sc_n = fmt_int(row["n_parents__Social/Casual"])
        sc_inc = fmt_pct(row["reply_incidence_pct__Social/Casual"])
        sc_t90 = fmt_min(row["conditional_t90_minutes__Social/Casual"])
        pm_n = fmt_int(row["n_parents__Philosophy/Meta"])
        pm_inc = fmt_pct(row["reply_incidence_pct__Philosophy/Meta"])
        pm_t90 = fmt_min(row["conditional_t90_minutes__Philosophy/Meta"])
        lines.append(
            f"{variant} & {filter_desc} & {sc_n} & {sc_inc} & {sc_t90} & "
            f"{pm_n} & {pm_inc} & {pm_t90} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survival-units", type=Path, default=DEFAULT_SURVIVAL_UNITS)
    parser.add_argument("--thread-events", type=Path, default=DEFAULT_THREAD_EVENTS)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-tex", type=Path, default=DEFAULT_OUT_TEX)
    parser.add_argument("--seed", type=int, default=20260210)
    parser.add_argument(
        "--min-category-n",
        type=int,
        default=1000,
        help="Minimum parents threshold for the 'drop small categories' filter rule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    survival = pd.read_parquet(
        args.survival_units, columns=["thread_id", "event_observed", "duration_hours"]
    )
    thread_events = pd.read_parquet(args.thread_events, columns=["thread_id", "submolt"])
    thread_submolt = thread_events.drop_duplicates("thread_id").set_index("thread_id")["submolt"]

    out = build_sensitivity_table(
        survival,
        thread_submolt,
        seed=int(args.seed),
        min_category_n=int(args.min_category_n),
        input_survival_path=args.survival_units,
        input_thread_events_path=args.thread_events,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    write_latex_table(out, args.out_tex)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
