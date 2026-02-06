#!/usr/bin/env python3
"""Run coarse matched Moltbook-vs-Reddit cross-platform analysis.

This script performs coarsened exact matching (CEM)-style stratification and
one-to-one deterministic matching between Moltbook and Reddit thread-level units.
Controls are:
1) deterministic coarse topic mapping,
2) UTC posting-hour bin,
3) first-30-minute comment-count bin.

Outputs include matched sample flow, pre/post balance diagnostics, paired effect
estimates, matched-sample half-life summaries, and run-scoped manifests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize, stats

DEFAULT_MOLTBOOK_THREAD_METRICS = Path(
    "data_features/moltbook_only/run_20260206-145240Z/thread_metrics.parquet"
)
DEFAULT_MOLTBOOK_THREAD_EVENTS = Path(
    "data_features/moltbook_only/run_20260206-145240Z/thread_events.parquet"
)
DEFAULT_MOLTBOOK_SURVIVAL = Path(
    "data_features/moltbook_only/run_20260206-145240Z/survival_units.parquet"
)

DEFAULT_REDDIT_THREAD_METRICS = Path(
    "data_features/reddit_only/attempt_scaled_20260206-142651Z/thread_metrics.parquet"
)
DEFAULT_REDDIT_THREAD_EVENTS = Path(
    "data_features/reddit_only/attempt_scaled_20260206-142651Z/thread_events.parquet"
)
DEFAULT_REDDIT_SURVIVAL = Path(
    "data_features/reddit_only/attempt_scaled_20260206-142651Z/survival_units.parquet"
)

DEFAULT_OUTPUTS_ROOT = Path("outputs/cross_platform_matched")
DEFAULT_SEED = 20260206
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_HALF_LIFE_BOOTSTRAP_REPS = 400
DEFAULT_CENSOR_BOUNDARY_HOURS = 4.0
PAIR_FLOAT_DECIMALS = 12
INFERENCE_DIFF_DECIMALS = 12

STRATUM_COLS = ["topic_coarse", "post_hour_bin", "early_engagement_bin"]

EARLY_BIN_EDGES = [-0.1, 0.0, 2.0, 5.0, 10.0, float("inf")]
EARLY_BIN_LABELS = ["0", "1-2", "3-5", "6-10", "11+"]

MOLTBOOK_TOPIC_MAP = {
    "Builder/Technical": "tech",
    "Philosophy/Meta": "meta",
    "Social/Casual": "general",
    "Creative": "general",
    "Other": "general",
    "Spam/Low-Signal": "spam",
}

REDDIT_TOPIC_MAP = {
    "MachineLearning": "tech",
    "Python": "tech",
    "datascience": "tech",
    "learnprogramming": "tech",
    "programming": "tech",
    "artificial": "meta",
}

OUTCOME_SPECS = {
    "n_comments": "Comments per thread",
    "depth_max": "Max depth",
    "n_unique_agents": "Unique participants",
    "thread_duration_hours": "Thread duration (hours)",
    "reentry_rate": "Re-entry rate",
}


@dataclass(frozen=True)
class Config:
    moltbook_thread_metrics_path: Path
    moltbook_thread_events_path: Path
    moltbook_survival_path: Path
    reddit_thread_metrics_path: Path
    reddit_thread_events_path: Path
    reddit_survival_path: Path
    outputs_root: Path
    run_id: str
    seed: int
    bootstrap_reps: int
    half_life_bootstrap_reps: int
    censor_boundary_hours: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moltbook-thread-metrics-path",
        type=Path,
        default=DEFAULT_MOLTBOOK_THREAD_METRICS,
        help="Path to Moltbook thread_metrics.parquet.",
    )
    parser.add_argument(
        "--moltbook-thread-events-path",
        type=Path,
        default=DEFAULT_MOLTBOOK_THREAD_EVENTS,
        help="Path to Moltbook thread_events.parquet.",
    )
    parser.add_argument(
        "--moltbook-survival-path",
        type=Path,
        default=DEFAULT_MOLTBOOK_SURVIVAL,
        help="Path to Moltbook survival_units.parquet.",
    )
    parser.add_argument(
        "--reddit-thread-metrics-path",
        type=Path,
        default=DEFAULT_REDDIT_THREAD_METRICS,
        help="Path to Reddit thread_metrics.parquet.",
    )
    parser.add_argument(
        "--reddit-thread-events-path",
        type=Path,
        default=DEFAULT_REDDIT_THREAD_EVENTS,
        help="Path to Reddit thread_events.parquet.",
    )
    parser.add_argument(
        "--reddit-survival-path",
        type=Path,
        default=DEFAULT_REDDIT_SURVIVAL,
        help="Path to Reddit survival_units.parquet.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Root directory for run-scoped outputs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run ID. Defaults to UTC timestamped run_*.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic RNG seed.")
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help="Bootstrap reps for paired mean-difference confidence intervals.",
    )
    parser.add_argument(
        "--half-life-bootstrap-reps",
        type=int,
        default=DEFAULT_HALF_LIFE_BOOTSTRAP_REPS,
        help="Cluster-bootstrap reps for matched-sample half-life confidence intervals.",
    )
    parser.add_argument(
        "--censor-boundary-hours",
        type=float,
        default=DEFAULT_CENSOR_BOUNDARY_HOURS,
        help="Boundary-censor exclusion threshold (hours) for matched-sample survival summaries.",
    )

    args = parser.parse_args()
    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%d-%H%M%SZ")

    return Config(
        moltbook_thread_metrics_path=args.moltbook_thread_metrics_path,
        moltbook_thread_events_path=args.moltbook_thread_events_path,
        moltbook_survival_path=args.moltbook_survival_path,
        reddit_thread_metrics_path=args.reddit_thread_metrics_path,
        reddit_thread_events_path=args.reddit_thread_events_path,
        reddit_survival_path=args.reddit_survival_path,
        outputs_root=args.outputs_root,
        run_id=run_id,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
        half_life_bootstrap_reps=args.half_life_bootstrap_reps,
        censor_boundary_hours=args.censor_boundary_hours,
    )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        if np.isnan(out):
            return None
        if np.isposinf(out):
            return "inf"
        if np.isneginf(out):
            return "-inf"
        return out
    return value


def quantize_float_columns(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["floating"]).columns
    if len(float_cols) > 0:
        out.loc[:, float_cols] = out.loc[:, float_cols].round(decimals)
    return out


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")
    return pd.read_parquet(path)


def add_coarsened_controls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["post_hour_bin"] = out["post_hour_utc"].astype("Int64")
    out["early_engagement_bin"] = pd.cut(
        out["early_comments_30m"].astype(float),
        bins=EARLY_BIN_EDGES,
        labels=EARLY_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def prepare_moltbook_threads(metrics: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    required_metrics = {
        "thread_id",
        "post_created_at_utc",
        "submolt",
        "submolt_category",
        "n_comments",
        "depth_max",
        "n_unique_agents",
        "thread_duration_hours",
        "reentry_rate",
    }
    required_events = {"thread_id", "lag_since_post_hours"}
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    missing_events = sorted(required_events - set(events.columns))
    if missing_metrics:
        raise ValueError(f"Moltbook thread_metrics missing columns: {missing_metrics}")
    if missing_events:
        raise ValueError(f"Moltbook thread_events missing columns: {missing_events}")

    early = (
        events.loc[events["lag_since_post_hours"].astype(float) <= 0.5, ["thread_id"]]
        .groupby("thread_id")
        .size()
        .rename("early_comments_30m")
    )

    out = metrics[
        [
            "thread_id",
            "post_created_at_utc",
            "submolt",
            "submolt_category",
            "n_comments",
            "depth_max",
            "n_unique_agents",
            "thread_duration_hours",
            "reentry_rate",
        ]
    ].copy()
    out = out.merge(early, on="thread_id", how="left")
    out["early_comments_30m"] = out["early_comments_30m"].fillna(0).astype(int)
    out["created_at_utc"] = to_utc(out["post_created_at_utc"])
    out["post_hour_utc"] = out["created_at_utc"].dt.hour
    out["topic_source"] = out["submolt_category"].astype("string")
    out["topic_coarse"] = out["topic_source"].map(MOLTBOOK_TOPIC_MAP)
    out["platform"] = "moltbook"
    out = add_coarsened_controls(out)
    return out


def prepare_reddit_threads(metrics: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    required_metrics = {
        "thread_id",
        "submission_created_at_utc",
        "subreddit",
        "n_comments",
        "depth_max",
        "n_unique_agents",
        "thread_duration_hours",
        "reentry_rate",
    }
    required_events = {"thread_id", "lag_since_submission_hours"}
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    missing_events = sorted(required_events - set(events.columns))
    if missing_metrics:
        raise ValueError(f"Reddit thread_metrics missing columns: {missing_metrics}")
    if missing_events:
        raise ValueError(f"Reddit thread_events missing columns: {missing_events}")

    early = (
        events.loc[events["lag_since_submission_hours"].astype(float) <= 0.5, ["thread_id"]]
        .groupby("thread_id")
        .size()
        .rename("early_comments_30m")
    )

    out = metrics[
        [
            "thread_id",
            "submission_created_at_utc",
            "subreddit",
            "n_comments",
            "depth_max",
            "n_unique_agents",
            "thread_duration_hours",
            "reentry_rate",
        ]
    ].copy()
    out = out.merge(early, on="thread_id", how="left")
    out["early_comments_30m"] = out["early_comments_30m"].fillna(0).astype(int)
    out["created_at_utc"] = to_utc(out["submission_created_at_utc"])
    out["post_hour_utc"] = out["created_at_utc"].dt.hour
    out["topic_source"] = out["subreddit"].astype("string")
    out["topic_coarse"] = out["topic_source"].map(REDDIT_TOPIC_MAP)
    out["platform"] = "reddit"
    out = add_coarsened_controls(out)
    return out


def eligible_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "thread_id",
        "created_at_utc",
        "topic_coarse",
        "post_hour_bin",
        "early_engagement_bin",
    ]
    out = df.dropna(subset=required).copy()
    out["post_hour_bin"] = out["post_hour_bin"].astype(int)
    out["early_engagement_bin"] = out["early_engagement_bin"].astype(str)
    return out


def stratum_tuple(df: pd.DataFrame) -> pd.Series:
    return pd.Series(list(zip(df["topic_coarse"], df["post_hour_bin"], df["early_engagement_bin"])))


def overlap_subset(
    molt: pd.DataFrame,
    reddit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, int, str]]]:
    m_keys = set(stratum_tuple(molt))
    r_keys = set(stratum_tuple(reddit))
    overlap = m_keys & r_keys
    m_mask = stratum_tuple(molt).isin(overlap)
    r_mask = stratum_tuple(reddit).isin(overlap)
    return molt[m_mask].copy(), reddit[r_mask].copy(), overlap


def exact_match_pairs(
    molt: pd.DataFrame,
    reddit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []

    m_groups = {
        key: grp.copy()
        for key, grp in molt.groupby(STRATUM_COLS, sort=True, observed=False)
        if len(grp) > 0
    }
    r_groups = {
        key: grp.copy()
        for key, grp in reddit.groupby(STRATUM_COLS, sort=True, observed=False)
        if len(grp) > 0
    }

    shared_keys = sorted(set(m_groups) & set(r_groups))
    pair_id = 1

    for key in shared_keys:
        m_stratum = m_groups[key].sort_values(
            ["early_comments_30m", "created_at_utc", "thread_id"], kind="stable"
        )
        r_stratum = r_groups[key].sort_values(
            ["early_comments_30m", "created_at_utc", "thread_id"], kind="stable"
        )

        n_m = len(m_stratum)
        n_r = len(r_stratum)
        n_pairs = min(n_m, n_r)

        topic, hour, early_bin = key
        strata_rows.append(
            {
                "topic_coarse": topic,
                "post_hour_bin": int(hour),
                "early_engagement_bin": str(early_bin),
                "moltbook_n": int(n_m),
                "reddit_n": int(n_r),
                "matched_pairs": int(n_pairs),
            }
        )

        if n_pairs == 0:
            continue

        m_used = m_stratum.iloc[:n_pairs].reset_index(drop=True)
        r_used = r_stratum.iloc[:n_pairs].reset_index(drop=True)

        for i in range(n_pairs):
            m_row = m_used.iloc[i]
            r_row = r_used.iloc[i]
            row = {
                "pair_id": pair_id,
                "topic_coarse": topic,
                "post_hour_bin": int(hour),
                "early_engagement_bin": str(early_bin),
                "moltbook_thread_id": str(m_row["thread_id"]),
                "reddit_thread_id": str(r_row["thread_id"]),
                "moltbook_created_at_utc": m_row["created_at_utc"],
                "reddit_created_at_utc": r_row["created_at_utc"],
                "moltbook_topic_source": m_row["topic_source"],
                "reddit_topic_source": r_row["topic_source"],
                "moltbook_early_comments_30m": int(m_row["early_comments_30m"]),
                "reddit_early_comments_30m": int(r_row["early_comments_30m"]),
            }
            for outcome in OUTCOME_SPECS:
                row[f"moltbook_{outcome}"] = float(m_row[outcome])
                row[f"reddit_{outcome}"] = float(r_row[outcome])
                row[f"diff_{outcome}"] = float(m_row[outcome]) - float(r_row[outcome])

            pair_rows.append(row)
            pair_id += 1

    pairs = pd.DataFrame(pair_rows)
    strata = pd.DataFrame(strata_rows).sort_values(
        ["matched_pairs", "moltbook_n", "reddit_n"], ascending=[False, False, False]
    )
    return pairs, strata


def standardized_mean_difference(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    pooled_sd = float(np.sqrt((var_a + var_b) / 2.0))
    mean_diff = float(np.mean(a) - np.mean(b))
    if pooled_sd <= 0:
        return 0.0 if abs(mean_diff) < 1e-12 else float("inf")
    return mean_diff / pooled_sd


def categorical_smd_levels(
    x: pd.Series,
    y: pd.Series,
    covariate: str,
) -> pd.DataFrame:
    levels = sorted(set(x.dropna().astype(str)) | set(y.dropna().astype(str)))
    rows: list[dict[str, Any]] = []
    for level in levels:
        x_bin = (x.astype(str) == level).astype(float)
        y_bin = (y.astype(str) == level).astype(float)
        smd = standardized_mean_difference(x_bin, y_bin)
        rows.append(
            {
                "covariate": covariate,
                "level": level,
                "smd": float(smd),
            }
        )
    return pd.DataFrame(rows)


def total_variation_distance(x: pd.Series, y: pd.Series) -> float:
    x_norm = x.astype(str).value_counts(normalize=True)
    y_norm = y.astype(str).value_counts(normalize=True)
    levels = sorted(set(x_norm.index) | set(y_norm.index))
    tvd = 0.5 * sum(
        abs(float(x_norm.get(level, 0.0)) - float(y_norm.get(level, 0.0))) for level in levels
    )
    return float(tvd)


def build_balance_tables(
    before_m: pd.DataFrame,
    before_r: pd.DataFrame,
    after_m: pd.DataFrame,
    after_r: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    level_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    numeric_covariates = ["post_hour_utc", "early_comments_30m"]
    categorical_covariates = ["topic_coarse", "post_hour_bin", "early_engagement_bin"]

    for cov in numeric_covariates:
        before_smd = standardized_mean_difference(before_m[cov], before_r[cov])
        after_smd = standardized_mean_difference(after_m[cov], after_r[cov])
        summary_rows.append(
            {
                "covariate": cov,
                "type": "numeric",
                "before_smd": float(before_smd),
                "after_smd": float(after_smd),
                "before_abs_smd": abs(float(before_smd)),
                "after_abs_smd": abs(float(after_smd)),
                "before_tvd": float("nan"),
                "after_tvd": float("nan"),
            }
        )

    for cov in categorical_covariates:
        before_levels = categorical_smd_levels(before_m[cov], before_r[cov], cov)
        before_levels = before_levels.rename(columns={"smd": "before_smd"})
        after_levels = categorical_smd_levels(after_m[cov], after_r[cov], cov)
        after_levels = after_levels.rename(columns={"smd": "after_smd"})
        merged = before_levels.merge(after_levels, on=["covariate", "level"], how="outer")
        merged["before_abs_smd"] = merged["before_smd"].abs()
        merged["after_abs_smd"] = merged["after_smd"].abs()
        level_rows.append(merged)

        summary_rows.append(
            {
                "covariate": cov,
                "type": "categorical",
                "before_smd": float("nan"),
                "after_smd": float("nan"),
                "before_abs_smd": float(merged["before_abs_smd"].max()),
                "after_abs_smd": float(merged["after_abs_smd"].max()),
                "before_tvd": total_variation_distance(before_m[cov], before_r[cov]),
                "after_tvd": total_variation_distance(after_m[cov], after_r[cov]),
            }
        )

    levels_df = (
        pd.concat(level_rows, ignore_index=True)
        if level_rows
        else pd.DataFrame(columns=["covariate", "level", "before_smd", "after_smd"])
    )
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, levels_df


def bootstrap_mean_diff_ci(
    diffs: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> tuple[float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan")
    boot = np.empty(reps, dtype=float)
    n = diffs.size
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return lo, hi


def paired_effects_table(
    pairs: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_reps: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for outcome, label in OUTCOME_SPECS.items():
        m = pd.to_numeric(pairs[f"moltbook_{outcome}"], errors="coerce")
        r = pd.to_numeric(pairs[f"reddit_{outcome}"], errors="coerce")
        diff_col = f"diff_{outcome}"
        if diff_col in pairs.columns:
            d_series = pd.to_numeric(pairs[diff_col], errors="coerce")
            keep = m.notna() & r.notna() & d_series.notna()
            d = d_series[keep].to_numpy(dtype=float)
        else:
            keep = m.notna() & r.notna()
            d = (m[keep] - r[keep]).to_numpy(dtype=float)

        m = m[keep].to_numpy(dtype=float)
        r = r[keep].to_numpy(dtype=float)
        # Quantize paired deltas to make rank-based inference stable across
        # CSV/parquet round-trips of promoted pair-level artifacts.
        d = np.round(d, INFERENCE_DIFF_DECIMALS)

        if d.size == 0:
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": label,
                    "n_pairs": 0,
                    "moltbook_mean": float("nan"),
                    "reddit_mean": float("nan"),
                    "mean_diff_m_minus_r": float("nan"),
                    "mean_diff_ci95_low": float("nan"),
                    "mean_diff_ci95_high": float("nan"),
                    "median_diff_m_minus_r": float("nan"),
                    "cohen_d_paired": float("nan"),
                    "wilcoxon_stat": float("nan"),
                    "wilcoxon_p": float("nan"),
                    "sign_test_p": float("nan"),
                }
            )
            continue

        ci_low, ci_high = bootstrap_mean_diff_ci(d, rng=rng, reps=bootstrap_reps)
        std_d = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
        cohen_d = float(np.mean(d) / std_d) if np.isfinite(std_d) and std_d > 0 else float("nan")

        nonzero = d[np.abs(d) > 1e-12]
        if nonzero.size > 0:
            wil = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
            n_pos = int(np.sum(nonzero > 0))
            sign_p = float(
                stats.binomtest(
                    n_pos,
                    n=nonzero.size,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            wil_stat = float(wil.statistic)
            wil_p = float(wil.pvalue)
        else:
            wil_stat = float("nan")
            wil_p = float("nan")
            sign_p = float("nan")

        rows.append(
            {
                "outcome": outcome,
                "outcome_label": label,
                "n_pairs": int(d.size),
                "moltbook_mean": float(np.mean(m)),
                "reddit_mean": float(np.mean(r)),
                "mean_diff_m_minus_r": float(np.mean(d)),
                "mean_diff_ci95_low": float(ci_low),
                "mean_diff_ci95_high": float(ci_high),
                "median_diff_m_minus_r": float(np.median(d)),
                "cohen_d_paired": cohen_d,
                "wilcoxon_stat": wil_stat,
                "wilcoxon_p": wil_p,
                "sign_test_p": sign_p,
            }
        )

    return pd.DataFrame(rows)


def fit_exponential_decay(
    durations_hours: np.ndarray,
    event_observed: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    s = np.asarray(durations_hours, dtype=float)
    d = np.asarray(event_observed, dtype=float)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(s) & np.isfinite(d) & np.isfinite(w) & (s > 0) & (w > 0)
    s = s[valid]
    d = d[valid]
    w = w[valid]
    if len(s) == 0 or np.sum(w * d) <= 0:
        return {"success": False, "message": "No valid uncensored observations."}

    event_rate = np.sum(w * d) / np.sum(w * s)
    observed = s[d > 0]
    beta0 = 1.0 / max(np.median(observed), 1e-3) if observed.size > 0 else 1.0
    alpha0 = max(event_rate, 1e-4)

    def neg_loglik(theta: np.ndarray) -> float:
        log_alpha, log_beta = theta
        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)
        integral = (alpha / beta) * (1.0 - np.exp(-beta * s))
        ll = w * (d * (log_alpha - beta * s) - integral)
        return -float(np.sum(ll))

    starts = [
        np.array([np.log(alpha0), np.log(beta0)]),
        np.array([np.log(alpha0 * 0.5), np.log(beta0 * 1.5)]),
        np.array([np.log(alpha0 * 1.5), np.log(beta0 * 0.5)]),
    ]

    best = None
    for start in starts:
        res = optimize.minimize(
            neg_loglik,
            start,
            method="L-BFGS-B",
            bounds=[(-20, 20), (-20, 20)],
        )
        if best is None or res.fun < best.fun:
            best = res

    assert best is not None
    if not best.success:
        return {"success": False, "message": str(best.message)}
    log_alpha_hat, log_beta_hat = best.x
    alpha_hat = float(np.exp(log_alpha_hat))
    beta_hat = float(np.exp(log_beta_hat))
    half_life_hours = float(np.log(2.0) / beta_hat)
    return {
        "success": True,
        "alpha": alpha_hat,
        "beta": beta_hat,
        "half_life_hours": half_life_hours,
        "log_likelihood": float(-best.fun),
        "n": int(len(s)),
        "events": float(np.sum(d)),
    }


def percentile_ci(values: list[float], alpha: float = 0.05) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    arr = np.asarray(values, dtype=float)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return [lo, hi]


def cluster_bootstrap_exponential(
    durations_hours: np.ndarray,
    event_observed: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    if n_clusters <= 0:
        return {
            "n_successful_bootstrap": 0,
            "beta_ci_95": [float("nan"), float("nan")],
            "half_life_ci_95": [float("nan"), float("nan")],
        }

    betas: list[float] = []
    half_lives: list[float] = []
    for _ in range(reps):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(sampled, minlength=n_clusters).astype(float)
        weights = counts[cluster_codes]
        if np.sum(weights * event_observed) <= 0:
            continue
        fit = fit_exponential_decay(durations_hours, event_observed, weights=weights)
        if fit.get("success"):
            betas.append(float(fit["beta"]))
            half_lives.append(float(fit["half_life_hours"]))

    return {
        "n_successful_bootstrap": len(betas),
        "beta_ci_95": percentile_ci(betas),
        "half_life_ci_95": percentile_ci(half_lives),
    }


def matched_half_life_summary(
    survival_df: pd.DataFrame,
    matched_thread_ids: set[str],
    rng: np.random.Generator,
    censor_boundary_hours: float,
    bootstrap_reps: int,
) -> dict[str, Any]:
    required_cols = {"thread_id", "duration_hours", "event_observed", "excluded_censor_boundary"}
    missing = sorted(required_cols - set(survival_df.columns))
    if missing:
        raise ValueError(f"Survival units missing columns: {missing}")

    surv = survival_df[survival_df["thread_id"].astype(str).isin(matched_thread_ids)].copy()
    if surv.empty:
        return {
            "n_rows": 0,
            "n_threads": 0,
            "n_primary": 0,
            "events_primary": 0,
            "censored_primary": 0,
            "excluded_boundary_comments": 0,
            "fit": {"success": False, "message": "No matched survival rows."},
            "bootstrap": {
                "n_successful_bootstrap": 0,
                "beta_ci_95": [float("nan"), float("nan")],
                "half_life_ci_95": [float("nan"), float("nan")],
            },
        }

    if "excluded_censor_boundary" not in surv.columns:
        end_time = to_utc(surv["created_at_utc"]).max()
        cutoff = end_time - pd.Timedelta(hours=censor_boundary_hours)
        surv["excluded_censor_boundary"] = to_utc(surv["created_at_utc"]) > cutoff

    primary = surv[~surv["excluded_censor_boundary"].astype(bool)].copy()

    d = primary["duration_hours"].to_numpy(dtype=float)
    e = primary["event_observed"].to_numpy(dtype=float)
    fit = fit_exponential_decay(d, e)

    clusters = primary["thread_id"].astype("category")
    cluster_codes = clusters.cat.codes.to_numpy(dtype=int)
    n_clusters = int(clusters.nunique())
    boot = cluster_bootstrap_exponential(
        durations_hours=d,
        event_observed=e,
        cluster_codes=cluster_codes,
        n_clusters=n_clusters,
        rng=rng,
        reps=bootstrap_reps,
    )

    return {
        "n_rows": int(len(surv)),
        "n_threads": int(surv["thread_id"].nunique()),
        "n_primary": int(len(primary)),
        "events_primary": int(primary["event_observed"].sum()),
        "censored_primary": int((1 - primary["event_observed"]).sum()),
        "excluded_boundary_comments": int(surv["excluded_censor_boundary"].sum()),
        "fit": fit,
        "bootstrap": boot,
    }


def build_sample_flow(
    m_all: pd.DataFrame,
    r_all: pd.DataFrame,
    m_eligible: pd.DataFrame,
    r_eligible: pd.DataFrame,
    m_overlap: pd.DataFrame,
    r_overlap: pd.DataFrame,
    n_pairs: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stages = [
        (1, "Input threads", len(m_all), len(r_all)),
        (2, "Covariates available", len(m_eligible), len(r_eligible)),
        (3, "In overlap strata", len(m_overlap), len(r_overlap)),
        (4, "Matched threads (1:1)", n_pairs, n_pairs),
    ]

    for stage_order, stage_name, m_count, r_count in stages:
        rows.append(
            {
                "stage_order": stage_order,
                "stage": stage_name,
                "platform": "moltbook",
                "n_threads": int(m_count),
                "share_of_input": float(m_count / len(m_all)) if len(m_all) else float("nan"),
            }
        )
        rows.append(
            {
                "stage_order": stage_order,
                "stage": stage_name,
                "platform": "reddit",
                "n_threads": int(r_count),
                "share_of_input": float(r_count / len(r_all)) if len(r_all) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def make_balance_love_plot(balance: pd.DataFrame, out_path: Path) -> None:
    plot_df = balance[["covariate", "before_abs_smd", "after_abs_smd"]].copy()
    plot_df = plot_df.sort_values("before_abs_smd", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(plot_df))

    ax.scatter(plot_df["before_abs_smd"], y, label="Before matching", color="#d65f5f", s=55)
    ax.scatter(plot_df["after_abs_smd"], y, label="After matching", color="#3c78d8", s=55)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(
            [row["before_abs_smd"], row["after_abs_smd"]],
            [i, i],
            color="#999999",
            linewidth=1,
            alpha=0.7,
        )

    ax.axvline(0.1, color="#666666", linestyle="--", linewidth=1, label="|SMD| = 0.1")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["covariate"])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title("Covariate balance before vs after matching")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_sample_flow_figure(sample_flow: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=sample_flow,
        x="stage_order",
        y="n_threads",
        hue="platform",
        marker="o",
        linewidth=2,
        palette={"moltbook": "#1f77b4", "reddit": "#ff7f0e"},
        ax=ax,
    )
    labels = (
        sample_flow[["stage_order", "stage"]].drop_duplicates().sort_values("stage_order")["stage"]
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Sample flow stage")
    ax.set_ylabel("Number of threads")
    ax.set_title("Matched-sample flow")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Platform", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_paired_effects_figure(effects: pd.DataFrame, out_path: Path) -> None:
    plot_df = effects.copy().sort_values("mean_diff_m_minus_r", ascending=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    means = plot_df["mean_diff_m_minus_r"].to_numpy(dtype=float)
    lows = plot_df["mean_diff_ci95_low"].to_numpy(dtype=float)
    highs = plot_df["mean_diff_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([means - lows, highs - means])

    ax.errorbar(
        means,
        y,
        xerr=xerr,
        fmt="o",
        color="#2a9d8f",
        ecolor="#2a9d8f",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["outcome_label"])
    ax.set_xlabel("Mean paired difference (Moltbook - Reddit)")
    ax.set_title("Paired outcome differences with bootstrap 95% CI")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    run_dir = cfg.outputs_root / cfg.run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    ensure_dirs(run_dir, figures_dir, tables_dir)

    m_metrics = read_parquet(cfg.moltbook_thread_metrics_path)
    m_events = read_parquet(cfg.moltbook_thread_events_path)
    m_survival = read_parquet(cfg.moltbook_survival_path)

    r_metrics = read_parquet(cfg.reddit_thread_metrics_path)
    r_events = read_parquet(cfg.reddit_thread_events_path)
    r_survival = read_parquet(cfg.reddit_survival_path)

    m_all = prepare_moltbook_threads(m_metrics, m_events)
    r_all = prepare_reddit_threads(r_metrics, r_events)

    m_eligible = eligible_for_matching(m_all)
    r_eligible = eligible_for_matching(r_all)

    m_overlap, r_overlap, overlap_keys = overlap_subset(m_eligible, r_eligible)
    pairs, strata = exact_match_pairs(m_overlap, r_overlap)
    pairs = quantize_float_columns(pairs, PAIR_FLOAT_DECIMALS)
    n_pairs = int(len(pairs))

    if n_pairs == 0:
        raise RuntimeError("No exact matches found under configured coarsened strata.")

    m_matched_ids = set(pairs["moltbook_thread_id"].astype(str))
    r_matched_ids = set(pairs["reddit_thread_id"].astype(str))
    m_after = m_eligible[m_eligible["thread_id"].astype(str).isin(m_matched_ids)].copy()
    r_after = r_eligible[r_eligible["thread_id"].astype(str).isin(r_matched_ids)].copy()

    sample_flow = build_sample_flow(
        m_all=m_all,
        r_all=r_all,
        m_eligible=m_eligible,
        r_eligible=r_eligible,
        m_overlap=m_overlap,
        r_overlap=r_overlap,
        n_pairs=n_pairs,
    )

    balance_summary, balance_levels = build_balance_tables(
        before_m=m_eligible,
        before_r=r_eligible,
        after_m=m_after,
        after_r=r_after,
    )

    paired_effects = paired_effects_table(pairs=pairs, rng=rng, bootstrap_reps=cfg.bootstrap_reps)

    m_half_life = matched_half_life_summary(
        survival_df=m_survival,
        matched_thread_ids=m_matched_ids,
        rng=rng,
        censor_boundary_hours=cfg.censor_boundary_hours,
        bootstrap_reps=cfg.half_life_bootstrap_reps,
    )
    r_half_life = matched_half_life_summary(
        survival_df=r_survival,
        matched_thread_ids=r_matched_ids,
        rng=rng,
        censor_boundary_hours=cfg.censor_boundary_hours,
        bootstrap_reps=cfg.half_life_bootstrap_reps,
    )

    half_life_rows = [
        {
            "platform": "moltbook",
            "n_threads": m_half_life["n_threads"],
            "n_primary": m_half_life["n_primary"],
            "events_primary": m_half_life["events_primary"],
            "censored_primary": m_half_life["censored_primary"],
            "half_life_hours": m_half_life["fit"].get("half_life_hours"),
            "half_life_ci95_low": m_half_life["bootstrap"]["half_life_ci_95"][0],
            "half_life_ci95_high": m_half_life["bootstrap"]["half_life_ci_95"][1],
        },
        {
            "platform": "reddit",
            "n_threads": r_half_life["n_threads"],
            "n_primary": r_half_life["n_primary"],
            "events_primary": r_half_life["events_primary"],
            "censored_primary": r_half_life["censored_primary"],
            "half_life_hours": r_half_life["fit"].get("half_life_hours"),
            "half_life_ci95_low": r_half_life["bootstrap"]["half_life_ci_95"][0],
            "half_life_ci95_high": r_half_life["bootstrap"]["half_life_ci_95"][1],
        },
    ]
    half_life_table = pd.DataFrame(half_life_rows)

    topic_mapping_rows = []
    for source, mapped in sorted(MOLTBOOK_TOPIC_MAP.items()):
        topic_mapping_rows.append(
            {"platform": "moltbook", "source_label": source, "coarse_topic": mapped}
        )
    for source, mapped in sorted(REDDIT_TOPIC_MAP.items()):
        topic_mapping_rows.append(
            {"platform": "reddit", "source_label": source, "coarse_topic": mapped}
        )
    topic_mapping_table = pd.DataFrame(topic_mapping_rows)

    # Write tables.
    pairs_out_csv = tables_dir / "matched_pairs.csv"
    pairs_out_parquet = tables_dir / "matched_pairs.parquet"
    strata_out_csv = tables_dir / "matching_strata_counts.csv"
    sample_flow_out_csv = tables_dir / "sample_flow.csv"
    balance_out_csv = tables_dir / "balance_diagnostics.csv"
    balance_levels_out_csv = tables_dir / "balance_diagnostics_levels.csv"
    paired_effects_out_csv = tables_dir / "paired_effects.csv"
    half_life_out_csv = tables_dir / "matched_sample_half_life.csv"
    topic_map_out_csv = tables_dir / "topic_mapping.csv"

    pairs.to_csv(pairs_out_csv, index=False, float_format=f"%.{PAIR_FLOAT_DECIMALS}f")
    pairs.to_parquet(pairs_out_parquet, index=False)
    strata.to_csv(strata_out_csv, index=False)
    sample_flow.to_csv(sample_flow_out_csv, index=False)
    balance_summary.to_csv(balance_out_csv, index=False)
    balance_levels.to_csv(balance_levels_out_csv, index=False)
    paired_effects.to_csv(paired_effects_out_csv, index=False)
    half_life_table.to_csv(half_life_out_csv, index=False)
    topic_mapping_table.to_csv(topic_map_out_csv, index=False)

    # Figures.
    balance_fig = figures_dir / "balance_love_plot.png"
    flow_fig = figures_dir / "matched_sample_flow.png"
    effects_fig = figures_dir / "paired_effects.png"

    make_balance_love_plot(balance_summary, balance_fig)
    make_sample_flow_figure(sample_flow, flow_fig)
    make_paired_effects_figure(paired_effects, effects_fig)

    matched_strata_with_pairs = int((strata["matched_pairs"] > 0).sum()) if not strata.empty else 0
    m_overlap_threads = int(len(m_overlap))
    r_overlap_threads = int(len(r_overlap))

    summary = {
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "inputs": {
            "moltbook_thread_metrics_path": str(cfg.moltbook_thread_metrics_path),
            "moltbook_thread_events_path": str(cfg.moltbook_thread_events_path),
            "moltbook_survival_path": str(cfg.moltbook_survival_path),
            "reddit_thread_metrics_path": str(cfg.reddit_thread_metrics_path),
            "reddit_thread_events_path": str(cfg.reddit_thread_events_path),
            "reddit_survival_path": str(cfg.reddit_survival_path),
        },
        "controls": {
            "topic_mapping": {
                "moltbook": MOLTBOOK_TOPIC_MAP,
                "reddit": REDDIT_TOPIC_MAP,
            },
            "post_hour_bin": "UTC hour of post/submission creation (0-23)",
            "early_engagement_proxy": {
                "definition": "comments in first 30 minutes since thread root",
                "bin_edges": EARLY_BIN_EDGES,
                "bin_labels": EARLY_BIN_LABELS,
            },
        },
        "sample_flow": {
            "moltbook_input_threads": int(len(m_all)),
            "reddit_input_threads": int(len(r_all)),
            "moltbook_covariates_available": int(len(m_eligible)),
            "reddit_covariates_available": int(len(r_eligible)),
            "moltbook_in_overlap_strata": m_overlap_threads,
            "reddit_in_overlap_strata": r_overlap_threads,
            "shared_strata_count": int(len(overlap_keys)),
            "shared_strata_with_pairs": matched_strata_with_pairs,
            "matched_pairs": n_pairs,
            "matched_threads_per_platform": n_pairs,
        },
        "balance": {
            "summary": balance_summary.to_dict(orient="records"),
            "max_abs_smd_before": float(balance_summary["before_abs_smd"].max()),
            "max_abs_smd_after": float(balance_summary["after_abs_smd"].max()),
        },
        "paired_effects": paired_effects.to_dict(orient="records"),
        "matched_sample_half_life": {
            "moltbook": m_half_life,
            "reddit": r_half_life,
        },
        "artifacts": {
            "tables": [
                str(pairs_out_csv),
                str(pairs_out_parquet),
                str(strata_out_csv),
                str(sample_flow_out_csv),
                str(balance_out_csv),
                str(balance_levels_out_csv),
                str(paired_effects_out_csv),
                str(half_life_out_csv),
                str(topic_map_out_csv),
            ],
            "figures": [
                str(balance_fig),
                str(flow_fig),
                str(effects_fig),
            ],
        },
    }

    summary_out = tables_dir / "analysis_summary.json"
    summary_out.write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": "analysis/08_cross_platform_matched_analysis.py",
        "config": asdict(cfg),
        "outputs_dir": str(run_dir),
        "analysis_summary_json": str(summary_out),
    }
    manifest_out = run_dir / "run_manifest.json"
    manifest_out.write_text(json.dumps(sanitize_for_json(manifest), indent=2), encoding="utf-8")

    print(f"Run ID: {cfg.run_id}")
    print(f"Matched pairs: {n_pairs}")
    print(f"Shared strata: {len(overlap_keys)} (with pairs: {matched_strata_with_pairs})")
    print(f"Summary: {summary_out}")
    print(f"Manifest: {manifest_out}")


if __name__ == "__main__":
    main()
