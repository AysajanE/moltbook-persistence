#!/usr/bin/env python3
"""Generate section-4 results-presentation artifacts for the EJOR revision.

Outputs:
- paper/tables/results_two_part_headline.csv
- paper/tables/results_timing_model_fit.csv
- paper/tables/results_periodicity_detectability_summary.csv
- paper/figures/reply_time_ecdf_logscale.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, special

DEFAULT_MOLTBOOK_SURVIVAL_PATH = Path(
    "data_features/moltbook_only/run_20260206-145240Z/survival_units.parquet"
)
DEFAULT_MOLTBOOK_THREAD_METRICS_PATH = Path(
    "data_features/moltbook_only/run_20260206-145240Z/thread_metrics.parquet"
)
DEFAULT_MOLTBOOK_THREAD_EVENTS_PATH = Path(
    "data_features/moltbook_only/run_20260206-145240Z/thread_events.parquet"
)
DEFAULT_MOLTBOOK_AGENTS_PATH = Path("data_curated/hf_archive/snapshot_20260204-234429Z/agents")
DEFAULT_REDDIT_SURVIVAL_PATH = Path(
    "data_features/reddit_only/attempt_scaled_20260206-142651Z/survival_units.parquet"
)
DEFAULT_REDDIT_THREAD_EVENTS_PATH = Path(
    "data_features/reddit_only/attempt_scaled_20260206-142651Z/thread_events.parquet"
)
DEFAULT_MOLTBOOK_DETECTABILITY_PATH = Path(
    "paper/tables/moltbook_periodicity_detectability_power.csv"
)
DEFAULT_MOLTBOOK_EVENT_TIME_TEST_PATH = Path(
    "paper/tables/moltbook_periodicity_event_time_test.csv"
)
DEFAULT_HEADLINE_OUT_PATH = Path("paper/tables/results_two_part_headline.csv")
DEFAULT_MODEL_FIT_OUT_PATH = Path("paper/tables/results_timing_model_fit.csv")
DEFAULT_PERIODICITY_SUMMARY_OUT_PATH = Path(
    "paper/tables/results_periodicity_detectability_summary.csv"
)
DEFAULT_CATEGORY_UNCERTAINTY_OUT_PATH = Path(
    "paper/tables/moltbook_results_category_uncertainty.csv"
)
DEFAULT_ECDF_OUT_PATH = Path("paper/figures/reply_time_ecdf_logscale.png")
DEFAULT_SEED = 20260208
DEFAULT_BOOTSTRAP_REPS = 400
DEFAULT_GAP_THRESHOLD_HOURS = 6.0
DEFAULT_PERIOD_HOURS = 4.0
DEFAULT_ALPHA_LEVEL = 0.05
DEFAULT_NULL_MC_REPS = 200_000
DEFAULT_POWER_SIM_REPS = 50_000
DEFAULT_KAPPA_GRID = tuple(np.round(np.arange(0.0, 3.0001, 0.2), 2).tolist())
DEFAULT_CATEGORY_ORDER = (
    "Social/Casual",
    "Philosophy/Meta",
    "Builder/Technical",
    "Creative",
    "Spam/Low-Signal",
    "Other",
)
REQUIRED_KEY_CATEGORIES = ("Social/Casual", "Philosophy/Meta")


@dataclass(frozen=True)
class Config:
    moltbook_survival_path: Path
    moltbook_thread_metrics_path: Path
    moltbook_thread_events_path: Path
    moltbook_agents_path: Path
    reddit_survival_path: Path
    reddit_thread_events_path: Path
    moltbook_detectability_path: Path
    moltbook_event_time_test_path: Path
    headline_out_path: Path
    model_fit_out_path: Path
    periodicity_summary_out_path: Path
    category_uncertainty_out_path: Path
    ecdf_out_path: Path
    seed: int
    bootstrap_reps: int
    gap_threshold_hours: float
    period_hours: float
    alpha_level: float
    null_mc_reps: int
    power_sim_reps: int
    kappa_grid: tuple[float, ...]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moltbook-survival-path", type=Path, default=DEFAULT_MOLTBOOK_SURVIVAL_PATH
    )
    parser.add_argument(
        "--moltbook-thread-metrics-path",
        type=Path,
        default=DEFAULT_MOLTBOOK_THREAD_METRICS_PATH,
    )
    parser.add_argument(
        "--moltbook-thread-events-path", type=Path, default=DEFAULT_MOLTBOOK_THREAD_EVENTS_PATH
    )
    parser.add_argument("--moltbook-agents-path", type=Path, default=DEFAULT_MOLTBOOK_AGENTS_PATH)
    parser.add_argument("--reddit-survival-path", type=Path, default=DEFAULT_REDDIT_SURVIVAL_PATH)
    parser.add_argument(
        "--reddit-thread-events-path", type=Path, default=DEFAULT_REDDIT_THREAD_EVENTS_PATH
    )
    parser.add_argument(
        "--moltbook-detectability-path", type=Path, default=DEFAULT_MOLTBOOK_DETECTABILITY_PATH
    )
    parser.add_argument(
        "--moltbook-event-time-test-path", type=Path, default=DEFAULT_MOLTBOOK_EVENT_TIME_TEST_PATH
    )
    parser.add_argument("--headline-out-path", type=Path, default=DEFAULT_HEADLINE_OUT_PATH)
    parser.add_argument("--model-fit-out-path", type=Path, default=DEFAULT_MODEL_FIT_OUT_PATH)
    parser.add_argument(
        "--periodicity-summary-out-path",
        type=Path,
        default=DEFAULT_PERIODICITY_SUMMARY_OUT_PATH,
    )
    parser.add_argument(
        "--category-uncertainty-out-path",
        type=Path,
        default=DEFAULT_CATEGORY_UNCERTAINTY_OUT_PATH,
    )
    parser.add_argument("--ecdf-out-path", type=Path, default=DEFAULT_ECDF_OUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--gap-threshold-hours", type=float, default=DEFAULT_GAP_THRESHOLD_HOURS)
    parser.add_argument("--period-hours", type=float, default=DEFAULT_PERIOD_HOURS)
    parser.add_argument("--alpha-level", type=float, default=DEFAULT_ALPHA_LEVEL)
    parser.add_argument("--null-mc-reps", type=int, default=DEFAULT_NULL_MC_REPS)
    parser.add_argument("--power-sim-reps", type=int, default=DEFAULT_POWER_SIM_REPS)
    parser.add_argument(
        "--kappa-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_KAPPA_GRID),
        help="Von Mises concentration values for detectability simulation.",
    )
    args = parser.parse_args()
    return Config(
        moltbook_survival_path=args.moltbook_survival_path,
        moltbook_thread_metrics_path=args.moltbook_thread_metrics_path,
        moltbook_thread_events_path=args.moltbook_thread_events_path,
        moltbook_agents_path=args.moltbook_agents_path,
        reddit_survival_path=args.reddit_survival_path,
        reddit_thread_events_path=args.reddit_thread_events_path,
        moltbook_detectability_path=args.moltbook_detectability_path,
        moltbook_event_time_test_path=args.moltbook_event_time_test_path,
        headline_out_path=args.headline_out_path,
        model_fit_out_path=args.model_fit_out_path,
        periodicity_summary_out_path=args.periodicity_summary_out_path,
        category_uncertainty_out_path=args.category_uncertainty_out_path,
        ecdf_out_path=args.ecdf_out_path,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
        gap_threshold_hours=args.gap_threshold_hours,
        period_hours=args.period_hours,
        alpha_level=args.alpha_level,
        null_mc_reps=args.null_mc_reps,
        power_sim_reps=args.power_sim_reps,
        kappa_grid=tuple(float(x) for x in args.kappa_grid),
    )


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def load_primary_survival(path: Path, *, drop_censor_boundary: bool) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = ["thread_id", "event_observed", "duration_hours"]
    if drop_censor_boundary:
        required.append("excluded_censor_boundary")
    require_columns(df, required, path.name)

    df["event_observed"] = (
        pd.to_numeric(df["event_observed"], errors="coerce").fillna(0).astype(int)
    )
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")

    if drop_censor_boundary:
        included = ~pd.to_numeric(df["excluded_censor_boundary"], errors="coerce").fillna(0).astype(
            bool
        )
        df = df.loc[included].copy()

    valid = np.isfinite(df["duration_hours"].to_numpy()) & (df["duration_hours"].to_numpy() > 0)
    return df.loc[valid].reset_index(drop=True)


def load_thread_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = [
        "thread_id",
        "submolt_category",
        "reentry_rate",
        "reciprocal_dyads_thread",
        "dyads_thread",
    ]
    require_columns(df, required, path.name)
    out = df[required].copy()
    out["thread_id"] = out["thread_id"].astype(str)
    out["submolt_category"] = out["submolt_category"].astype("string").fillna("Unknown").astype(str)
    out["reentry_rate"] = pd.to_numeric(out["reentry_rate"], errors="coerce")
    out["reciprocal_dyads_thread"] = pd.to_numeric(
        out["reciprocal_dyads_thread"], errors="coerce"
    ).fillna(0.0)
    out["dyads_thread"] = pd.to_numeric(out["dyads_thread"], errors="coerce").fillna(0.0)
    if out["thread_id"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate thread_id rows.")
    return out.reset_index(drop=True)


def ordered_category_labels(observed_labels: set[str]) -> list[str]:
    out: list[str] = []
    observed = {label for label in observed_labels if isinstance(label, str) and label.strip()}
    for label in DEFAULT_CATEGORY_ORDER:
        if label in observed or label in REQUIRED_KEY_CATEGORIES:
            out.append(label)
    out.extend(sorted(observed.difference(out)))
    return out


def attach_claimed_group(survival_df: pd.DataFrame, agents_path: Path) -> pd.DataFrame:
    out = survival_df.copy()
    require_columns(out, ["comment_agent_id"], "moltbook_survival_for_claims")
    agents = pd.read_parquet(agents_path)
    require_columns(agents, ["id", "is_claimed"], agents_path.name)
    claims = agents[["id", "is_claimed"]].rename(columns={"id": "comment_agent_id"})
    claims["is_claimed"] = pd.to_numeric(claims["is_claimed"], errors="coerce")
    out = out.merge(claims, on="comment_agent_id", how="left", validate="many_to_one")
    out["claimed_group"] = "Unknown"
    out.loc[out["is_claimed"] == 1, "claimed_group"] = "Claimed"
    out.loc[out["is_claimed"] == 0, "claimed_group"] = "Unclaimed"
    return out


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
        result = optimize.minimize(
            neg_loglik,
            start,
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-20.0, 20.0)],
        )
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    if not best.success:
        return {"success": False, "message": str(best.message)}

    alpha_hat = float(np.exp(best.x[0]))
    beta_hat = float(np.exp(best.x[1]))
    return {
        "success": True,
        "alpha": alpha_hat,
        "beta": beta_hat,
        "half_life_hours": float(np.log(2.0) / beta_hat),
    }


def implied_eventual_reply_probability(exponential_fit: dict[str, Any]) -> float:
    if not exponential_fit.get("success"):
        return float("nan")
    alpha = float(exponential_fit["alpha"])
    beta = float(exponential_fit["beta"])
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
        return float("nan")
    return float(1.0 - np.exp(-(alpha / beta)))


def conditional_quantile_from_fit_seconds(exponential_fit: dict[str, Any], p: float) -> float:
    if not exponential_fit.get("success"):
        return float("nan")
    alpha = float(exponential_fit["alpha"])
    beta = float(exponential_fit["beta"])
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
        return float("nan")
    mu = alpha / beta
    event_prob = float(1.0 - np.exp(-mu))
    if not np.isfinite(event_prob) or event_prob <= 0 or event_prob >= 1:
        return float("nan")
    target_hazard = -np.log(1.0 - p * event_prob)
    inside = 1.0 - (target_hazard / mu)
    if inside <= 0:
        return float("nan")
    return float((-np.log(inside) / beta) * 3600.0)


def percentile_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1.0 - alpha / 2))


def weighted_quantiles_from_sorted(
    sorted_values: np.ndarray,
    sorted_weights: np.ndarray,
    probs: tuple[float, ...],
) -> list[float]:
    total_weight = float(np.sum(sorted_weights))
    if total_weight <= 0:
        return [float("nan")] * len(probs)
    cumulative = np.cumsum(sorted_weights)
    out: list[float] = []
    for p in probs:
        threshold = p * total_weight
        idx = int(np.searchsorted(cumulative, threshold, side="left"))
        idx = min(max(idx, 0), len(sorted_values) - 1)
        out.append(float(sorted_values[idx]))
    return out


def bootstrap_headline_metrics(
    survival_primary: pd.DataFrame,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    durations_seconds = survival_primary["duration_hours"].to_numpy(dtype=float) * 3600.0
    event_observed = survival_primary["event_observed"].to_numpy(dtype=int)
    cluster = survival_primary["thread_id"].astype("category")
    cluster_codes = cluster.cat.codes.to_numpy(dtype=int)
    n_clusters = int(cluster.cat.categories.size)

    n_parent_comments = int(len(survival_primary))
    n_replied = int(np.sum(event_observed))
    incidence_prob = float(n_replied / n_parent_comments) if n_parent_comments else float("nan")

    conditional_seconds = durations_seconds[event_observed.astype(bool)]
    if conditional_seconds.size:
        obs_q10, obs_q50, obs_q90 = np.quantile(conditional_seconds, [0.10, 0.50, 0.90])
    else:
        obs_q10, obs_q50, obs_q90 = np.nan, np.nan, np.nan

    incidence_boot: list[float] = []
    q10_boot: list[float] = []
    q50_boot: list[float] = []
    q90_boot: list[float] = []

    if conditional_seconds.size:
        event_cluster_codes = cluster_codes[event_observed.astype(bool)]
        order = np.argsort(conditional_seconds, kind="stable")
        sorted_event_seconds = conditional_seconds[order]
        sorted_event_cluster_codes = event_cluster_codes[order]
    else:
        sorted_event_seconds = np.array([], dtype=float)
        sorted_event_cluster_codes = np.array([], dtype=int)

    for _ in range(reps):
        sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
        sampled_cluster_counts = np.bincount(sampled_clusters, minlength=n_clusters).astype(float)
        weights = sampled_cluster_counts[cluster_codes]
        weight_sum = float(np.sum(weights))
        incidence_boot.append(float(np.sum(weights * event_observed) / weight_sum))

        if sorted_event_seconds.size:
            sorted_event_weights = sampled_cluster_counts[sorted_event_cluster_codes]
            q10, q50, q90 = weighted_quantiles_from_sorted(
                sorted_values=sorted_event_seconds,
                sorted_weights=sorted_event_weights,
                probs=(0.10, 0.50, 0.90),
            )
            q10_boot.append(q10)
            q50_boot.append(q50)
            q90_boot.append(q90)

    incidence_ci_low, incidence_ci_high = percentile_ci(incidence_boot)
    q10_ci_low, q10_ci_high = percentile_ci(q10_boot)
    q50_ci_low, q50_ci_high = percentile_ci(q50_boot)
    q90_ci_low, q90_ci_high = percentile_ci(q90_boot)

    return {
        "n_parent_comments": n_parent_comments,
        "n_replied": n_replied,
        "reply_incidence_prob": incidence_prob,
        "reply_incidence_ci_low": incidence_ci_low,
        "reply_incidence_ci_high": incidence_ci_high,
        "conditional_p10_seconds": float(obs_q10),
        "conditional_p10_ci_low_seconds": q10_ci_low,
        "conditional_p10_ci_high_seconds": q10_ci_high,
        "conditional_p50_seconds": float(obs_q50),
        "conditional_p50_ci_low_seconds": q50_ci_low,
        "conditional_p50_ci_high_seconds": q50_ci_high,
        "conditional_p90_seconds": float(obs_q90),
        "conditional_p90_ci_low_seconds": q90_ci_low,
        "conditional_p90_ci_high_seconds": q90_ci_high,
        "conditional_event_seconds": conditional_seconds,
    }


def bootstrap_category_uncertainty_metrics(
    survival_subset: pd.DataFrame,
    thread_metrics_subset: pd.DataFrame,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    n_parents = int(len(survival_subset))
    n_replied = int(np.sum(survival_subset["event_observed"].to_numpy(dtype=int)))
    n_threads = int(thread_metrics_subset["thread_id"].nunique())
    if n_parents == 0 or n_threads == 0:
        return {
            "n_parents": n_parents,
            "n_replied": n_replied,
            "n_threads": n_threads,
            "reply_incidence": float("nan"),
            "reply_incidence_ci_low": float("nan"),
            "reply_incidence_ci_high": float("nan"),
            "conditional_t50_seconds": float("nan"),
            "conditional_t50_ci_low_seconds": float("nan"),
            "conditional_t50_ci_high_seconds": float("nan"),
            "conditional_t90_seconds": float("nan"),
            "conditional_t90_ci_low_seconds": float("nan"),
            "conditional_t90_ci_high_seconds": float("nan"),
            "pooled_reciprocity": float("nan"),
            "pooled_reciprocity_ci_low": float("nan"),
            "pooled_reciprocity_ci_high": float("nan"),
            "reentry_mean": float("nan"),
            "reentry_mean_ci_low": float("nan"),
            "reentry_mean_ci_high": float("nan"),
            "reentry_median": float("nan"),
            "reentry_median_ci_low": float("nan"),
            "reentry_median_ci_high": float("nan"),
        }

    thread_ids = pd.Index(
        sorted(
            set(survival_subset["thread_id"].astype(str).tolist())
            | set(thread_metrics_subset["thread_id"].astype(str).tolist())
        )
    )
    n_clusters = int(len(thread_ids))
    if n_clusters == 0:
        raise ValueError("Bootstrap requires at least one thread cluster.")
    cluster_map = pd.Series(np.arange(n_clusters), index=thread_ids)

    event_observed = survival_subset["event_observed"].to_numpy(dtype=float)
    durations_seconds = survival_subset["duration_hours"].to_numpy(dtype=float) * 3600.0
    survival_cluster_codes_raw = survival_subset["thread_id"].astype(str).map(cluster_map)
    if survival_cluster_codes_raw.isna().any():
        raise ValueError("Failed to map survival rows to thread-cluster ids.")
    survival_cluster_codes = survival_cluster_codes_raw.astype(int).to_numpy()

    event_mask = event_observed > 0
    conditional_seconds = durations_seconds[event_mask]
    conditional_cluster_codes = survival_cluster_codes[event_mask]
    if conditional_seconds.size:
        cond_order = np.argsort(conditional_seconds, kind="stable")
        conditional_seconds_sorted = conditional_seconds[cond_order]
        conditional_cluster_codes_sorted = conditional_cluster_codes[cond_order]
        conditional_t50_obs, conditional_t90_obs = np.quantile(conditional_seconds, [0.50, 0.90])
    else:
        conditional_seconds_sorted = np.array([], dtype=float)
        conditional_cluster_codes_sorted = np.array([], dtype=int)
        conditional_t50_obs, conditional_t90_obs = np.nan, np.nan

    reciprocal_dyads = thread_metrics_subset["reciprocal_dyads_thread"].to_numpy(dtype=float)
    dyads = thread_metrics_subset["dyads_thread"].to_numpy(dtype=float)
    metric_cluster_codes_raw = thread_metrics_subset["thread_id"].astype(str).map(cluster_map)
    if metric_cluster_codes_raw.isna().any():
        raise ValueError("Failed to map thread-metric rows to thread-cluster ids.")
    metric_cluster_codes = metric_cluster_codes_raw.astype(int).to_numpy()
    dyads_total = float(np.sum(dyads))
    pooled_reciprocity_obs = (
        float(np.sum(reciprocal_dyads) / dyads_total) if dyads_total > 0 else float("nan")
    )

    reentry_values = thread_metrics_subset["reentry_rate"].to_numpy(dtype=float)
    reentry_valid = np.isfinite(reentry_values)
    if np.any(reentry_valid):
        reentry_values_valid = reentry_values[reentry_valid]
        reentry_cluster_codes_valid = metric_cluster_codes[reentry_valid]
        reentry_order = np.argsort(reentry_values_valid, kind="stable")
        reentry_values_sorted = reentry_values_valid[reentry_order]
        reentry_cluster_codes_sorted = reentry_cluster_codes_valid[reentry_order]
        reentry_mean_obs = float(np.mean(reentry_values_valid))
        reentry_median_obs = float(np.median(reentry_values_valid))
    else:
        reentry_values_sorted = np.array([], dtype=float)
        reentry_cluster_codes_sorted = np.array([], dtype=int)
        reentry_mean_obs = float("nan")
        reentry_median_obs = float("nan")

    incidence_obs = float(n_replied / n_parents)

    incidence_boot: list[float] = []
    t50_boot: list[float] = []
    t90_boot: list[float] = []
    reciprocity_boot: list[float] = []
    reentry_mean_boot: list[float] = []
    reentry_median_boot: list[float] = []

    for _ in range(reps):
        sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
        sampled_cluster_counts = np.bincount(sampled_clusters, minlength=n_clusters).astype(float)

        surv_weights = sampled_cluster_counts[survival_cluster_codes]
        surv_weight_sum = float(np.sum(surv_weights))
        if surv_weight_sum > 0:
            incidence_boot.append(float(np.sum(surv_weights * event_observed) / surv_weight_sum))

        if conditional_seconds_sorted.size:
            cond_weights = sampled_cluster_counts[conditional_cluster_codes_sorted]
            t50, t90 = weighted_quantiles_from_sorted(
                sorted_values=conditional_seconds_sorted,
                sorted_weights=cond_weights,
                probs=(0.50, 0.90),
            )
            t50_boot.append(t50)
            t90_boot.append(t90)

        metric_weights = sampled_cluster_counts[metric_cluster_codes]
        dyads_weighted = float(np.sum(metric_weights * dyads))
        if dyads_weighted > 0:
            reciprocal_weighted = float(np.sum(metric_weights * reciprocal_dyads))
            reciprocity_boot.append(reciprocal_weighted / dyads_weighted)

        if reentry_values_sorted.size:
            reentry_weights = sampled_cluster_counts[reentry_cluster_codes_sorted]
            reentry_weight_sum = float(np.sum(reentry_weights))
            if reentry_weight_sum > 0:
                reentry_mean_boot.append(
                    float(np.sum(reentry_weights * reentry_values_sorted) / reentry_weight_sum)
                )
                reentry_median_boot.append(
                    weighted_quantiles_from_sorted(
                        sorted_values=reentry_values_sorted,
                        sorted_weights=reentry_weights,
                        probs=(0.50,),
                    )[0]
                )

    incidence_ci_low, incidence_ci_high = percentile_ci(incidence_boot)
    t50_ci_low, t50_ci_high = percentile_ci(t50_boot)
    t90_ci_low, t90_ci_high = percentile_ci(t90_boot)
    reciprocity_ci_low, reciprocity_ci_high = percentile_ci(reciprocity_boot)
    reentry_mean_ci_low, reentry_mean_ci_high = percentile_ci(reentry_mean_boot)
    reentry_median_ci_low, reentry_median_ci_high = percentile_ci(reentry_median_boot)

    return {
        "n_parents": n_parents,
        "n_replied": n_replied,
        "n_threads": n_threads,
        "reply_incidence": incidence_obs,
        "reply_incidence_ci_low": incidence_ci_low,
        "reply_incidence_ci_high": incidence_ci_high,
        "conditional_t50_seconds": float(conditional_t50_obs),
        "conditional_t50_ci_low_seconds": t50_ci_low,
        "conditional_t50_ci_high_seconds": t50_ci_high,
        "conditional_t90_seconds": float(conditional_t90_obs),
        "conditional_t90_ci_low_seconds": t90_ci_low,
        "conditional_t90_ci_high_seconds": t90_ci_high,
        "pooled_reciprocity": pooled_reciprocity_obs,
        "pooled_reciprocity_ci_low": reciprocity_ci_low,
        "pooled_reciprocity_ci_high": reciprocity_ci_high,
        "reentry_mean": reentry_mean_obs,
        "reentry_mean_ci_low": reentry_mean_ci_low,
        "reentry_mean_ci_high": reentry_mean_ci_high,
        "reentry_median": reentry_median_obs,
        "reentry_median_ci_low": reentry_median_ci_low,
        "reentry_median_ci_high": reentry_median_ci_high,
    }


def build_moltbook_category_uncertainty_table(
    cfg: Config,
    moltbook_survival: pd.DataFrame,
    thread_metrics: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        moltbook_survival,
        ["thread_id", "submolt_category", "event_observed", "duration_hours"],
        "moltbook_survival_for_category_uncertainty",
    )
    require_columns(
        thread_metrics,
        [
            "thread_id",
            "submolt_category",
            "reentry_rate",
            "reciprocal_dyads_thread",
            "dyads_thread",
        ],
        "moltbook_thread_metrics_for_category_uncertainty",
    )

    survival = moltbook_survival.copy()
    survival["thread_id"] = survival["thread_id"].astype(str)
    survival["submolt_category"] = (
        survival["submolt_category"].astype("string").fillna("Unknown").astype(str)
    )

    metrics = thread_metrics.copy()
    metrics["thread_id"] = metrics["thread_id"].astype(str)
    metrics["submolt_category"] = (
        metrics["submolt_category"].astype("string").fillna("Unknown").astype(str)
    )

    observed_categories = set(survival["submolt_category"].tolist()) | set(
        metrics["submolt_category"].tolist()
    )
    category_labels = ordered_category_labels(observed_categories)

    group_specs: list[tuple[str, str]] = [("overall", "Overall")]
    group_specs.extend(("submolt_category", category) for category in category_labels)

    rows: list[dict[str, Any]] = []
    for i, (group_family, label) in enumerate(group_specs):
        if group_family == "overall":
            survival_subset = survival
            thread_subset = metrics
        else:
            survival_subset = survival[survival["submolt_category"] == label].copy()
            thread_subset = metrics[metrics["submolt_category"] == label].copy()

        bootstrap_seed = int(cfg.seed + 2101 + i)
        row = bootstrap_category_uncertainty_metrics(
            survival_subset=survival_subset,
            thread_metrics_subset=thread_subset,
            rng=np.random.default_rng(bootstrap_seed),
            reps=cfg.bootstrap_reps,
        )
        row.update(
            {
                "group_family": group_family,
                "submolt_category": label,
                "bootstrap_reps": int(cfg.bootstrap_reps),
                "bootstrap_cluster": "thread_id",
                "ci_alpha": 0.05,
                "analysis_seed": int(cfg.seed),
                "bootstrap_seed": bootstrap_seed,
                "input_survival_path": str(cfg.moltbook_survival_path),
                "input_thread_metrics_path": str(cfg.moltbook_thread_metrics_path),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out["group_order"] = out["group_family"].map({"overall": 0, "submolt_category": 1}).fillna(2)
    out["category_order"] = out["submolt_category"].map(
        {label: i for i, label in enumerate(["Overall", *category_labels])}
    ).fillna(10_000)
    out = out.sort_values(["group_order", "category_order"], kind="stable").drop(
        columns=["group_order", "category_order"]
    )
    return out.reset_index(drop=True)


def to_minutes(seconds: float) -> float:
    return float(seconds / 60.0) if np.isfinite(seconds) else float("nan")


def to_hours(seconds: float) -> float:
    return float(seconds / 3600.0) if np.isfinite(seconds) else float("nan")


def format_half_life_minutes_or_hours(half_life_hours: float) -> str:
    if not np.isfinite(half_life_hours):
        return "nan"
    if half_life_hours < 1.0:
        return f"{half_life_hours * 60.0:.6g} minutes"
    return f"{half_life_hours:.6g} hours"


def longest_contiguous_segment(times_utc: pd.Series, gap_threshold_hours: float) -> dict[str, Any]:
    timestamps = pd.to_datetime(times_utc, utc=True, errors="coerce").dropna().sort_values()
    if timestamps.empty:
        raise ValueError("No valid timestamps for periodicity detectability computation.")

    arr_ns = timestamps.astype("int64", copy=False).to_numpy()
    gap_seconds = np.diff(arr_ns) / 1e9
    break_idx = np.where(gap_seconds > gap_threshold_hours * 3600.0)[0]

    starts = np.r_[0, break_idx + 1]
    ends = np.r_[break_idx, len(arr_ns) - 1]
    sizes = (ends - starts + 1).astype(int)
    durations = (arr_ns[ends] - arr_ns[starts]) / 1e9

    choice = int(np.lexsort((durations, sizes))[-1])
    start_i = int(starts[choice])
    end_i = int(ends[choice])
    chosen = arr_ns[start_i : end_i + 1]

    return {
        "segment_event_count": int(chosen.size),
        "segment_duration_hours": float((chosen[-1] - chosen[0]) / (3600.0 * 1e9)),
    }


def null_z_monte_carlo_normal(n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(n / 2.0)
    c = rng.normal(loc=0.0, scale=std, size=reps)
    s = rng.normal(loc=0.0, scale=std, size=reps)
    return (c * c + s * s) / n


def simulate_detectability(
    n_events: int,
    z_critical: float,
    kappa_grid: tuple[float, ...],
    reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kappa in kappa_grid:
        rho = 0.0 if kappa < 1e-12 else float(special.i1(kappa) / special.i0(kappa))
        noncentrality = 2.0 * n_events * rho * rho
        z_sim = 0.5 * rng.noncentral_chisquare(df=2.0, nonc=noncentrality, size=reps)
        rows.append({"kappa": float(kappa), "estimated_power": float(np.mean(z_sim >= z_critical))})
    return pd.DataFrame(rows).sort_values("kappa", kind="stable").reset_index(drop=True)


def compute_detectability_for_events(
    cfg: Config, thread_events_path: Path, seed_offset: int
) -> dict[str, Any]:
    events = pd.read_parquet(thread_events_path)
    require_columns(events, ["created_at_utc"], thread_events_path.name)
    segment = longest_contiguous_segment(events["created_at_utc"], cfg.gap_threshold_hours)
    n_events = int(segment["segment_event_count"])

    rng_null = np.random.default_rng(cfg.seed + seed_offset)
    z_null = null_z_monte_carlo_normal(n=n_events, reps=cfg.null_mc_reps, rng=rng_null)
    z_critical = float(np.quantile(z_null, 1.0 - cfg.alpha_level))

    rng_power = np.random.default_rng(cfg.seed + seed_offset + 1)
    power_table = simulate_detectability(
        n_events=n_events,
        z_critical=z_critical,
        kappa_grid=cfg.kappa_grid,
        reps=cfg.power_sim_reps,
        rng=rng_power,
    )
    above = power_table.loc[power_table["estimated_power"] >= 0.8, "kappa"]
    kappa_star = float(above.iloc[0]) if not above.empty else float("nan")
    null_size = float(power_table.loc[power_table["kappa"] == 0.0, "estimated_power"].iloc[0])

    return {
        "period_hours": float(cfg.period_hours),
        "segment_duration_hours": float(segment["segment_duration_hours"]),
        "segment_event_count": n_events,
        "kappa_star_power_0p8": kappa_star,
        "null_size_at_kappa0": null_size,
        "critical_z_at_alpha": z_critical,
        "alpha_level": float(cfg.alpha_level),
        "power_sim_reps": int(cfg.power_sim_reps),
        "null_mc_reps": int(cfg.null_mc_reps),
        "seed": int(cfg.seed),
    }


def maybe_extract_moltbook_detectability(cfg: Config) -> dict[str, Any] | None:
    if not cfg.moltbook_detectability_path.exists():
        return None
    table = pd.read_csv(cfg.moltbook_detectability_path)
    if table.empty:
        return None
    if "kappa_power_crosses_0p8" in table.columns:
        kappa_vals = pd.to_numeric(table["kappa_power_crosses_0p8"], errors="coerce").dropna()
        kappa_star = float(kappa_vals.iloc[0]) if not kappa_vals.empty else float("nan")
    else:
        valid = table.loc[pd.to_numeric(table["estimated_power"], errors="coerce") >= 0.8, "kappa"]
        kappa_star = (
            float(pd.to_numeric(valid, errors="coerce").iloc[0]) if len(valid) else float("nan")
        )
    kappa0 = table.loc[np.isclose(pd.to_numeric(table["kappa"], errors="coerce"), 0.0)]
    null_size = (
        float(pd.to_numeric(kappa0["estimated_power"], errors="coerce").iloc[0])
        if not kappa0.empty
        else float("nan")
    )
    first = table.iloc[0]
    row = {
        "period_hours": float(first.get("period_hours", np.nan)),
        "segment_event_count": float(first.get("segment_event_count", np.nan)),
        "critical_z_at_alpha": float(first.get("critical_z_at_alpha", np.nan)),
        "alpha_level": float(first.get("alpha_level", np.nan)),
        "kappa_star_power_0p8": kappa_star,
        "null_size_at_kappa0": null_size,
        "power_sim_reps": int(cfg.power_sim_reps),
        "null_mc_reps": int(cfg.null_mc_reps),
        "seed": int(cfg.seed),
    }
    if cfg.moltbook_event_time_test_path.exists():
        event_time = pd.read_csv(cfg.moltbook_event_time_test_path)
        if not event_time.empty and "segment_duration_hours" in event_time.columns:
            row["segment_duration_hours"] = float(
                pd.to_numeric(event_time["segment_duration_hours"], errors="coerce").iloc[0]
            )
    return row


def make_ecdf_figure(event_seconds_by_platform: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for platform, seconds in event_seconds_by_platform.items():
        vals = np.sort(seconds[np.isfinite(seconds) & (seconds > 0)])
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        ax.step(vals, y, where="post", linewidth=2.0, label=f"{platform} (n={vals.size:,})")

    for x, label in ((10.0, "10 s"), (60.0, "1 m"), (3600.0, "1 h")):
        ax.axvline(x=x, color="0.4", linestyle="--", linewidth=1.0)
        ax.text(x, 0.02, label, rotation=90, va="bottom", ha="right", fontsize=9, color="0.35")

    ax.set_xscale("log")
    ax.set_xlabel("Conditional time to first reply (seconds, log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title("Conditional reply-time ECDF by platform")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    cfg = parse_args()

    moltbook_survival = load_primary_survival(
        cfg.moltbook_survival_path, drop_censor_boundary=False
    )
    moltbook_thread_metrics = load_thread_metrics(cfg.moltbook_thread_metrics_path)
    moltbook_survival = attach_claimed_group(moltbook_survival, cfg.moltbook_agents_path)
    reddit_survival = load_primary_survival(cfg.reddit_survival_path, drop_censor_boundary=True)

    headline_specs = [
        ("Moltbook", "Overall", moltbook_survival),
        (
            "Moltbook",
            "Claimed",
            moltbook_survival.loc[moltbook_survival["claimed_group"] == "Claimed"].copy(),
        ),
        (
            "Moltbook",
            "Unclaimed",
            moltbook_survival.loc[moltbook_survival["claimed_group"] == "Unclaimed"].copy(),
        ),
        ("Reddit", "Overall", reddit_survival),
    ]

    headline_rows: list[dict[str, Any]] = []
    ecdf_data: dict[str, np.ndarray] = {}
    for i, (platform, group_label, survival_df) in enumerate(headline_specs):
        bootstrap_rng = np.random.default_rng(cfg.seed + 11 + i)
        headline = bootstrap_headline_metrics(
            survival_primary=survival_df,
            rng=bootstrap_rng,
            reps=cfg.bootstrap_reps,
        )
        exp_fit = fit_exponential_decay(
            durations_hours=survival_df["duration_hours"].to_numpy(dtype=float),
            event_observed=survival_df["event_observed"].to_numpy(dtype=float),
        )
        implied_prob = implied_eventual_reply_probability(exp_fit)
        half_life_hours = (
            float(exp_fit["half_life_hours"]) if exp_fit.get("success") else float("nan")
        )
        q10_obs = float(headline["conditional_p10_seconds"])
        q50_obs = float(headline["conditional_p50_seconds"])
        q90_obs = float(headline["conditional_p90_seconds"])

        headline_rows.append(
            {
                "platform": platform,
                "group_label": group_label,
                "n_parent_comments": int(headline["n_parent_comments"]),
                "n_replied": int(headline["n_replied"]),
                "reply_incidence_prob": float(headline["reply_incidence_prob"]),
                "reply_incidence_ci_low": float(headline["reply_incidence_ci_low"]),
                "reply_incidence_ci_high": float(headline["reply_incidence_ci_high"]),
                "conditional_p10_seconds": q10_obs,
                "conditional_p10_ci_low_seconds": float(headline["conditional_p10_ci_low_seconds"]),
                "conditional_p10_ci_high_seconds": float(
                    headline["conditional_p10_ci_high_seconds"]
                ),
                "conditional_p50_seconds": q50_obs,
                "conditional_p50_ci_low_seconds": float(headline["conditional_p50_ci_low_seconds"]),
                "conditional_p50_ci_high_seconds": float(
                    headline["conditional_p50_ci_high_seconds"]
                ),
                "conditional_p90_seconds": q90_obs,
                "conditional_p90_ci_low_seconds": float(headline["conditional_p90_ci_low_seconds"]),
                "conditional_p90_ci_high_seconds": float(
                    headline["conditional_p90_ci_high_seconds"]
                ),
                "conditional_p10_minutes": to_minutes(q10_obs),
                "conditional_p10_ci_low_minutes": to_minutes(
                    float(headline["conditional_p10_ci_low_seconds"])
                ),
                "conditional_p10_ci_high_minutes": to_minutes(
                    float(headline["conditional_p10_ci_high_seconds"])
                ),
                "conditional_p50_minutes": to_minutes(q50_obs),
                "conditional_p50_ci_low_minutes": to_minutes(
                    float(headline["conditional_p50_ci_low_seconds"])
                ),
                "conditional_p50_ci_high_minutes": to_minutes(
                    float(headline["conditional_p50_ci_high_seconds"])
                ),
                "conditional_p90_minutes": to_minutes(q90_obs),
                "conditional_p90_ci_low_minutes": to_minutes(
                    float(headline["conditional_p90_ci_low_seconds"])
                ),
                "conditional_p90_ci_high_minutes": to_minutes(
                    float(headline["conditional_p90_ci_high_seconds"])
                ),
                "conditional_p10_hours": to_hours(q10_obs),
                "conditional_p50_hours": to_hours(q50_obs),
                "conditional_p90_hours": to_hours(q90_obs),
                "half_life_minutes_or_hours": format_half_life_minutes_or_hours(half_life_hours),
                "half_life_minutes": half_life_hours * 60.0
                if np.isfinite(half_life_hours)
                else np.nan,
                "half_life_hours": half_life_hours,
                "implied_eventual_reply_probability": implied_prob,
                "implied_eventual_reply_probability_note": (
                    "model-implied diagnostic under exponential kernel; not an estimand"
                ),
                "bootstrap_reps": int(cfg.bootstrap_reps),
                "bootstrap_cluster": "thread_id",
                "analysis_seed": int(cfg.seed),
                "bootstrap_seed": int(cfg.seed + 11 + i),
                "input_survival_path": (
                    str(cfg.moltbook_survival_path)
                    if platform == "Moltbook"
                    else str(cfg.reddit_survival_path)
                ),
                "input_agents_path": str(cfg.moltbook_agents_path),
            }
        )
        if group_label == "Overall":
            ecdf_data[platform] = np.asarray(headline["conditional_event_seconds"], dtype=float)

    make_ecdf_figure(ecdf_data, cfg.ecdf_out_path)

    model_fit_rows: list[dict[str, Any]] = []
    for platform, survival_df, input_path in (
        ("Moltbook", moltbook_survival, cfg.moltbook_survival_path),
        ("Reddit", reddit_survival, cfg.reddit_survival_path),
    ):
        headline = bootstrap_headline_metrics(
            survival_primary=survival_df,
            rng=np.random.default_rng(cfg.seed + 901),
            reps=1,
        )
        exp_fit = fit_exponential_decay(
            durations_hours=survival_df["duration_hours"].to_numpy(dtype=float),
            event_observed=survival_df["event_observed"].to_numpy(dtype=float),
        )
        implied_prob = implied_eventual_reply_probability(exp_fit)
        q10_obs = float(headline["conditional_p10_seconds"])
        q50_obs = float(headline["conditional_p50_seconds"])
        q90_obs = float(headline["conditional_p90_seconds"])
        q10_fit = conditional_quantile_from_fit_seconds(exp_fit, 0.10)
        q50_fit = conditional_quantile_from_fit_seconds(exp_fit, 0.50)
        q90_fit = conditional_quantile_from_fit_seconds(exp_fit, 0.90)
        model_fit_rows.append(
            {
                "platform": platform,
                "n_parent_comments": int(len(survival_df)),
                "n_replied": int(np.sum(survival_df["event_observed"].to_numpy(dtype=int))),
                "event_probability_observed": float(headline["reply_incidence_prob"]),
                "event_probability_fitted": implied_prob,
                "event_probability_residual_fitted_minus_observed": implied_prob
                - float(headline["reply_incidence_prob"]),
                "conditional_p10_seconds_observed": q10_obs,
                "conditional_p10_seconds_fitted": q10_fit,
                "conditional_p50_seconds_observed": q50_obs,
                "conditional_p50_seconds_fitted": q50_fit,
                "conditional_p90_seconds_observed": q90_obs,
                "conditional_p90_seconds_fitted": q90_fit,
                "conditional_p10_minutes_observed": to_minutes(q10_obs),
                "conditional_p10_minutes_fitted": to_minutes(q10_fit),
                "conditional_p50_minutes_observed": to_minutes(q50_obs),
                "conditional_p50_minutes_fitted": to_minutes(q50_fit),
                "conditional_p90_minutes_observed": to_minutes(q90_obs),
                "conditional_p90_minutes_fitted": to_minutes(q90_fit),
                "conditional_p10_hours_observed": to_hours(q10_obs),
                "conditional_p10_hours_fitted": to_hours(q10_fit),
                "conditional_p50_hours_observed": to_hours(q50_obs),
                "conditional_p50_hours_fitted": to_hours(q50_fit),
                "conditional_p90_hours_observed": to_hours(q90_obs),
                "conditional_p90_hours_fitted": to_hours(q90_fit),
                "timing_model_alpha_per_hour": float(exp_fit["alpha"])
                if exp_fit.get("success")
                else np.nan,
                "timing_model_beta_per_hour": float(exp_fit["beta"])
                if exp_fit.get("success")
                else np.nan,
                "timing_model_half_life_hours": (
                    float(exp_fit["half_life_hours"]) if exp_fit.get("success") else np.nan
                ),
                "analysis_seed": int(cfg.seed),
                "input_survival_path": str(input_path),
            }
        )

    moltbook_detect = maybe_extract_moltbook_detectability(cfg)
    if moltbook_detect is None:
        moltbook_detect = compute_detectability_for_events(
            cfg=cfg,
            thread_events_path=cfg.moltbook_thread_events_path,
            seed_offset=101,
        )
        moltbook_detect["detectability_source"] = "computed_from_thread_events"
    else:
        moltbook_detect["detectability_source"] = str(cfg.moltbook_detectability_path)

    reddit_detect = compute_detectability_for_events(
        cfg=cfg,
        thread_events_path=cfg.reddit_thread_events_path,
        seed_offset=303,
    )
    reddit_detect["detectability_source"] = "computed_from_thread_events"

    periodicity_df = pd.DataFrame(
        [
            {
                "platform": "Moltbook",
                **moltbook_detect,
                "input_thread_events_path": str(cfg.moltbook_thread_events_path),
            },
            {
                "platform": "Reddit",
                **reddit_detect,
                "input_thread_events_path": str(cfg.reddit_thread_events_path),
            },
        ]
    )

    headline_df = pd.DataFrame(headline_rows)
    headline_df["platform_order"] = headline_df["platform"].map({"Moltbook": 0, "Reddit": 1})
    headline_df["group_order"] = headline_df["group_label"].map(
        {"Overall": 0, "Claimed": 1, "Unclaimed": 2}
    )
    headline_df = headline_df.sort_values(["platform_order", "group_order"]).drop(
        columns=["platform_order", "group_order"]
    )

    model_fit_df = pd.DataFrame(model_fit_rows).sort_values("platform")
    category_uncertainty_df = build_moltbook_category_uncertainty_table(
        cfg=cfg,
        moltbook_survival=moltbook_survival,
        thread_metrics=moltbook_thread_metrics,
    )

    write_csv(headline_df, cfg.headline_out_path)
    write_csv(model_fit_df, cfg.model_fit_out_path)
    write_csv(periodicity_df, cfg.periodicity_summary_out_path)
    write_csv(category_uncertainty_df, cfg.category_uncertainty_out_path)

    print(f"Wrote {cfg.headline_out_path}")
    print(f"Wrote {cfg.model_fit_out_path}")
    print(f"Wrote {cfg.periodicity_summary_out_path}")
    print(f"Wrote {cfg.category_uncertainty_out_path}")
    print(f"Wrote {cfg.ecdf_out_path}")


if __name__ == "__main__":
    main()
