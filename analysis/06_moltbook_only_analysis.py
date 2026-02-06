#!/usr/bin/env python3
"""Run Moltbook-only analysis and generate paper-ready artifacts.

This script implements a reproducible, end-to-end analysis workflow on the curated
Moltbook Observatory Archive snapshot, including:
1) thread reconstruction and geometry metrics,
2) interaction half-life estimation (exponential + Weibull),
3) periodicity analysis on aggregate arrivals,
4) manuscript-facing figures/tables and machine-readable summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter, WeibullFitter
from scipy import optimize, signal

DEFAULT_SNAPSHOT_ROOT = Path("data_curated/hf_archive/snapshot_20260204-234429Z")
DEFAULT_DATA_FEATURES_ROOT = Path("data_features/moltbook_only")
DEFAULT_OUTPUTS_ROOT = Path("outputs/moltbook_only")
DEFAULT_SEED = 20260206
DEFAULT_BOOTSTRAP_REPS = 400
DEFAULT_AR1_SIMS = 2000
DEFAULT_CENSOR_BOUNDARY_HOURS = 4.0
DEFAULT_GAP_THRESHOLD_HOURS = 6.0
DEFAULT_PERIODICITY_BIN_MINUTES = 15
DEFAULT_MIN_AGENT_COMMENTS_FOR_ACF = 10

CATEGORY_ORDER = [
    "Builder/Technical",
    "Philosophy/Meta",
    "Social/Casual",
    "Creative",
    "Spam/Low-Signal",
    "Other",
]

# Simple, deterministic keyword dictionaries used for transparent categorization.
SPAM_KEYWORDS = {
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
}
BUILDER_KEYWORDS = {
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
}
PHILOSOPHY_KEYWORDS = {
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
}
CREATIVE_KEYWORDS = {
    "creative",
    "creativeprojects",
    "music",
    "poetry",
    "shakespeare",
    "story",
    "theatre",
    "writing",
}
SOCIAL_KEYWORDS = {
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
}


@dataclass(frozen=True)
class Config:
    snapshot_root: Path
    data_features_root: Path
    outputs_root: Path
    run_id: str
    seed: int
    bootstrap_reps: int
    ar1_sims: int
    censor_boundary_hours: float
    gap_threshold_hours: float
    periodicity_bin_minutes: int
    min_agent_comments_for_acf: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Path to curated HF snapshot root.",
    )
    parser.add_argument(
        "--data-features-root",
        type=Path,
        default=DEFAULT_DATA_FEATURES_ROOT,
        help="Root for derived feature outputs.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Root for analysis tables/figures/manifests.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run identifier. Defaults to UTC timestamp.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed.")
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help="Cluster-bootstrap repetitions for uncertainty intervals.",
    )
    parser.add_argument(
        "--ar1-sims",
        type=int,
        default=DEFAULT_AR1_SIMS,
        help="AR(1) simulation count for Fisher g null calibration.",
    )
    parser.add_argument(
        "--censor-boundary-hours",
        type=float,
        default=DEFAULT_CENSOR_BOUNDARY_HOURS,
        help=(
            "Exclude survival units with parent comment created within this many hours "
            "of observation end in the primary estimate."
        ),
    )
    parser.add_argument(
        "--gap-threshold-hours",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_HOURS,
        help="Timestamp gap threshold used to segment periodicity windows.",
    )
    parser.add_argument(
        "--periodicity-bin-minutes",
        type=int,
        default=DEFAULT_PERIODICITY_BIN_MINUTES,
        help="Bin width (minutes) for aggregate periodicity time series.",
    )
    parser.add_argument(
        "--min-agent-comments-for-acf",
        type=int,
        default=DEFAULT_MIN_AGENT_COMMENTS_FOR_ACF,
        help="Minimum comments for agent-level lag-4h autocorrelation eligibility.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%d-%H%M%SZ")
    return Config(
        snapshot_root=args.snapshot_root,
        data_features_root=args.data_features_root,
        outputs_root=args.outputs_root,
        run_id=run_id,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
        ar1_sims=args.ar1_sims,
        censor_boundary_hours=args.censor_boundary_hours,
        gap_threshold_hours=args.gap_threshold_hours,
        periodicity_bin_minutes=args.periodicity_bin_minutes,
        min_agent_comments_for_acf=args.min_agent_comments_for_acf,
    )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def dedupe_latest(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if "dump_date" not in df.columns:
        return df.drop_duplicates(subset=[key], keep="last").copy()
    work = df.copy()
    work["dump_date"] = pd.to_datetime(work["dump_date"], errors="coerce")
    work = work.sort_values(["dump_date", key], kind="stable")
    work = work.drop_duplicates(subset=[key], keep="last")
    return work.reset_index(drop=True)


def tokenize_submolt(name: str) -> set[str]:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    return {tok for tok in tokens if tok}


def categorize_submolt(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Other"
    lower = name.strip().lower()
    tokens = tokenize_submolt(lower)

    def hit(keywords: set[str]) -> bool:
        return bool(tokens & keywords) or any(k in lower for k in keywords)

    if hit(SPAM_KEYWORDS):
        return "Spam/Low-Signal"
    if hit(BUILDER_KEYWORDS):
        return "Builder/Technical"
    if hit(PHILOSOPHY_KEYWORDS):
        return "Philosophy/Meta"
    if hit(CREATIVE_KEYWORDS):
        return "Creative"
    if hit(SOCIAL_KEYWORDS):
        return "Social/Casual"
    return "Other"


def prepare_tables(
    snapshot_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comments = pd.read_parquet(
        snapshot_root / "canonical/comments_latest.parquet",
        columns=["id", "post_id", "parent_id", "agent_id", "created_at_utc"],
    ).rename(columns={"id": "comment_id", "agent_id": "comment_agent_id"})
    comments["created_at_utc"] = to_utc(comments["created_at_utc"])

    posts = pd.read_parquet(
        snapshot_root / "posts",
        columns=[
            "id",
            "agent_id",
            "submolt",
            "score",
            "comment_count",
            "created_at_utc",
            "dump_date",
        ],
    )
    posts = dedupe_latest(posts, key="id").rename(
        columns={"id": "thread_id", "agent_id": "post_agent_id", "score": "post_score"}
    )
    posts["post_created_at_utc"] = to_utc(posts["created_at_utc"])
    posts = posts.drop(columns=["created_at_utc"])
    posts["submolt_category"] = posts["submolt"].map(categorize_submolt)

    agents = pd.read_parquet(
        snapshot_root / "agents",
        columns=["id", "karma", "follower_count", "dump_date"],
    )
    agents = dedupe_latest(agents, key="id").rename(columns={"id": "agent_id"})

    submolts = pd.read_parquet(
        snapshot_root / "submolts",
        columns=["name", "subscriber_count", "post_count", "dump_date"],
    )
    submolts = dedupe_latest(submolts, key="name").rename(columns={"name": "submolt"})

    return comments, posts, agents, submolts


def compute_depths(events: pd.DataFrame) -> pd.Series:
    depth = pd.Series(np.nan, index=events.index, dtype="float64")
    grouped = events.groupby("thread_id", sort=False).groups
    for thread_id, idx in grouped.items():
        _ = thread_id
        sub = events.loc[list(idx), ["comment_id", "parent_id", "created_at_utc"]].copy()
        sub = sub.sort_values(["created_at_utc", "comment_id"], kind="stable")
        depth_map: dict[str, int] = {}
        for row in sub.itertuples(index=False):
            if pd.isna(row.parent_id):
                this_depth = 1
            else:
                parent_depth = depth_map.get(row.parent_id)
                this_depth = parent_depth + 1 if parent_depth is not None else -1
            depth_map[row.comment_id] = this_depth
        sub_depth = pd.Series(depth_map, name="depth")
        mapped = events.loc[list(idx), "comment_id"].map(sub_depth)
        depth.loc[list(idx)] = mapped.to_numpy(dtype="float64")
    return depth


def build_thread_events(comments: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    events = comments.merge(
        posts[
            [
                "thread_id",
                "post_agent_id",
                "submolt",
                "submolt_category",
                "post_score",
                "comment_count",
                "post_created_at_utc",
            ]
        ],
        left_on="post_id",
        right_on="thread_id",
        how="inner",
        validate="many_to_one",
    ).drop(columns=["post_id"])

    parent_time = events.set_index("comment_id")["created_at_utc"]
    parent_agent = events.set_index("comment_id")["comment_agent_id"]

    events["parent_type"] = np.where(events["parent_id"].isna(), "post", "comment")
    events["parent_created_at_utc"] = events["parent_id"].map(parent_time)
    events["parent_agent_id"] = events["parent_id"].map(parent_agent)

    root_mask = events["parent_id"].isna()
    events.loc[root_mask, "parent_created_at_utc"] = events.loc[root_mask, "post_created_at_utc"]
    events.loc[root_mask, "parent_agent_id"] = events.loc[root_mask, "post_agent_id"]

    events["lag_since_parent_hours"] = (
        events["created_at_utc"] - events["parent_created_at_utc"]
    ).dt.total_seconds() / 3600.0
    events["lag_since_post_hours"] = (
        events["created_at_utc"] - events["post_created_at_utc"]
    ).dt.total_seconds() / 3600.0

    events["depth"] = compute_depths(events).astype("Int64")

    child_count_by_comment = events["parent_id"].value_counts(dropna=True)
    events["n_children"] = events["comment_id"].map(child_count_by_comment).fillna(0).astype(int)

    return events


def reentry_stats_for_thread(group: pd.DataFrame) -> tuple[float, int, int]:
    ordered = group.sort_values(["created_at_utc", "comment_id"], kind="stable")
    seen: set[str] = set()
    reentry_count = 0
    known_count = 0
    for agent_id in ordered["comment_agent_id"]:
        if pd.isna(agent_id):
            continue
        known_count += 1
        if agent_id in seen:
            reentry_count += 1
        seen.add(str(agent_id))
    rate = reentry_count / known_count if known_count > 0 else np.nan
    return rate, reentry_count, known_count


def reciprocal_chain_lengths_for_thread(group: pd.DataFrame) -> list[int]:
    ordered = group.sort_values(["created_at_utc", "comment_id"], kind="stable")
    edge_seq: list[tuple[str, str]] = []
    for source, target in zip(ordered["comment_agent_id"], ordered["parent_agent_id"], strict=True):
        if pd.isna(source) or pd.isna(target):
            continue
        s = str(source)
        t = str(target)
        if s == t:
            continue
        edge_seq.append((s, t))

    if not edge_seq:
        return []

    out: list[int] = []
    prev_pair: tuple[str, str] | None = None
    prev_dir: tuple[str, str] | None = None
    chain_len = 0
    for edge in edge_seq:
        pair = tuple(sorted(edge))
        if prev_pair == pair and prev_dir == (edge[1], edge[0]):
            chain_len += 1
        else:
            if chain_len >= 2:
                out.append(chain_len)
            chain_len = 1
        prev_pair = pair
        prev_dir = edge
    if chain_len >= 2:
        out.append(chain_len)
    return out


def reciprocity_for_thread(group: pd.DataFrame) -> tuple[float, int, int]:
    edges = []
    for source, target in zip(group["comment_agent_id"], group["parent_agent_id"], strict=True):
        if pd.isna(source) or pd.isna(target):
            continue
        s = str(source)
        t = str(target)
        if s == t:
            continue
        edges.append((s, t))
    if not edges:
        return np.nan, 0, 0
    edge_set = set(edges)
    dyads = {tuple(sorted((a, b))) for a, b in edge_set}
    reciprocal = 0
    for a, b in dyads:
        if (a, b) in edge_set and (b, a) in edge_set:
            reciprocal += 1
    return reciprocal / len(dyads), reciprocal, len(dyads)


def build_thread_metrics(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = (
        events.groupby("thread_id", as_index=False)
        .agg(
            n_comments=("comment_id", "size"),
            n_unique_agents=("comment_agent_id", pd.Series.nunique),
            depth_max=("depth", "max"),
            depth_mean=("depth", "mean"),
            first_comment_at=("created_at_utc", "min"),
            last_comment_at=("created_at_utc", "max"),
            post_created_at_utc=("post_created_at_utc", "first"),
            submolt=("submolt", "first"),
            submolt_category=("submolt_category", "first"),
        )
        .copy()
    )
    base["thread_duration_hours"] = (
        base["last_comment_at"] - base["post_created_at_utc"]
    ).dt.total_seconds() / 3600.0

    reentry_rate_map: dict[str, float] = {}
    reentry_count_map: dict[str, int] = {}
    reentry_known_map: dict[str, int] = {}
    reciprocity_rate_map: dict[str, float] = {}
    reciprocity_dyads_map: dict[str, int] = {}
    reciprocity_reciprocal_map: dict[str, int] = {}
    chain_lengths_all: list[int] = []

    for thread_id, group in events.groupby("thread_id", sort=False):
        tid = str(thread_id)
        rate, cnt, known = reentry_stats_for_thread(group)
        reentry_rate_map[tid] = rate
        reentry_count_map[tid] = cnt
        reentry_known_map[tid] = known

        recip_rate, recip_count, dyads = reciprocity_for_thread(group)
        reciprocity_rate_map[tid] = recip_rate
        reciprocity_reciprocal_map[tid] = recip_count
        reciprocity_dyads_map[tid] = dyads

        chain_lengths_all.extend(reciprocal_chain_lengths_for_thread(group))

    thread_id_str = base["thread_id"].astype(str)
    base["reentry_rate"] = thread_id_str.map(reentry_rate_map)
    base["reentry_comment_count"] = thread_id_str.map(reentry_count_map).fillna(0).astype(int)
    base["known_agent_comment_count"] = thread_id_str.map(reentry_known_map).fillna(0).astype(int)
    base["reciprocity_thread"] = thread_id_str.map(reciprocity_rate_map)
    base["reciprocal_dyads_thread"] = (
        thread_id_str.map(reciprocity_reciprocal_map).fillna(0).astype(int)
    )
    base["dyads_thread"] = thread_id_str.map(reciprocity_dyads_map).fillna(0).astype(int)

    summary = {
        "reciprocal_chain_count": len(chain_lengths_all),
        "reciprocal_chain_median_length": float(np.median(chain_lengths_all))
        if chain_lengths_all
        else float("nan"),
        "reciprocal_chain_mean_length": float(np.mean(chain_lengths_all))
        if chain_lengths_all
        else float("nan"),
    }
    return base, summary


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
        return {
            "success": False,
            "message": "No valid uncensored observations.",
        }

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
        return {
            "success": False,
            "message": best.message,
        }
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


def fit_weibull(
    durations_hours: np.ndarray,
    event_observed: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    s = np.asarray(durations_hours, dtype=float)
    d = np.asarray(event_observed, dtype=float).astype(int)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(s) & np.isfinite(d) & np.isfinite(w) & (s > 0) & (w > 0)
    s = s[valid]
    d = d[valid]
    w = w[valid]
    if len(s) == 0 or np.sum(w * d) <= 0:
        return {"success": False, "message": "No valid uncensored observations."}
    wf = WeibullFitter()
    wf.fit(s, event_observed=d, weights=w)
    shape = float(wf.rho_)
    scale = float(wf.lambda_)
    half_life = float(scale * (np.log(2.0) ** (1.0 / shape)))
    return {
        "success": True,
        "shape": shape,
        "scale": scale,
        "half_life_hours": half_life,
        "log_likelihood": float(wf.log_likelihood_),
    }


def cluster_bootstrap_exponential(
    durations_hours: np.ndarray,
    event_observed: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    betas = []
    half_lives = []
    for _ in range(reps):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(sampled, minlength=n_clusters).astype(float)
        weights = counts[cluster_codes]
        if np.sum(weights * event_observed) <= 0:
            continue
        fit = fit_exponential_decay(durations_hours, event_observed, weights=weights)
        if fit.get("success"):
            betas.append(fit["beta"])
            half_lives.append(fit["half_life_hours"])
    return {
        "n_successful_bootstrap": len(betas),
        "beta_ci_95": percentile_ci(betas),
        "half_life_ci_95": percentile_ci(half_lives),
    }


def cluster_bootstrap_weibull_shape(
    durations_hours: np.ndarray,
    event_observed: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    shapes = []
    for _ in range(reps):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(sampled, minlength=n_clusters).astype(float)
        weights = counts[cluster_codes]
        if np.sum(weights * event_observed) <= 0:
            continue
        fit = fit_weibull(durations_hours, event_observed, weights=weights)
        if fit.get("success"):
            shapes.append(fit["shape"])
    return {
        "n_successful_bootstrap": len(shapes),
        "shape_ci_95": percentile_ci(shapes),
    }


def percentile_ci(values: Iterable[float], alpha: float = 0.05) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return [float("nan"), float("nan")]
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return [lo, hi]


def summarize_descriptive_table(thread_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "Comments per post": thread_metrics["n_comments"],
        "Max depth per thread": thread_metrics["depth_max"].astype(float),
        "Thread duration (hours)": thread_metrics["thread_duration_hours"],
        "Unique agents per thread": thread_metrics["n_unique_agents"].astype(float),
        "Re-entry rate": thread_metrics["reentry_rate"],
    }
    rows = []
    for name, series in metrics.items():
        s = pd.Series(series).dropna()
        rows.append(
            {
                "metric": name,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_thread_depth_stats(
    depth_max: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    n = len(depth_max)
    mean_samples = []
    p5_samples = []
    p10_samples = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        sample = depth_max[idx]
        mean_samples.append(float(np.mean(sample)))
        p5_samples.append(float(np.mean(sample >= 5)))
        p10_samples.append(float(np.mean(sample >= 10)))
    return {
        "mean_depth_ci_95": percentile_ci(mean_samples),
        "p_depth_ge_5_ci_95": percentile_ci(p5_samples),
        "p_depth_ge_10_ci_95": percentile_ci(p10_samples),
    }


def fit_depth_tail_mu(depth_max: np.ndarray) -> dict[str, Any]:
    if depth_max.size == 0:
        return {"mu_hat": float("nan"), "k_values": [], "tail_probs": []}
    max_k = int(np.max(depth_max))
    k_values = np.arange(1, max_k + 1)
    tail_probs = np.array([np.mean(depth_max >= k) for k in k_values], dtype=float)
    valid = tail_probs > 0
    k_valid = k_values[valid]
    p_valid = tail_probs[valid]
    if len(k_valid) == 0:
        return {"mu_hat": float("nan"), "k_values": [], "tail_probs": []}
    # Fit log P(D >= k) ~= k log(mu) with zero intercept.
    log_p = np.log(p_valid)
    slope = float(np.sum(k_valid * log_p) / np.sum(k_valid**2))
    mu_hat = float(np.exp(slope))
    return {
        "mu_hat": mu_hat,
        "k_values": k_valid.astype(int).tolist(),
        "tail_probs": p_valid.tolist(),
    }


def detect_time_gaps(comment_times: pd.Series, threshold_hours: float) -> list[dict[str, Any]]:
    ordered = comment_times.sort_values(kind="stable").reset_index(drop=True)
    diffs = ordered.diff().dt.total_seconds().fillna(0.0) / 3600.0
    gaps = []
    for idx in np.where(diffs > threshold_hours)[0]:
        gaps.append(
            {
                "gap_hours": float(diffs.iloc[idx]),
                "before": ordered.iloc[idx - 1].isoformat(),
                "after": ordered.iloc[idx].isoformat(),
            }
        )
    return gaps


def longest_contiguous_segment(
    comment_times: pd.Series,
    threshold_hours: float,
) -> tuple[pd.Timestamp, pd.Timestamp, list[dict[str, Any]]]:
    ordered = comment_times.sort_values(kind="stable").reset_index(drop=True)
    diffs = ordered.diff().dt.total_seconds().fillna(0.0) / 3600.0
    breakpoints = np.where(diffs > threshold_hours)[0].tolist()
    starts = [0] + breakpoints
    ends = breakpoints + [len(ordered)]

    segments: list[dict[str, Any]] = []
    for s_idx, e_idx in zip(starts, ends, strict=True):
        start_time = ordered.iloc[s_idx]
        end_time = ordered.iloc[e_idx - 1]
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        segments.append(
            {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": float(duration_hours),
                "n_comments": int(e_idx - s_idx),
            }
        )
    longest = max(segments, key=lambda x: x["duration_hours"])
    return (
        pd.Timestamp(longest["start"]),
        pd.Timestamp(longest["end"]),
        segments,
    )


def build_periodicity_series(
    comment_times: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bin_minutes: int,
) -> pd.Series:
    mask = (comment_times >= start) & (comment_times <= end)
    subset = comment_times[mask]
    freq = f"{bin_minutes}min"
    bins = pd.date_range(start=start.floor(freq), end=end.ceil(freq), freq=freq, tz="UTC")
    binned = pd.Series(1, index=subset).groupby(pd.Grouper(freq=freq)).sum()
    binned = binned.reindex(bins, fill_value=0).astype(float)
    return binned


def detrend_series_for_psd(series: np.ndarray, moving_window_bins: int) -> np.ndarray:
    s = pd.Series(series, dtype=float)
    transformed = np.log1p(s)
    trend = transformed.rolling(moving_window_bins, center=True, min_periods=1).mean()
    y = (transformed - trend).to_numpy()
    y = y - np.mean(y)
    return y


def periodogram_positive(y: np.ndarray, fs_per_hour: float) -> tuple[np.ndarray, np.ndarray]:
    freqs, power = signal.periodogram(y, fs=fs_per_hour, window="hann", detrend="constant")
    mask = freqs > 0
    return freqs[mask], power[mask]


def fit_ar1(y: np.ndarray) -> tuple[float, float]:
    if len(y) < 3:
        return 0.0, 1.0
    y0 = y[:-1]
    y1 = y[1:]
    denom = float(np.dot(y0, y0))
    phi = float(np.dot(y0, y1) / denom) if denom > 0 else 0.0
    phi = float(np.clip(phi, -0.99, 0.99))
    resid = y1 - phi * y0
    sigma = float(np.std(resid, ddof=1))
    return phi, max(sigma, 1e-6)


def ar1_simulation(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def analyze_periodicity(
    comment_times: pd.Series,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    segment_start, segment_end, segments = longest_contiguous_segment(
        comment_times,
        threshold_hours=cfg.gap_threshold_hours,
    )
    binned = build_periodicity_series(
        comment_times,
        start=segment_start,
        end=segment_end,
        bin_minutes=cfg.periodicity_bin_minutes,
    )
    fs_per_hour = 60.0 / cfg.periodicity_bin_minutes
    moving_window_bins = int(round((24 * 60) / cfg.periodicity_bin_minutes))
    y = detrend_series_for_psd(binned.to_numpy(), moving_window_bins=moving_window_bins)

    freqs, power = signal.welch(
        y,
        fs=fs_per_hour,
        window="hann",
        nperseg=min(128, len(y)),
        noverlap=min(64, max(0, len(y) // 2)),
        detrend="constant",
    )
    mask = freqs > 0
    freqs = freqs[mask]
    power = power[mask]

    target_freq = 0.25
    target_idx = int(np.argmin(np.abs(freqs - target_freq)))
    target_freq_nearest = float(freqs[target_idx])
    target_power = float(power[target_idx])

    exclusion = np.abs(freqs - target_freq_nearest) <= 0.03
    if np.any(~exclusion):
        background = float(np.median(power[~exclusion]))
    else:
        background = float(np.median(power))
    peak_to_background = float(target_power / background) if background > 0 else float("nan")

    per_freqs, per_power = periodogram_positive(y, fs_per_hour=fs_per_hour)
    g_obs = float(np.max(per_power) / np.sum(per_power))
    periodogram_target_idx = int(np.argmin(np.abs(per_freqs - target_freq)))
    p_target_obs = float(per_power[periodogram_target_idx])

    phi, sigma = fit_ar1(y)
    g_null = []
    target_null = []
    for _ in range(cfg.ar1_sims):
        sim = ar1_simulation(len(y), phi, sigma, rng)
        sim_freq, sim_power = periodogram_positive(sim, fs_per_hour=fs_per_hour)
        g_null.append(float(np.max(sim_power) / np.sum(sim_power)))
        sim_target_idx = int(np.argmin(np.abs(sim_freq - target_freq)))
        target_null.append(float(sim_power[sim_target_idx]))
    g_null_arr = np.asarray(g_null, dtype=float)
    target_null_arr = np.asarray(target_null, dtype=float)

    g_p = float(np.mean(g_null_arr >= g_obs))
    target_p = float(np.mean(target_null_arr >= p_target_obs))

    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx])
    dominant_period_hours = float(1.0 / dominant_freq) if dominant_freq > 0 else float("nan")

    psd_df = pd.DataFrame({"frequency_per_hour": freqs, "power": power})
    summary = {
        "segment_start": segment_start.isoformat(),
        "segment_end": segment_end.isoformat(),
        "segment_duration_hours": float((segment_end - segment_start).total_seconds() / 3600.0),
        "segments_detected": segments,
        "bin_minutes": cfg.periodicity_bin_minutes,
        "target_frequency_per_hour": target_freq,
        "target_frequency_nearest_per_hour": target_freq_nearest,
        "target_period_hours_nearest": float(1.0 / target_freq_nearest),
        "target_power": target_power,
        "background_power_median": background,
        "peak_to_background_ratio": peak_to_background,
        "dominant_frequency_per_hour": dominant_freq,
        "dominant_period_hours": dominant_period_hours,
        "fisher_g": g_obs,
        "fisher_g_p_value_ar1": g_p,
        "target_power_p_value_ar1": target_p,
        "ar1_phi": phi,
    }
    return summary, psd_df


def agent_lag_autocorrelation(
    events: pd.DataFrame,
    segment_start: pd.Timestamp,
    segment_end: pd.Timestamp,
    bin_minutes: int,
    min_comments: int,
    rng: np.random.Generator,
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seg = events[
        (events["created_at_utc"] >= segment_start) & (events["created_at_utc"] <= segment_end)
    ][["comment_agent_id", "created_at_utc"]].dropna()
    seg = seg.copy()
    lag_bins = int(round((4 * 60) / bin_minutes))

    start_floor = segment_start.floor(f"{bin_minutes}min")
    n_bins = int(np.floor((segment_end - start_floor).total_seconds() / (bin_minutes * 60))) + 1
    if n_bins <= lag_bins + 1:
        return pd.DataFrame(columns=["agent_id", "n_comments", "acf_lag_4h"]), {
            "eligible_agents": 0,
            "mean_acf_lag_4h": float("nan"),
            "acf_lag_4h_ci_95": [float("nan"), float("nan")],
            "lag_hours": 4.0,
            "lag_bins": lag_bins,
        }

    offset_bins = (
        (seg["created_at_utc"] - start_floor).dt.total_seconds() // (bin_minutes * 60)
    ).astype(int)
    seg["bin_idx"] = offset_bins

    out = []
    for agent_id, group in seg.groupby("comment_agent_id", sort=False):
        n_comments = len(group)
        if n_comments < min_comments:
            continue
        counts = np.bincount(group["bin_idx"], minlength=n_bins).astype(float)
        x = counts[:-lag_bins]
        y = counts[lag_bins:]
        if np.std(x) <= 0 or np.std(y) <= 0:
            continue
        acf = float(np.corrcoef(x, y)[0, 1])
        out.append({"agent_id": str(agent_id), "n_comments": n_comments, "acf_lag_4h": acf})
    agent_acf = pd.DataFrame(out)
    if agent_acf.empty:
        summary = {
            "eligible_agents": 0,
            "mean_acf_lag_4h": float("nan"),
            "acf_lag_4h_ci_95": [float("nan"), float("nan")],
            "lag_hours": 4.0,
            "lag_bins": lag_bins,
        }
        return agent_acf, summary

    vals = agent_acf["acf_lag_4h"].to_numpy()
    means = []
    n = len(vals)
    for _ in range(bootstrap_reps):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(vals[idx])))
    summary = {
        "eligible_agents": int(n),
        "mean_acf_lag_4h": float(np.mean(vals)),
        "acf_lag_4h_ci_95": percentile_ci(means),
        "lag_hours": 4.0,
        "lag_bins": lag_bins,
    }
    return agent_acf, summary


def make_depth_figure(thread_metrics: pd.DataFrame, mu_hat: float, out_path: Path) -> None:
    depth_counts = (
        thread_metrics["depth_max"].dropna().astype(int).value_counts().sort_index().rename("count")
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(depth_counts.index, depth_counts.values, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Maximum thread depth")
    ax.set_ylabel("Thread count")
    ax.set_title("Distribution of maximum thread depth")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    if np.isfinite(mu_hat):
        ax.text(
            0.98,
            0.95,
            f"Tail fit $\\hat{{\\mu}}$ = {mu_hat:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#bbbbbb"},
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_branching_figure(branching_by_depth: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(
        data=branching_by_depth,
        x="depth",
        y="mean_children",
        marker="o",
        linewidth=2.0,
        color="#f58518",
        ax=ax,
    )
    ax.set_xlabel("Node depth")
    ax.set_ylabel("Mean direct children")
    ax.set_title("Mean branching factor by depth")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_reentry_figure(thread_metrics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    vals = thread_metrics["reentry_rate"].dropna().to_numpy()
    ax.hist(vals, bins=np.linspace(0, 1, 31), color="#54a24b", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Thread re-entry rate")
    ax.set_ylabel("Thread count")
    ax.set_title("Distribution of thread re-entry rates")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_survival_figure(
    survival_units: pd.DataFrame,
    exponential_fit: dict[str, Any],
    out_path: Path,
) -> None:
    kmf = KaplanMeierFitter()
    s = survival_units["duration_hours"].to_numpy()
    d = survival_units["event_observed"].to_numpy()
    kmf.fit(s, event_observed=d, label="Kaplan-Meier")

    t_grid = np.linspace(0, np.quantile(s, 0.99), 400)
    alpha = exponential_fit["alpha"]
    beta = exponential_fit["beta"]
    surv_model = np.exp(-(alpha / beta) * (1.0 - np.exp(-beta * t_grid)))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    kmf.plot_survival_function(ax=ax, ci_show=False, color="#4c78a8", linewidth=2.0)
    ax.plot(t_grid, surv_model, color="#e45756", linewidth=2.0, label="Exponential fit")
    ax.set_xlabel("Time to first direct reply (hours)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Time-to-first-reply survival curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_psd_figure(psd_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(psd_df["frequency_per_hour"], psd_df["power"], color="#72b7b2", linewidth=2.0)
    ax.axvline(
        0.25,
        color="#e45756",
        linestyle="--",
        linewidth=1.5,
        label="4-hour target (0.25 hr$^{-1}$)",
    )
    ax.set_xlabel("Frequency (cycles per hour)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("Aggregate comment activity PSD")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def overall_dyadic_reciprocity(events: pd.DataFrame) -> dict[str, Any]:
    edges = events[["thread_id", "comment_agent_id", "parent_agent_id"]].dropna().copy()
    edges["comment_agent_id"] = edges["comment_agent_id"].astype(str)
    edges["parent_agent_id"] = edges["parent_agent_id"].astype(str)
    edges = edges[edges["comment_agent_id"] != edges["parent_agent_id"]]
    if edges.empty:
        return {
            "dyads": 0,
            "reciprocal_dyads": 0,
            "reciprocity_rate": float("nan"),
        }

    directed = (
        edges.groupby(["thread_id", "comment_agent_id", "parent_agent_id"], as_index=False)
        .size()
        .rename(columns={"size": "edge_count"})
    )
    directed["a"] = directed[["comment_agent_id", "parent_agent_id"]].min(axis=1)
    directed["b"] = directed[["comment_agent_id", "parent_agent_id"]].max(axis=1)
    directed["direction"] = (directed["comment_agent_id"] > directed["parent_agent_id"]).astype(int)

    dyad_status = directed.groupby(["thread_id", "a", "b"], as_index=False)["direction"].nunique()
    reciprocal = int((dyad_status["direction"] == 2).sum())
    total = int(len(dyad_status))
    rate = float(reciprocal / total) if total > 0 else float("nan")
    return {
        "dyads": total,
        "reciprocal_dyads": reciprocal,
        "reciprocity_rate": rate,
    }


def make_branching_table(events: pd.DataFrame) -> pd.DataFrame:
    root_children = (
        events.assign(is_root_reply=events["parent_id"].isna())
        .groupby("thread_id", as_index=False)["is_root_reply"]
        .sum()
        .rename(columns={"is_root_reply": "n_children"})
    )
    root_children["depth"] = 0

    comment_nodes = events[["depth", "n_children"]].copy()
    comment_nodes["depth"] = comment_nodes["depth"].astype(int)

    nodes = pd.concat(
        [
            root_children[["depth", "n_children"]],
            comment_nodes[["depth", "n_children"]],
        ],
        ignore_index=True,
    )

    by_depth = (
        nodes.groupby("depth", as_index=False)
        .agg(
            mean_children=("n_children", "mean"),
            median_children=("n_children", "median"),
            n_nodes=("n_children", "size"),
        )
        .sort_values("depth")
        .reset_index(drop=True)
    )
    return by_depth


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    run_features_dir = cfg.data_features_root / cfg.run_id
    run_outputs_dir = cfg.outputs_root / cfg.run_id
    figures_dir = run_outputs_dir / "figures"
    tables_dir = run_outputs_dir / "tables"
    ensure_dirs(run_features_dir, figures_dir, tables_dir)

    comments, posts, agents, submolts = prepare_tables(cfg.snapshot_root)
    events = build_thread_events(comments=comments, posts=posts)

    qc = {
        "n_posts": int(len(posts)),
        "n_comments": int(len(events)),
        "n_agents": int(len(agents)),
        "n_submolts": int(len(submolts)),
        "n_threads_with_comments": int(events["thread_id"].nunique()),
        "missing_parent_timestamp": int(events["parent_created_at_utc"].isna().sum()),
        "negative_lag_since_parent": int((events["lag_since_parent_hours"] < 0).sum()),
        "negative_lag_since_post": int((events["lag_since_post_hours"] < 0).sum()),
        "missing_depth": int(events["depth"].isna().sum()),
        "timestamp_start": events["created_at_utc"].min().isoformat(),
        "timestamp_end": events["created_at_utc"].max().isoformat(),
        "time_gaps_gt_threshold": detect_time_gaps(
            events["created_at_utc"], threshold_hours=cfg.gap_threshold_hours
        ),
    }

    thread_metrics, reciprocity_chain_summary = build_thread_metrics(events)
    branching_by_depth = make_branching_table(events)

    depth_max = thread_metrics["depth_max"].dropna().astype(int).to_numpy()
    depth_boot = bootstrap_thread_depth_stats(depth_max=depth_max, rng=rng, reps=cfg.bootstrap_reps)
    depth_tail = fit_depth_tail_mu(depth_max=depth_max)

    overall_reciprocity = overall_dyadic_reciprocity(events)

    # Survival units: one row per parent comment.
    first_reply_at = (
        events.loc[events["parent_id"].notna(), ["parent_id", "created_at_utc"]]
        .groupby("parent_id")["created_at_utc"]
        .min()
    )
    observation_end = events["created_at_utc"].max()

    survival = events[
        [
            "thread_id",
            "comment_id",
            "comment_agent_id",
            "created_at_utc",
            "submolt_category",
        ]
    ].copy()
    survival["first_reply_at"] = survival["comment_id"].map(first_reply_at)
    survival["event_observed"] = survival["first_reply_at"].notna().astype(int)
    survival["duration_hours"] = np.where(
        survival["event_observed"].astype(bool),
        (survival["first_reply_at"] - survival["created_at_utc"]).dt.total_seconds() / 3600.0,
        (observation_end - survival["created_at_utc"]).dt.total_seconds() / 3600.0,
    )
    boundary_cutoff = observation_end - pd.Timedelta(hours=cfg.censor_boundary_hours)
    survival["excluded_censor_boundary"] = survival["created_at_utc"] > boundary_cutoff
    survival = survival[survival["duration_hours"] > 0].copy()

    survival_primary = survival[~survival["excluded_censor_boundary"]].copy()
    clusters = survival_primary["thread_id"].astype("category")
    cluster_codes = clusters.cat.codes.to_numpy()
    n_clusters = int(clusters.nunique())

    exp_fit = fit_exponential_decay(
        durations_hours=survival_primary["duration_hours"].to_numpy(),
        event_observed=survival_primary["event_observed"].to_numpy(),
    )
    if not exp_fit.get("success"):
        raise RuntimeError(f"Exponential fit failed: {exp_fit.get('message')}")
    exp_boot = cluster_bootstrap_exponential(
        durations_hours=survival_primary["duration_hours"].to_numpy(),
        event_observed=survival_primary["event_observed"].to_numpy(),
        cluster_codes=cluster_codes,
        n_clusters=n_clusters,
        rng=rng,
        reps=cfg.bootstrap_reps,
    )

    weib_fit = fit_weibull(
        durations_hours=survival_primary["duration_hours"].to_numpy(),
        event_observed=survival_primary["event_observed"].to_numpy(),
    )
    if not weib_fit.get("success"):
        raise RuntimeError(f"Weibull fit failed: {weib_fit.get('message')}")
    weib_boot = cluster_bootstrap_weibull_shape(
        durations_hours=survival_primary["duration_hours"].to_numpy(),
        event_observed=survival_primary["event_observed"].to_numpy(),
        cluster_codes=cluster_codes,
        n_clusters=n_clusters,
        rng=rng,
        reps=cfg.bootstrap_reps,
    )

    # Sensitivity: no boundary exclusion.
    exp_fit_all = fit_exponential_decay(
        durations_hours=survival["duration_hours"].to_numpy(),
        event_observed=survival["event_observed"].to_numpy(),
    )

    # Stratified half-life by category.
    strat_rows = []
    for category in CATEGORY_ORDER:
        subset = survival_primary[survival_primary["submolt_category"] == category].copy()
        if subset.empty:
            continue
        subset_clusters = subset["thread_id"].astype("category")
        subset_codes = subset_clusters.cat.codes.to_numpy()
        subset_n_clusters = int(subset_clusters.nunique())
        fit = fit_exponential_decay(
            durations_hours=subset["duration_hours"].to_numpy(),
            event_observed=subset["event_observed"].to_numpy(),
        )
        if not fit.get("success"):
            continue
        boot = cluster_bootstrap_exponential(
            durations_hours=subset["duration_hours"].to_numpy(),
            event_observed=subset["event_observed"].to_numpy(),
            cluster_codes=subset_codes,
            n_clusters=subset_n_clusters,
            rng=rng,
            reps=max(150, cfg.bootstrap_reps // 2),
        )
        strat_rows.append(
            {
                "category": category,
                "threads": int(subset["thread_id"].nunique()),
                "comments": int(subset["comment_id"].nunique()),
                "half_life_hours": float(fit["half_life_hours"]),
                "half_life_ci_low": float(boot["half_life_ci_95"][0]),
                "half_life_ci_high": float(boot["half_life_ci_95"][1]),
            }
        )
    half_life_by_category = pd.DataFrame(strat_rows).sort_values("half_life_hours", ascending=False)

    periodicity_summary, psd_df = analyze_periodicity(
        comment_times=events["created_at_utc"],
        cfg=cfg,
        rng=rng,
    )
    segment_start = pd.Timestamp(periodicity_summary["segment_start"])
    segment_end = pd.Timestamp(periodicity_summary["segment_end"])
    agent_acf, agent_acf_summary = agent_lag_autocorrelation(
        events=events,
        segment_start=segment_start,
        segment_end=segment_end,
        bin_minutes=cfg.periodicity_bin_minutes,
        min_comments=cfg.min_agent_comments_for_acf,
        rng=rng,
        bootstrap_reps=cfg.bootstrap_reps,
    )

    # Karma-stratified half-life (bottom vs top quartile among known karma).
    survival_agent = survival_primary.merge(
        agents.rename(columns={"agent_id": "comment_agent_id"}),
        on="comment_agent_id",
        how="left",
    )
    survival_agent = survival_agent.dropna(subset=["karma"]).copy()
    q1 = float(survival_agent["karma"].quantile(0.25))
    q3 = float(survival_agent["karma"].quantile(0.75))
    low = survival_agent[survival_agent["karma"] <= q1].copy()
    high = survival_agent[survival_agent["karma"] >= q3].copy()

    karma_compare = {}
    for label, subset in [("low_q1", low), ("high_q4", high)]:
        fit = fit_exponential_decay(
            durations_hours=subset["duration_hours"].to_numpy(),
            event_observed=subset["event_observed"].to_numpy(),
        )
        karma_compare[label] = {
            "n_comments": int(subset["comment_id"].nunique()),
            "n_threads": int(subset["thread_id"].nunique()),
            "half_life_hours": float(fit["half_life_hours"]) if fit.get("success") else None,
        }

    # Write derived features.
    events_out = run_features_dir / "thread_events.parquet"
    thread_metrics_out = run_features_dir / "thread_metrics.parquet"
    survival_out = run_features_dir / "survival_units.parquet"
    agent_acf_out = run_features_dir / "agent_acf.parquet"
    branching_out = run_features_dir / "branching_by_depth.parquet"
    events.to_parquet(events_out, index=False)
    thread_metrics.to_parquet(thread_metrics_out, index=False)
    survival.to_parquet(survival_out, index=False)
    agent_acf.to_parquet(agent_acf_out, index=False)
    branching_by_depth.to_parquet(branching_out, index=False)

    # Figures.
    depth_fig = figures_dir / "moltbook_depth_distribution.png"
    branch_fig = figures_dir / "moltbook_branching_by_depth.png"
    reentry_fig = figures_dir / "moltbook_reentry_distribution.png"
    surv_fig = figures_dir / "moltbook_survival_curve.png"
    psd_fig = figures_dir / "moltbook_psd.png"
    make_depth_figure(
        thread_metrics=thread_metrics,
        mu_hat=depth_tail["mu_hat"],
        out_path=depth_fig,
    )
    make_branching_figure(branching_by_depth=branching_by_depth, out_path=branch_fig)
    make_reentry_figure(thread_metrics=thread_metrics, out_path=reentry_fig)
    make_survival_figure(
        survival_units=survival_primary,
        exponential_fit=exp_fit,
        out_path=surv_fig,
    )
    make_psd_figure(psd_df=psd_df, out_path=psd_fig)

    # Tables / JSON summaries.
    descriptive_table = summarize_descriptive_table(thread_metrics)
    descriptive_out = tables_dir / "descriptive_stats.csv"
    branching_csv_out = tables_dir / "branching_by_depth.csv"
    half_life_by_cat_out = tables_dir / "half_life_by_category.csv"
    psd_out = tables_dir / "psd_curve.csv"
    descriptive_table.to_csv(descriptive_out, index=False)
    branching_by_depth.to_csv(branching_csv_out, index=False)
    half_life_by_category.to_csv(half_life_by_cat_out, index=False)
    psd_df.to_csv(psd_out, index=False)

    summary = {
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "input_snapshot_root": str(cfg.snapshot_root),
        "censor_boundary_hours": cfg.censor_boundary_hours,
        "bootstrap_reps": cfg.bootstrap_reps,
        "qc": qc,
        "geometry": {
            "n_threads": int(len(thread_metrics)),
            "mean_depth": float(np.mean(depth_max)),
            "median_depth": float(np.median(depth_max)),
            "p_depth_ge_5": float(np.mean(depth_max >= 5)),
            "p_depth_ge_10": float(np.mean(depth_max >= 10)),
            "mu_hat_depth_tail": float(depth_tail["mu_hat"]),
            "depth_bootstrap": depth_boot,
            "reciprocity_overall": overall_reciprocity,
            "reentry_mean": float(thread_metrics["reentry_rate"].mean()),
            "reentry_median": float(thread_metrics["reentry_rate"].median()),
            "reciprocal_chains": reciprocity_chain_summary,
        },
        "half_life": {
            "primary_sample_comments": int(len(survival_primary)),
            "primary_events": int(survival_primary["event_observed"].sum()),
            "primary_censored": int((1 - survival_primary["event_observed"]).sum()),
            "excluded_boundary_comments": int(survival["excluded_censor_boundary"].sum()),
            "observation_end_utc": observation_end.isoformat(),
            "exponential": exp_fit,
            "exponential_bootstrap": exp_boot,
            "exponential_sensitivity_no_boundary_exclusion": exp_fit_all,
            "weibull": weib_fit,
            "weibull_bootstrap": weib_boot,
            "category_table_path": str(half_life_by_cat_out),
            "karma_quartile_comparison": {
                "q1_cutoff": q1,
                "q3_cutoff": q3,
                "estimates": karma_compare,
            },
        },
        "periodicity": periodicity_summary,
        "agent_acf": agent_acf_summary,
        "artifacts": {
            "thread_events_parquet": str(events_out),
            "thread_metrics_parquet": str(thread_metrics_out),
            "survival_units_parquet": str(survival_out),
            "agent_acf_parquet": str(agent_acf_out),
            "branching_by_depth_parquet": str(branching_out),
            "figures": [
                str(depth_fig),
                str(branch_fig),
                str(reentry_fig),
                str(surv_fig),
                str(psd_fig),
            ],
            "tables": [
                str(descriptive_out),
                str(branching_csv_out),
                str(half_life_by_cat_out),
                str(psd_out),
            ],
        },
    }
    summary_out = tables_dir / "analysis_summary.json"
    summary_out.write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": "analysis/06_moltbook_only_analysis.py",
        "config": asdict(cfg),
        "outputs_dir": str(run_outputs_dir),
        "data_features_dir": str(run_features_dir),
        "analysis_summary_json": str(summary_out),
    }
    manifest_out = run_outputs_dir / "run_manifest.json"
    manifest_out.write_text(json.dumps(sanitize_for_json(manifest), indent=2), encoding="utf-8")

    print(f"Run ID: {cfg.run_id}")
    print(f"Summary: {summary_out}")
    print(f"Manifest: {manifest_out}")
    print(f"Mean depth: {summary['geometry']['mean_depth']:.3f}")
    print(f"Half-life (hours): {summary['half_life']['exponential']['half_life_hours']:.3f}")
    print(
        "Half-life CI (95%): "
        f"{summary['half_life']['exponential_bootstrap']['half_life_ci_95'][0]:.3f}, "
        f"{summary['half_life']['exponential_bootstrap']['half_life_ci_95'][1]:.3f}"
    )


if __name__ == "__main__":
    main()
