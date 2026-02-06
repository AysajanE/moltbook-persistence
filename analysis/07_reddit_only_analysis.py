#!/usr/bin/env python3
"""Run Reddit-only analysis and generate reproducible, run-scoped artifacts.

This script mirrors the Moltbook-only analysis flow for curated Reddit data and
computes, from a pinned run_id:
1) thread depth distribution,
2) mean branching factor by depth,
3) thread re-entry rate distribution,
4) time-to-first-reply survival + exponential half-life estimate
   (primary censor-boundary exclusion + no-boundary sensitivity),
5) aggregate comment activity PSD with AR(1)-calibrated Fisher-g p-value.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import seaborn as sns
from scipy import optimize, signal

DEFAULT_CURATED_ROOT = Path("data_curated/reddit")
DEFAULT_DATA_FEATURES_ROOT = Path("data_features/reddit_only")
DEFAULT_OUTPUTS_ROOT = Path("outputs/reddit_only")
DEFAULT_OPS_ROOT = Path("outputs/ops")

DEFAULT_SEED = 20260206
DEFAULT_CENSOR_BOUNDARY_HOURS = 4.0
DEFAULT_GAP_THRESHOLD_HOURS = 6.0
DEFAULT_PERIODICITY_BIN_MINUTES = 15
DEFAULT_BOOTSTRAP_REPS = 400
DEFAULT_AR1_SIMS = 2000

TARGET_PERIODICITY_FREQ_PER_HOUR = 0.25  # 4-hour hypothesis frequency.


@dataclass(frozen=True)
class Config:
    curated_root: Path
    outputs_root: Path
    data_features_root: Path
    ops_root: Path
    run_id: str
    seed: int
    censor_boundary_hours: float
    gap_threshold_hours: float
    periodicity_bin_minutes: int
    bootstrap_reps: int
    ar1_sims: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curated-root",
        type=Path,
        default=DEFAULT_CURATED_ROOT,
        help="Curated Reddit root containing partitioned submissions/comments tables.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Root for manuscript-facing figures/tables/manifests.",
    )
    parser.add_argument(
        "--data-features-root",
        type=Path,
        default=DEFAULT_DATA_FEATURES_ROOT,
        help="Root for run-scoped derived feature outputs.",
    )
    parser.add_argument(
        "--ops-root",
        type=Path,
        default=DEFAULT_OPS_ROOT,
        help="Optional ops artifact root used for warning extraction.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Pinned run_id partition to analyze.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic RNG seed.")
    parser.add_argument(
        "--censor-boundary-hours",
        type=float,
        default=DEFAULT_CENSOR_BOUNDARY_HOURS,
        help=(
            "Exclude parent comments created within this many hours of observation end "
            "for the primary survival estimate."
        ),
    )
    parser.add_argument(
        "--gap-threshold-hours",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_HOURS,
        help="Gap threshold for contiguous periodicity segment selection.",
    )
    parser.add_argument(
        "--periodicity-bin-minutes",
        type=int,
        default=DEFAULT_PERIODICITY_BIN_MINUTES,
        help="Bin width in minutes for aggregate activity periodicity analysis.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help="Cluster-bootstrap repetitions for half-life uncertainty.",
    )
    parser.add_argument(
        "--ar1-sims",
        type=int,
        default=DEFAULT_AR1_SIMS,
        help="AR(1) simulation count for Fisher-g null calibration.",
    )
    args = parser.parse_args()
    return Config(
        curated_root=args.curated_root,
        outputs_root=args.outputs_root,
        data_features_root=args.data_features_root,
        ops_root=args.ops_root,
        run_id=args.run_id,
        seed=args.seed,
        censor_boundary_hours=args.censor_boundary_hours,
        gap_threshold_hours=args.gap_threshold_hours,
        periodicity_bin_minutes=args.periodicity_bin_minutes,
        bootstrap_reps=args.bootstrap_reps,
        ar1_sims=args.ar1_sims,
    )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def percentile_ci(values: Iterable[float], alpha: float = 0.05) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return [float("nan"), float("nan")]
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return [lo, hi]


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


def read_partitioned_run_table(curated_root: Path, subset: str, run_id: str) -> pd.DataFrame:
    subset_path = curated_root / subset
    if not subset_path.exists():
        raise FileNotFoundError(f"Curated subset directory does not exist: {subset_path}")
    dataset = ds.dataset(str(subset_path), format="parquet", partitioning="hive")
    table = dataset.to_table(filter=ds.field("run_id") == run_id)
    df = table.to_pandas()
    if not df.empty and "run_id" in df.columns:
        df = df[df["run_id"].astype(str) == run_id].copy()
    return df.reset_index(drop=True)


def normalize_author(author: pd.Series, author_is_deleted: pd.Series | None = None) -> pd.Series:
    out = author.astype("string")
    if author_is_deleted is not None:
        out = out.mask(author_is_deleted.fillna(False))
    out = out.str.strip()
    out = out.mask(out.isna() | (out == "") | (out == "[deleted]") | (out == "[removed]"))
    return out.astype(object)


def prepare_tables(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    submissions = read_partitioned_run_table(cfg.curated_root, "submissions", run_id=cfg.run_id)
    comments = read_partitioned_run_table(cfg.curated_root, "comments", run_id=cfg.run_id)

    if submissions.empty:
        raise ValueError(
            "No submissions found for run_id="
            f"{cfg.run_id} under {cfg.curated_root / 'submissions'}."
        )
    if comments.empty:
        raise ValueError(
            f"No comments found for run_id={cfg.run_id} under {cfg.curated_root / 'comments'}."
        )

    required_sub_cols = {"submission_id", "created_at_utc", "subreddit", "author"}
    required_comment_cols = {
        "comment_id",
        "submission_id",
        "parent_id",
        "author",
        "created_at_utc",
    }
    missing_sub = sorted(required_sub_cols - set(submissions.columns))
    missing_com = sorted(required_comment_cols - set(comments.columns))
    if missing_sub:
        raise ValueError(f"Submissions table missing required columns: {missing_sub}")
    if missing_com:
        raise ValueError(f"Comments table missing required columns: {missing_com}")

    submissions = submissions.copy()
    comments = comments.copy()
    submissions["submission_id"] = submissions["submission_id"].astype(str)
    comments["submission_id"] = comments["submission_id"].astype(str)
    comments["comment_id"] = comments["comment_id"].astype(str)
    submissions["created_at_utc"] = to_utc(submissions["created_at_utc"])
    comments["created_at_utc"] = to_utc(comments["created_at_utc"])

    before_sub = len(submissions)
    before_com = len(comments)
    submissions = submissions.dropna(subset=["submission_id", "created_at_utc"]).copy()
    comments = comments.dropna(subset=["comment_id", "submission_id", "created_at_utc"]).copy()
    if submissions.empty or comments.empty:
        raise ValueError(
            "No analyzable rows after dropping null IDs/timestamps for submissions/comments."
        )

    # Stable dedupe guards against accidental duplicate rows in partition files.
    submissions = submissions.drop_duplicates(subset=["submission_id"], keep="first").copy()
    comments = comments.drop_duplicates(subset=["comment_id"], keep="first").copy()

    submissions["submission_author_clean"] = normalize_author(submissions["author"])
    comments["comment_author"] = normalize_author(
        comments["author"],
        comments["author_is_deleted"] if "author_is_deleted" in comments.columns else None,
    )

    if before_sub != len(submissions):
        submissions = submissions.reset_index(drop=True)
    if before_com != len(comments):
        comments = comments.reset_index(drop=True)

    return comments, submissions


def parse_parent_columns(parent_id: pd.Series) -> pd.DataFrame:
    raw = parent_id.fillna("").astype(str)
    is_comment_parent = raw.str.startswith("t1_")
    is_submission_parent = raw.str.startswith("t3_")
    parent_type = np.select(
        [is_comment_parent, is_submission_parent],
        ["comment", "submission"],
        default="unknown",
    )
    parent_comment_id = raw.where(is_comment_parent).str.removeprefix("t1_")
    parent_submission_id = raw.where(is_submission_parent).str.removeprefix("t3_")
    return pd.DataFrame(
        {
            "parent_type": parent_type,
            "parent_comment_id": parent_comment_id,
            "parent_submission_id": parent_submission_id,
        }
    )


def compute_depths(events: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    depth = pd.Series(np.nan, index=events.index, dtype="float64")
    grouped = events.groupby("thread_id", sort=False).groups

    missing_t1_parent_count = 0
    unknown_parent_prefix_count = 0
    t3_parent_submission_mismatch_count = 0
    cycle_detected_count = 0

    for thread_id, idx in grouped.items():
        _ = thread_id
        loc = list(idx)
        sub = events.loc[
            loc,
            [
                "comment_id",
                "thread_id",
                "parent_type",
                "parent_comment_id",
                "parent_submission_id",
            ],
        ].copy()

        unknown_parent_prefix_count += int((sub["parent_type"] == "unknown").sum())
        t3_parent_submission_mismatch_count += int(
            (
                (sub["parent_type"] == "submission")
                & sub["parent_submission_id"].notna()
                & (sub["parent_submission_id"].astype(str) != sub["thread_id"].astype(str))
            ).sum()
        )

        parent_map: dict[str, str | None] = {}
        for row in sub.itertuples(index=False):
            cid = str(row.comment_id)
            if row.parent_type == "submission":
                parent_map[cid] = None
            elif row.parent_type == "comment":
                if pd.isna(row.parent_comment_id) or str(row.parent_comment_id) == "":
                    parent_map[cid] = "__MISSING__"
                else:
                    parent_map[cid] = str(row.parent_comment_id)
            else:
                parent_map[cid] = "__UNKNOWN__"

        depth_map: dict[str, float] = {}
        state: dict[str, int] = {}

        def resolve_depth(comment_id: str) -> float:
            nonlocal missing_t1_parent_count
            nonlocal cycle_detected_count

            status = state.get(comment_id, 0)
            if status == 2:
                return depth_map[comment_id]
            if status == 1:
                cycle_detected_count += 1
                depth_map[comment_id] = float("nan")
                state[comment_id] = 2
                return depth_map[comment_id]

            state[comment_id] = 1
            parent = parent_map[comment_id]

            if parent is None:
                this_depth = 1.0
            elif parent == "__UNKNOWN__":
                this_depth = float("nan")
            elif parent == "__MISSING__":
                missing_t1_parent_count += 1
                this_depth = float("nan")
            elif parent not in parent_map:
                missing_t1_parent_count += 1
                this_depth = float("nan")
            else:
                parent_depth = resolve_depth(parent)
                this_depth = parent_depth + 1.0 if np.isfinite(parent_depth) else float("nan")

            depth_map[comment_id] = this_depth
            state[comment_id] = 2
            return this_depth

        for comment_id in parent_map:
            _ = resolve_depth(comment_id)

        mapped = events.loc[loc, "comment_id"].astype(str).map(depth_map)
        depth.loc[loc] = mapped.to_numpy(dtype="float64")

    stats = {
        "missing_t1_parent_count": missing_t1_parent_count,
        "unknown_parent_prefix_count": unknown_parent_prefix_count,
        "t3_parent_submission_mismatch_count": t3_parent_submission_mismatch_count,
        "cycle_detected_count": cycle_detected_count,
    }
    return depth, stats


def build_thread_events(
    comments: pd.DataFrame,
    submissions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    sub_keep = [
        "submission_id",
        "created_at_utc",
        "subreddit",
        "author",
        "submission_author_clean",
        "score",
        "num_comments",
    ]
    submissions_sub = submissions[sub_keep].rename(
        columns={
            "created_at_utc": "submission_created_at_utc",
            "author": "submission_author",
            "score": "submission_score",
            "num_comments": "submission_num_comments",
        }
    )

    comment_keep = [
        "comment_id",
        "submission_id",
        "parent_id",
        "author",
        "comment_author",
        "created_at_utc",
        "score",
        "body_is_deleted_or_removed",
    ]
    keep_present = [col for col in comment_keep if col in comments.columns]
    comments_sub = comments[keep_present].copy()

    events = comments_sub.merge(
        submissions_sub,
        on="submission_id",
        how="inner",
        validate="many_to_one",
    )
    events["thread_id"] = events["submission_id"]

    parent_cols = parse_parent_columns(events["parent_id"])
    events = pd.concat([events, parent_cols], axis=1)

    parent_time = events.set_index("comment_id")["created_at_utc"]
    events["parent_created_at_utc"] = events["parent_comment_id"].map(parent_time)
    root_mask = events["parent_type"] == "submission"
    events.loc[root_mask, "parent_created_at_utc"] = events.loc[
        root_mask, "submission_created_at_utc"
    ]

    events["lag_since_parent_hours"] = (
        events["created_at_utc"] - events["parent_created_at_utc"]
    ).dt.total_seconds() / 3600.0
    events["lag_since_submission_hours"] = (
        events["created_at_utc"] - events["submission_created_at_utc"]
    ).dt.total_seconds() / 3600.0

    depth, depth_stats = compute_depths(events)
    events["depth"] = depth
    events["depth_int"] = pd.Series(depth, index=events.index).round().astype("Int64")

    child_count_by_comment = events["parent_comment_id"].value_counts(dropna=True)
    events["n_children"] = events["comment_id"].map(child_count_by_comment).fillna(0).astype(int)

    return events, depth_stats


def parent_completeness_stats(events: pd.DataFrame) -> dict[str, Any]:
    t1_mask = events["parent_type"] == "comment"
    t1_count = int(t1_mask.sum())
    if t1_count == 0:
        return {
            "t1_parent_count": 0,
            "missing_parent_comment_count": 0,
            "missing_fraction": float("nan"),
        }
    comment_ids = set(events["comment_id"].astype(str).tolist())
    parent_ids = events.loc[t1_mask, "parent_comment_id"]
    missing_mask = parent_ids.isna() | ~parent_ids.astype(str).isin(comment_ids)
    missing_count = int(missing_mask.sum())
    missing_fraction = float(missing_count / t1_count)
    return {
        "t1_parent_count": t1_count,
        "missing_parent_comment_count": missing_count,
        "missing_fraction": missing_fraction,
    }


def reentry_stats_for_thread(group: pd.DataFrame) -> tuple[float, int, int]:
    ordered = group.sort_values(["created_at_utc", "comment_id"], kind="stable")
    seen: set[str] = set()
    reentry_count = 0
    known_count = 0
    for author in ordered["comment_author"]:
        if pd.isna(author):
            continue
        aid = str(author)
        known_count += 1
        if aid in seen:
            reentry_count += 1
        seen.add(aid)
    rate = reentry_count / known_count if known_count > 0 else float("nan")
    return rate, reentry_count, known_count


def build_thread_metrics(events: pd.DataFrame) -> pd.DataFrame:
    base = (
        events.groupby("thread_id", as_index=False)
        .agg(
            n_comments=("comment_id", "size"),
            n_unique_agents=("comment_author", pd.Series.nunique),
            depth_max=("depth", "max"),
            depth_mean=("depth", "mean"),
            first_comment_at=("created_at_utc", "min"),
            last_comment_at=("created_at_utc", "max"),
            submission_created_at_utc=("submission_created_at_utc", "first"),
            subreddit=("subreddit", "first"),
        )
        .copy()
    )
    base["thread_duration_hours"] = (
        base["last_comment_at"] - base["submission_created_at_utc"]
    ).dt.total_seconds() / 3600.0

    reentry_rate_map: dict[str, float] = {}
    reentry_count_map: dict[str, int] = {}
    reentry_known_map: dict[str, int] = {}
    for thread_id, group in events.groupby("thread_id", sort=False):
        tid = str(thread_id)
        rate, count, known = reentry_stats_for_thread(group)
        reentry_rate_map[tid] = rate
        reentry_count_map[tid] = count
        reentry_known_map[tid] = known

    thread_id_str = base["thread_id"].astype(str)
    base["reentry_rate"] = thread_id_str.map(reentry_rate_map)
    base["reentry_comment_count"] = thread_id_str.map(reentry_count_map).fillna(0).astype(int)
    base["known_agent_comment_count"] = thread_id_str.map(reentry_known_map).fillna(0).astype(int)
    return base.sort_values("thread_id").reset_index(drop=True)


def make_depth_distribution_table(thread_metrics: pd.DataFrame) -> pd.DataFrame:
    depth = thread_metrics["depth_max"].dropna().round().astype(int)
    if depth.empty:
        return pd.DataFrame(columns=["depth_max", "thread_count", "thread_share"])
    counts = depth.value_counts().sort_index()
    out = counts.rename_axis("depth_max").reset_index(name="thread_count")
    out["thread_share"] = out["thread_count"] / out["thread_count"].sum()
    return out


def make_reentry_distribution_table(thread_metrics: pd.DataFrame) -> pd.DataFrame:
    vals = thread_metrics["reentry_rate"].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "thread_count", "thread_share"])
    bins = np.linspace(0.0, 1.0, 21)
    counts, edges = np.histogram(vals, bins=bins)
    out = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "thread_count": counts.astype(int),
        }
    )
    total = int(out["thread_count"].sum())
    out["thread_share"] = out["thread_count"] / total if total > 0 else float("nan")
    return out


def make_branching_table(events: pd.DataFrame) -> pd.DataFrame:
    root_children = (
        events.assign(is_root_reply=events["parent_type"] == "submission")
        .groupby("thread_id", as_index=False)["is_root_reply"]
        .sum()
        .rename(columns={"is_root_reply": "n_children"})
    )
    root_children["depth"] = 0

    comment_nodes = events.loc[events["depth"].notna(), ["depth", "n_children"]].copy()
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


def summarize_descriptive_table(thread_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "Comments per thread": thread_metrics["n_comments"],
        "Max depth per thread": thread_metrics["depth_max"],
        "Thread duration (hours)": thread_metrics["thread_duration_hours"],
        "Unique agents per thread": thread_metrics["n_unique_agents"].astype(float),
        "Re-entry rate": thread_metrics["reentry_rate"],
    }
    rows = []
    for name, series in metrics.items():
        s = pd.Series(series).dropna()
        if s.empty:
            rows.append(
                {
                    "metric": name,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            )
            continue
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


def build_survival_units(
    events: pd.DataFrame,
    censor_boundary_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    first_reply_at = (
        events.loc[events["parent_type"] == "comment", ["parent_comment_id", "created_at_utc"]]
        .groupby("parent_comment_id")["created_at_utc"]
        .min()
    )
    observation_end = pd.Timestamp(events["created_at_utc"].max())
    survival = events[
        [
            "thread_id",
            "comment_id",
            "comment_author",
            "created_at_utc",
            "subreddit",
        ]
    ].copy()
    survival["first_reply_at"] = survival["comment_id"].map(first_reply_at)
    survival["event_observed"] = survival["first_reply_at"].notna().astype(int)
    survival["duration_hours"] = np.where(
        survival["event_observed"].astype(bool),
        (survival["first_reply_at"] - survival["created_at_utc"]).dt.total_seconds() / 3600.0,
        (observation_end - survival["created_at_utc"]).dt.total_seconds() / 3600.0,
    )
    boundary_cutoff = observation_end - pd.Timedelta(hours=censor_boundary_hours)
    survival["excluded_censor_boundary"] = survival["created_at_utc"] > boundary_cutoff
    survival = survival[survival["duration_hours"] > 0].copy()
    survival_primary = survival[~survival["excluded_censor_boundary"]].copy()
    return survival, survival_primary, observation_end


def detect_time_gaps(comment_times: pd.Series, threshold_hours: float) -> list[dict[str, Any]]:
    ordered = comment_times.dropna().sort_values(kind="stable").reset_index(drop=True)
    if ordered.empty:
        return []
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
    ordered = comment_times.dropna().sort_values(kind="stable").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("No timestamps available for periodicity analysis.")
    if len(ordered) == 1:
        ts = pd.Timestamp(ordered.iloc[0])
        one_point_segment = {
            "start": ts.isoformat(),
            "end": ts.isoformat(),
            "duration_hours": 0.0,
            "n_comments": 1,
        }
        return ts, ts, [one_point_segment]
    diffs = ordered.diff().dt.total_seconds().fillna(0.0) / 3600.0
    breakpoints = np.where(diffs > threshold_hours)[0].tolist()
    starts = [0] + breakpoints
    ends = breakpoints + [len(ordered)]

    segments: list[dict[str, Any]] = []
    for s_idx, e_idx in zip(starts, ends, strict=True):
        start_time = pd.Timestamp(ordered.iloc[s_idx])
        end_time = pd.Timestamp(ordered.iloc[e_idx - 1])
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
    return pd.Timestamp(longest["start"]), pd.Timestamp(longest["end"]), segments


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
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    warnings: list[str] = []
    ordered = comment_times.dropna().sort_values(kind="stable")
    if ordered.size < 4:
        warnings.append(
            "Insufficient timestamp count for stable PSD/Fisher-g periodicity estimates."
        )
        summary = {
            "segment_start": None,
            "segment_end": None,
            "segment_duration_hours": float("nan"),
            "segments_detected": [],
            "bin_minutes": cfg.periodicity_bin_minutes,
            "target_frequency_per_hour": TARGET_PERIODICITY_FREQ_PER_HOUR,
            "target_frequency_nearest_per_hour": float("nan"),
            "target_period_hours_nearest": float("nan"),
            "target_power": float("nan"),
            "dominant_frequency_per_hour": float("nan"),
            "dominant_period_hours": float("nan"),
            "fisher_g": float("nan"),
            "fisher_g_p_value_ar1": float("nan"),
            "target_power_p_value_ar1": float("nan"),
            "ar1_phi": float("nan"),
        }
        return summary, pd.DataFrame(columns=["frequency_per_hour", "power"]), warnings

    segment_start, segment_end, segments = longest_contiguous_segment(
        ordered,
        threshold_hours=cfg.gap_threshold_hours,
    )
    binned = build_periodicity_series(
        ordered,
        start=segment_start,
        end=segment_end,
        bin_minutes=cfg.periodicity_bin_minutes,
    )

    if len(binned) < 8:
        warnings.append(
            "Longest contiguous segment too short for stable PSD/Fisher-g periodicity estimates."
        )
        summary = {
            "segment_start": segment_start.isoformat(),
            "segment_end": segment_end.isoformat(),
            "segment_duration_hours": float((segment_end - segment_start).total_seconds() / 3600.0),
            "segments_detected": segments,
            "bin_minutes": cfg.periodicity_bin_minutes,
            "target_frequency_per_hour": TARGET_PERIODICITY_FREQ_PER_HOUR,
            "target_frequency_nearest_per_hour": float("nan"),
            "target_period_hours_nearest": float("nan"),
            "target_power": float("nan"),
            "dominant_frequency_per_hour": float("nan"),
            "dominant_period_hours": float("nan"),
            "fisher_g": float("nan"),
            "fisher_g_p_value_ar1": float("nan"),
            "target_power_p_value_ar1": float("nan"),
            "ar1_phi": float("nan"),
        }
        return summary, pd.DataFrame(columns=["frequency_per_hour", "power"]), warnings

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
    positive = freqs > 0
    freqs = freqs[positive]
    power = power[positive]
    psd_df = pd.DataFrame({"frequency_per_hour": freqs, "power": power})

    if freqs.size == 0 or power.size == 0:
        warnings.append("PSD returned no positive frequencies.")
        summary = {
            "segment_start": segment_start.isoformat(),
            "segment_end": segment_end.isoformat(),
            "segment_duration_hours": float((segment_end - segment_start).total_seconds() / 3600.0),
            "segments_detected": segments,
            "bin_minutes": cfg.periodicity_bin_minutes,
            "target_frequency_per_hour": TARGET_PERIODICITY_FREQ_PER_HOUR,
            "target_frequency_nearest_per_hour": float("nan"),
            "target_period_hours_nearest": float("nan"),
            "target_power": float("nan"),
            "dominant_frequency_per_hour": float("nan"),
            "dominant_period_hours": float("nan"),
            "fisher_g": float("nan"),
            "fisher_g_p_value_ar1": float("nan"),
            "target_power_p_value_ar1": float("nan"),
            "ar1_phi": float("nan"),
        }
        return summary, psd_df, warnings

    target_idx = int(np.argmin(np.abs(freqs - TARGET_PERIODICITY_FREQ_PER_HOUR)))
    target_freq_nearest = float(freqs[target_idx])
    target_power = float(power[target_idx])

    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx])
    dominant_period_hours = float(1.0 / dominant_freq) if dominant_freq > 0 else float("nan")

    per_freqs, per_power = periodogram_positive(y, fs_per_hour=fs_per_hour)
    fisher_g = float("nan")
    fisher_g_p = float("nan")
    target_p = float("nan")
    ar1_phi = float("nan")
    if per_power.size > 0 and float(np.sum(per_power)) > 0:
        fisher_g = float(np.max(per_power) / np.sum(per_power))
        periodogram_target_idx = int(
            np.argmin(np.abs(per_freqs - TARGET_PERIODICITY_FREQ_PER_HOUR))
        )
        target_power_obs = float(per_power[periodogram_target_idx])

        phi, sigma = fit_ar1(y)
        ar1_phi = phi
        if cfg.ar1_sims > 0:
            fisher_g_null = []
            target_power_null = []
            for _ in range(cfg.ar1_sims):
                sim = ar1_simulation(len(y), phi, sigma, rng)
                sim_freqs, sim_power = periodogram_positive(sim, fs_per_hour=fs_per_hour)
                if sim_power.size == 0 or float(np.sum(sim_power)) <= 0:
                    continue
                fisher_g_null.append(float(np.max(sim_power) / np.sum(sim_power)))
                sim_target_idx = int(
                    np.argmin(np.abs(sim_freqs - TARGET_PERIODICITY_FREQ_PER_HOUR))
                )
                target_power_null.append(float(sim_power[sim_target_idx]))
            if fisher_g_null:
                fisher_g_p = float(np.mean(np.asarray(fisher_g_null) >= fisher_g))
            if target_power_null:
                target_p = float(np.mean(np.asarray(target_power_null) >= target_power_obs))
    else:
        warnings.append("Periodogram power sum is zero; Fisher-g is undefined.")

    summary = {
        "segment_start": segment_start.isoformat(),
        "segment_end": segment_end.isoformat(),
        "segment_duration_hours": float((segment_end - segment_start).total_seconds() / 3600.0),
        "segments_detected": segments,
        "bin_minutes": cfg.periodicity_bin_minutes,
        "target_frequency_per_hour": TARGET_PERIODICITY_FREQ_PER_HOUR,
        "target_frequency_nearest_per_hour": target_freq_nearest,
        "target_period_hours_nearest": float(1.0 / target_freq_nearest)
        if target_freq_nearest > 0
        else float("nan"),
        "target_power": target_power,
        "dominant_frequency_per_hour": dominant_freq,
        "dominant_period_hours": dominant_period_hours,
        "fisher_g": fisher_g,
        "fisher_g_p_value_ar1": fisher_g_p,
        "target_power_p_value_ar1": target_p,
        "ar1_phi": ar1_phi,
    }
    return summary, psd_df, warnings


def plot_no_data(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def kaplan_meier_curve(
    durations_hours: np.ndarray,
    event_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations_hours, dtype=float)
    events = np.asarray(event_observed, dtype=int)
    valid = np.isfinite(durations) & np.isfinite(events) & (durations >= 0)
    durations = durations[valid]
    events = events[valid]
    if durations.size == 0:
        return np.array([0.0]), np.array([1.0])

    event_times = np.sort(np.unique(durations[events == 1]))
    if event_times.size == 0:
        max_t = float(np.max(durations))
        return np.array([0.0, max_t]), np.array([1.0, 1.0])

    times = [0.0]
    survival = [1.0]
    s_prob = 1.0
    for t in event_times:
        at_risk = int(np.sum(durations >= t))
        d_t = int(np.sum((durations == t) & (events == 1)))
        if at_risk <= 0:
            continue
        s_prob *= 1.0 - (d_t / at_risk)
        times.append(float(t))
        survival.append(float(s_prob))
    return np.asarray(times, dtype=float), np.asarray(survival, dtype=float)


def make_depth_figure(depth_distribution: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if depth_distribution.empty:
        plot_no_data(ax, "Distribution of maximum thread depth", "No depth data available")
    else:
        ax.bar(
            depth_distribution["depth_max"],
            depth_distribution["thread_count"],
            color="#4c78a8",
            alpha=0.85,
        )
        ax.set_xlabel("Maximum thread depth")
        ax.set_ylabel("Thread count")
        ax.set_title("Distribution of maximum thread depth")
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_branching_figure(branching_by_depth: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if branching_by_depth.empty:
        plot_no_data(ax, "Mean branching factor by depth", "No branching data available")
    else:
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
    vals = thread_metrics["reentry_rate"].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        plot_no_data(ax, "Distribution of thread re-entry rates", "No re-entry data available")
    else:
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
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if survival_units.empty or not exponential_fit.get("success"):
        plot_no_data(ax, "Time-to-first-reply survival curve", "Insufficient data for survival fit")
    else:
        s = survival_units["duration_hours"].to_numpy(dtype=float)
        d = survival_units["event_observed"].to_numpy(dtype=int)
        km_t, km_s = kaplan_meier_curve(s, d)

        t_grid = np.linspace(0, np.quantile(s, 0.99), 400)
        alpha = float(exponential_fit["alpha"])
        beta = float(exponential_fit["beta"])
        surv_model = np.exp(-(alpha / beta) * (1.0 - np.exp(-beta * t_grid)))

        ax.step(km_t, km_s, where="post", color="#4c78a8", linewidth=2.0, label="Kaplan-Meier")
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
    if psd_df.empty:
        plot_no_data(ax, "Aggregate comment activity PSD", "Insufficient data for PSD")
    else:
        ax.plot(psd_df["frequency_per_hour"], psd_df["power"], color="#72b7b2", linewidth=2.0)
        ax.axvline(
            TARGET_PERIODICITY_FREQ_PER_HOUR,
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


def find_ops_artifact(ops_root: Path, run_id: str, filename: str) -> Path | None:
    if not ops_root.exists():
        return None
    candidates: list[Path] = []
    for pattern in [f"*/item5/{run_id}/{filename}", f"**/item5/{run_id}/{filename}"]:
        candidates.extend(ops_root.glob(pattern))
    if not candidates:
        return None
    dedup = {str(path): path for path in candidates if path.is_file()}
    if not dedup:
        return None
    return max(dedup.values(), key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_upstream_warnings(run_id: str, ops_root: Path) -> tuple[list[str], dict[str, Any]]:
    warnings: list[str] = []
    details: dict[str, Any] = {"ops_root": str(ops_root)}

    curation_path = find_ops_artifact(ops_root, run_id, "curation_manifest.json")
    details["curation_manifest_path"] = str(curation_path) if curation_path else None
    if curation_path is None:
        warnings.append(
            "Upstream curation manifest not found under outputs/ops; "
            "rows_dropped_missing_submission is unavailable."
        )
    else:
        try:
            curation_payload = load_json(curation_path)
            dropped = (
                curation_payload.get("stats", {})
                .get("comments", {})
                .get("rows_dropped_missing_submission")
            )
            details["rows_dropped_missing_submission"] = dropped
            if dropped is None:
                warnings.append(
                    "Curation manifest present but rows_dropped_missing_submission key is missing."
                )
            elif int(dropped) > 0:
                warnings.append(
                    "Curation manifest reports rows_dropped_missing_submission="
                    f"{int(dropped)}."
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                "Failed to parse curation manifest for warning extraction: "
                f"{type(exc).__name__}: {exc}"
            )

    validation_path = find_ops_artifact(ops_root, run_id, "validation_results.json")
    details["validation_results_path"] = str(validation_path) if validation_path else None
    if validation_path is None:
        warnings.append(
            "Upstream validation_results.json not found under outputs/ops; "
            "request_log non-200 warning signal is unavailable."
        )
    else:
        try:
            validation_payload = load_json(validation_path)
            req_details = (
                validation_payload.get("checks", {})
                .get("request_log_exists", {})
                .get("details", {})
            )
            non_200 = req_details.get("non_200_or_error")
            details["request_log_non_200_or_error"] = non_200
            if non_200 is None:
                warnings.append(
                    "Validation results present but request_log non_200_or_error key is missing."
                )
            elif int(non_200) > 0:
                warnings.append(
                    f"Validation request log includes non_200_or_error={int(non_200)}."
                )

            parent_details = (
                validation_payload.get("checks", {})
                .get("comment_parent_id_completeness", {})
                .get("details", {})
            )
            details["validator_parent_completeness"] = parent_details
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                "Failed to parse validation results for warning extraction: "
                f"{type(exc).__name__}: {exc}"
            )

    return warnings, details


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    run_features_dir = cfg.data_features_root / cfg.run_id
    run_outputs_dir = cfg.outputs_root / cfg.run_id
    figures_dir = run_outputs_dir / "figures"
    tables_dir = run_outputs_dir / "tables"
    ensure_dirs(run_features_dir, figures_dir, tables_dir)

    comments, submissions = prepare_tables(cfg)
    upstream_warnings, upstream_warning_details = collect_upstream_warnings(
        cfg.run_id, cfg.ops_root
    )

    events, depth_diagnostics = build_thread_events(comments=comments, submissions=submissions)
    parent_completeness = parent_completeness_stats(events)
    parent_caveat = (
        "Parent completeness caveat: "
        f"{parent_completeness['missing_parent_comment_count']}/"
        f"{parent_completeness['t1_parent_count']} t1_ parent references are missing in "
        "run-scoped comments; depth/branching estimates are conditional on observed comments."
    )

    warnings: list[str] = list(upstream_warnings)
    warnings.append(parent_caveat)

    thread_metrics = build_thread_metrics(events)
    depth_distribution = make_depth_distribution_table(thread_metrics)
    branching_by_depth = make_branching_table(events)
    reentry_distribution = make_reentry_distribution_table(thread_metrics)
    descriptive_table = summarize_descriptive_table(thread_metrics)

    survival, survival_primary, observation_end = build_survival_units(
        events=events,
        censor_boundary_hours=cfg.censor_boundary_hours,
    )
    primary_clusters = survival_primary["thread_id"].astype("category")
    primary_cluster_codes = primary_clusters.cat.codes.to_numpy()
    primary_n_clusters = int(primary_clusters.nunique())

    exp_fit_primary = fit_exponential_decay(
        durations_hours=survival_primary["duration_hours"].to_numpy(dtype=float),
        event_observed=survival_primary["event_observed"].to_numpy(dtype=float),
    )
    if not exp_fit_primary.get("success"):
        warnings.append(
            "Primary exponential half-life fit failed: "
            f"{exp_fit_primary.get('message', 'unknown error')}"
        )
    exp_boot_primary = cluster_bootstrap_exponential(
        durations_hours=survival_primary["duration_hours"].to_numpy(dtype=float),
        event_observed=survival_primary["event_observed"].to_numpy(dtype=float),
        cluster_codes=primary_cluster_codes,
        n_clusters=primary_n_clusters,
        rng=rng,
        reps=cfg.bootstrap_reps,
    )

    exp_fit_all = fit_exponential_decay(
        durations_hours=survival["duration_hours"].to_numpy(dtype=float),
        event_observed=survival["event_observed"].to_numpy(dtype=float),
    )
    if not exp_fit_all.get("success"):
        warnings.append(
            "No-boundary sensitivity exponential fit failed: "
            f"{exp_fit_all.get('message', 'unknown error')}"
        )

    periodicity_summary, psd_df, periodicity_warnings = analyze_periodicity(
        comment_times=events["created_at_utc"],
        cfg=cfg,
        rng=rng,
    )
    warnings.extend(periodicity_warnings)

    # Write run-scoped feature artifacts.
    events_out = run_features_dir / "thread_events.parquet"
    thread_metrics_out = run_features_dir / "thread_metrics.parquet"
    survival_out = run_features_dir / "survival_units.parquet"
    branching_out = run_features_dir / "branching_by_depth.parquet"
    depth_dist_out = run_features_dir / "depth_distribution.parquet"
    reentry_dist_out = run_features_dir / "reentry_rate_distribution.parquet"
    psd_out_parquet = run_features_dir / "psd_curve.parquet"

    events.to_parquet(events_out, index=False)
    thread_metrics.to_parquet(thread_metrics_out, index=False)
    survival.to_parquet(survival_out, index=False)
    branching_by_depth.to_parquet(branching_out, index=False)
    depth_distribution.to_parquet(depth_dist_out, index=False)
    reentry_distribution.to_parquet(reentry_dist_out, index=False)
    psd_df.to_parquet(psd_out_parquet, index=False)

    # Manuscript-facing figures.
    depth_fig = figures_dir / "reddit_depth_distribution.png"
    branching_fig = figures_dir / "reddit_branching_by_depth.png"
    reentry_fig = figures_dir / "reddit_reentry_distribution.png"
    survival_fig = figures_dir / "reddit_survival_curve.png"
    psd_fig = figures_dir / "reddit_psd.png"

    make_depth_figure(depth_distribution=depth_distribution, out_path=depth_fig)
    make_branching_figure(branching_by_depth=branching_by_depth, out_path=branching_fig)
    make_reentry_figure(thread_metrics=thread_metrics, out_path=reentry_fig)
    make_survival_figure(
        survival_units=survival_primary,
        exponential_fit=exp_fit_primary,
        out_path=survival_fig,
    )
    make_psd_figure(psd_df=psd_df, out_path=psd_fig)

    # Manuscript-facing tables.
    descriptive_out = tables_dir / "descriptive_stats.csv"
    depth_csv_out = tables_dir / "depth_distribution.csv"
    branching_csv_out = tables_dir / "branching_by_depth.csv"
    reentry_csv_out = tables_dir / "reentry_rate_distribution.csv"
    psd_csv_out = tables_dir / "psd_curve.csv"

    descriptive_table.to_csv(descriptive_out, index=False)
    depth_distribution.to_csv(depth_csv_out, index=False)
    branching_by_depth.to_csv(branching_csv_out, index=False)
    reentry_distribution.to_csv(reentry_csv_out, index=False)
    psd_df.to_csv(psd_csv_out, index=False)

    depth_vals = thread_metrics["depth_max"].dropna().to_numpy(dtype=float)
    reentry_vals = thread_metrics["reentry_rate"].dropna().to_numpy(dtype=float)

    qc = {
        "n_submissions_curated_filtered_run": int(len(submissions)),
        "n_comments_curated_filtered_run": int(len(comments)),
        "n_threads_with_comments": int(events["thread_id"].nunique()),
        "timestamp_start": events["created_at_utc"].min().isoformat(),
        "timestamp_end": events["created_at_utc"].max().isoformat(),
        "time_gaps_gt_threshold": detect_time_gaps(
            events["created_at_utc"], threshold_hours=cfg.gap_threshold_hours
        ),
        "missing_parent_timestamp": int(events["parent_created_at_utc"].isna().sum()),
        "negative_lag_since_parent": int((events["lag_since_parent_hours"] < 0).sum()),
        "negative_lag_since_submission": int((events["lag_since_submission_hours"] < 0).sum()),
        "missing_depth": int(events["depth"].isna().sum()),
        "depth_diagnostics": depth_diagnostics,
        "parent_completeness": parent_completeness,
    }

    summary = {
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "input_curated_root": str(cfg.curated_root),
        "censor_boundary_hours": cfg.censor_boundary_hours,
        "gap_threshold_hours": cfg.gap_threshold_hours,
        "periodicity_bin_minutes": cfg.periodicity_bin_minutes,
        "bootstrap_reps": cfg.bootstrap_reps,
        "ar1_sims": cfg.ar1_sims,
        "qc": qc,
        "geometry": {
            "n_threads": int(len(thread_metrics)),
            "mean_depth_max": float(np.mean(depth_vals)) if depth_vals.size else float("nan"),
            "median_depth_max": float(np.median(depth_vals)) if depth_vals.size else float("nan"),
            "p_depth_ge_5": float(np.mean(depth_vals >= 5)) if depth_vals.size else float("nan"),
            "p_depth_ge_10": float(np.mean(depth_vals >= 10)) if depth_vals.size else float("nan"),
            "mean_reentry_rate": float(np.mean(reentry_vals))
            if reentry_vals.size
            else float("nan"),
            "median_reentry_rate": float(np.median(reentry_vals))
            if reentry_vals.size
            else float("nan"),
        },
        "half_life": {
            "primary_sample_comments": int(len(survival_primary)),
            "primary_events": int(survival_primary["event_observed"].sum()),
            "primary_censored": int((1 - survival_primary["event_observed"]).sum()),
            "excluded_boundary_comments": int(survival["excluded_censor_boundary"].sum()),
            "observation_end_utc": observation_end.isoformat(),
            "exponential_primary": exp_fit_primary,
            "exponential_primary_bootstrap": exp_boot_primary,
            "exponential_sensitivity_no_boundary_exclusion": exp_fit_all,
        },
        "periodicity": periodicity_summary,
        "warnings": warnings,
        "warning_details": {
            **upstream_warning_details,
            "parent_completeness_local": parent_completeness,
        },
        "artifacts": {
            "thread_events_parquet": str(events_out),
            "thread_metrics_parquet": str(thread_metrics_out),
            "survival_units_parquet": str(survival_out),
            "branching_by_depth_parquet": str(branching_out),
            "depth_distribution_parquet": str(depth_dist_out),
            "reentry_rate_distribution_parquet": str(reentry_dist_out),
            "psd_curve_parquet": str(psd_out_parquet),
            "figures": [
                str(depth_fig),
                str(branching_fig),
                str(reentry_fig),
                str(survival_fig),
                str(psd_fig),
            ],
            "tables": [
                str(descriptive_out),
                str(depth_csv_out),
                str(branching_csv_out),
                str(reentry_csv_out),
                str(psd_csv_out),
            ],
        },
    }
    summary_out = tables_dir / "analysis_summary.json"
    summary_out.write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": "analysis/07_reddit_only_analysis.py",
        "config": asdict(cfg),
        "inputs": {
            "curated_root": str(cfg.curated_root),
            "run_id": cfg.run_id,
            "ops_root": str(cfg.ops_root),
            "submissions_rows": int(len(submissions)),
            "comments_rows": int(len(comments)),
            "ops_warning_sources": upstream_warning_details,
        },
        "outputs_dir": str(run_outputs_dir),
        "data_features_dir": str(run_features_dir),
        "analysis_summary_json": str(summary_out),
        "warnings": warnings,
        "warning_details": {
            **upstream_warning_details,
            "parent_completeness_local": parent_completeness,
        },
    }
    manifest_out = run_outputs_dir / "run_manifest.json"
    manifest_out.write_text(json.dumps(sanitize_for_json(manifest), indent=2), encoding="utf-8")

    print(f"Run ID: {cfg.run_id}")
    print(f"Summary: {summary_out}")
    print(f"Manifest: {manifest_out}")
    mean_depth = summary["geometry"]["mean_depth_max"]
    print(f"Mean max depth: {mean_depth:.3f}" if np.isfinite(mean_depth) else "Mean max depth: NA")
    half_life = summary["half_life"]["exponential_primary"].get("half_life_hours")
    if half_life is not None and np.isfinite(float(half_life)):
        print(f"Half-life (hours): {float(half_life):.3f}")
    else:
        print("Half-life (hours): NA (fit unavailable)")


if __name__ == "__main__":
    main()
