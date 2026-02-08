#!/usr/bin/env python3
"""Compute methodology-revision quantitative artifacts for the Moltbook paper tables.

Outputs (overwritten):
- paper/tables/moltbook_two_part_metrics.csv
- paper/tables/moltbook_incidence_cloglog.csv
- paper/tables/moltbook_periodicity_event_time_test.csv
- paper/tables/moltbook_periodicity_detectability_power.csv
- paper/tables/moltbook_model_observable_validation.csv
- paper/tables/moltbook_dependence_robustness.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, special, stats
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

DEFAULT_SURVIVAL_PATH = Path(
    "data_features/moltbook_only/run_20260206-145240Z/survival_units.parquet"
)
DEFAULT_THREAD_EVENTS_PATH = Path(
    "data_features/moltbook_only/run_20260206-145240Z/thread_events.parquet"
)
DEFAULT_AGENTS_PATH = Path("data_curated/hf_archive/snapshot_20260204-234429Z/agents")
DEFAULT_SEED = 20260208
DEFAULT_GAP_THRESHOLD_HOURS = 6.0
DEFAULT_PERIOD_HOURS = 4.0
DEFAULT_NULL_MC_REPS = 200_000
DEFAULT_POWER_SIM_REPS = 50_000
DEFAULT_ALPHA_LEVEL = 0.05
DEFAULT_KAPPA_GRID = tuple(np.round(np.arange(0.0, 3.0001, 0.2), 2).tolist())


@dataclass(frozen=True)
class Config:
    survival_path: Path
    thread_events_path: Path
    agents_path: Path
    seed: int
    gap_threshold_hours: float
    period_hours: float
    null_mc_reps: int
    power_sim_reps: int
    alpha_level: float
    kappa_grid: tuple[float, ...]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survival-path", type=Path, default=DEFAULT_SURVIVAL_PATH)
    parser.add_argument("--thread-events-path", type=Path, default=DEFAULT_THREAD_EVENTS_PATH)
    parser.add_argument("--agents-path", type=Path, default=DEFAULT_AGENTS_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gap-threshold-hours", type=float, default=DEFAULT_GAP_THRESHOLD_HOURS)
    parser.add_argument("--period-hours", type=float, default=DEFAULT_PERIOD_HOURS)
    parser.add_argument("--null-mc-reps", type=int, default=DEFAULT_NULL_MC_REPS)
    parser.add_argument("--power-sim-reps", type=int, default=DEFAULT_POWER_SIM_REPS)
    parser.add_argument("--alpha-level", type=float, default=DEFAULT_ALPHA_LEVEL)
    parser.add_argument(
        "--kappa-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_KAPPA_GRID),
        help="Von Mises concentration values for detectability simulation.",
    )
    args = parser.parse_args()
    return Config(
        survival_path=args.survival_path,
        thread_events_path=args.thread_events_path,
        agents_path=args.agents_path,
        seed=args.seed,
        gap_threshold_hours=args.gap_threshold_hours,
        period_hours=args.period_hours,
        null_mc_reps=args.null_mc_reps,
        power_sim_reps=args.power_sim_reps,
        alpha_level=args.alpha_level,
        kappa_grid=tuple(float(x) for x in args.kappa_grid),
    )


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def attach_claim_status(
    survival_df: pd.DataFrame,
    events_df: pd.DataFrame,
    agents_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    claims = agents_df[["id", "is_claimed"]].rename(columns={"id": "comment_agent_id"}).copy()
    claims["is_claimed"] = pd.to_numeric(claims["is_claimed"], errors="coerce")

    surv = survival_df.merge(claims, on="comment_agent_id", how="left", validate="many_to_one")
    evt = events_df.merge(claims, on="comment_agent_id", how="left", validate="many_to_one")

    for df in (surv, evt):
        claimed_numeric = pd.to_numeric(df["is_claimed"], errors="coerce")
        df["claimed_group"] = np.where(claimed_numeric.isna(), "Unknown", "Unclaimed")
        df.loc[claimed_numeric == 1, "claimed_group"] = "Claimed"

    return surv, evt


def positive_minutes(hours: pd.Series) -> pd.Series:
    minutes = pd.to_numeric(hours, errors="coerce") * 60.0
    return minutes[(minutes > 0) & np.isfinite(minutes)]


def fit_weibull_conditional(event_minutes: np.ndarray) -> dict[str, float | int | bool | str]:
    x = np.asarray(event_minutes, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 10:
        return {"success": False, "message": "Too few positive event times for Weibull fit."}

    mean_x = float(np.mean(x))
    std_x = float(np.std(x))
    shape_guess = 1.0
    if mean_x > 0 and std_x > 0:
        shape_guess = float(max(0.1, min(10.0, (mean_x / std_x) ** 1.2)))
    scale_guess = max(float(np.median(x)), 1e-6)

    def neg_loglik(theta: np.ndarray) -> float:
        log_shape, log_scale = theta
        k = np.exp(log_shape)
        logx = np.log(x)
        logz = logx - log_scale
        # Clip exponent argument to avoid overflow in exp(k * logz).
        expo = np.exp(np.clip(k * logz, -745.0, 700.0))
        ll = log_shape - log_scale + (k - 1.0) * logz - expo
        return -float(np.sum(ll))

    opt = optimize.minimize(
        neg_loglik,
        x0=np.array([np.log(shape_guess), np.log(scale_guess)]),
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0), (-20.0, 20.0)],
    )

    if not opt.success:
        return {"success": False, "message": str(opt.message)}

    shape = float(np.exp(opt.x[0]))
    scale = float(np.exp(opt.x[1]))
    return {
        "success": True,
        "shape_k": shape,
        "scale_minutes": scale,
        "log_likelihood": float(-opt.fun),
        "n_events": int(x.size),
    }


def fit_decay_alpha_beta(
    s_hours: np.ndarray,
    d_event: np.ndarray,
) -> dict[str, float | int | bool | str]:
    s = np.asarray(s_hours, dtype=float)
    d = np.asarray(d_event, dtype=float)
    valid = np.isfinite(s) & np.isfinite(d) & (s > 0)
    s = s[valid]
    d = d[valid]
    if s.size < 20 or float(np.sum(d)) < 5:
        return {"success": False, "message": "Insufficient survival observations/events."}

    event_rate = float(np.sum(d) / np.sum(s))
    observed = s[d > 0]
    beta0 = float(1.0 / max(float(np.median(observed)) if observed.size > 0 else 1.0, 1e-4))
    alpha0 = max(event_rate, 1e-6)

    def neg_loglik(theta: np.ndarray) -> float:
        log_alpha, log_beta = theta
        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)
        integral = (alpha / beta) * (1.0 - np.exp(-beta * s))
        ll = d * (log_alpha - beta * s) - integral
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
            x0=start,
            method="L-BFGS-B",
            bounds=[(-25.0, 25.0), (-25.0, 25.0)],
        )
        if best is None or res.fun < best.fun:
            best = res

    if best is None or not best.success:
        return {"success": False, "message": "Decay model optimization failed."}

    alpha = float(np.exp(best.x[0]))
    beta = float(np.exp(best.x[1]))
    mu_timing = alpha / beta if beta > 0 else np.nan
    return {
        "success": True,
        "alpha_per_hour": alpha,
        "beta_per_hour": beta,
        "mu_timing_alpha_over_beta": float(mu_timing),
        "log_likelihood": float(-best.fun),
        "n_units": int(s.size),
        "n_events": int(np.sum(d)),
    }


def one_group_two_part_metrics(df: pd.DataFrame) -> dict[str, Any]:
    n_units = int(len(df))
    n_replied = int(df["event_observed"].sum()) if n_units else 0
    incidence = float(n_replied / n_units) if n_units else np.nan

    replied_minutes = positive_minutes(
        df.loc[df["event_observed"].astype(bool), "duration_hours"]
    )
    conditional_p50 = (
        float(np.quantile(replied_minutes, 0.50)) if not replied_minutes.empty else np.nan
    )
    conditional_p90 = (
        float(np.quantile(replied_minutes, 0.90)) if not replied_minutes.empty else np.nan
    )
    conditional_p95 = (
        float(np.quantile(replied_minutes, 0.95)) if not replied_minutes.empty else np.nan
    )

    p_within_30s_uncond = (
        float(((df["event_observed"] == 1) & (df["duration_hours"] * 3600.0 <= 30.0)).mean())
        if n_units
        else np.nan
    )
    p_within_5m_uncond = (
        float(((df["event_observed"] == 1) & (df["duration_hours"] * 60.0 <= 5.0)).mean())
        if n_units
        else np.nan
    )

    if not replied_minutes.empty:
        p_within_30s_cond = float((replied_minutes <= 0.5).mean())
        p_within_5m_cond = float((replied_minutes <= 5.0).mean())
        exp_rate = float(1.0 / replied_minutes.mean())
        exp_half_life = float(np.log(2.0) / exp_rate)
        weibull_fit = fit_weibull_conditional(replied_minutes.to_numpy(dtype=float))
    else:
        p_within_30s_cond = np.nan
        p_within_5m_cond = np.nan
        exp_rate = np.nan
        exp_half_life = np.nan
        weibull_fit = {"success": False, "message": "No observed replies in group."}

    out = {
        "n_units_parent_comments": n_units,
        "n_replied_parent_comments": n_replied,
        "reply_incidence_unconditional_prob": incidence,
        "conditional_reply_time_p50_minutes": conditional_p50,
        "conditional_reply_time_p90_minutes": conditional_p90,
        "conditional_reply_time_p95_minutes": conditional_p95,
        "conditional_t90_minutes": conditional_p90,
        "conditional_t95_minutes": conditional_p95,
        "reply_within_30_seconds_unconditional_prob": p_within_30s_uncond,
        "reply_within_5_minutes_unconditional_prob": p_within_5m_uncond,
        "reply_within_30_seconds_conditional_prob": p_within_30s_cond,
        "reply_within_5_minutes_conditional_prob": p_within_5m_cond,
        "exponential_rate_per_minute": exp_rate,
        "exponential_half_life_minutes": exp_half_life,
        "weibull_success": bool(weibull_fit.get("success", False)),
        "weibull_shape_k": float(weibull_fit.get("shape_k", np.nan)),
        "weibull_scale_minutes": float(weibull_fit.get("scale_minutes", np.nan)),
        "weibull_log_likelihood": float(weibull_fit.get("log_likelihood", np.nan)),
        "weibull_fit_n_events": int(weibull_fit.get("n_events", 0)),
        "weibull_message": str(weibull_fit.get("message", "")),
    }
    return out


def build_two_part_table(survival_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_group(group_family: str, group_label: str, sub: pd.DataFrame) -> None:
        row = {"group_family": group_family, "group_label": group_label}
        row.update(one_group_two_part_metrics(sub))
        rows.append(row)

    append_group("overall", "Overall", survival_df)

    for label, sub in (
        survival_df.groupby("submolt_category", dropna=False, sort=True)
        if "submolt_category" in survival_df.columns
        else []
    ):
        append_group("submolt_category", str(label), sub)

    claimable = survival_df[survival_df["claimed_group"].isin(["Claimed", "Unclaimed"])].copy()
    if not claimable.empty:
        for label, sub in claimable.groupby("claimed_group", sort=True):
            append_group("claimed_status", str(label), sub)

    out = pd.DataFrame(rows)
    return out.sort_values(["group_family", "group_label"], kind="stable").reset_index(drop=True)


def build_incidence_cloglog_table(survival_df: pd.DataFrame) -> pd.DataFrame:
    modeled = survival_df[survival_df["claimed_group"].isin(["Claimed", "Unclaimed"])].copy()
    modeled["event_observed"] = (
        pd.to_numeric(modeled["event_observed"], errors="coerce").fillna(0).astype(int)
    )
    modeled = modeled.dropna(
        subset=[
            "event_observed",
            "submolt_category",
            "claimed_group",
            "thread_id",
            "comment_agent_id",
        ]
    )
    modeled["submolt_category"] = modeled["submolt_category"].astype(str)
    modeled["claimed_group"] = modeled["claimed_group"].astype(str)
    if modeled.empty:
        raise ValueError("No rows available for incidence cloglog model.")

    reference_submolt = "Social/Casual"
    covariance_note = "two_way_cluster_thread_author"
    if reference_submolt not in set(modeled["submolt_category"]):
        reference_submolt = sorted(modeled["submolt_category"].unique().tolist())[0]
        covariance_note = (
            f"two_way_cluster_thread_author;"
            f"reference_submolt_fallback:{reference_submolt}"
        )

    formula = (
        "event_observed ~ "
        f"C(submolt_category, Treatment(reference='{reference_submolt}')) + "
        "C(claimed_group, Treatment(reference='Unclaimed'))"
    )
    result = smf.glm(
        formula=formula,
        data=modeled,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit()

    params = result.params.astype(float)
    se_model = result.bse.astype(float)

    n_terms = len(params)
    thread_codes, _ = pd.factorize(modeled["thread_id"], sort=False)
    author_codes, _ = pd.factorize(modeled["comment_agent_id"], sort=False)

    se_thread = np.full(n_terms, np.nan, dtype=float)
    se_twoway = np.full(n_terms, np.nan, dtype=float)

    try:
        cov_thread = np.asarray(cov_cluster(result, thread_codes), dtype=float)
        diag_thread = np.diag(cov_thread)
        if np.all(np.isfinite(diag_thread)) and np.all(diag_thread >= 0):
            se_thread = np.sqrt(diag_thread)
        else:
            covariance_note = f"{covariance_note};thread_cov_diag_invalid_fallback_model"
            se_thread = se_model.to_numpy(dtype=float)
    except Exception as exc:  # pragma: no cover - defensive fallback
        covariance_note = f"{covariance_note};thread_cov_failed_fallback_model:{type(exc).__name__}"
        se_thread = se_model.to_numpy(dtype=float)

    try:
        cov_twoway_raw = cov_cluster_2groups(result, thread_codes, author_codes)
        cov_twoway = cov_twoway_raw[0] if isinstance(cov_twoway_raw, tuple) else cov_twoway_raw
        cov_twoway = np.asarray(cov_twoway, dtype=float)
        diag_twoway = np.diag(cov_twoway)
        if np.all(np.isfinite(diag_twoway)) and np.all(diag_twoway >= 0):
            se_twoway = np.sqrt(diag_twoway)
        else:
            covariance_note = f"{covariance_note};twoway_cov_diag_invalid_fallback_thread"
            se_twoway = se_thread.copy()
    except Exception as exc:  # pragma: no cover - defensive fallback
        covariance_note = (
            f"{covariance_note};"
            f"twoway_cov_failed_fallback_thread:{type(exc).__name__}"
        )
        se_twoway = se_thread.copy()

    coef_arr = params.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_twoway = coef_arr / se_twoway
    p_twoway = 2.0 * stats.norm.sf(np.abs(z_twoway))
    ci95_low = coef_arr - 1.96 * se_twoway
    ci95_high = coef_arr + 1.96 * se_twoway

    n_obs = int(len(modeled))
    n_events = int(modeled["event_observed"].sum())
    n_threads = int(modeled["thread_id"].nunique())
    n_authors = int(modeled["comment_agent_id"].nunique())

    rows: list[dict[str, Any]] = []
    for i, term in enumerate(params.index.tolist()):
        rows.append(
            {
                "row_type": "coefficient",
                "term": term,
                "coef": coef_arr[i],
                "se_model": se_model.iloc[i],
                "se_cluster_thread": se_thread[i],
                "se_cluster_twoway_thread_author": se_twoway[i],
                "z_twoway": z_twoway[i],
                "p_twoway": p_twoway[i],
                "ci95_twoway_low": ci95_low[i],
                "ci95_twoway_high": ci95_high[i],
                "observed_mean_incidence_prob": np.nan,
                "predicted_mean_incidence_prob": np.nan,
                "incidence_prediction_error_prob": np.nan,
                "model_n_obs": n_obs,
                "model_n_events": n_events,
                "model_n_threads": n_threads,
                "model_n_authors": n_authors,
                "model_link": "cloglog",
                "reference_submolt_category": reference_submolt,
                "reference_claimed_group": "Unclaimed",
                "formula": formula,
                "covariance_note": covariance_note,
                "model_converged": bool(result.converged),
            }
        )

    predicted = np.asarray(result.predict(modeled), dtype=float)
    observed_mean = float(modeled["event_observed"].mean())
    predicted_mean = float(np.mean(predicted))
    rows.append(
        {
            "row_type": "summary",
            "term": "__summary_mean_incidence__",
            "coef": np.nan,
            "se_model": np.nan,
            "se_cluster_thread": np.nan,
            "se_cluster_twoway_thread_author": np.nan,
            "z_twoway": np.nan,
            "p_twoway": np.nan,
            "ci95_twoway_low": np.nan,
            "ci95_twoway_high": np.nan,
            "observed_mean_incidence_prob": observed_mean,
            "predicted_mean_incidence_prob": predicted_mean,
            "incidence_prediction_error_prob": predicted_mean - observed_mean,
            "model_n_obs": n_obs,
            "model_n_events": n_events,
            "model_n_threads": n_threads,
            "model_n_authors": n_authors,
            "model_link": "cloglog",
            "reference_submolt_category": reference_submolt,
            "reference_claimed_group": "Unclaimed",
            "formula": formula,
            "covariance_note": covariance_note,
            "model_converged": bool(result.converged),
        }
    )

    return pd.DataFrame(rows)


def longest_contiguous_segment(times_utc: pd.Series, gap_threshold_hours: float) -> dict[str, Any]:
    t = to_utc(times_utc).dropna().sort_values()
    if t.empty:
        raise ValueError("No valid event timestamps for periodicity analysis.")

    arr_ns = t.astype("int64", copy=False).to_numpy()
    gaps_sec = np.diff(arr_ns) / 1e9
    break_idx = np.where(gaps_sec > gap_threshold_hours * 3600.0)[0]

    starts = np.r_[0, break_idx + 1]
    ends = np.r_[break_idx, len(arr_ns) - 1]
    seg_sizes = (ends - starts + 1).astype(int)
    seg_durations = (arr_ns[ends] - arr_ns[starts]) / 1e9

    order = np.lexsort((seg_durations, seg_sizes))
    best = int(order[-1])

    start_i = int(starts[best])
    end_i = int(ends[best])
    segment_ns = arr_ns[start_i : end_i + 1]

    return {
        "segment_index": best,
        "segment_start_utc": pd.to_datetime(segment_ns[0], utc=True, unit="ns"),
        "segment_end_utc": pd.to_datetime(segment_ns[-1], utc=True, unit="ns"),
        "segment_event_count": int(segment_ns.size),
        "segment_duration_hours": float((segment_ns[-1] - segment_ns[0]) / (3600.0 * 1e9)),
        "segment_epoch_seconds": segment_ns / 1e9,
    }


def rayleigh_statistic(epoch_seconds: np.ndarray, period_hours: float) -> dict[str, float]:
    period_seconds = period_hours * 3600.0
    phases = np.mod(epoch_seconds, period_seconds) / period_seconds
    angles = 2.0 * np.pi * phases
    n = int(angles.size)
    c = float(np.sum(np.cos(angles)))
    s = float(np.sum(np.sin(angles)))
    r = float(np.sqrt(c * c + s * s) / n)
    z = float(n * r * r)
    mu = float(np.arctan2(s, c) % (2.0 * np.pi))
    return {
        "n": n,
        "rayleigh_r": r,
        "rayleigh_z": z,
        "mean_phase_rad": mu,
        "mean_phase_minutes": float((mu / (2.0 * np.pi)) * period_hours * 60.0),
    }


def null_z_monte_carlo_normal(n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(n / 2.0)
    c = rng.normal(loc=0.0, scale=std, size=reps)
    s = rng.normal(loc=0.0, scale=std, size=reps)
    return (c * c + s * s) / n


def detectability_power_table(
    n: int,
    z_critical: float,
    kappa_grid: tuple[float, ...],
    reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kappa in kappa_grid:
        if kappa < 1e-12:
            rho = 0.0
        else:
            rho = float(special.i1(kappa) / special.i0(kappa))
        nc = 2.0 * n * rho * rho
        z_sim = 0.5 * rng.noncentral_chisquare(df=2.0, nonc=nc, size=reps)
        power = float(np.mean(z_sim >= z_critical))
        rows.append(
            {
                "kappa": float(kappa),
                "mean_resultant_length_rho": rho,
                "estimated_power": power,
                "sim_reps": int(reps),
            }
        )

    out = pd.DataFrame(rows).sort_values("kappa", kind="stable").reset_index(drop=True)
    above = out.loc[out["estimated_power"] >= 0.8, "kappa"]
    out["kappa_power_crosses_0p8"] = float(above.iloc[0]) if not above.empty else np.nan
    return out


def model_observables_for_group(
    survival_sub: pd.DataFrame,
    events_sub: pd.DataFrame,
) -> dict[str, Any]:
    fit = fit_decay_alpha_beta(
        s_hours=survival_sub["duration_hours"].to_numpy(dtype=float),
        d_event=survival_sub["event_observed"].to_numpy(dtype=float),
    )

    obs_reply_incidence = (
        float(survival_sub["event_observed"].mean()) if len(survival_sub) else np.nan
    )

    depths = pd.to_numeric(events_sub.get("depth", pd.Series([], dtype="float64")), errors="coerce")
    depth_valid = depths[np.isfinite(depths)]

    observed_non_root_branching = (
        float(pd.to_numeric(events_sub.get("n_children"), errors="coerce").mean())
        if len(events_sub)
        else np.nan
    )
    observed_depth_ge2 = float((depth_valid >= 2).mean()) if not depth_valid.empty else np.nan
    observed_depth_ge3 = float((depth_valid >= 3).mean()) if not depth_valid.empty else np.nan
    observed_depth_ge5 = float((depth_valid >= 5).mean()) if not depth_valid.empty else np.nan

    if fit.get("success"):
        mu = float(fit["mu_timing_alpha_over_beta"])
        pred_reply_incidence = float(1.0 - np.exp(-mu))
        pred_branching = mu
        pred_depth_ge2 = float(np.clip(mu**1, 0.0, 1.0))
        pred_depth_ge3 = float(np.clip(mu**2, 0.0, 1.0))
        pred_depth_ge5 = float(np.clip(mu**4, 0.0, 1.0))
    else:
        mu = np.nan
        pred_reply_incidence = np.nan
        pred_branching = np.nan
        pred_depth_ge2 = np.nan
        pred_depth_ge3 = np.nan
        pred_depth_ge5 = np.nan

    return {
        "n_survival_units": int(len(survival_sub)),
        "n_survival_replies": int(survival_sub["event_observed"].sum()) if len(survival_sub) else 0,
        "n_event_rows": int(len(events_sub)),
        "alpha_per_hour": float(fit.get("alpha_per_hour", np.nan)),
        "beta_per_hour": float(fit.get("beta_per_hour", np.nan)),
        "mu_timing_alpha_over_beta": mu,
        "pred_reply_incidence_prob": pred_reply_incidence,
        "obs_reply_incidence_prob": obs_reply_incidence,
        "pred_non_root_branching_factor": pred_branching,
        "obs_non_root_branching_factor": observed_non_root_branching,
        "pred_depth_tail_ge_2_prob": pred_depth_ge2,
        "obs_depth_tail_ge_2_prob": observed_depth_ge2,
        "pred_depth_tail_ge_3_prob": pred_depth_ge3,
        "obs_depth_tail_ge_3_prob": observed_depth_ge3,
        "pred_depth_tail_ge_5_prob": pred_depth_ge5,
        "obs_depth_tail_ge_5_prob": observed_depth_ge5,
        "abs_error_reply_incidence": abs(pred_reply_incidence - obs_reply_incidence)
        if np.isfinite(pred_reply_incidence) and np.isfinite(obs_reply_incidence)
        else np.nan,
        "abs_error_non_root_branching": abs(pred_branching - observed_non_root_branching)
        if np.isfinite(pred_branching) and np.isfinite(observed_non_root_branching)
        else np.nan,
        "fit_success": bool(fit.get("success", False)),
        "fit_message": str(fit.get("message", "")),
    }


def build_model_validation_table(
    survival_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_group(
        group_family: str,
        group_label: str,
        surv_sub: pd.DataFrame,
        evt_sub: pd.DataFrame,
    ) -> None:
        row = {"group_family": group_family, "group_label": group_label}
        row.update(model_observables_for_group(surv_sub, evt_sub))
        rows.append(row)

    add_group("overall", "Overall", survival_df, events_df)

    if "submolt_category" in survival_df.columns and "submolt_category" in events_df.columns:
        surv_groups = {
            str(k): v for k, v in survival_df.groupby("submolt_category", dropna=False, sort=True)
        }
        evt_groups = {
            str(k): v for k, v in events_df.groupby("submolt_category", dropna=False, sort=True)
        }
        for label in sorted(surv_groups.keys()):
            add_group(
                "submolt_category",
                label,
                surv_groups[label],
                evt_groups.get(label, events_df.iloc[0:0]),
            )

    surv_claim = survival_df[survival_df["claimed_group"].isin(["Claimed", "Unclaimed"])].copy()
    evt_claim = events_df[events_df["claimed_group"].isin(["Claimed", "Unclaimed"])].copy()
    if not surv_claim.empty and not evt_claim.empty:
        surv_groups = {str(k): v for k, v in surv_claim.groupby("claimed_group", sort=True)}
        evt_groups = {str(k): v for k, v in evt_claim.groupby("claimed_group", sort=True)}
        for label in sorted(surv_groups.keys()):
            add_group(
                "claimed_status",
                label,
                surv_groups[label],
                evt_groups.get(label, events_df.iloc[0:0]),
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["group_family", "group_label"], kind="stable").reset_index(drop=True)


def dependence_robustness_table(
    survival_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    primary = one_group_two_part_metrics(survival_df)

    sampled = survival_df.groupby("thread_id", group_keys=False).sample(
        n=1,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    robust = one_group_two_part_metrics(sampled)

    comparisons = [
        ("reply_incidence_unconditional_prob", "probability"),
        ("conditional_reply_time_p50_minutes", "minutes"),
        ("conditional_t90_minutes", "minutes"),
        ("exponential_half_life_minutes", "minutes"),
    ]

    rows = []
    for metric, unit in comparisons:
        p = float(primary.get(metric, np.nan))
        r = float(robust.get(metric, np.nan))
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "primary_value": p,
                "one_parent_per_thread_value": r,
                "absolute_difference": abs(r - p) if np.isfinite(p) and np.isfinite(r) else np.nan,
                "relative_difference_percent": (
                    100.0 * (r - p) / p if np.isfinite(p) and p != 0 and np.isfinite(r) else np.nan
                ),
                "primary_n_units": int(primary["n_units_parent_comments"]),
                "robust_n_units": int(robust["n_units_parent_comments"]),
                "sampling_rule": "one_parent_per_thread_random_with_seed",
            }
        )

    return pd.DataFrame(rows)


def append_metadata(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["analysis_seed"] = int(cfg.seed)
    out["input_survival_path"] = str(cfg.survival_path)
    out["input_thread_events_path"] = str(cfg.thread_events_path)
    out["input_agents_path"] = str(cfg.agents_path)
    out["input_features_run_id"] = cfg.survival_path.parent.name
    out["input_snapshot_id"] = cfg.agents_path.parent.name
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    survival_df = pd.read_parquet(cfg.survival_path)
    events_df = pd.read_parquet(cfg.thread_events_path)
    agents_df = pd.read_parquet(cfg.agents_path)

    require_columns(
        survival_df,
        [
            "thread_id",
            "comment_id",
            "comment_agent_id",
            "submolt_category",
            "event_observed",
            "duration_hours",
        ],
        "survival_units",
    )
    require_columns(
        events_df,
        [
            "comment_id",
            "comment_agent_id",
            "created_at_utc",
            "submolt_category",
            "depth",
            "n_children",
        ],
        "thread_events",
    )
    require_columns(agents_df, ["id", "is_claimed"], "agents")

    survival_df = survival_df.copy()
    events_df = events_df.copy()
    survival_df["event_observed"] = (
        pd.to_numeric(survival_df["event_observed"], errors="coerce").fillna(0).astype(int)
    )

    survival_df, events_df = attach_claim_status(survival_df, events_df, agents_df)

    two_part = build_two_part_table(survival_df)
    incidence_cloglog = build_incidence_cloglog_table(survival_df)

    segment = longest_contiguous_segment(events_df["created_at_utc"], cfg.gap_threshold_hours)
    rayleigh = rayleigh_statistic(segment["segment_epoch_seconds"], cfg.period_hours)

    z_null = null_z_monte_carlo_normal(rayleigh["n"], cfg.null_mc_reps, rng)
    p_mc_raw = float(np.mean(z_null >= rayleigh["rayleigh_z"]))
    p_mc = max(1.0 / float(cfg.null_mc_reps), p_mc_raw)
    z_crit = float(np.quantile(z_null, 1.0 - cfg.alpha_level))

    periodicity_table = pd.DataFrame(
        [
            {
                "period_hours": float(cfg.period_hours),
                "gap_threshold_hours": float(cfg.gap_threshold_hours),
                "segment_index": int(segment["segment_index"]),
                "segment_start_utc": str(segment["segment_start_utc"]),
                "segment_end_utc": str(segment["segment_end_utc"]),
                "segment_duration_hours": float(segment["segment_duration_hours"]),
                "segment_event_count": int(segment["segment_event_count"]),
                "rayleigh_r": float(rayleigh["rayleigh_r"]),
                "rayleigh_z": float(rayleigh["rayleigh_z"]),
                "rayleigh_p_value_monte_carlo": p_mc,
                "null_calibration_method": "normal_approx_monte_carlo",
                "null_monte_carlo_reps": int(cfg.null_mc_reps),
                "mean_phase_rad": float(rayleigh["mean_phase_rad"]),
                "mean_phase_minutes": float(rayleigh["mean_phase_minutes"]),
                "alpha_level": float(cfg.alpha_level),
                "critical_z_at_alpha": z_crit,
            }
        ]
    )

    detectability = detectability_power_table(
        n=rayleigh["n"],
        z_critical=z_crit,
        kappa_grid=cfg.kappa_grid,
        reps=cfg.power_sim_reps,
        rng=rng,
    )
    detectability["period_hours"] = float(cfg.period_hours)
    detectability["segment_event_count"] = int(segment["segment_event_count"])
    detectability["critical_z_at_alpha"] = z_crit
    detectability["alpha_level"] = float(cfg.alpha_level)
    detectability["power_simulation_method"] = "noncentral_chi_square_monte_carlo"

    model_validation = build_model_validation_table(survival_df, events_df)

    robustness = dependence_robustness_table(survival_df, rng)

    two_part = append_metadata(two_part, cfg)
    incidence_cloglog = append_metadata(incidence_cloglog, cfg)
    periodicity_table = append_metadata(periodicity_table, cfg)
    detectability = append_metadata(detectability, cfg)
    model_validation = append_metadata(model_validation, cfg)
    robustness = append_metadata(robustness, cfg)

    write_csv(two_part, Path("paper/tables/moltbook_two_part_metrics.csv"))
    write_csv(incidence_cloglog, Path("paper/tables/moltbook_incidence_cloglog.csv"))
    write_csv(periodicity_table, Path("paper/tables/moltbook_periodicity_event_time_test.csv"))
    write_csv(detectability, Path("paper/tables/moltbook_periodicity_detectability_power.csv"))
    write_csv(model_validation, Path("paper/tables/moltbook_model_observable_validation.csv"))
    write_csv(robustness, Path("paper/tables/moltbook_dependence_robustness.csv"))

    print("Wrote 6 tables:")
    print("- paper/tables/moltbook_two_part_metrics.csv")
    print("- paper/tables/moltbook_incidence_cloglog.csv")
    print("- paper/tables/moltbook_periodicity_event_time_test.csv")
    print("- paper/tables/moltbook_periodicity_detectability_power.csv")
    print("- paper/tables/moltbook_model_observable_validation.csv")
    print("- paper/tables/moltbook_dependence_robustness.csv")


if __name__ == "__main__":
    main()
