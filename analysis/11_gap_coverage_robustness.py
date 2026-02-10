#!/usr/bin/env python3
"""Coverage-gap robustness diagnostics for Moltbook persistence measurements.

This script generates manuscript-facing robustness tables for:
1) gap-disambiguation evidence across raw archive tables,
2) two-part decomposition stability across contiguous windows, and
3) horizon-standardized reply probabilities with explicit risk sets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SURVIVAL_PATH = Path(
    "data_features/moltbook_only/run_20260206-145240Z/survival_units.parquet"
)
DEFAULT_RAW_ARCHIVE_ROOT = Path("data_raw/hf_archive/snapshot_20260204-234429Z")
DEFAULT_WINDOWS_OUT = Path("paper/tables/moltbook_gap_window_robustness.csv")
DEFAULT_HORIZON_OUT = Path("paper/tables/moltbook_gap_horizon_standardized.csv")
DEFAULT_GAP_EVIDENCE_OUT = Path("paper/tables/moltbook_gap_disambiguation_evidence.csv")
DEFAULT_SUMMARY_OUT = Path("paper/tables/moltbook_gap_robustness_summary.json")

HORIZONS_SECONDS = (30, 300, 3600)


@dataclass(frozen=True)
class GapInterval:
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    duration_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate coverage-gap robustness outputs for manuscript integration."
    )
    parser.add_argument("--survival-path", type=Path, default=DEFAULT_SURVIVAL_PATH)
    parser.add_argument("--raw-archive-root", type=Path, default=DEFAULT_RAW_ARCHIVE_ROOT)
    parser.add_argument("--windows-out", type=Path, default=DEFAULT_WINDOWS_OUT)
    parser.add_argument("--horizon-out", type=Path, default=DEFAULT_HORIZON_OUT)
    parser.add_argument("--gap-evidence-out", type=Path, default=DEFAULT_GAP_EVIDENCE_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument(
        "--gap-threshold-hours",
        type=float,
        default=6.0,
        help="Minimum inter-event gap to define a coverage break.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_survival(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = {
        "created_at_utc",
        "first_reply_at",
        "event_observed",
        "duration_hours",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required survival columns: {sorted(missing)}")
    df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True)
    df["first_reply_at"] = pd.to_datetime(df["first_reply_at"], utc=True, errors="coerce")
    df["event_observed"] = pd.to_numeric(df["event_observed"], errors="coerce").fillna(0).astype(int)
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    df["reply_seconds"] = np.where(
        df["event_observed"].astype(bool),
        (df["first_reply_at"] - df["created_at_utc"]).dt.total_seconds(),
        np.nan,
    )
    return df


def detect_primary_gap(created_times: pd.Series, threshold_hours: float) -> GapInterval:
    arr = pd.to_datetime(created_times, utc=True).dropna().sort_values().reset_index(drop=True)
    if arr.empty:
        raise ValueError("No valid timestamps found in survival data.")
    gaps_hours = arr.diff().dt.total_seconds().div(3600.0)
    mask = gaps_hours > threshold_hours
    if not mask.any():
        raise ValueError(
            f"No gaps larger than {threshold_hours:.2f} hours were found in survival timestamps."
        )
    idx = int(gaps_hours[mask].idxmax())
    start = arr.iloc[idx - 1]
    end = arr.iloc[idx]
    duration = float(gaps_hours.iloc[idx])
    return GapInterval(start_utc=start, end_utc=end, duration_hours=duration)


def summarize_two_part(df: pd.DataFrame, *, event_col: str, reply_seconds_col: str) -> dict[str, float]:
    n_parents = int(len(df))
    if n_parents == 0:
        return {
            "n_parents": 0,
            "n_replies": 0,
            "reply_incidence_pct": np.nan,
            "conditional_t50_seconds": np.nan,
            "conditional_t90_seconds": np.nan,
            "p_reply_le_30s_uncond_pct": np.nan,
            "p_reply_le_5m_uncond_pct": np.nan,
            "p_reply_le_1h_uncond_pct": np.nan,
            "p_reply_le_30s_cond_pct": np.nan,
            "p_reply_le_5m_cond_pct": np.nan,
            "p_reply_le_1h_cond_pct": np.nan,
        }

    event = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    reply_seconds = pd.to_numeric(df[reply_seconds_col], errors="coerce")
    replied = reply_seconds[event.astype(bool)].dropna()

    def uncond_prob(threshold_seconds: float) -> float:
        return float((event.astype(bool) & (reply_seconds <= threshold_seconds)).mean()) * 100.0

    def cond_prob(threshold_seconds: float) -> float:
        if replied.empty:
            return np.nan
        return float((replied <= threshold_seconds).mean()) * 100.0

    return {
        "n_parents": n_parents,
        "n_replies": int(event.sum()),
        "reply_incidence_pct": float(event.mean()) * 100.0,
        "conditional_t50_seconds": float(replied.quantile(0.5)) if not replied.empty else np.nan,
        "conditional_t90_seconds": float(replied.quantile(0.9)) if not replied.empty else np.nan,
        "p_reply_le_30s_uncond_pct": uncond_prob(30.0),
        "p_reply_le_5m_uncond_pct": uncond_prob(300.0),
        "p_reply_le_1h_uncond_pct": uncond_prob(3600.0),
        "p_reply_le_30s_cond_pct": cond_prob(30.0),
        "p_reply_le_5m_cond_pct": cond_prob(300.0),
        "p_reply_le_1h_cond_pct": cond_prob(3600.0),
    }


def summarize_horizon_standardized(
    df: pd.DataFrame,
    *,
    scenario: str,
    event_col: str,
    reply_seconds_col: str,
    followup_hours_col: str,
) -> list[dict[str, Any]]:
    event = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    reply_seconds = pd.to_numeric(df[reply_seconds_col], errors="coerce")
    followup_seconds = pd.to_numeric(df[followup_hours_col], errors="coerce") * 3600.0

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS_SECONDS:
        eligible = (followup_seconds >= horizon) | (event.astype(bool) & (reply_seconds <= horizon))
        risk_set_n = int(eligible.sum())
        if risk_set_n == 0:
            prob = np.nan
        else:
            numerator = int((event.astype(bool) & (reply_seconds <= horizon) & eligible).sum())
            prob = (numerator / risk_set_n) * 100.0
        rows.append(
            {
                "scenario": scenario,
                "horizon_seconds": int(horizon),
                "risk_set_n": risk_set_n,
                "p_reply_within_horizon_pct": prob,
            }
        )
    return rows


def build_gap_evidence(raw_archive_root: Path, gap: GapInterval) -> pd.DataFrame:
    specs = [
        ("comments", "created_at"),
        ("posts", "created_at"),
        ("snapshots", "timestamp"),
        ("word_frequency", "hour"),
    ]
    rows: list[dict[str, Any]] = []

    for table_name, ts_col in specs:
        table_path = raw_archive_root / table_name / "archive.parquet"
        frame = pd.read_parquet(table_path)
        ts = pd.to_datetime(frame[ts_col], utc=True, errors="coerce").dropna().sort_values().reset_index(drop=True)
        gaps_hours = ts.diff().dt.total_seconds().div(3600.0)
        if len(gaps_hours) > 0:
            idx = int(gaps_hours.fillna(-np.inf).idxmax())
            max_gap = float(gaps_hours.iloc[idx])
            max_gap_start = ts.iloc[idx - 1] if idx > 0 else pd.NaT
            max_gap_end = ts.iloc[idx]
        else:
            max_gap = np.nan
            max_gap_start = pd.NaT
            max_gap_end = pd.NaT

        in_gap = int(((ts > gap.start_utc) & (ts < gap.end_utc)).sum())
        rows.append(
            {
                "table": table_name,
                "timestamp_field": ts_col,
                "n_records": int(len(ts)),
                "n_records_in_comment_gap_interval": in_gap,
                "max_interevent_gap_hours": max_gap,
                "max_gap_start_utc": max_gap_start.isoformat() if pd.notna(max_gap_start) else "",
                "max_gap_end_utc": max_gap_end.isoformat() if pd.notna(max_gap_end) else "",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    survival = load_survival(args.survival_path)
    gap = detect_primary_gap(survival["created_at_utc"], threshold_hours=args.gap_threshold_hours)

    observation_end = pd.concat(
        [survival["created_at_utc"], survival["first_reply_at"].dropna()]
    ).max()

    # Full-window baseline
    windows_rows: list[dict[str, Any]] = []
    full = survival.copy()
    windows_rows.append(
        {
            "scenario": "full_window",
            "window_start_utc": full["created_at_utc"].min().isoformat(),
            "window_end_utc": observation_end.isoformat(),
            **summarize_two_part(full, event_col="event_observed", reply_seconds_col="reply_seconds"),
        }
    )

    # Pre-gap contiguous window with administrative censoring at gap start.
    pre = survival[survival["created_at_utc"] <= gap.start_utc].copy()
    pre["event_window"] = pre["first_reply_at"].notna() & (pre["first_reply_at"] <= gap.start_utc)
    pre["reply_window_seconds"] = np.where(
        pre["event_window"],
        (pre["first_reply_at"] - pre["created_at_utc"]).dt.total_seconds(),
        np.nan,
    )
    pre["followup_window_hours"] = np.where(
        pre["event_window"],
        pre["reply_window_seconds"] / 3600.0,
        (gap.start_utc - pre["created_at_utc"]).dt.total_seconds() / 3600.0,
    )
    windows_rows.append(
        {
            "scenario": "pre_gap_contiguous",
            "window_start_utc": pre["created_at_utc"].min().isoformat(),
            "window_end_utc": gap.start_utc.isoformat(),
            **summarize_two_part(pre, event_col="event_window", reply_seconds_col="reply_window_seconds"),
        }
    )

    # Post-gap contiguous window.
    post = survival[survival["created_at_utc"] >= gap.end_utc].copy()
    post["event_window"] = post["first_reply_at"].notna() & (post["first_reply_at"] <= observation_end)
    post["reply_window_seconds"] = np.where(
        post["event_window"],
        (post["first_reply_at"] - post["created_at_utc"]).dt.total_seconds(),
        np.nan,
    )
    post["followup_window_hours"] = np.where(
        post["event_window"],
        post["reply_window_seconds"] / 3600.0,
        (observation_end - post["created_at_utc"]).dt.total_seconds() / 3600.0,
    )
    windows_rows.append(
        {
            "scenario": "post_gap_contiguous",
            "window_start_utc": gap.end_utc.isoformat(),
            "window_end_utc": observation_end.isoformat(),
            **summarize_two_part(post, event_col="event_window", reply_seconds_col="reply_window_seconds"),
        }
    )

    # Gap-overlap exclusions based on parent creation time before gap start.
    for hours in (6, 24):
        keep = ~(
            (survival["created_at_utc"] > (gap.start_utc - pd.Timedelta(hours=hours)))
            & (survival["created_at_utc"] <= gap.start_utc)
        )
        subset = survival[keep].copy()
        windows_rows.append(
            {
                "scenario": f"exclude_gap_overlap_{hours}h",
                "window_start_utc": subset["created_at_utc"].min().isoformat(),
                "window_end_utc": observation_end.isoformat(),
                **summarize_two_part(
                    subset,
                    event_col="event_observed",
                    reply_seconds_col="reply_seconds",
                ),
            }
        )

    windows_df = pd.DataFrame(windows_rows)
    ensure_parent(args.windows_out)
    windows_df.to_csv(args.windows_out, index=False)

    horizon_rows: list[dict[str, Any]] = []
    horizon_rows.extend(
        summarize_horizon_standardized(
            full,
            scenario="full_window",
            event_col="event_observed",
            reply_seconds_col="reply_seconds",
            followup_hours_col="duration_hours",
        )
    )
    horizon_rows.extend(
        summarize_horizon_standardized(
            pre,
            scenario="pre_gap_contiguous",
            event_col="event_window",
            reply_seconds_col="reply_window_seconds",
            followup_hours_col="followup_window_hours",
        )
    )
    horizon_rows.extend(
        summarize_horizon_standardized(
            post,
            scenario="post_gap_contiguous",
            event_col="event_window",
            reply_seconds_col="reply_window_seconds",
            followup_hours_col="followup_window_hours",
        )
    )

    for hours in (6, 24):
        keep = ~(
            (survival["created_at_utc"] > (gap.start_utc - pd.Timedelta(hours=hours)))
            & (survival["created_at_utc"] <= gap.start_utc)
        )
        subset = survival[keep].copy()
        horizon_rows.extend(
            summarize_horizon_standardized(
                subset,
                scenario=f"exclude_gap_overlap_{hours}h",
                event_col="event_observed",
                reply_seconds_col="reply_seconds",
                followup_hours_col="duration_hours",
            )
        )

    horizon_df = pd.DataFrame(horizon_rows)
    ensure_parent(args.horizon_out)
    horizon_df.to_csv(args.horizon_out, index=False)

    gap_evidence_df = build_gap_evidence(args.raw_archive_root, gap=gap)
    ensure_parent(args.gap_evidence_out)
    gap_evidence_df.to_csv(args.gap_evidence_out, index=False)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "survival_input_path": str(args.survival_path),
        "raw_archive_root": str(args.raw_archive_root),
        "gap_threshold_hours": float(args.gap_threshold_hours),
        "detected_comment_gap": {
            "start_utc": gap.start_utc.isoformat(),
            "end_utc": gap.end_utc.isoformat(),
            "duration_hours": gap.duration_hours,
        },
        "windows_output_csv": str(args.windows_out),
        "horizon_output_csv": str(args.horizon_out),
        "gap_evidence_output_csv": str(args.gap_evidence_out),
    }
    ensure_parent(args.summary_out)
    args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
