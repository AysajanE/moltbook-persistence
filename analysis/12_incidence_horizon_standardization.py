#!/usr/bin/env python3
"""Horizon-incidence standardization for censoring-robust persistence reporting.

Outputs manuscript-facing tables that report:
- descriptive ever-reply incidence (secondary), and
- horizon-standardized incidence at 5 minutes and 1 hour (primary),
with explicit risk-set denominators.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SURVIVAL_PATH = Path(
    "data_features/moltbook_only/run_20260206-145240Z/survival_units.parquet"
)
DEFAULT_RAW_AGENTS_PATH = Path(
    "data_raw/hf_archive/snapshot_20260204-234429Z/agents/archive.parquet"
)
DEFAULT_OUT_TABLE = Path("paper/tables/moltbook_horizon_incidence_by_group.csv")
DEFAULT_OUT_SUMMARY = Path("paper/tables/moltbook_horizon_incidence_summary.json")
HORIZONS = ((300, "5m"), (3600, "1h"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute censoring-robust horizon incidence by reporting group."
    )
    parser.add_argument("--survival-path", type=Path, default=DEFAULT_SURVIVAL_PATH)
    parser.add_argument("--raw-agents-path", type=Path, default=DEFAULT_RAW_AGENTS_PATH)
    parser.add_argument("--out-table", type=Path, default=DEFAULT_OUT_TABLE)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_survival(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = {
        "comment_agent_id",
        "submolt_category",
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


def attach_claim_status(survival: pd.DataFrame, raw_agents_path: Path) -> pd.DataFrame:
    agents = pd.read_parquet(raw_agents_path, columns=["id", "is_claimed", "dump_date"]).copy()
    agents["dump_date"] = pd.to_datetime(agents["dump_date"], errors="coerce")
    agents = agents.sort_values(["id", "dump_date"], kind="stable").drop_duplicates("id", keep="last")
    claims = agents.rename(columns={"id": "comment_agent_id"})[["comment_agent_id", "is_claimed"]]

    out = survival.merge(claims, on="comment_agent_id", how="left", validate="many_to_one")
    out["claimed_group"] = "Unknown"
    out.loc[out["is_claimed"] == 1, "claimed_group"] = "Claimed"
    out.loc[out["is_claimed"] == 0, "claimed_group"] = "Unclaimed"
    return out


def horizon_incidence(df: pd.DataFrame, horizon_seconds: int) -> tuple[int, float]:
    event = df["event_observed"].astype(bool)
    reply_seconds = pd.to_numeric(df["reply_seconds"], errors="coerce")
    followup_seconds = pd.to_numeric(df["duration_hours"], errors="coerce") * 3600.0
    risk_set = (followup_seconds >= horizon_seconds) | (event & (reply_seconds <= horizon_seconds))
    n_risk = int(risk_set.sum())
    if n_risk == 0:
        return n_risk, np.nan
    p = float((event & (reply_seconds <= horizon_seconds) & risk_set).sum()) / float(n_risk)
    return n_risk, p * 100.0


def summarize_group(df: pd.DataFrame, group_family: str, group_label: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group_family": group_family,
        "group_label": group_label,
        "n_parents": int(len(df)),
        "n_reply_any": int(df["event_observed"].sum()),
        "p_obs_any_reply_pct": float(df["event_observed"].mean()) * 100.0 if len(df) else np.nan,
    }
    for horizon_seconds, suffix in HORIZONS:
        risk_set_n, p_h = horizon_incidence(df, horizon_seconds=horizon_seconds)
        row[f"risk_set_n_{suffix}"] = risk_set_n
        row[f"p_reply_within_{suffix}_pct"] = p_h
    return row


def main() -> None:
    args = parse_args()
    survival = load_survival(args.survival_path)
    survival = attach_claim_status(survival, raw_agents_path=args.raw_agents_path)

    rows: list[dict[str, Any]] = []
    rows.append(summarize_group(survival, "overall", "Overall"))

    claimable = survival[survival["claimed_group"].isin(["Claimed", "Unclaimed"])].copy()
    for label in ["Claimed", "Unclaimed"]:
        subset = claimable[claimable["claimed_group"] == label].copy()
        rows.append(summarize_group(subset, "claimed_status", label))

    for label, subset in survival.groupby("submolt_category", sort=True):
        rows.append(summarize_group(subset.copy(), "submolt_category", str(label)))

    out_df = pd.DataFrame(rows)
    order = {"overall": 0, "claimed_status": 1, "submolt_category": 2}
    out_df["__order"] = out_df["group_family"].map(order).fillna(99).astype(int)
    out_df = out_df.sort_values(["__order", "group_label"], kind="stable").drop(columns="__order")

    ensure_parent(args.out_table)
    out_df.to_csv(args.out_table, index=False)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "survival_input_path": str(args.survival_path),
        "raw_agents_input_path": str(args.raw_agents_path),
        "horizons_seconds": [h for h, _ in HORIZONS],
        "output_table_csv": str(args.out_table),
    }
    ensure_parent(args.out_summary)
    args.out_summary.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
