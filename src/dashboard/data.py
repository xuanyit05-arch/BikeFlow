"""Load the 2024 demand and frozen notebook results without fitting models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.eda import DAY_NAMES, FALL_REPEAT, MONTH_NAMES, SPRING_GAP


ROOT = Path(__file__).resolve().parents[2]
DEMAND_PATH = ROOT / "data" / "interim" / "hourly_demand_2024.csv"
ARTIFACT_DIR = ROOT / "results" / "dashboard"
HORIZONS = (1, 3, 6, 12, 24, 48, 72, 168)
STRATEGIES = ("Final Hybrid", "No-weather Direct", "Weekly Seasonal Naive")
COLORS = {
    "Final Hybrid": "#176B59",
    "No-weather Direct": "#457699",
    "Weekly Seasonal Naive": "#818895",
}
SHADOW_START = pd.Timestamp("2024-10-19 18:00")
SHADOW_END = pd.Timestamp("2024-12-31 23:00")


def load_demand() -> pd.DataFrame:
    """Use notebook 01's demand calendar and DST treatment, without raw weather."""
    frame = pd.read_csv(DEMAND_PATH, parse_dates=["timestamp"])
    if len(frame) != 8784 or frame["timestamp"].duplicated().any():
        raise ValueError("Expected the complete 2024 hourly demand source")
    if (frame["timestamp"].min() != pd.Timestamp("2024-01-01 00:00")
            or frame["timestamp"].max() != pd.Timestamp("2024-12-31 23:00")
            or frame["ride_count"].isna().any()):
        raise ValueError("2024 demand range or ride counts are incomplete")
    spring = frame.loc[frame["timestamp"].eq(SPRING_GAP), "ride_count"]
    if spring.tolist() != [0]:
        raise ValueError("The spring DST placeholder is not the expected zero row")
    frame = frame.loc[frame["timestamp"].ne(SPRING_GAP)].copy()
    frame["date"] = frame["timestamp"].dt.date
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["day_name"] = frame["day_of_week"].map(dict(enumerate(DAY_NAMES)))
    frame["day_type"] = frame["day_of_week"].ge(5).map({False: "Weekday", True: "Weekend"})
    frame["month"] = frame["timestamp"].dt.month
    frame["month_name"] = frame["month"].map(dict(enumerate(MONTH_NAMES, 1)))
    frame["demand_period"] = (
        (frame["day_of_week"] < 5) & frame["hour"].isin((7, 8, 9, 16, 17, 18, 19))
    ).map({False: "Off-peak", True: "Peak"})
    frame["dst_fallback_anomaly"] = frame["timestamp"].eq(FALL_REPEAT)
    return frame.reset_index(drop=True)


def load_result(filename: str, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    path = ARTIFACT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing dashboard artifact: {path.name}")
    return pd.read_csv(path, parse_dates=list(date_columns))


def load_prediction_log() -> pd.DataFrame:
    """Read the validated 2024 historical replay; prefer compact Parquet."""
    parquet_path = ARTIFACT_DIR / "prediction_log.parquet"
    csv_path = ARTIFACT_DIR / "prediction_log.csv"
    if parquet_path.is_file():
        frame = pd.read_parquet(parquet_path)
    elif csv_path.is_file():
        frame = pd.read_csv(csv_path, parse_dates=["forecast_origin", "target_time"])
    else:
        raise FileNotFoundError("The complete historical prediction log is missing")
    required = {
        "forecast_origin", "target_time", "horizon", "horizon_hours", "actual",
        "prediction", "strategy", "absolute_error",
    }
    if not required.issubset(frame):
        raise ValueError(f"Prediction log missing columns: {sorted(required - set(frame))}")
    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    frame["horizon"] = frame["horizon"].astype(int)
    if frame.duplicated(["forecast_origin", "horizon", "strategy"]).any():
        raise ValueError("Prediction log has duplicate origin/horizon/strategy rows")
    if not frame["horizon"].isin(HORIZONS).all() or not frame["horizon_hours"].eq(frame["horizon"]).all():
        raise ValueError("Prediction log includes an unsupported horizon")
    expected_target = frame["forecast_origin"] + pd.to_timedelta(frame["horizon"], unit="h")
    if not frame["target_time"].eq(expected_target).all():
        raise ValueError("Prediction log target times do not match origin + horizon")
    counts = frame.groupby(["forecast_origin", "strategy"])["horizon"].nunique()
    paired = frame.groupby(["forecast_origin", "horizon"])["strategy"].nunique()
    if (len(frame) != 34_152 or frame["forecast_origin"].nunique() != 1_423
            or not counts.eq(len(HORIZONS)).all() or not paired.eq(len(STRATEGIES)).all()
            or set(frame["strategy"]) != set(STRATEGIES)):
        raise ValueError("The historical prediction log has incomplete strategy bundles")
    if not np.isfinite(frame[["actual", "prediction", "absolute_error"]].to_numpy()).all():
        raise ValueError("The historical prediction log contains missing or nonfinite predictions")
    if not np.allclose(frame["absolute_error"], (frame["actual"] - frame["prediction"]).abs(), rtol=0, atol=1e-9):
        raise ValueError("Historical absolute errors do not match predictions")
    return frame


def eligible_shadow_origins(demand: pd.DataFrame) -> pd.DatetimeIndex:
    """Recreate notebook 12's eligible hourly schedule, without model fitting."""
    valid_targets = set(demand.loc[demand["timestamp"].ne(FALL_REPEAT), "timestamp"])
    first_future = pd.date_range(
        SHADOW_START, SHADOW_END - pd.Timedelta(hours=167), freq="h"
    )
    starts = pd.DatetimeIndex([
        start for start in first_future
        if all(start + pd.Timedelta(hours=step) in valid_targets for step in range(168))
    ])
    if len(starts) != 1423:
        raise ValueError("The frozen shadow origin schedule did not reproduce 1,423 origins")
    return starts - pd.Timedelta(hours=1)


def weekly_naive_bundle(origin: pd.Timestamp, demand: pd.DataFrame) -> pd.DataFrame:
    """Exact frozen baseline: observed demand at target_time minus 168 hours."""
    target_times = pd.DatetimeIndex([
        origin + pd.Timedelta(hours=horizon) for horizon in HORIZONS
    ])
    weekly_sources = target_times - pd.Timedelta(hours=168)
    if not (weekly_sources <= origin).all():
        raise ValueError("Weekly baseline attempted to use post-origin demand")
    demand_lookup = demand.set_index("timestamp")["ride_count"]
    predictions = demand_lookup.reindex(weekly_sources).to_numpy()
    actual = demand_lookup.reindex(target_times).to_numpy()
    if pd.isna(predictions).any() or pd.isna(actual).any():
        raise ValueError("The selected origin lacks a frozen weekly baseline source")
    return pd.DataFrame({
        "horizon": HORIZONS,
        "target_time": target_times,
        "prediction": predictions,
        "actual": actual,
    })
