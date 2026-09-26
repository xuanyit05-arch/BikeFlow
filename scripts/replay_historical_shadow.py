"""Export notebook 12's frozen 2024 Historical Shadow Simulation predictions.

This is an artifact replay, not a model search or a new test. The selected
notebook cells are executed verbatim, in their original order, against the
unchanged 2024 source. Nothing is published unless all saved-result and
leakage checks pass.

Run from the repository root: python scripts/replay_historical_shadow.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "12_shadow_validation.ipynb"
ROUTE_NOTEBOOK = ROOT / "notebooks" / "11_final_hybrid_strategy.ipynb"
SOURCE = ROOT / "data" / "processed" / "bikeflow_2024.csv"
OUTPUT = ROOT / "results" / "dashboard"

# Pin the exact frozen notebook versions already referenced by dashboard/manifest.json.
EXPECTED_NOTEBOOK_SHA256 = "6b74a309d79a94f4a43e6ab6f23208f9c3027f1427ad54e5082d714b13efe3be"
EXPECTED_ROUTE_SHA256 = "5e9d84d2977f14edd63bd84add9fc44cfd05ecba1ff9064d1835c3d428a92fdb"
EXPECTED_LOCK = "6f2460944230dfd5"
CELLS = (3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 18, 19)
STRATEGIES = ("Final Hybrid", "No-weather Direct", "Weekly Seasonal Naive")
HORIZONS = (1, 3, 6, 12, 24, 48, 72, 168)
METRICS = ("N", "MAE", "RMSE", "Median AE", "P90 AE", "Worst-decile MAE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_namespace() -> dict:
    """Execute only the existing prediction, audit and scoring cells."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sha256(NOTEBOOK) != EXPECTED_NOTEBOOK_SHA256:
        raise ValueError("Notebook 12 differs from the pinned frozen source")
    if sha256(ROUTE_NOTEBOOK) != EXPECTED_ROUTE_SHA256:
        raise ValueError("Notebook 11 differs from the pinned frozen route")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__historical_shadow_replay__", "display": lambda *_: None}
    for number in CELLS:
        cell = notebook["cells"][number]
        if cell["cell_type"] != "code":
            raise ValueError(f"Frozen notebook cell {number} is not code")
        print(f"Replaying frozen notebook 12 cell {number}", flush=True)
        exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell_{number}", "exec"), namespace)
        namespace["display"] = lambda *_: None
    if namespace["lock_fingerprint"] != EXPECTED_LOCK:
        raise ValueError("Frozen lock fingerprint changed")
    return namespace


def verify_log(frame: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Check exact notebook population, pair integrity and input provenance."""
    log = frame.rename(columns={"origin_time": "forecast_origin"}).copy()
    log["horizon_hours"] = log["horizon"]
    if len(log) != 34_152:
        raise ValueError(f"Expected 34,152 prediction rows, found {len(log)}")
    starts = pd.DatetimeIndex(frozen["shadow_starts"])
    expected_origins = starts - pd.Timedelta(hours=1)
    if len(starts) != 1_423 or not pd.DatetimeIndex(log["forecast_origin"].unique()).sort_values().equals(expected_origins):
        raise ValueError("Forecast origins differ from notebook 12's eligible schedule")
    if set(log["strategy"]) != set(STRATEGIES) or set(log["horizon"]) != set(HORIZONS):
        raise ValueError("Unexpected strategy or horizon")
    if log.duplicated(["forecast_origin", "horizon", "strategy"]).any():
        raise ValueError("Duplicate origin/horizon/strategy prediction")
    expected = pd.MultiIndex.from_product(
        [expected_origins, HORIZONS, STRATEGIES],
        names=["forecast_origin", "horizon", "strategy"],
    )
    found = pd.MultiIndex.from_frame(log[["forecast_origin", "horizon", "strategy"]])
    if not expected.difference(found).empty or not found.difference(expected).empty:
        raise ValueError("Incomplete or extra origin/horizon/strategy pairs")
    if not log["target_time"].eq(
        log["forecast_origin"] + pd.to_timedelta(log["horizon"], unit="h")
    ).all():
        raise ValueError("Target time differs from forecast origin plus horizon")
    if not log["horizon_hours"].eq(log["horizon"]).all():
        raise ValueError("Horizon-hour labels disagree")
    if not log["latest_actual_demand_source"].le(log["forecast_origin"]).all():
        raise ValueError("Post-origin observed demand used as input")
    weather_rows = log["latest_actual_weather_source"].notna()
    if not log.loc[weather_rows, "latest_actual_weather_source"].le(
        log.loc[weather_rows, "forecast_origin"]
    ).all():
        raise ValueError("Post-origin observed weather used as input")
    if not log["model_fit_label_cutoff"].dropna().le(frozen["TRAIN_END"]).all():
        raise ValueError("A model was fit with post-training labels")
    for column in ("actual", "prediction", "absolute_error"):
        if not np.isfinite(log[column].to_numpy()).all():
            raise ValueError(f"Nonfinite {column} in prediction log")
    if not np.allclose(log["absolute_error"], np.abs(log["actual"] - log["prediction"]), rtol=0, atol=1e-10):
        raise ValueError("Absolute errors do not match predictions")
    weekly = log.loc[log["strategy"].eq("Weekly Seasonal Naive"),
                     ["forecast_origin", "horizon", "prediction"]].set_index(["forecast_origin", "horizon"])
    hybrid_weekly = log.loc[log["strategy"].eq("Final Hybrid") & log["horizon"].ge(6),
                         ["forecast_origin", "horizon", "prediction"]].set_index(["forecast_origin", "horizon"])
    if not np.array_equal(hybrid_weekly["prediction"].to_numpy(), weekly.loc[hybrid_weekly.index, "prediction"].to_numpy()):
        raise ValueError("Hybrid weekly-route forecasts differ from the frozen weekly baseline")
    return log


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Recalculate notebook 12's definitions from prediction rows."""
    errors = frame["absolute_error"].to_numpy()
    return {
        "N": float(len(errors)),
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean((frame["actual"].to_numpy() - frame["prediction"].to_numpy()) ** 2))),
        "Median AE": float(np.median(errors)),
        "P90 AE": float(np.quantile(errors, 0.9)),
        "Worst-decile MAE": float(np.sort(errors)[-max(1, math.ceil(len(errors) * 0.10)):].mean()),
    }


def validation_table(log: pd.DataFrame) -> pd.DataFrame:
    original_overall = pd.read_csv(OUTPUT / "overall_metrics.csv").set_index("Strategy")
    original_horizon = pd.read_csv(OUTPUT / "horizon_mae.csv").set_index("Horizon")
    if list(original_overall.index) != list(STRATEGIES) or list(original_horizon.index) != list(HORIZONS):
        raise ValueError("Saved notebook 12 reference tables changed")
    records = []

    def add(scope: str, strategy: str, horizon: int | None, metric: str, original: float, replayed: float) -> None:
        difference = abs(float(original) - replayed)
        # Saved notebook display tables contain 2-decimal rounded values.
        tolerance = 0 if metric == "N" else 0.005001
        records.append({
            "scope": scope, "strategy": strategy, "horizon": horizon, "metric": metric,
            "original_value": float(original), "replayed_value": replayed,
            "absolute_difference": difference,
            "relative_difference": difference / abs(float(original)) if original else (0.0 if difference == 0 else np.nan),
            "status": "PASS" if difference <= tolerance else "FAIL",
        })

    for strategy in STRATEGIES:
        scores = metrics(log.loc[log["strategy"].eq(strategy)])
        for metric in METRICS:
            add("overall", strategy, None, metric, original_overall.loc[strategy, metric], scores[metric])
    for horizon in HORIZONS:
        for strategy in STRATEGIES:
            scores = metrics(log.loc[log["strategy"].eq(strategy) & log["horizon"].eq(horizon)])
            add("horizon", strategy, horizon, "MAE", original_horizon.loc[horizon, strategy], scores["MAE"])
    return pd.DataFrame.from_records(records)


def verify_saved_sample(log: pd.DataFrame) -> None:
    sample = pd.read_csv(OUTPUT / "sample_prediction_log.csv", parse_dates=["origin_time", "target_time"])
    sample = sample.rename(columns={"origin_time": "forecast_origin"})
    joined = sample.merge(
        log, on=["forecast_origin", "target_time", "horizon", "strategy"],
        how="left", validate="one_to_one", suffixes=("_original", "_replayed"),
    )
    if len(joined) != len(sample) or joined["prediction_replayed"].isna().any():
        raise ValueError("Saved notebook 12 sample rows are missing from replay")
    for field in ("prediction", "actual", "absolute_error"):
        if not np.allclose(joined[f"{field}_original"], joined[f"{field}_replayed"], rtol=0, atol=0.00000051):
            raise ValueError(f"Replay disagrees with saved notebook 12 sample {field}")


def replace_output(path: Path, writer) -> None:
    """Write a validated output in the destination directory before replacement."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError("Run the historical replay from the repository root")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = frozen_namespace()
    log = verify_log(frozen["prediction_log"], frozen)
    verify_saved_sample(log)
    validation = validation_table(log)
    if not validation["status"].eq("PASS").all():
        print(validation.loc[validation["status"].eq("FAIL")].to_string(index=False))
        raise ValueError("Historical replay did not reproduce the saved notebook 12 metrics")

    columns = (
        "forecast_origin", "target_time", "horizon", "horizon_hours", "actual",
        "prediction", "strategy", "absolute_error", "first_future_hour",
        "weather_source", "latest_actual_demand_source", "latest_actual_weather_source",
        "model_fit_label_cutoff", "feedback_time", "squared_error",
    )
    log = log.loc[:, columns]
    parquet = OUTPUT / "prediction_log.parquet"
    csv = OUTPUT / "prediction_log.csv"
    validation_path = OUTPUT / "replay_validation.csv"
    replace_output(parquet, lambda path: log.to_parquet(path, index=False))
    replace_output(csv, lambda path: log.to_csv(path, index=False, float_format="%.12f"))
    replace_output(validation_path, lambda path: validation.to_csv(path, index=False, float_format="%.12f"))
    manifest = {
        "scope": "2024 Historical Shadow Simulation",
        "purpose": "Prediction-level artifact export from notebook 12's frozen historical replay; not a new blind test",
        "notebook_12_sha256": sha256(NOTEBOOK),
        "notebook_11_sha256": sha256(ROUTE_NOTEBOOK),
        "source_data_sha256": sha256(SOURCE),
        "frozen_lock_fingerprint": frozen["lock_fingerprint"],
        "executed_notebook_12_code_cells": CELLS,
        "training_label_cutoff": str(frozen["TRAIN_END"]),
        "shadow_target_window": [str(frozen["SHADOW_START"]), str(frozen["SHADOW_END"])],
        "forecast_origin_count": len(frozen["shadow_starts"]),
        "prediction_rows": len(log),
        "strategies": STRATEGIES,
        "horizons_hours": HORIZONS,
        "leakage_checks": "Notebook 12 cell 16 provenance assertions and three post-origin demand/weather perturbation checks passed; replay pair and source checks passed",
        "saved_sample_check": "Nine notebook 12 displayed rows matched to 6-decimal precision",
        "saved_metrics_check": "All overall metrics and per-horizon MAE matched saved 2-decimal outputs",
        "files_bytes": {parquet.name: parquet.stat().st_size, csv.name: csv.stat().st_size,
                        validation_path.name: validation_path.stat().st_size},
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    replace_output(OUTPUT / "replay_manifest.json", lambda path: path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    ))
    print(f"PASS: {len(log):,} predictions across {len(frozen['shadow_starts']):,} historical origins")
    print(f"Files: {parquet.stat().st_size:,} B Parquet; {csv.stat().st_size:,} B CSV")


if __name__ == "__main__":
    main()
