"""按纽约本地小时标签合并 2024 年 Citi Bike 需求与天气数据。

骑行需求的小时标签是不带时区的钟表时间（wall-clock time）：春季 02:00 不存在，
秋季两个真实的 01:00 已汇总成一个 ``ride_count``。天气先保留 UTC 物理时刻，
再将秋季两条观测聚合（aggregation）到相同的本地小时粒度。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMAND_PATH = PROJECT_ROOT / "data" / "interim" / "hourly_demand_2024.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "interim" / "hourly_weather_2024.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bikeflow_2024.csv"

TIMEZONE = "America/New_York"
YEAR_START = pd.Timestamp("2024-01-01 00:00:00")
YEAR_END = pd.Timestamp("2025-01-01 00:00:00")
SPRING_GAP = pd.Timestamp("2024-03-10 02:00:00")
FALL_REPEAT = pd.Timestamp("2024-11-03 01:00:00")

MEAN_COLUMNS = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
)
SUM_COLUMNS = ("precipitation", "rain", "snowfall")
WEATHER_COLUMNS = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
)
FINAL_COLUMNS = ("timestamp", "ride_count", *WEATHER_COLUMNS)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """确认输入字段齐全，避免后续合并时悄然遗漏数据。"""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _parse_local_timestamps(values: pd.Series, label: str) -> pd.Series:
    """解析不带时区的本地时间标签，并拒绝格式不一致的输入。"""
    if values.isna().any():
        raise ValueError(f"{label} contains empty timestamps")
    try:
        parsed = pd.to_datetime(values, format="mixed", errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains an invalid timestamp: {exc}") from exc
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype) and not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError(f"{label} must contain consistent naive local timestamps")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{label} must contain naive local timestamps")
    return parsed


def _read_demand() -> pd.DataFrame:
    """读取小时需求，核对全年本地时间标签与春季 DST 空缺。"""
    demand = pd.read_csv(DEMAND_PATH)
    _require_columns(demand, ("timestamp", "ride_count"), "Citi Bike demand")
    demand = demand.loc[:, ["timestamp", "ride_count"]].copy()
    demand["timestamp"] = _parse_local_timestamps(demand["timestamp"], "Demand timestamp")
    if demand["timestamp"].duplicated().any():
        raise ValueError("Citi Bike demand contains duplicate timestamps")

    counts = pd.to_numeric(demand["ride_count"], errors="raise")
    if counts.isna().any() or counts.lt(0).any() or counts.mod(1).ne(0).any():
        raise ValueError("ride_count must contain nonnegative whole numbers without nulls")
    demand["ride_count"] = counts.astype("int64")

    # 需求表按本地钟表时间列出 8,784 个标签，包括实际不存在的春季 02:00。
    expected_labels = pd.date_range(YEAR_START, YEAR_END, freq="h", inclusive="left")
    observed_labels = pd.DatetimeIndex(demand["timestamp"])
    if not observed_labels.sort_values().equals(expected_labels):
        raise ValueError("Demand does not cover each 2024 local clock-hour label exactly once")
    spring_count = demand.loc[demand["timestamp"].eq(SPRING_GAP), "ride_count"].iloc[0]
    if spring_count != 0:
        raise ValueError("The nonexistent spring DST hour has a nonzero ride_count")
    return demand


def _read_weather() -> pd.DataFrame:
    """读取天气，并用 UTC 时间检查真实小时是否连续且完整。"""
    weather = pd.read_csv(WEATHER_PATH)
    _require_columns(weather, ("timestamp", "timestamp_utc", *WEATHER_COLUMNS), "Hourly weather")
    weather = weather.loc[:, ["timestamp", "timestamp_utc", *WEATHER_COLUMNS]].copy()
    weather["timestamp"] = _parse_local_timestamps(weather["timestamp"], "Weather timestamp")

    utc_text = weather["timestamp_utc"].astype("string").str.strip()
    if not utc_text.str.contains(r"(?:Z|\+00:?00)$", case=False, regex=True, na=False).all():
        raise ValueError("timestamp_utc must contain explicit UTC offsets (Z or +00:00)")
    try:
        weather["timestamp_utc"] = pd.to_datetime(
            utc_text, format="mixed", utc=True, errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Weather contains an invalid UTC timestamp: {exc}") from exc
    if weather["timestamp_utc"].isna().any() or weather["timestamp_utc"].duplicated().any():
        raise ValueError("Weather UTC timestamps must be present and unique")

    # UTC 是无歧义的真实时刻；反向转换可验证本地标签没有标错。
    actual_local = weather["timestamp_utc"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    mismatched = weather["timestamp"].ne(actual_local)
    if mismatched.any():
        sample = weather.loc[mismatched, ["timestamp", "timestamp_utc"]].head(3)
        raise ValueError(
            "Weather local timestamps disagree with UTC converted to America/New_York: "
            + sample.to_string(index=False)
        )
    if weather["timestamp"].lt(YEAR_START).any() or weather["timestamp"].ge(YEAR_END).any():
        raise ValueError("Weather contains local timestamps outside 2024")

    # 在 UTC 上检查真实小时覆盖范围，才能发现漏测；本地 DST 日期并非都恰好 24 小时。
    expected_utc = pd.date_range(
        YEAR_START.tz_localize(TIMEZONE).tz_convert("UTC"),
        YEAR_END.tz_localize(TIMEZONE).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    observed_utc = pd.DatetimeIndex(weather["timestamp_utc"])
    missing_utc = expected_utc.difference(observed_utc)
    extra_utc = observed_utc.difference(expected_utc)
    if len(missing_utc) or len(extra_utc):
        raise ValueError(
            f"Weather UTC coverage is incomplete or outside the local 2024 year: "
            f"missing={list(missing_utc[:3])} ({len(missing_utc)} total), "
            f"extra={list(extra_utc[:3])} ({len(extra_utc)} total)"
        )

    for column in WEATHER_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="raise")
    reported_codes = weather["weather_code"].dropna()
    if reported_codes.mod(1).ne(0).any():
        raise ValueError("weather_code contains a non-integer code")
    return weather


def _sum_with_missing(values: pd.Series) -> float:
    """源小时全部缺失时仍保留缺失值，避免误当作零降水。"""
    return values.sum(min_count=1)


def _collapse_local_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """把秋季重复的两条天气观测合并到同一个本地小时标签。"""
    label_sizes = weather.groupby("timestamp", sort=True).size()
    duplicates = label_sizes[label_sizes.gt(1)]
    if len(duplicates) != 1 or duplicates.index[0] != FALL_REPEAT or duplicates.iloc[0] != 2:
        raise ValueError(
            "Expected exactly two weather observations at 2024-11-03 01:00 "
            f"and no other repeated local hours; found {duplicates.to_dict()}"
        )
    if SPRING_GAP in label_sizes.index:
        raise ValueError("Weather includes the nonexistent 2024-03-10 02:00 local hour")

    # 秋季两个真实的 01:00 对应一个 Citi Bike 钟表小时计数。
    # 温度等状态量取平均，降水等累计量求和；weather_code 取最大值仅为确定性
    # 的分类摘要，并不表示代码数字越大，天气就一定越严重。
    aggregations = {column: "mean" for column in MEAN_COLUMNS}
    aggregations.update({column: _sum_with_missing for column in SUM_COLUMNS})
    aggregations["weather_code"] = "max"
    hourly = weather.groupby("timestamp", sort=True, as_index=False).agg(aggregations)
    hourly = hourly.loc[:, ["timestamp", *WEATHER_COLUMNS]]
    hourly["weather_code"] = hourly["weather_code"].astype("Int64")
    if hourly["timestamp"].duplicated().any():
        raise AssertionError("Weather still has duplicate local timestamps after aggregation")
    return hourly


def build_model_dataset() -> None:
    """合并需求和天气，确保骑行总量及 DST 特殊小时保持一致。"""
    demand = _read_demand()
    weather = _read_weather()
    hourly_weather = _collapse_local_weather(weather)

    demand_labels = pd.DatetimeIndex(demand["timestamp"])
    weather_labels = pd.DatetimeIndex(hourly_weather["timestamp"])
    missing_labels = demand_labels.difference(weather_labels)
    extra_labels = weather_labels.difference(demand_labels)
    if not missing_labels.equals(pd.DatetimeIndex([SPRING_GAP])) or len(extra_labels):
        raise ValueError(
            "Weather and demand local-hour coverage differs beyond the spring DST gap: "
            f"missing={list(missing_labels[:3])} ({len(missing_labels)} total), "
            f"extra={list(extra_labels[:3])} ({len(extra_labels)} total)"
        )

    # 以需求小时为基准；春季不存在的 02:00 保留零骑行与缺失天气。
    merged = demand.merge(hourly_weather, on="timestamp", how="left", validate="one_to_one", indicator=True)
    unmatched = merged.loc[merged["_merge"].eq("left_only"), "timestamp"]
    if not pd.DatetimeIndex(unmatched).equals(pd.DatetimeIndex([SPRING_GAP])):
        raise AssertionError(f"Unexpected unmatched demand hours: {unmatched.tolist()}")
    merged = merged.loc[:, FINAL_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    if merged["timestamp"].duplicated().any():
        raise AssertionError("Final dataset has duplicate timestamps")
    before_total = int(demand["ride_count"].sum())
    after_total = int(merged["ride_count"].sum())
    if before_total != after_total:
        raise AssertionError(f"Ride counts changed during merge: {before_total} -> {after_total}")
    if len(merged) != len(demand):
        raise AssertionError("The merge changed the number of demand rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    fall_source = weather.loc[weather["timestamp"].eq(FALL_REPEAT), "timestamp_utc"]
    print("2024 BikeFlow demand and weather merge quality report")
    print(f"Demand input shape: {demand.shape}")
    print(f"Weather physical-hour input shape: {weather.shape}")
    print(f"Weather local-hour shape after fall aggregation: {hourly_weather.shape}")
    print(f"Final shape: {merged.shape}")
    print(f"Time range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    print(f"Spring gap: {SPRING_GAP}; ride_count=0, weather remains missing")
    print(f"Fall repeated hour: {FALL_REPEAT}; UTC observations={list(fall_source)}")
    print("Fall weather rule: mean measured states, sum precipitation/rain/snowfall, max reported weather code")
    print(f"Duplicate final timestamps: {int(merged['timestamp'].duplicated().sum())}")
    print(f"Missing values by column: {merged.isna().sum().to_dict()}")
    print(f"ride_count before/after: {before_total:,} / {after_total:,} (PASS)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_model_dataset()
