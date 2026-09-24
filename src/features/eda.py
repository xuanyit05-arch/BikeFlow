"""为 BikeFlow 的 2024 年探索性数据分析（EDA）准备可核对的小时数据。

处理后的 CSV 保留 Open-Meteo 原始时间标签下的天气观测。``precipitation``、
``rain`` 和 ``snowfall`` 记录的是时间标签 *之前一小时* 的累计量，而
``ride_count[t]`` 对应 [t, t + 1 小时) 内开始的骑行。因此需要在 UTC
物理时间上先对齐降水，再合并纽约秋季重复的本地小时。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TIMEZONE = "America/New_York"
YEAR_START = pd.Timestamp("2024-01-01 00:00:00")
YEAR_END = pd.Timestamp("2025-01-01 00:00:00")
SPRING_GAP = pd.Timestamp("2024-03-10 02:00:00")
FALL_REPEAT = pd.Timestamp("2024-11-03 01:00:00")
ACCUMULATION_COLUMNS = ("precipitation", "rain", "snowfall")
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
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def aligned_hourly_accumulations(raw_weather_path: Path) -> pd.DataFrame:
    """把原始天气的「前一小时累计量」映射到对应的纽约骑行小时起点。

    例如 ``ride_count@08:00`` 统计 08:00–09:00 开始的骑行，故对应
    Open-Meteo 在 09:00 标记的累计降水。先在 UTC 中减去一小时，可正确处理
    两次夏令时（DST）切换。原始响应包含 2025-01-01 00:00 本地时间这个
    终点，因此也能覆盖 2024-12-31 23:00 的骑行小时。
    """
    with Path(raw_weather_path).open(encoding="utf-8") as source:
        payload = json.load(source)
    hourly = payload["hourly"]
    required = ("time", *ACCUMULATION_COLUMNS)
    missing = set(required).difference(hourly)
    if missing:
        raise ValueError(f"Raw Open-Meteo response is missing: {sorted(missing)}")

    utc_end = pd.to_datetime(hourly["time"], unit="s", utc=True)
    if utc_end.has_duplicates or not utc_end.is_monotonic_increasing:
        raise ValueError("Raw weather UTC times must be unique and ordered")
    if any(len(hourly[name]) != len(utc_end) for name in ACCUMULATION_COLUMNS):
        raise ValueError("Raw weather arrays have different lengths")

    physical = pd.DataFrame(
        {name: pd.to_numeric(hourly[name], errors="coerce") for name in ACCUMULATION_COLUMNS}
    )
    # Open-Meteo 的时间标签是累计区间的终点；减去一小时才是骑行区间的起点。
    # 先在 UTC 中移动，再转本地时间，避免 DST 重复或缺失小时造成错位。
    physical.insert(
        0,
        "timestamp",
        (utc_end - pd.Timedelta(hours=1)).tz_convert(TIMEZONE).tz_localize(None),
    )
    physical = physical.loc[
        physical["timestamp"].ge(YEAR_START) & physical["timestamp"].lt(YEAR_END)
    ]
    if len(physical) != 8784:
        raise ValueError(f"Expected 8,784 real ride hours in 2024, got {len(physical)}")

    # 秋季两个真实的 01:00 共享同一本地标签；累计量需相加后对应一条骑行计数。
    grouped = physical.groupby("timestamp", sort=True)
    aligned = grouped.agg(
        **{
            f"{name}_during_hour": (name, lambda values: values.sum(min_count=1))
            for name in ACCUMULATION_COLUMNS
        }
    ).reset_index()
    aligned["physical_weather_hours"] = grouped.size().to_numpy()
    if len(aligned) != 8783:
        raise ValueError(f"Expected 8,783 real local hour labels, got {len(aligned)}")
    if aligned.loc[aligned["timestamp"].eq(FALL_REPEAT), "physical_weather_hours"].tolist() != [2]:
        raise ValueError("Fall-back 01:00 must represent two physical hours")
    if SPRING_GAP in set(aligned["timestamp"]):
        raise ValueError("The nonexistent spring hour appeared in aligned weather")
    return aligned


def prepare_eda_data(processed_path: Path, raw_weather_path: Path) -> pd.DataFrame:
    """返回 2024 年真实本地小时的 EDA 数据及对齐后的累计天气字段。

    只读取源 CSV，原有八个天气字段保持不变。春季不存在的小时仅从分析表中
    排除；秋季 01:00 的合并骑行计数保留，并以标志字段提示其特殊性。
    """
    source = pd.read_csv(processed_path, parse_dates=["timestamp"])
    required = {"timestamp", "ride_count", *WEATHER_COLUMNS}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing: {sorted(missing)}")
    if len(source) != 8784 or source["timestamp"].duplicated().any():
        raise ValueError("Processed dataset must have 8,784 unique local hour labels")
    spring = source.loc[source["timestamp"].eq(SPRING_GAP)]
    if len(spring) != 1 or spring["ride_count"].iloc[0] != 0:
        raise ValueError("Spring DST gap must be one zero-ride row")
    if not spring.loc[:, WEATHER_COLUMNS].isna().all().all():
        raise ValueError("Spring DST gap weather must remain missing")

    # 2024-03-10 02:00 是日历占位行，并非真实发生的小时；EDA 不应将其计入均值。
    analysis = source.loc[source["timestamp"].ne(SPRING_GAP)].copy()
    aligned = aligned_hourly_accumulations(raw_weather_path)
    analysis = analysis.merge(aligned, on="timestamp", how="left", validate="one_to_one")
    if analysis["physical_weather_hours"].isna().any():
        raise ValueError("Some real ride hours have no aligned raw weather")
    if analysis.loc[:, [f"{name}_during_hour" for name in ACCUMULATION_COLUMNS]].isna().any().any():
        raise ValueError("Some aligned accumulation values are missing")
    if analysis.loc[:, WEATHER_COLUMNS].isna().any().any():
        raise ValueError("Unexpected missing original weather on real ride hours")
    if analysis["ride_count"].sum() != source["ride_count"].sum():
        raise ValueError("Ride total changed after excluding the zero-ride spring gap")

    # 时间特征用于比较通勤时段、星期与季节；周末标志便于单独观察休闲需求。
    analysis["dst_fallback_anomaly"] = analysis["timestamp"].eq(FALL_REPEAT)
    analysis["hour"] = analysis["timestamp"].dt.hour
    analysis["day_of_week"] = analysis["timestamp"].dt.dayofweek
    analysis["day_name"] = analysis["day_of_week"].map(dict(enumerate(DAY_NAMES)))
    analysis["month"] = analysis["timestamp"].dt.month
    analysis["month_name"] = analysis["month"].map(dict(enumerate(MONTH_NAMES, start=1)))
    analysis["date"] = analysis["timestamp"].dt.date
    analysis["is_weekend"] = analysis["day_of_week"].ge(5)
    return analysis.reset_index(drop=True)
