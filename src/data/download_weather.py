"""下载 Open-Meteo 历史天气，并按纽约本地时间选出 2024 年数据。

Open-Meteo 返回的 ISO 时间标签在夏令时（DST）切换前后可能沿用固定偏移量。
UNIX 时间戳对应真实发生的小时，因此先从 UTC 转到 America/New_York，
再按纽约本地日历筛选年份。
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "weather" / "open_meteo_nyc_2024.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "hourly_weather_2024.csv"
TIMEZONE = "America/New_York"
LATITUDE = 40.7580
LONGITUDE = -73.9855
VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
)

# 请求额外的边界日，是因为 Open-Meteo 的 UNIX 时间响应边界会使用固定 UTC 偏移量，
# 即使请求范围跨越夏令时也是如此；转成纽约本地时间后再剔除年份外的小时。
PARAMETERS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
    "hourly": ",".join(VARIABLES),
    "timezone": TIMEZONE,
    "timeformat": "unixtime",
    "temperature_unit": "celsius",
    "precipitation_unit": "mm",
    "wind_speed_unit": "kmh",
}
SOURCE_URL = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(PARAMETERS)


def load_raw_weather() -> dict:
    """原样保存并复用官方 JSON 响应，便于复现且避免重复下载。"""
    if RAW_PATH.exists():
        raw_bytes = RAW_PATH.read_bytes()
        print(f"Using existing raw response: {RAW_PATH}")
    else:
        request = Request(SOURCE_URL, headers={"User-Agent": "BikeFlow/1.0"})
        with urlopen(request, timeout=120) as response:
            raw_bytes = response.read()
        payload = json.loads(raw_bytes)
        if payload.get("error"):
            raise RuntimeError(f"Open-Meteo API error: {payload}")
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = RAW_PATH.with_suffix(".json.part")
        temp_path.write_bytes(raw_bytes)
        temp_path.replace(RAW_PATH)
        print(f"Saved official raw response: {RAW_PATH}")
    payload = json.loads(raw_bytes)
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo API error: {payload}")
    if payload.get("timezone") != TIMEZONE:
        raise ValueError(f"Unexpected API timezone: {payload.get('timezone')}")
    return payload


def build_hourly_weather(payload: dict) -> pd.DataFrame:
    """用 UTC 时间戳定位真实小时，再生成纽约本地 2024 年天气表。"""
    hourly = payload["hourly"]
    missing_keys = [name for name in ("time", *VARIABLES) if name not in hourly]
    if missing_keys:
        raise ValueError(f"Missing API hourly arrays: {missing_keys}")
    raw_hours = len(hourly["time"])
    if any(len(hourly[name]) != raw_hours for name in VARIABLES):
        raise ValueError("Open-Meteo hourly arrays have different lengths")

    # 先用唯一的 UTC 物理时刻排序和校验，避免秋季重复的本地 01:00 产生歧义。
    utc = pd.to_datetime(hourly["time"], unit="s", utc=True)
    if utc.has_duplicates or not utc.is_monotonic_increasing:
        raise ValueError("API UNIX timestamps are duplicated or unsorted")
    local = utc.tz_convert(TIMEZONE)
    local_year = local.year == 2024

    frame = pd.DataFrame({name: pd.to_numeric(hourly[name], errors="coerce") for name in VARIABLES})
    # 同时保留 UTC 和本地标签：前者区分真实小时，后者用于和骑行需求对齐。
    frame.insert(0, "timestamp_utc", utc)
    frame.insert(0, "timestamp", local.tz_localize(None))
    frame = frame.loc[local_year].copy().reset_index(drop=True)
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if len(frame) != 8784:
        raise ValueError(f"Expected 8,784 physical NYC hours in 2024; got {len(frame)}")
    unique_utc = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if unique_utc.duplicated().any() or not unique_utc.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("The 2024 physical UTC hours are not unique and consecutive")
    # DST 开始当天只有 23 个真实小时；结束当天有 25 个，且 01:00 出现两次。
    spring = frame.loc[frame["timestamp"].dt.strftime("%Y-%m-%d").eq("2024-03-10")]
    fall = frame.loc[frame["timestamp"].dt.strftime("%Y-%m-%d").eq("2024-11-03")]
    if len(spring) != 23 or len(fall) != 25:
        raise ValueError(f"Unexpected DST day lengths: spring={len(spring)}, fall={len(fall)}")
    if frame["timestamp"].duplicated().sum() != 1:
        raise ValueError("Expected exactly one repeated NYC clock-hour in 2024")
    return frame


def main() -> None:
    """生成小时天气 CSV，并输出覆盖范围与 DST 检查结果。"""
    payload = load_raw_weather()
    frame = build_hourly_weather(payload)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)

    print(f"Source: {SOURCE_URL}")
    print(f"Requested coordinates: {LATITUDE}, {LONGITUDE}")
    print(f"API raw hourly shape: ({len(payload['hourly']['time'])}, {len(VARIABLES) + 1})")
    print(f"API hourly units: {payload.get('hourly_units')}")
    print(f"Local 2024 weather shape: {frame.shape}")
    print(f"Local timestamp range: {frame['timestamp'].min()} to {frame['timestamp'].max()}")
    print(f"Physical hours: {len(frame)}")
    print(f"Repeated local timestamp count: {int(frame['timestamp'].duplicated().sum())}")
    print(f"Unique UTC timestamp count: {int(frame['timestamp_utc'].nunique())}")
    print(f"Missing values: {frame.isna().sum().to_dict()}")
    for date in ("2024-03-10", "2024-11-03"):
        day = frame.loc[frame["timestamp"].dt.strftime("%Y-%m-%d").eq(date)]
        print(f"{date}: {len(day)} physical hour(s)")
        print(day.loc[day["timestamp"].dt.hour.isin((0, 1, 2, 3, 4)), ["timestamp", "timestamp_utc"]].to_string(index=False))
    print(f"Saved local 2024 hourly weather: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
