"""从 Citi Bike 官方骑行文件生成完整的 2024 年小时需求数据。

原始月度 ZIP 保留在 data/raw/citibike/。CSV 采用分块读取（chunk processing），
避免将全年四千多万条骑行记录同时加载到内存。
"""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterator
from zipfile import ZipFile

import pandas as pd

if __package__:
    from .clean_citibike import clean_citibike_chunk
else:
    from clean_citibike import clean_citibike_chunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "citibike"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "hourly_demand_2024.csv"
TARGET_YEAR = 2024
MONTHS = tuple(f"2024{month:02d}" for month in range(1, 13))
CHUNK_SIZE = 250_000
START = pd.Timestamp("2024-01-01 00:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
REQUIRED_COLUMNS = ("ride_id", "started_at")


def _is_data_member(name: str) -> bool:
    """排除 macOS 元数据；ZIP 目录由调用方单独跳过。"""
    parts = Path(name.replace("\\", "/")).parts
    return not any(part == "__MACOSX" or part.startswith("._") for part in parts)


def iter_zip_csvs(
    archive: ZipFile, label: str, depth: int = 0
) -> Iterator[tuple[str, BinaryIO]]:
    """依次提供 ZIP 内的 CSV 流，包括嵌套 ZIP 中的 CSV。

    ``ZipFile`` 需要可定位（seekable）的输入，所以嵌套 ZIP 会暂存到临时磁盘文件；
    这样既不必把整个压缩包放入内存，也不会留下另一份原始数据。
    """
    for member in archive.infolist():
        if member.is_dir() or not _is_data_member(member.filename):
            continue
        member_label = f"{label}!{member.filename}"
        suffix = Path(member.filename).suffix.lower()
        if suffix == ".csv":
            with archive.open(member) as csv_stream:
                yield member_label, csv_stream
        elif suffix == ".zip":
            if depth >= 3:
                raise ValueError(f"ZIP nesting is unexpectedly deep: {member_label}")
            with tempfile.TemporaryFile(mode="w+b") as nested_file:
                with archive.open(member) as nested_stream:
                    shutil.copyfileobj(nested_stream, nested_file, length=1024 * 1024)
                nested_file.seek(0)
                with ZipFile(nested_file) as nested_archive:
                    yield from iter_zip_csvs(nested_archive, member_label, depth + 1)


def find_monthly_sources() -> dict[str, tuple[list[Path], str]]:
    """优先使用每月官方 ZIP；仅在 ZIP 缺失时使用已解压的 CSV。"""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Citi Bike raw directory does not exist: {RAW_DIR}")

    paths = [path for path in RAW_DIR.rglob("*") if path.is_file()]
    sources: dict[str, tuple[list[Path], str]] = {}
    missing: list[str] = []
    for month in MONTHS:
        official_zip_name = f"{month}-citibike-tripdata.zip"
        archives = sorted(
            path for path in paths if path.name.lower() == official_zip_name.lower()
        )
        if len(archives) > 1:
            raise ValueError(f"Multiple copies of the same official ZIP: {archives}")
        if archives:
            sources[month] = (archives, "zip")
            continue

        # 同时存在 ZIP 和解压后的 CSV 时只读 ZIP，避免同一批骑行被重复计数。
        csvs = sorted(
            path
            for path in paths
            if path.suffix.lower() == ".csv"
            and path.name.lower().startswith(f"{month}-citibike-tripdata")
            and _is_data_member(str(path.relative_to(RAW_DIR)))
        )
        if csvs:
            sources[month] = (csvs, "csv")
        else:
            missing.append(month)

    if missing:
        raise FileNotFoundError(
            "Missing official Citi Bike trip data for months: " + ", ".join(missing)
        )
    return sources


def aggregate_csv(
    csv_stream: BinaryIO,
    label: str,
    hourly_counts: Counter[pd.Timestamp],
) -> tuple[int, int, int, int]:
    """分块统计每小时骑行量，并返回（原始、无效时间、年份外、有效）记录数。"""
    raw_rides = invalid_dates = outside_year = valid_rides = 0
    try:
        # 原始骑行量很大；每次只读取所需字段和一个数据块，控制内存占用。
        chunks = pd.read_csv(
            csv_stream,
            usecols=list(REQUIRED_COLUMNS),
            dtype={column: "string" for column in REQUIRED_COLUMNS},
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
        )
        for chunk in chunks:
            raw_rides += len(chunk)
            cleaned, invalid, outside = clean_citibike_chunk(
                chunk, year=TARGET_YEAR, return_counts=True
            )
            invalid_dates += invalid
            outside_year += outside
            valid_rides += len(cleaned)
            # 一行原始数据代表一次骑行；以 started_at 所在小时汇总为 ride_count。
            hourly_counts.update(
                cleaned["started_at"].dt.floor("h").value_counts(sort=False).to_dict()
            )
    except Exception as exc:
        raise RuntimeError(f"Could not process {label}: {exc}") from exc
    return raw_rides, invalid_dates, outside_year, valid_rides


def build_hourly_demand() -> None:
    """汇总全年骑行，并补齐 2024 年本地日历上的每个小时时间标签。"""
    monthly_sources = find_monthly_sources()
    year_counts: Counter[pd.Timestamp] = Counter()
    csv_file_count = 0
    raw_rides = invalid_dates = outside_year = valid_rides = 0

    for month in MONTHS:
        files, kind = monthly_sources[month]
        month_counts: Counter[pd.Timestamp] = Counter()
        month_csv_count = 0
        for path in files:
            if kind == "zip":
                with ZipFile(path) as archive:
                    for label, csv_stream in iter_zip_csvs(archive, str(path)):
                        rides, invalid, outside, valid = aggregate_csv(
                            csv_stream, label, month_counts
                        )
                        csv_file_count += 1
                        month_csv_count += 1
                        raw_rides += rides
                        invalid_dates += invalid
                        outside_year += outside
                        valid_rides += valid
            else:
                with path.open("rb") as csv_stream:
                    rides, invalid, outside, valid = aggregate_csv(
                        csv_stream, str(path), month_counts
                    )
                    csv_file_count += 1
                    month_csv_count += 1
                    raw_rides += rides
                    invalid_dates += invalid
                    outside_year += outside
                    valid_rides += valid
        if month_csv_count == 0:
            raise ValueError(f"No CSV data found for {month} in {files}")
        year_counts.update(month_counts)
        print(f"{month}: processed {month_csv_count} CSV file(s)", flush=True)

    if raw_rides != invalid_dates + outside_year + valid_rides:
        raise AssertionError("Raw ride accounting does not balance")

    # started_at 是不带时区偏移的纽约本地钟表时间（wall-clock time）。
    # 2024 年有 366 天，因此本地日历列出 366 × 24 = 8,784 个小时标签；
    # 春季不存在的 02:00 仍保留为零骑行标签，秋季重复的 01:00 已合并计数。
    full_hours = pd.date_range(START, END, freq="h", inclusive="left")
    if len(full_hours) != 8784:
        raise AssertionError(f"Expected 8,784 hours; got {len(full_hours)}")
    result = pd.DataFrame(
        {
            "timestamp": full_hours,
            "ride_count": [year_counts.get(timestamp, 0) for timestamp in full_hours],
        }
    )
    duplicate_timestamps = int(result["timestamp"].duplicated().sum())
    aggregated_rides = int(result["ride_count"].sum())
    missing = result.isna().sum().to_dict()
    if duplicate_timestamps:
        raise AssertionError(f"Duplicate timestamps: {duplicate_timestamps}")
    if aggregated_rides != valid_rides:
        raise AssertionError(
            f"Ride count mismatch: aggregated={aggregated_rides}, valid={valid_rides}"
        )
    if any(missing.values()):
        raise AssertionError(f"Missing output values: {missing}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n2024 Citi Bike hourly demand data quality report")
    print(f"Raw CSV files: {csv_file_count:,}")
    print(f"Raw rides read: {raw_rides:,}")
    print(f"Invalid started_at dropped: {invalid_dates:,}")
    print(f"Outside-2024 rides dropped: {outside_year:,}")
    print(f"Valid 2024 rides: {valid_rides:,}")
    print(f"Aggregated hours: {len(result):,}")
    print(f"Minimum timestamp: {result['timestamp'].min()}")
    print(f"Maximum timestamp: {result['timestamp'].max()}")
    print(f"Total ride_count: {aggregated_rides:,}")
    print(f"Duplicate timestamps: {duplicate_timestamps:,}")
    print(f"Zero-ride hours: {int(result['ride_count'].eq(0).sum()):,}")
    print(f"Missing values: {missing}")
    print("Ride count consistency: PASS")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_hourly_demand()
