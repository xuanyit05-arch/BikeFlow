"""在按小时聚合前，清洗单个 Citi Bike 骑行数据块（chunk）。"""

import pandas as pd


def clean_citibike_chunk(
    df: pd.DataFrame, year: int = 2024, *, return_counts: bool = False
):
    """保留 ``started_at`` 可解析且属于 ``year`` 的骑行记录。

    原始 DataFrame 不会被修改。设置 ``return_counts`` 后，同时返回时间无法解析、
    不属于目标年份的记录数，供后续数据质量报告核对原始记录总量。
    """
    if "started_at" not in df.columns:
        raise ValueError("Citi Bike data is missing the started_at column")

    # 先复制数据，避免日期转换和筛选意外修改调用方持有的原始数据块。
    cleaned = df.copy()
    # 无法解析的时间设为 NaT，随后与年份范围一起筛除。
    cleaned["started_at"] = pd.to_datetime(
        cleaned["started_at"], format="mixed", errors="coerce"
    )
    parseable = cleaned["started_at"].notna()
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    in_year = parseable & cleaned["started_at"].ge(start) & cleaned["started_at"].lt(end)

    # 分别计数，便于核对「无效时间 + 年份外记录 + 有效记录 = 原始记录」。
    invalid_dates = int((~parseable).sum())
    outside_year = int((parseable & ~in_year).sum())
    cleaned = cleaned.loc[in_year].copy()
    if return_counts:
        return cleaned, invalid_dates, outside_year
    return cleaned
