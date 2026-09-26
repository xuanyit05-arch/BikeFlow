"""Export saved notebook 12 display outputs for the Streamlit dashboard.

This script reads the .ipynb JSON. It never executes a notebook cell, fits a
model, or recomputes a forecast. Only values visibly saved in notebook outputs
are exported, so the sample prediction file is intentionally incomplete.

Run from any directory: python scripts/export_dashboard_artifacts.py
"""

from __future__ import annotations

import base64
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "12_shadow_validation.ipynb"
ROUTE_NOTEBOOK_PATH = ROOT / "notebooks" / "11_final_hybrid_strategy.ipynb"
OUTPUT_DIR = ROOT / "results" / "dashboard"
STRATEGIES = ("Final Hybrid", "No-weather Direct", "Weekly Seasonal Naive")
HORIZONS = (1, 3, 6, 12, 24, 48, 72, 168)
METRIC_COLUMNS = (
    "Strategy", "N", "MAE", "RMSE", "Median AE", "P90 AE", "Worst-decile MAE"
)


class TableBodyParser(HTMLParser):
    """Read cells in a pandas display table's tbody, including its index."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_body = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.row: list[str] | None = None
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tbody":
            self.in_body = True
        elif self.in_body and tag == "tr":
            self.row = []
        elif self.in_body and self.row is not None and tag in ("td", "th"):
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_body and self.in_cell and tag in ("td", "th"):
            assert self.row is not None
            self.row.append("".join(self.cell_parts).strip())
            self.in_cell = False
        elif self.in_body and tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "tbody":
            self.in_body = False


def output(notebook: dict, cell_number: int, output_number: int) -> dict:
    try:
        return notebook["cells"][cell_number]["outputs"][output_number]
    except (IndexError, KeyError) as exc:
        raise ValueError(
            f"Saved output {output_number} in cell {cell_number} is missing"
        ) from exc


def display_rows(notebook: dict, cell_number: int, output_number: int) -> list[list[str]]:
    rendered = output(notebook, cell_number, output_number).get("data", {}).get("text/html")
    if not rendered:
        raise ValueError(f"No saved HTML table at cell {cell_number}, output {output_number}")
    parser = TableBodyParser()
    parser.feed("".join(rendered) if isinstance(rendered, list) else rendered)
    if not parser.rows:
        raise ValueError(f"Empty saved HTML table at cell {cell_number}, output {output_number}")
    return parser.rows


def checked_rows(rows: list[list[str]], count: int, width: int, label: str) -> list[list[str]]:
    if len(rows) != count or any(len(row) != width for row in rows):
        raise ValueError(f"Unexpected {label} table shape: {len(rows)} rows, widths {[len(r) for r in rows]}")
    return rows


def write_csv(name: str, columns: tuple[str, ...], rows: list[list[str]]) -> None:
    with (OUTPUT_DIR / name).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def write_png(notebook: dict, cell_number: int, name: str) -> None:
    outputs = notebook["cells"][cell_number].get("outputs", [])
    encoded = next(
        (item["data"]["image/png"] for item in outputs if "image/png" in item.get("data", {})),
        None,
    )
    if encoded is None:
        raise ValueError(f"No saved PNG in cell {cell_number}")
    data = base64.b64decode("".join(encoded) if isinstance(encoded, list) else encoded)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Invalid saved PNG in cell {cell_number}")
    (OUTPUT_DIR / name).write_bytes(data)


def main() -> None:
    notebook_bytes = NOTEBOOK_PATH.read_bytes()
    notebook = json.loads(notebook_bytes)
    route_notebook_bytes = ROUTE_NOTEBOOK_PATH.read_bytes()
    route_notebook = json.loads(route_notebook_bytes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall = [r[1:] for r in checked_rows(display_rows(notebook, 19, 0), 3, 8, "overall metrics")]
    recent = [r[1:] for r in checked_rows(display_rows(notebook, 25, 1), 3, 8, "recent metrics")]
    if [r[0] for r in overall] != list(STRATEGIES) or [r[0] for r in recent] != list(STRATEGIES):
        raise ValueError("Saved metric strategy order changed")
    expected = {
        "Final Hybrid": (1104.69, 1772.44, 595.00, 2903.00, 4339.94),
        "No-weather Direct": (1140.38, 1761.82, 630.06, 2981.11, 4306.20),
        "Weekly Seasonal Naive": (1241.80, 1925.91, 709.00, 3110.00, 4610.30),
    }
    for row in overall:
        if int(row[1]) != 11384 or tuple(float(v) for v in row[2:]) != expected[row[0]]:
            raise ValueError(f"Saved overall metrics do not match validated results: {row[0]}")
    if float(recent[0][2]) != 1604.29 or float(recent[1][2]) != 1160.06:
        raise ValueError("Saved recent matured-window MAE changed")

    horizon = checked_rows(display_rows(notebook, 19, 1), 8, 4, "horizon MAE")
    if tuple(int(row[0]) for row in horizon) != HORIZONS:
        raise ValueError("Saved horizon set changed")
    rolling_tail = checked_rows(display_rows(notebook, 23, 1), 8, 7, "rolling checkpoint tail")
    warnings = checked_rows(display_rows(notebook, 25, 4), 12, 5, "warning days")
    sample = checked_rows(display_rows(notebook, 18, 0), 9, 10, "sample prediction log")
    sample = [[row[i] for i in (1, 2, 3, 4, 5, 6, 7)] for row in sample]
    if len({row[0] for row in sample}) != 1 or set(int(row[2]) for row in sample) != {1, 3, 6}:
        raise ValueError("Saved sample prediction preview changed")
    route = [r[1:] for r in checked_rows(
        display_rows(route_notebook, 34, 2), 8, 4, "frozen route"
    )]
    if tuple(int(row[0]) for row in route) != HORIZONS:
        raise ValueError("Saved route horizons changed")
    if ([row[1] for row in route] != [
        "Forecasted Direct", "Forecasted Recursive", *(["Weekly Seasonal Naive"] * 6)
    ] or [row[2] for row in route] != ["precipitation", "precipitation", *(["none"] * 6)]):
        raise ValueError("Saved route no longer matches notebook 12's frozen mapping")

    write_csv("route.csv", ("Horizon", "Selected strategy", "Weather inputs"), route)
    write_csv("overall_metrics.csv", METRIC_COLUMNS, overall)
    write_csv("horizon_mae.csv", ("Horizon", *STRATEGIES), horizon)
    write_csv("recent_metrics.csv", METRIC_COLUMNS, recent)
    write_csv(
        "rolling_checkpoints_tail.csv",
        ("feedback_time", *STRATEGIES, "Hybrid minus No-weather Direct",
         "Hybrid minus Weekly Naive", "Matured origins in window"),
        rolling_tail,
    )
    write_csv(
        "warning_days.csv",
        ("feedback_time", "Final Hybrid", "No-weather Direct",
         "clear_loss_streak_days", "warning_episode_start"),
        warnings,
    )
    write_csv(
        "sample_prediction_log.csv",
        ("origin_time", "target_time", "horizon", "strategy",
         "prediction", "actual", "absolute_error"),
        sample,
    )
    for cell_number, name in (
        (27, "monitoring_rolling_mae.png"),
        (28, "monitoring_hybrid_minus_no_weather.png"),
        (30, "monitoring_cumulative_ae.png"),
        (31, "monitoring_hybrid_minus_weekly.png"),
    ):
        write_png(notebook, cell_number, name)

    manifest = {
        "source": "notebooks/12_shadow_validation.ipynb",
        "source_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "route_source": "notebooks/11_final_hybrid_strategy.ipynb",
        "route_source_sha256": hashlib.sha256(route_notebook_bytes).hexdigest(),
        "method": "Extract saved HTML tables and PNG outputs from notebook JSON; no code execution or model fitting",
        "scope": "2024 historical shadow simulation; not live production or a 2025 blind test",
        "limitations": [
            "Displayed metric and rolling-table values are rounded to two decimal places.",
            "Rolling checkpoint CSV contains only the last 8 displayed daily checkpoints, not the full 53 checkpoints or 1233 hourly windows.",
            "Sample prediction CSV contains only 9 displayed rows for one origin and horizons 1, 3, and 6 hours; it is not the 34152-row full prediction log.",
            "Monitoring figures are saved notebook images; their underlying plotted series are not serialized in the notebook outputs.",
        ],
        "files": {
            "route.csv": "notebook 11 cell 34 output 2; locked horizon route",
            "overall_metrics.csv": "cell 19 output 0; 3 strategies",
            "horizon_mae.csv": "cell 19 output 1; 8 horizons",
            "recent_metrics.csv": "cell 25 output 1; 3 strategies, latest fully matured 168-origin window",
            "rolling_checkpoints_tail.csv": "cell 23 output 1; last 8 of 53 daily checkpoints",
            "warning_days.csv": "cell 25 output 4; 12 performance warning days",
            "sample_prediction_log.csv": "cell 18 output 0; first 9 of 34152 prediction rows",
            "monitoring_rolling_mae.png": "cell 27 saved image",
            "monitoring_hybrid_minus_no_weather.png": "cell 28 saved image",
            "monitoring_cumulative_ae.png": "cell 30 saved image",
            "monitoring_hybrid_minus_weekly.png": "cell 31 saved Hybrid minus Weekly Naive image",
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(manifest['files'])} saved artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
