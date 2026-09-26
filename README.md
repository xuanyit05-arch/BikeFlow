# BikeFlow

BikeFlow currently contains 2024 Citi Bike hourly ride demand and historical New York weather. The processed source is `data/processed/bikeflow_2024.csv` (44,302,799 valid rides).

## Exploratory analysis

Open `notebooks/01_eda.ipynb` and run it from the project root or the `notebooks` directory with a Python environment containing pandas, NumPy, Matplotlib, Jupyter, and IPython. The notebook saves its charts in `results/eda/`.

The source CSV is read only. `src/features/eda.py` excludes the nonexistent spring DST hour in the analysis frame, flags the combined fall 01:00 bucket, and aligns Open-Meteo's preceding-hour precipitation, rain, and snowfall to each ride interval using the retained raw weather response. Original weather columns remain available for audit. The analysis does not train a model.

## Forecasting dashboard

Install the dashboard dependencies, then launch from the repository root:

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

The app has Overview, Demand Explorer, Forecast, Model Comparison, and Monitoring pages. It reads the tracked 2024 hourly demand CSV and files in `results/dashboard/`; it does not run notebook code or fit models when the app starts. Paths are resolved from the app source, so the launch directory does not change its data sources.

For Streamlit Community Cloud, use `app.py` as the entrypoint and choose Python 3.11 in Advanced settings to match the validated local runtime. The cloud app installs `requirements.txt`; `requirements-replay.txt` is only for regenerating the offline artifacts.

The frozen route was extracted from `notebooks/11_final_hybrid_strategy.ipynb`; the fixed result CSVs and monitoring figures were extracted from the saved outputs in `notebooks/12_shadow_validation.ipynb`. To repeat that read-only export after the notebook outputs change, run `python scripts/export_dashboard_artifacts.py`. `results/dashboard/manifest.json` records both notebook hashes and the exact output cells used.

**Forecast data availability:** `results/dashboard/prediction_log.parquet` contains all 34,152 prediction rows for 1,423 historical forecast origins, three frozen strategies, and eight horizons. A CSV copy is also available as a deployment fallback. The Forecast page reads the complete log without fitting a model and supports any valid origin, strategy, and horizon. `replay_validation.csv` compares the replayed overall metrics and per-horizon MAE with notebook 12's saved, two-decimal results; all 42 checks pass. `replay_manifest.json` records the source hashes, frozen lock, execution cells, leakage checks, and file sizes.

To regenerate the prediction-level artifacts from the unchanged notebooks, install `requirements-replay.txt` and run `python scripts/replay_historical_shadow.py` from the repository root. The runner executes only notebook 12's frozen training, prediction, leakage-audit, and scoring cells. It verifies the saved sample and metrics before writing any prediction log. This offline export does not run in the Streamlit app.

This dashboard reports a **2024 Historical Shadow Simulation**. It is not a live production system or a new 2025 blind test. The frozen Final Hybrid is a research/shadow strategy; No-weather Direct is the low-complexity deployment candidate; Weekly Seasonal Naive is the baseline. Forecasted weather is not a production dependency. Dashboard presentation does not change notebook 11's route or notebook 12's monitoring rule.
