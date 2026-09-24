# BikeFlow

BikeFlow currently contains 2024 Citi Bike hourly ride demand and historical New York weather. The processed source is `data/processed/bikeflow_2024.csv` (44,302,799 valid rides).

## Exploratory analysis

Open `notebooks/01_eda.ipynb` and run it from the project root or the `notebooks` directory with a Python environment containing pandas, NumPy, Matplotlib, Jupyter, and IPython. The notebook saves its charts in `results/eda/`.

The source CSV is read only. `src/features/eda.py` excludes the nonexistent spring DST hour in the analysis frame, flags the combined fall 01:00 bucket, and aligns Open-Meteo's preceding-hour precipitation, rain, and snowfall to each ride interval using the retained raw weather response. Original weather columns remain available for audit. The analysis does not train a model.
