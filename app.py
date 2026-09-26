"""BikeFlow: a read-only 2024 forecasting portfolio dashboard.

Run with ``streamlit run app.py``. All model results come from the frozen
historical shadow study; opening this application never trains a model.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import data as source


st.set_page_config(
    page_title="BikeFlow | Forecasting Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1500px;}
    h1, h2, h3 {color: #183044;}
    [data-testid="stMetric"] {background: #f6f8fa; border: 1px solid #e5ebef;
        border-radius: 10px; padding: 14px 16px;}
    [data-testid="stMetricValue"] {font-size: clamp(1.25rem, 2.1vw, 2rem);}
    [data-testid="stMetricLabel"] p {white-space: normal;}
    [data-testid="stSidebar"] {background: #f7f9fa;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def demand_data() -> pd.DataFrame:
    return source.load_demand()


@st.cache_data(show_spinner=False)
def result_data(filename: str, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    return source.load_result(filename, date_columns)


@st.cache_data(show_spinner=False)
def forecast_data() -> pd.DataFrame:
    return source.load_prediction_log()


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=36, b=12),
        legend_title_text="",
        hovermode="x unified",
        font=dict(color="#304557"),
    )
    return fig


def metric_lookup(frame: pd.DataFrame, strategy: str, metric: str) -> float:
    return float(frame.set_index("Strategy").loc[strategy, metric])


def show_overview() -> None:
    overall = result_data("overall_metrics.csv")
    recent = result_data("recent_metrics.csv")
    st.title("BikeFlow")
    st.subheader("NYC Citi Bike hourly demand forecasting")
    st.caption(
        "2024 hourly ride demand · eight horizons: 1h / 3h / 6h / 12h / 24h / 48h / 72h / 168h"
    )
    st.info(
        "2024 Historical Shadow Simulation. This is not a live production system "
        "or a new 2025 blind test."
    )

    left, middle, right = st.columns(3)
    left.metric("Final Hybrid MAE", f"{metric_lookup(overall, 'Final Hybrid', 'MAE'):,.2f}",
                help="Mean absolute error, rides/hour; historical shadow simulation")
    middle.metric("No-weather Direct MAE", f"{metric_lookup(overall, 'No-weather Direct', 'MAE'):,.2f}",
                  help="Mean absolute error, rides/hour; historical shadow simulation")
    right.metric("Weekly Seasonal Naive MAE",
                 f"{metric_lookup(overall, 'Weekly Seasonal Naive', 'MAE'):,.2f}",
                 help="Mean absolute error, rides/hour; historical shadow simulation")
    paired_points = int(overall.iloc[0]["N"])
    st.caption(
        f"All three MAEs use the same {paired_points // len(source.HORIZONS):,} historical origins "
        f"and {paired_points:,} paired forecasts per strategy."
    )

    status, candidate = st.columns(2)
    status.metric("Hybrid status", "Shadow Monitoring")
    candidate.metric("Deployment candidate", "No-weather Direct")

    st.markdown("### Why the lowest MAE is still under review")
    st.write(
        "The frozen Hybrid has the lowest overall MAE, but its forecasted-weather branches "
        "lose to No-weather Direct at 1h and 3h. No-weather also has lower overall RMSE "
        "and worst-decile MAE. Hybrid's recent matured seven-day MAE rises to "
        f"{metric_lookup(recent, 'Final Hybrid', 'MAE'):,.2f} versus "
        f"{metric_lookup(recent, 'No-weather Direct', 'MAE'):,.2f} for No-weather, "
        "and the research warning rule is active. The available weather history does not "
        "contain real forecast issue times."
    )
    st.caption(
        "Final Hybrid is a research/shadow strategy. No-weather Direct is the "
        "low-complexity deployment candidate. Weekly Seasonal Naive remains the baseline. "
        "Forecasted weather is not a production dependency."
    )
    with st.expander("Frozen routing from notebook 11"):
        route = result_data("route.csv")
        route["Horizon"] = route["Horizon"].map(lambda h: f"{h}h")
        st.dataframe(route, hide_index=True, width="stretch")


def show_demand_explorer() -> None:
    demand = demand_data()
    st.title("Demand Explorer")
    st.caption("2024 Citi Bike ride starts per local hour · source: existing hourly demand CSV")
    selected = st.date_input(
        "Date range",
        value=(date(2024, 1, 1), date(2024, 12, 31)),
        min_value=date(2024, 1, 1),
        max_value=date(2024, 12, 31),
    )
    if not isinstance(selected, tuple) or len(selected) != 2:
        st.info("Select both a start and end date to explore demand.")
        return
    start, end = selected
    if start > end:
        st.warning("The start date must be on or before the end date.")
        return
    view = demand.loc[demand["date"].between(start, end)].copy()
    if view.empty:
        st.info("No demand hours are available in this range.")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric("Ride starts", f"{view['ride_count'].sum():,.0f}")
    k2.metric("Average per local-hour label", f"{view['ride_count'].mean():,.0f}")
    peak_share = view.loc[view["demand_period"].eq("Peak"), "ride_count"].sum() / view["ride_count"].sum()
    k3.metric("Peak-hour share of rides", f"{peak_share:.1%}")

    fig = px.line(view, x="timestamp", y="ride_count", labels={
        "timestamp": "Local hour", "ride_count": "Ride starts per local-hour label"
    }, title="Hourly demand over the selected period")
    fig.update_traces(line_color=source.COLORS["No-weather Direct"], line_width=1)
    st.plotly_chart(chart_layout(fig, 390), width="stretch")

    by_hour = view.groupby("hour", as_index=False)["ride_count"].mean()
    by_type_hour = view.groupby(["day_type", "hour"], as_index=False)["ride_count"].mean()
    a, b = st.columns(2)
    with a:
        fig = px.bar(by_hour, x="hour", y="ride_count", title="Average demand by hour",
                     labels={"hour": "Hour of day", "ride_count": "Mean rides per local-hour label"},
                     color_discrete_sequence=[source.COLORS["Final Hybrid"]])
        st.plotly_chart(chart_layout(fig), width="stretch")
    with b:
        fig = px.line(by_type_hour, x="hour", y="ride_count", color="day_type", markers=True,
                      category_orders={"day_type": ["Weekday", "Weekend"]},
                      title="Weekday vs weekend hourly profile",
                      labels={"hour": "Hour of day", "ride_count": "Mean rides per local-hour label", "day_type": "Day type"})
        st.plotly_chart(chart_layout(fig), width="stretch")

    by_month = view.groupby(["month", "month_name"], as_index=False).agg(
        total_rides=("ride_count", "sum"), mean_rides=("ride_count", "mean")
    )
    by_month = by_month.sort_values("month")
    by_period = view.groupby("demand_period", as_index=False)["ride_count"].mean()
    c, d = st.columns(2)
    with c:
        month_measure = st.selectbox(
            "Monthly comparison", ("Total ride starts", "Average per local-hour label")
        )
        month_column = "total_rides" if month_measure == "Total ride starts" else "mean_rides"
        fig = px.bar(by_month, x="month_name", y=month_column,
                     title=f"{month_measure} by month · selected dates",
                     labels={"month_name": "Month", month_column: month_measure},
                     color_discrete_sequence=[source.COLORS["No-weather Direct"]])
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(chart_layout(fig), width="stretch")
        st.caption("Totals cover selected days only when a month is partly included.")
    with d:
        fig = px.bar(by_period, x="demand_period", y="ride_count", title="Peak vs off-peak demand",
                     labels={"demand_period": "Demand period", "ride_count": "Mean rides per local-hour label"},
                     color="demand_period", color_discrete_map={
                         "Peak": source.COLORS["Final Hybrid"], "Off-peak": source.COLORS["Weekly Seasonal Naive"]
                     })
        st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption(
        "Peak = weekday target hours 07–09 and 16–19 (not a demand threshold). "
        "The nonexistent 2024-03-10 02:00 spring DST hour is excluded; "
        "2024-11-03 01:00 combines the two fall-back hours."
    )


def show_forecast() -> None:
    log = forecast_data()
    st.title("Forecast")
    st.caption(
        "2024 Historical Shadow Simulation · New York local clock labels · "
        "historical forecasts only"
    )
    st.info(
        "These predictions were recreated from notebook 12's frozen process and checked "
        "against its saved results. Changing a selection does not run a model."
    )
    origin_choice, strategy_choice, horizon_choice = st.columns(3)
    origins = list(pd.DatetimeIndex(log["forecast_origin"].unique()).sort_values(ascending=False))
    origin = origin_choice.selectbox(
        "Forecast Origin", origins,
        format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M"),
    )
    origin = pd.Timestamp(origin)
    strategy = strategy_choice.selectbox("Strategy", source.STRATEGIES)
    horizon = horizon_choice.selectbox(
        "Horizon", source.HORIZONS, format_func=lambda value: f"{value}h",
    )
    origin_log = log.loc[log["forecast_origin"].eq(origin)]
    path = origin_log.loc[origin_log["strategy"].eq(strategy)].sort_values("horizon").copy()
    selected = path.loc[path["horizon"].eq(horizon)].iloc[0]

    st.subheader("Prediction Snapshot")
    snapshot = pd.DataFrame([{
        "Forecast Origin": origin.strftime("%Y-%m-%d %H:%M"),
        "Target Time": selected["target_time"].strftime("%Y-%m-%d %H:%M"),
        "Horizon": f"{horizon}h",
        "Actual": f"{selected['actual']:,.0f}",
        "Prediction": f"{selected['prediction']:,.2f}",
        "Absolute Error": f"{selected['absolute_error']:,.2f}",
        "Strategy": strategy,
    }])
    st.dataframe(snapshot, hide_index=True, width="stretch")
    st.caption("Actual and error are historical outcomes, shown for evaluation only. Counts are ride starts per hour.")

    st.subheader("Forecast Path")
    labels = [f"{value}h" for value in path["horizon"]]
    target_labels = path["target_time"].dt.strftime("%Y-%m-%d %H:%M")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=path["actual"], customdata=target_labels,
        mode="lines+markers", name="Actual", line=dict(color="#263B4C", width=2),
        hovertemplate="Target %{customdata}<br>Actual %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=path["prediction"], customdata=target_labels,
        mode="lines+markers", name=strategy,
        line=dict(color=source.COLORS[strategy], width=3),
        hovertemplate="Target %{customdata}<br>Prediction %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Forecast horizon", yaxis_title="Ride starts per hour")
    st.plotly_chart(chart_layout(fig, 420), width="stretch")
    st.caption(
        "The eight points are the frozen forecast horizons; connecting lines do not represent "
        "predictions at intervening hours."
    )

    st.subheader("Strategy Comparison")
    st.caption(f"Same forecast origin · selected {horizon}h target")
    comparison = origin_log.loc[origin_log["horizon"].eq(horizon)].set_index("strategy").loc[
        list(source.STRATEGIES)
    ].reset_index()
    comparison_table = pd.DataFrame({
        "Strategy": comparison["strategy"],
        "Prediction": comparison["prediction"].map(lambda value: f"{value:,.2f}"),
        "Actual": comparison["actual"].map(lambda value: f"{value:,.0f}"),
        "Absolute Error": comparison["absolute_error"].map(lambda value: f"{value:,.2f}"),
    })
    st.dataframe(comparison_table, hide_index=True, width="stretch")

    all_strategies = origin_log.copy()
    all_strategies["Horizon"] = all_strategies["horizon"].map(lambda value: f"{value}h")
    fig = px.bar(
        all_strategies, x="Horizon", y="prediction", color="strategy", barmode="group",
        category_orders={"Horizon": [f"{value}h" for value in source.HORIZONS],
                         "strategy": list(source.STRATEGIES)},
        color_discrete_map=source.COLORS,
        labels={"prediction": "Predicted ride starts per hour", "strategy": "Strategy"},
    )
    fig.update_layout(xaxis_title="Forecast horizon", yaxis_title="Predicted ride starts per hour")
    st.plotly_chart(chart_layout(fig, 430), width="stretch")


def show_model_comparison() -> None:
    overall = result_data("overall_metrics.csv")
    horizon = result_data("horizon_mae.csv")
    st.title("Model Comparison")
    st.caption("Fixed results from notebook 12 · hourly historical shadow origins · lower error is better")
    st.subheader("Overall metrics")
    table = overall[["Strategy", "N", "MAE", "RMSE", "Median AE", "P90 AE", "Worst-decile MAE"]].copy()
    table["N"] = table["N"].map(lambda value: f"{value:,.0f}")
    for column in ("MAE", "RMSE", "Median AE", "P90 AE", "Worst-decile MAE"):
        table[column] = table[column].map(lambda value: f"{value:,.2f}")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Error units: rides/hour. Worst-decile MAE is the mean AE of the worst 10% of paired predictions.")

    hybrid = metric_lookup(overall, "Final Hybrid", "MAE")
    no_weather = metric_lookup(overall, "No-weather Direct", "MAE")
    weekly = metric_lookup(overall, "Weekly Seasonal Naive", "MAE")
    a, b = st.columns(2)
    a.metric("Hybrid vs No-weather · MAE improvement", f"{(no_weather - hybrid) / no_weather:.2%}")
    b.metric("Hybrid vs Weekly Naive · MAE improvement", f"{(weekly - hybrid) / weekly:.2%}")

    long = horizon.melt(id_vars="Horizon", value_vars=list(source.STRATEGIES),
                        var_name="Strategy", value_name="MAE")
    fig = px.line(long, x="Horizon", y="MAE", color="Strategy", markers=True,
                  color_discrete_map=source.COLORS, title="MAE by forecast horizon",
                  labels={"Horizon": "Forecast horizon (hours)", "MAE": "MAE (rides/hour)"})
    fig.update_xaxes(type="category")
    st.plotly_chart(chart_layout(fig, 420), width="stretch")

    gain = pd.DataFrame({
        "Horizon": horizon["Horizon"].astype(str) + "h",
        "Hybrid vs No-weather": horizon["No-weather Direct"] - horizon["Final Hybrid"],
        "Hybrid vs Weekly Naive": horizon["Weekly Seasonal Naive"] - horizon["Final Hybrid"],
    }).melt(id_vars="Horizon", var_name="Comparison", value_name="MAE advantage")
    fig = px.bar(gain, x="Horizon", y="MAE advantage", color="Comparison", barmode="group",
                 title="Hybrid MAE advantage by horizon",
                 labels={"MAE advantage": "Benchmark − Hybrid MAE (rides/hour)"},
                 color_discrete_map={"Hybrid vs No-weather": source.COLORS["No-weather Direct"],
                                     "Hybrid vs Weekly Naive": source.COLORS["Weekly Seasonal Naive"]})
    fig.add_hline(y=0, line_color="#34424E", line_width=1)
    st.plotly_chart(chart_layout(fig, 390), width="stretch")
    st.caption(
        "Positive advantage means Hybrid has lower MAE. The frozen Hybrid equals Weekly Naive "
        "at 6–168h; its 1h and 3h weather branches trail No-weather Direct."
    )


def show_monitoring() -> None:
    recent = result_data("recent_metrics.csv")
    checkpoint = result_data("rolling_checkpoints_tail.csv", ("feedback_time",))
    warnings = result_data("warning_days.csv", ("feedback_time",))
    st.title("Monitoring")
    st.caption("2024 Historical Shadow Simulation · not live production monitoring")
    latest = checkpoint.sort_values("feedback_time").iloc[-1]
    latest_time = pd.Timestamp(latest["feedback_time"])
    warning_latest = warnings["feedback_time"].eq(latest_time).any()
    clear_loss = latest["Final Hybrid"] > 1.02 * latest["No-weather Direct"]
    state = "Performance Warning" if warning_latest else ("Watch" if clear_loss else "Healthy")
    if state == "Performance Warning":
        st.error(f"{state} · last matured checkpoint: {latest_time:%Y-%m-%d %H:%M}")
    elif state == "Watch":
        st.warning(f"{state} · last matured checkpoint: {latest_time:%Y-%m-%d %H:%M}")
    else:
        st.success(f"{state} · last matured checkpoint: {latest_time:%Y-%m-%d %H:%M}")
    st.caption(
        "Notebook 12 rule: seven calendar-day rolling MAE with at least 96 matured origins; "
        "Performance Warning begins when the daily last valid Hybrid MAE exceeds "
        "1.02 × No-weather Direct for three consecutive calendar days. Watch denotes a "
        "clear-loss day before that streak matures; Healthy denotes no clear loss."
    )

    a, b, c = st.columns(3)
    a.metric("Recent matured · Hybrid MAE", f"{metric_lookup(recent, 'Final Hybrid', 'MAE'):,.2f}")
    b.metric("Recent matured · No-weather MAE", f"{metric_lookup(recent, 'No-weather Direct', 'MAE'):,.2f}")
    c.metric("Hybrid − No-weather", f"{latest['Hybrid minus No-weather Direct']:+,.2f}")
    st.caption("Recent matured seven-day window; MAE and difference are in rides/hour. All eight horizons must have feedback.")

    st.subheader("Rolling performance")
    st.image(str(source.ARTIFACT_DIR / "monitoring_rolling_mae.png"), width="stretch")
    st.image(str(source.ARTIFACT_DIR / "monitoring_hybrid_minus_no_weather.png"), width="stretch")
    st.caption("Saved figures from notebook 12. The gap around the DST-affected maturity window is intentional.")
    with st.expander("Hybrid minus Weekly Naive · rolling MAE"):
        st.image(str(source.ARTIFACT_DIR / "monitoring_hybrid_minus_weekly.png"), width="stretch")
        st.caption("Positive values mean the Hybrid's rolling MAE was higher than the Weekly baseline.")

    st.subheader("Cumulative absolute error")
    st.image(str(source.ARTIFACT_DIR / "monitoring_cumulative_ae.png"), width="stretch")
    st.caption("Cumulative AE sums paired origin/horizon predictions by target time; a target hour can appear in several pairs.")

    st.subheader("Research warning periods")
    ordered = warnings.sort_values("feedback_time").copy()
    ordered["episode"] = ordered["warning_episode_start"].astype(bool).cumsum()
    periods = ordered.groupby("episode", as_index=False).agg(
        start=("feedback_time", "min"), end=("feedback_time", "max"),
        warning_days=("feedback_time", "size"),
    )
    periods["start"] = periods["start"].dt.strftime("%Y-%m-%d")
    periods["end"] = periods["end"].dt.strftime("%Y-%m-%d")
    periods = periods.rename(columns={"episode": "Episode", "start": "Start",
                                      "end": "End", "warning_days": "Warning days"})
    st.dataframe(periods, hide_index=True, width="stretch")
    st.caption(
        f"{len(periods)} warning episodes and {len(ordered)} warning days "
        "in the saved 2024 shadow results."
    )


PAGES = {
    "Overview": show_overview,
    "Demand Explorer": show_demand_explorer,
    "Forecast": show_forecast,
    "Model Comparison": show_model_comparison,
    "Monitoring": show_monitoring,
}

with st.sidebar:
    st.markdown("## 🚲 BikeFlow")
    requested_page = st.query_params.get("page", "Overview")
    initial_page = requested_page if requested_page in PAGES else "Overview"
    page = st.radio(
        "Navigate", list(PAGES), index=list(PAGES).index(initial_page), label_visibility="collapsed"
    )
    st.divider()
    st.caption("NYC Citi Bike · 2024")
    st.caption("2024 Historical Shadow Simulation")

try:
    PAGES[page]()
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"This page could not load its saved source data: {error}")
    st.caption("Run the dashboard artifact export script from the repository before launching the app.")
