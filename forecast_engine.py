"""
forecast_engine.py – Short-term AQI forecasting

AQICN's /feed endpoint includes a multi-day forecast (per-pollutant
avg/min/max) built from real monitoring station + model data. This module
extracts that into a clean DataFrame and derives a simple risk trend so the
app can show "what's coming", not just "what is now" — the predictive layer
required for decision intelligence.
"""

import pandas as pd


def extract_forecast(raw_payload: dict) -> pd.DataFrame:
    """
    Parse the 'forecast.daily' block from an AQICN raw payload into a
    tidy DataFrame with columns: date, pollutant, avg, min, max.
    Returns an empty DataFrame if no forecast is available.
    """
    forecast = raw_payload.get("forecast", {}).get("daily", {})
    if not forecast:
        return pd.DataFrame()

    rows = []
    for pollutant, entries in forecast.items():
        for e in entries:
            rows.append({
                "date": e.get("day"),
                "pollutant": pollutant,
                "avg": e.get("avg"),
                "min": e.get("min"),
                "max": e.get("max"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["pollutant", "date"])


def daily_overall_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-pollutant forecast into a single overall-AQI-per-day
    estimate by taking the max avg across pollutants for each date
    (mirrors how real AQI = max of sub-indices).
    """
    if df.empty:
        return df
    pivot = df.pivot_table(index="date", columns="pollutant", values="avg", aggfunc="max")
    pivot["overall_aqi_est"] = pivot.max(axis=1)
    return pivot.reset_index()


def trend_direction(pivot_df: pd.DataFrame) -> str:
    """Return 'rising', 'falling', or 'stable' based on first vs last forecast day."""
    if pivot_df.empty or len(pivot_df) < 2:
        return "stable"
    first = pivot_df["overall_aqi_est"].iloc[0]
    last = pivot_df["overall_aqi_est"].iloc[-1]
    delta = last - first
    if delta > 8:
        return "rising"
    elif delta < -8:
        return "falling"
    return "stable"
