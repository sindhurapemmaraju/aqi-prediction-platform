"""
live_data.py – Real-time AQI data ingestion
Pulls live pollutant readings from the AQICN (World Air Quality Index) API
for a given city, so the platform reflects real conditions, not just
hand-set sliders.

Get a free token: https://aqicn.org/data-platform/token/
Set it as an environment variable AQICN_TOKEN, or paste into st.secrets.
"""

import os
import requests
import streamlit as st

AQICN_BASE = "https://api.waqi.info/feed"


def _get_token():
    """Look for the AQICN token in env vars or Streamlit secrets."""
    token = os.environ.get("AQICN_TOKEN")
    if not token:
        try:
            token = st.secrets.get("AQICN_TOKEN", None)
        except Exception:
            token = None
    return token


@st.cache_data(ttl=600, show_spinner="Fetching live AQI data…")
def fetch_live_aqi(city: str):
    """
    Fetch real-time AQI + pollutant breakdown for a city.

    Returns a dict:
    {
        "aqi": int,
        "city": str,
        "co": float, "ozone": float, "no2": float, "pm25": float,
        "dominant_pollutant": str,
        "time": str,
        "raw": dict   # full API payload for debugging
    }
    or None if the fetch failed.
    """
    token = _get_token()
    if not token:
        return {"error": "no_token"}

    url = f"{AQICN_BASE}/{city}/?token={token}"
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
    except Exception as e:
        return {"error": str(e)}

    if data.get("status") != "ok":
        return {"error": data.get("data", "unknown_error")}

    d = data["data"]
    iaqi = d.get("iaqi", {})

    def sub(key):
        return iaqi.get(key, {}).get("v", None)

    return {
        "aqi": d.get("aqi"),
        "city": d.get("city", {}).get("name", city),
        "co": sub("co") or 0.0,
        "ozone": sub("o3") or 0.0,
        "no2": sub("no2") or 0.0,
        "pm25": sub("pm25") or 0.0,
        "dominant_pollutant": d.get("dominentpol", "n/a"),
        "time": d.get("time", {}).get("s", "n/a"),
        "raw": d,
    }


# Common India cities for a quick-select dropdown
SUGGESTED_CITIES = [
    "Hyderabad", "Delhi", "Mumbai", "Bengaluru", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]
