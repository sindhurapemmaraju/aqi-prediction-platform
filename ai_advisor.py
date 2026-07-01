"""
ai_advisor.py – Gemini-powered decision intelligence layer

Turns raw AQI numbers into a natural-language answer to the question
that actually matters to a user: "what should I do?"

Two entry points:
  - answer_question(): free-form NL Q&A grounded in current + forecast data
  - generate_advisory(): auto-generated recommendation card for a user profile

Set GEMINI_API_KEY as an environment variable or in st.secrets.
Get a free key at https://aistudio.google.com/apikey
"""

import os
import streamlit as st

try:
    import google.generativeai as genai
except ImportError:
    genai = None

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _get_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None
    return key


def _configured_model(model_name=None):
    if genai is None:
        return None
    key = _get_key()
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel(model_name or MODEL_NAME)


def _generate_with_fallback(prompt):
    """Try to generate content using primary model, fallback to 1.5-flash on failure."""
    model = _configured_model()
    if model is None:
        return None
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"[AI Fallback] Primary model failed: {e}. Trying gemini-1.5-flash...")
        try:
            fallback_model = _configured_model("gemini-1.5-flash")
            if fallback_model:
                resp = fallback_model.generate_content(prompt)
                return resp.text.strip()
        except Exception as fe:
            print(f"[AI Fallback] Fallback model failed: {fe}")
    return None


def _context_block(aqi_value, aqi_label, co, ozone, no2, pm25, city=None, forecast_summary=None):
    ctx = f"""Current air quality data{f" for {city}" if city else ""}:
- Overall AQI: {aqi_value:.0f} ({aqi_label})
- CO sub-index: {co}
- Ozone sub-index: {ozone}
- NO2 sub-index: {no2}
- PM2.5 sub-index: {pm25}
"""
    if forecast_summary:
        ctx += f"\nForecast trend over the next few days: {forecast_summary}\n"
    return ctx


def answer_question(question, aqi_value, aqi_label, co, ozone, no2, pm25,
                     city=None, forecast_summary=None, user_profile=None):
    """
    Free-form natural language Q&A grounded in the current dashboard state.
    Falls back to a rule-based answer if no API key is configured.
    """
    context = _context_block(aqi_value, aqi_label, co, ozone, no2, pm25, city, forecast_summary)
    profile_line = f"\nThe user is: {user_profile}." if user_profile else ""

    prompt = f"""You are an air quality decision assistant. Be direct, concrete, and brief
(3-5 sentences max). Base your answer only on the data given. If the question
requires a decision, give a clear recommendation, not just a description of the data.

{context}{profile_line}

Question: {question}
"""
    result = _generate_with_fallback(prompt)
    if result:
        return result
    return _fallback_answer(aqi_value, aqi_label)


def generate_advisory(aqi_value, aqi_label, co, ozone, no2, pm25,
                       city=None, forecast_summary=None, user_profile="general public"):
    """
    Auto-generated short advisory card, e.g. for a homepage alert banner.
    """
    context = _context_block(aqi_value, aqi_label, co, ozone, no2, pm25, city, forecast_summary)

    prompt = f"""You are an air quality decision assistant writing a short (2-3 sentence)
advisory for {user_profile}. Be specific and actionable — mention a concrete action
(e.g. timing, activity to avoid/do, precaution) rather than generic advice like "stay safe".

{context}
"""
    result = _generate_with_fallback(prompt)
    if result:
        return result
    return _fallback_answer(aqi_value, aqi_label)


def _fallback_answer(aqi_value, aqi_label):
    """Rule-based backup so the app still works without an API key."""
    if aqi_value <= 50:
        return f"AQI is {aqi_value:.0f} ({aqi_label}) — air quality is good. Fine for outdoor activity, exercise, and keeping windows open."
    elif aqi_value <= 100:
        return f"AQI is {aqi_value:.0f} ({aqi_label}) — acceptable for most people. Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi_value <= 150:
        return f"AQI is {aqi_value:.0f} ({aqi_label}) — sensitive groups (asthma, children, elderly) should reduce prolonged outdoor exertion. General public can continue normal activity."
    elif aqi_value <= 200:
        return f"AQI is {aqi_value:.0f} ({aqi_label}) — everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor activity entirely."
    elif aqi_value <= 300:
        return f"AQI is {aqi_value:.0f} ({aqi_label}) — avoid outdoor exertion. Keep windows closed, use an air purifier or N95 mask if you must go outside."
    else:
        return f"AQI is {aqi_value:.0f} ({aqi_label}) — hazardous. Stay indoors, avoid all outdoor activity, and use air filtration if available."
