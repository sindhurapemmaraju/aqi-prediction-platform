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

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")


def _get_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None
    return key


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_generate_content(model_name, prompt):
    """Cached content generator to prevent repetitive API calls on Streamlit reruns."""
    if genai is None:
        raise RuntimeError("google-generativeai package not imported")
    key = _get_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return resp.text.strip()


def _generate_with_fallback(prompt):
    """Try to generate content using primary model, fallback to 2.5-flash-lite on failure."""
    try:
        return _cached_generate_content(MODEL_NAME, prompt)
    except Exception as e:
        print(f"[AI Fallback] Primary model ({MODEL_NAME}) failed: {e}. Trying fallback...")
        try:
            return _cached_generate_content("gemini-2.5-flash-lite", prompt)
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
