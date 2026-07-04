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

import requests

try:
    import google.generativeai as genai
except ImportError:
    genai = None

def _get_api_keys():
    """Retrieve keys from environment variables or Streamlit secrets."""
    keys = {
        "gemini": os.environ.get("GEMINI_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
    }
    
    # Try loading from streamlit secrets
    for provider in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        key_name = provider.split("_")[0].lower()
        if not keys[key_name]:
            try:
                if provider in st.secrets:
                    keys[key_name] = st.secrets[provider]
            except Exception:
                pass
                
    # Clean up empty strings or falsy values
    for k in keys:
        if keys[k]:
            keys[k] = str(keys[k]).strip()
            if not keys[k] or keys[k].lower() in ["none", "null", "false", ""]:
                keys[k] = None
        else:
            keys[k] = None
            
    return keys

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_generate_content(prompt, provider, keys_hash):
    """Cached generator to fetch and route prompts to a specific API credential."""
    keys = _get_api_keys()
    
    # 1. OpenAI API
    if provider == "OpenAI" and keys["openai"]:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {keys['openai']}"
            }
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 200
            }
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {"status": "success", "text": data["choices"][0]["message"]["content"].strip()}
            else:
                return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}

    # 2. Gemini API
    elif provider == "Gemini" and keys["gemini"]:
        if genai is not None:
            try:
                genai.configure(api_key=keys["gemini"])
                model = genai.GenerativeModel("gemini-2.5-flash")
                resp = model.generate_content(prompt)
                return {"status": "success", "text": resp.text.strip()}
            except Exception as e:
                print(f"[AI Advisor] Gemini SDK call failed: {e}")
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={keys['gemini']}"
            headers = {"Content-Type": "application/json"}
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {"status": "success", "text": data["candidates"][0]["content"]["parts"][0]["text"].strip()}
            else:
                return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}

    # 3. Anthropic API
    elif provider == "Anthropic" and keys["anthropic"]:
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "content-type": "application/json",
                "x-api-key": keys["anthropic"],
                "anthropic-version": "2023-06-01"
            }
            body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {"status": "success", "text": data["content"][0]["text"].strip()}
            else:
                return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}

    return {"status": "no_key", "message": "No keys configured for this provider."}

def _generate_with_fallback(prompt):
    """Router utility for prompt generation with automatic key failover and error reporting."""
    keys = _get_api_keys()
    keys_hash = (keys["gemini"], keys["openai"], keys["anthropic"])
    errors = []
    
    # 1. Try Gemini first (since Google AI Studio keys are free-tier and active)
    if keys["gemini"]:
        res = _cached_generate_content(prompt, "Gemini", keys_hash)
        if res["status"] == "success":
            return res["text"]
        else:
            errors.append(f"Gemini: {res['message']}")
            
    # 2. Try OpenAI second
    if keys["openai"]:
        res = _cached_generate_content(prompt, "OpenAI", keys_hash)
        if res["status"] == "success":
            return res["text"]
        else:
            errors.append(f"OpenAI: {res['message']}")
            
    # 3. Try Anthropic third
    if keys["anthropic"]:
        res = _cached_generate_content(prompt, "Anthropic", keys_hash)
        if res["status"] == "success":
            return res["text"]
        else:
            errors.append(f"Anthropic: {res['message']}")
            
    if errors:
        return "⚠️ **AI Provider Connection Error**\n\n" + "\n".join([f"- {err}" for err in errors])
        
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
                     city=None, forecast_summary=None, user_profile=None, chat_history=None):
    """
    Free-form natural language Q&A grounded in the current dashboard state, supporting conversation history.
    Falls back to a rule-based answer if no API key is configured.
    """
    context = _context_block(aqi_value, aqi_label, co, ozone, no2, pm25, city, forecast_summary)
    profile_line = f"\nThe user is: {user_profile}." if user_profile else ""

    history_str = ""
    if chat_history:
        history_str = "\nConversation history:\n"
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    prompt = f"""You are an air quality decision assistant. Be direct, concrete, and brief
(3-5 sentences max). Base your answer only on the data given. If the question
requires a decision, give a clear recommendation, not just a description of the data.

{context}{profile_line}
{history_str}
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
