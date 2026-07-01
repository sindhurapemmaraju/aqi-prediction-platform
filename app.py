"""
app.py  –  AI-Driven AQI Analytics Platform
Streamlit + Plotly | Data Science Project
"""

import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

from live_data import fetch_live_aqi
from forecast_engine import extract_forecast, daily_overall_forecast, trend_direction
from ai_advisor import answer_question, generate_advisory

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AQI Analytics Platform",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Hide Streamlit header & top white bar */
[data-testid="stHeader"], .stAppHeader {
    display: none !important;
}

/* ── App background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    color: #e6edf3;
    padding-top: 0px !important;
}

/* ── Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #79c0ff;
}

/* ── Metric cards (Equal height & flexbox centered) */
.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2533);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 145px;
    width: 100%;
}
.metric-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-sub {
    font-size: 0.88rem;
    margin-top: 8px;
    font-weight: 500;
}

/* ── Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #79c0ff;
    text-transform: uppercase;
    margin-bottom: 4px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
}

/* ── Info box */
.info-box {
    background: #161b22;
    border-left: 3px solid #388bfd;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    font-size: 0.87rem;
    color: #c9d1d9;
    line-height: 1.7;
}

/* ── Category badge */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 30px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ── Plotly chart containers */
.chart-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 10px;
}

/* ── Sidebar slider labels */
.stSlider label { color: #c9d1d9 !important; font-size: 0.9rem !important; }
div[data-testid="stSlider"] .stMarkdown { color: #8b949e; }

/* ── Top title bar */
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #79c0ff, #56d364);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: -6px;
}
</style>
""", unsafe_allow_html=True)

# ── AQI Helper Functions ──────────────────────────────────────────────────────

AQI_SCALE = [
    (50,  "Good",                  "#56d364", "#0d2c1a"),
    (100, "Moderate",              "#e3b341", "#2c200a"),
    (150, "Unhealthy for Sensitive","#f0883e", "#2c150a"),
    (200, "Unhealthy",             "#f85149", "#2c0b0b"),
    (300, "Very Unhealthy",        "#bc8cff", "#1a0d2c"),
    (500, "Hazardous",             "#da3633", "#2c0505"),
]

def classify_aqi(val):
    for ceiling, label, color, bg in AQI_SCALE:
        if val <= ceiling:
            return label, color, bg
    return "Hazardous", "#da3633", "#2c0505"

# ── Load Model ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    if not os.path.exists("aqi_model.pkl"):
        st.error("⚠️ `aqi_model.pkl` not found. Run `python train_model.py` first.")
        st.stop()
    model  = joblib.load("aqi_model.pkl")
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
    return model, scaler

model, scaler = load_artifacts()
FEATURES = ["co aqi value", "ozone aqi value", "no2 aqi value", "pm2.5 aqi value"]
FEATURE_LABELS = ["CO", "Ozone", "NO₂", "PM 2.5"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## :material/eco: AQI Platform")
    st.markdown("**AI-Driven Analytics**")
    st.markdown("---")
    
    st.markdown("### :material/public: Location Settings")
    st.caption("Type any city name around the world to fetch live AQI.")
    city_query = st.text_input("Enter city name", "Hyderabad").strip()

    st.markdown("---")
    st.markdown("### :material/person: Who's asking?")
    user_profile = st.selectbox(
        "Personalize advisories for:",
        ["General public", "Asthma / respiratory condition", "Parent of young children",
         "Elderly", "Outdoor athlete / runner"],
        index=0,
    )

    st.markdown("---")
    st.markdown("##### :material/legend_toggle: AQI Scale Reference")
    for ceil, lbl, col, _ in AQI_SCALE:
        st.markdown(
            f'<span style="color:{col};">●</span> **{lbl}** '
            f'<span style="color:#8b949e;font-size:0.8rem;">(≤{ceil})</span>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Model: Random Forest Regressor")

# ── Fetch Live City Data ──────────────────────────────────────────────────────
live = fetch_live_aqi(city_query)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">AQI Decision Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time air quality tracking, forecast analysis, and model-driven AI decision advisories</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab_live, tab_forecast, tab_ai = st.tabs([
    ":material/public: Live City AQI", ":material/calendar_today: Forecast", ":material/smart_toy: Ask AI"
])

# ── Tab: Live City AQI ──────────────────────────────────────────────────────
with tab_live:
    st.markdown('<div class="section-header">Real-Time City AQI</div>', unsafe_allow_html=True)
    st.caption("Live data ingested from the AQICN global monitoring network — this is what makes the platform a real decision tool, not just a demo.")

    if live is None or live.get("error") == "no_token":
        st.warning(
            "No AQICN API token configured. Live city lookup needs a free token from "
            "[aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/), "
            "set as the `AQICN_TOKEN` environment variable or in `st.secrets`."
        )
    elif live.get("error"):
        st.error(f"Couldn't fetch data for '{city_query}': {live['error']}")
    else:
        live_label, live_color, live_bg = classify_aqi(live["aqi"])
        
        # Calculate Random Forest prediction on real-time city pollutants
        live_input = np.array([[live["co"], live["ozone"], live["no2"], live["pm25"]]])
        if scaler:
            live_input = scaler.transform(live_input)
        predicted_aqi = float(model.predict(live_input)[0])
        pred_label, pred_color, pred_bg = classify_aqi(predicted_aqi)

        # ── Metric Row ──
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        with lc1:
            st.markdown(f"""<div class="metric-card" style="border-color:{live_color}33;background:linear-gradient(135deg,{live_bg},{live_bg}aa);">
                <div class="metric-title">{live['city']}</div>
                <div class="metric-value" style="color:{live_color};">{live['aqi']}</div>
                <div class="metric-sub" style="color:{live_color};font-size:0.8rem;margin-top:2px;">{live_label}</div></div>""", unsafe_allow_html=True)
        for col, label, val in [(lc2, "CO", live["co"]), (lc3, "Ozone", live["ozone"]),
                                 (lc4, "NO₂", live["no2"]), (lc5, "PM 2.5", live["pm25"])]:
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value" style="font-size:1.8rem;color:#c9d1d9;">{val:.1f}</div></div>""", unsafe_allow_html=True)

        st.caption(f"Dominant pollutant: **{live['dominant_pollutant']}** · Last updated: {live['time']}")

        # ── Gauge and Model Comparison Row ──
        left, right = st.columns([1, 1], gap="medium")
        
        with left:
            st.markdown('<div class="section-header">AQI Gauge (Actual Live)</div>', unsafe_allow_html=True)
            gauge_steps = [
                {"range": [0,   50],  "color": "#0d2c1a"},
                {"range": [50,  100], "color": "#2c200a"},
                {"range": [100, 150], "color": "#2c150a"},
                {"range": [150, 200], "color": "#2c0b0b"},
                {"range": [200, 300], "color": "#1a0d2c"},
                {"range": [300, 500], "color": "#2c0505"},
            ]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=live["aqi"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"<b>{live_label}</b>", "font": {"size": 18, "color": live_color}},
                number={"font": {"size": 52, "color": live_color}, "suffix": ""},
                gauge={
                    "axis": {
                        "range": [0, 500],
                        "tickwidth": 1,
                        "tickcolor": "#30363d",
                        "tickfont": {"color": "#8b949e", "size": 11},
                    },
                    "bar": {"color": live_color, "thickness": 0.25},
                    "bgcolor": "#0d1117",
                    "borderwidth": 0,
                    "steps": gauge_steps,
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 2},
                        "thickness": 0.75,
                        "value": live["aqi"],
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#161b22",
                font={"family": "Space Grotesk"},
                height=280,
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with right:
            st.markdown('<div class="section-header">ML Predictor Comparison</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card" style="border-color:{pred_color}33;background:linear-gradient(135deg,{pred_bg},{pred_bg}aa);height:280px;width:100%;">
                <div class="metric-title" style="font-size:0.9rem;">Random Forest Prediction</div>
                <div class="metric-value" style="color:{pred_color};font-size:4rem;margin-top:10px;">{predicted_aqi:.1f}</div>
                <div class="metric-sub" style="color:{pred_color};font-size:1.1rem;font-weight:600;margin-top:10px;">{pred_label}</div>
                <div style="font-size:0.8rem;color:#8b949e;margin-top:20px;">
                    Model inputs: Live readings for CO, Ozone, NO₂, PM 2.5
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Contribution and Feature Importance Row ──
        c_left, c_right = st.columns([1, 1], gap="medium")
        
        with c_left:
            st.markdown('<div class="section-header">Live Pollutant Contribution</div>', unsafe_allow_html=True)
            raw_vals   = [live["co"], live["ozone"], live["no2"], live["pm25"]]
            fig_inputs = go.Figure()
            colors_map = ["#388bfd", "#56d364", "#e3b341", "#f85149"]

            for i, (label, val, color) in enumerate(zip(FEATURE_LABELS, raw_vals, colors_map)):
                fig_inputs.add_trace(go.Bar(
                    name=label,
                    x=[label],
                    y=[val],
                    marker_color=color,
                    text=f"{val:.1f}",
                    textposition="outside",
                    textfont={"color": "#c9d1d9"},
                ))

            fig_inputs.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font={"family": "Space Grotesk", "color": "#c9d1d9"},
                height=260,
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=False,
                xaxis=dict(showgrid=False, tickfont={"size": 13}),
                yaxis=dict(
                    showgrid=True, gridcolor="#21262d",
                    title={"text": "AQI Sub-Index Value", "font": {"color": "#8b949e"}},
                    tickfont={"color": "#8b949e"},
                ),
                bargap=0.35,
            )
            st.plotly_chart(fig_inputs, use_container_width=True)

        with c_right:
            st.markdown('<div class="section-header">Model Feature Importance</div>', unsafe_allow_html=True)
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "Pollutant":   FEATURE_LABELS,
                "Importance":  importances,
                "Feature":     FEATURES,
            }).sort_values("Importance", ascending=True)

            bar_colors = ["#388bfd", "#56d364", "#e3b341", "#f85149"]
            fig_bar = go.Figure(go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Pollutant"],
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0),
                ),
                text=[f"{v:.3f}" for v in fi_df["Importance"]],
                textposition="outside",
                textfont={"color": "#c9d1d9", "size": 12},
            ))
            fig_bar.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font={"family": "Space Grotesk", "color": "#c9d1d9"},
                height=260,
                margin=dict(t=20, b=20, l=10, r=60),
                xaxis=dict(
                    showgrid=True, gridcolor="#21262d",
                    title={"text": "Relative Importance", "font": {"color": "#8b949e"}},
                    tickfont={"color": "#8b949e"},
                ),
                yaxis=dict(showgrid=False, tickfont={"size": 13}),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── AI Advisory Card ──
        st.markdown('<div class="section-header">AI Advisory — Live Conditions</div>', unsafe_allow_html=True)
        with st.spinner("Generating advisory…"):
            live_advisory = generate_advisory(
                live["aqi"], live_label, live["co"], live["ozone"], live["no2"], live["pm25"],
                city=live["city"], user_profile=user_profile,
            )
        st.markdown(f'<div class="info-box">{live_advisory}</div>', unsafe_allow_html=True)

# ── Tab: Forecast ─────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown('<div class="section-header">Multi-Day AQI Forecast</div>', unsafe_allow_html=True)
    st.caption("Forecast derived from the same live feed's projection data — shows what's coming, not just what is now.")

    if live is None or live.get("error"):
        st.info("Please select or search for a valid city in the sidebar to see its forecast.")
    else:
        fdf = extract_forecast(live["raw"])
        if fdf.empty:
            st.warning("No forecast data available for this city from the live feed.")
        else:
            pivot = daily_overall_forecast(fdf)
            direction = trend_direction(pivot)
            arrow = {"rising": ":material/trending_up: Rising", "falling": ":material/trending_down: Falling", "stable": ":material/trending_flat: Stable"}[direction]
            st.markdown(f"**Trend over forecast window: {arrow}**")

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=pivot["date"], y=pivot["overall_aqi_est"],
                mode="lines+markers", name="Estimated overall AQI",
                line=dict(color="#79c0ff", width=3), marker=dict(size=8),
            ))
            fig_fc.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                font={"family": "Space Grotesk", "color": "#c9d1d9"},
                height=340, margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(showgrid=False, tickfont={"color": "#8b949e"}),
                yaxis=dict(showgrid=True, gridcolor="#21262d", title="Estimated AQI", tickfont={"color": "#8b949e"}),
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            st.dataframe(pivot, use_container_width=True, hide_index=True)

            st.markdown('<div class="section-header">AI Advisory — Forecast-Aware</div>', unsafe_allow_html=True)
            summary = f"{direction} trend, moving from {pivot['overall_aqi_est'].iloc[0]:.0f} to {pivot['overall_aqi_est'].iloc[-1]:.0f} over the forecast window."
            with st.spinner("Generating advisory…"):
                fc_advisory = answer_question(
                    "Given this forecast trend, what should I plan for over the next few days?",
                    live["aqi"], classify_aqi(live["aqi"])[0],
                    live["co"], live["ozone"], live["no2"], live["pm25"],
                    city=live["city"], forecast_summary=summary, user_profile=user_profile,
                )
            st.markdown(f'<div class="info-box">{fc_advisory}</div>', unsafe_allow_html=True)

# ── Tab: Ask AI ────────────────────────────────────────────────────────────
with tab_ai:
    st.markdown('<div class="section-header">Ask the AQI Assistant</div>', unsafe_allow_html=True)
    st.caption("Ask a natural-language question grounded in the selected city's live air quality metrics.")

    question = st.text_area(
        "Your question",
        placeholder="e.g. Is it safe to let my kids play outside this evening?",
        height=90,
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            if live and not live.get("error"):
                q_aqi = live["aqi"]
                q_label = classify_aqi(live["aqi"])[0]
                q_co, q_ozone, q_no2, q_pm25 = live["co"], live["ozone"], live["no2"], live["pm25"]
                q_city = live["city"]
                
                with st.spinner("Thinking…"):
                    answer = answer_question(
                        question, q_aqi, q_label, q_co, q_ozone, q_no2, q_pm25,
                        city=q_city, user_profile=user_profile,
                    )
                st.markdown(f'<div class="info-box">{answer}</div>', unsafe_allow_html=True)
            else:
                st.error("No valid live city data loaded to ground the AI assistant.")

    st.caption("Needs a Gemini API key (`GEMINI_API_KEY`) to use the real model — without one, you'll get a rule-based fallback answer so the app still works.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#484f58;font-size:0.78rem;margin-top:20px;">'
    'AI-Driven AQI Analytics Platform &nbsp;|&nbsp; Built with Streamlit + Plotly + Scikit-Learn'
    '</div>',
    unsafe_allow_html=True,
)
