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
from ai_advisor import answer_question, generate_advisory, _get_api_keys

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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* ── Hide Streamlit header & top white bar */
[data-testid="stHeader"], .stAppHeader {
    display: none !important;
}

/* ── App background with a clean, light pastel gradient */
.stApp {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 50%, #cbd5e1 100%);
    color: #0f172a;
    padding-top: 0px !important;
}

/* ── Glassmorphism Sidebar (Light) */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.45) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.4);
}
/* Ensure sidebar labels, titles, and reference text are highly visible */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #1e293b !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4,
[data-testid="stSidebar"] .stMarkdown h5 {
    color: #0f172a !important;
    font-weight: 700;
}

/* ── Metric Grid (Responsive CSS Flexbox grid) */
.metrics-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 24px;
    width: 100%;
}

/* ── Metric cards (Glassmorphic light with hover lift) */
.metric-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 145px;
    flex: 1 1 180px;
    min-width: 140px;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.08);
    background: rgba(255, 255, 255, 0.85);
}

.metric-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    color: #0f172a;
}
.metric-sub {
    font-size: 0.88rem;
    margin-top: 8px;
    font-weight: 500;
}

/* ── Comparison Card (Grounded Model Predictor) */
.comparison-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 280px;
    width: 100%;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.comparison-card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.08);
    background: rgba(255, 255, 255, 0.85);
}

/* ── Plotly chart containers wrapper styling via Streamlit selector */
div[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.45) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.02);
    padding: 12px;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
div[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(99, 102, 241, 0.25);
    box-shadow: 0 15px 35px rgba(99, 102, 241, 0.06);
    background: rgba(255, 255, 255, 0.6) !important;
}

/* ── Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #4f46e5;
    text-transform: uppercase;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    padding-bottom: 8px;
}

/* ── Info box */
.info-box {
    background: rgba(255, 255, 255, 0.5);
    border-left: 4px solid #4f46e5;
    border-radius: 4px 16px 16px 4px;
    padding: 18px 24px;
    font-size: 0.92rem;
    color: #334155;
    line-height: 1.7;
    border-top: 1px solid rgba(255, 255, 255, 0.4);
    border-right: 1px solid rgba(255, 255, 255, 0.4);
    border-bottom: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    transition: border-color 0.3s ease;
}
.info-box:hover {
    border-left-color: #6366f1;
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

/* ── Sidebar slider labels */
.stSlider label { color: #334155 !important; font-size: 0.9rem !important; }
div[data-testid="stSlider"] .stMarkdown { color: #64748b; }

/* ── Top title bar */
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6366f1, #ec4899, #f43f5e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #475569;
    font-size: 1.05rem;
    margin-top: -4px;
}

/* Style primary buttons (Active Tab navigation) */
button[kind="primary"] {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.15) !important;
    transition: all 0.3s ease !important;
}
button[kind="primary"]:hover {
    background-color: #4338ca !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.25) !important;
}

/* Style secondary buttons (Inactive Tabs navigation) */
button[kind="secondary"] {
    background-color: rgba(255, 255, 255, 0.6) !important;
    color: #475569 !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
}
button[kind="secondary"]:hover {
    background-color: rgba(255, 255, 255, 0.85) !important;
    color: #1e293b !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── AQI Helper Functions ──────────────────────────────────────────────────────

AQI_SCALE = [
    (50,  "Good",                  "#059669", "#e6fbf3"),
    (100, "Moderate",              "#d97706", "#fef3c7"),
    (150, "Unhealthy for Sensitive","#ea580c", "#ffedd5"),
    (200, "Unhealthy",             "#dc2626", "#fee2e2"),
    (300, "Very Unhealthy",        "#9333ea", "#f3e8ff"),
    (500, "Hazardous",             "#be123c", "#ffe4e6"),
]

def classify_aqi(val):
    for ceiling, label, color, bg in AQI_SCALE:
        if val <= ceiling:
            return label, color, bg
    return "Hazardous", "#be123c", "#ffe4e6"

# ── Load Model ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    if not os.path.exists("aqi_model.pkl"):
        # Auto-train if files are missing (essential for direct Cloud deployments)
        try:
            import subprocess
            subprocess.run(["python", "train_model.py"], check=True)
        except Exception as e:
            st.error(f"⚠️ Model files missing and auto-training failed: {e}")
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
    st.markdown("**Air Quality Insights**")
    st.markdown("---")
    
    st.markdown("##### :material/legend_toggle: AQI Scale Reference")
    for ceil, lbl, col, _ in AQI_SCALE:
        st.markdown(
            f'<span style="color:{col};">●</span> **{lbl}** '
            f'<span style="color:#8b949e;font-size:0.8rem;">(≤{ceil})</span>',
            unsafe_allow_html=True,
        )



# Define default user profile
user_profile = "General public"

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">AQI Decision Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time air quality tracking, forecast analysis, and AI health recommendations</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Search Bar on Main Page ──
col_search, col_spacer = st.columns([2, 3])
with col_search:
    city_query = st.text_input(
        "Search City", 
        value="Hyderabad", 
        placeholder="Type any city (e.g. New York, London, Tokyo)...",
        label_visibility="collapsed",
        help="Type a city and press Enter to search air quality details."
    ).strip()



# ── Fetch Live City Data ──────────────────────────────────────────────────────
live = fetch_live_aqi(city_query)

# Initialize active tab in session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Live City AQI"

# Custom segmented tab navigation bar
nav_cols = st.columns([1, 1, 1])
with nav_cols[0]:
    if st.button("Live City AQI", icon=":material/public:", type="primary" if st.session_state.active_tab == "Live City AQI" else "secondary", use_container_width=True):
        st.session_state.active_tab = "Live City AQI"
        st.rerun()
with nav_cols[1]:
    if st.button("Forecast", icon=":material/calendar_today:", type="primary" if st.session_state.active_tab == "Forecast" else "secondary", use_container_width=True):
        st.session_state.active_tab = "Forecast"
        st.rerun()
with nav_cols[2]:
    if st.button("Ask AI", icon=":material/smart_toy:", type="primary" if st.session_state.active_tab == "Ask AI" else "secondary", use_container_width=True):
        st.session_state.active_tab = "Ask AI"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Tab Content rendering based on active tab ─────────────────────────────────
if st.session_state.active_tab == "Live City AQI":
    st.markdown('<div class="section-header">Real-Time City AQI</div>', unsafe_allow_html=True)
    st.caption("Live data ingested from the AQICN global monitoring network — this is what makes the platform a real decision tool, not just a demo.")

    if live is None or live.get("error") == "no_token":
        try:
            loaded_keys = list(st.secrets.keys())
        except Exception as e:
            loaded_keys = f"Error reading secrets: {e}"
        st.warning(
            "No AQICN API token configured. Live city lookup needs a free token from "
            "[aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/), "
            "set as the `AQICN_TOKEN` environment variable or in `st.secrets`."
        )
        st.error(f"🔧 **Streamlit Cloud Secrets Debug**: Loaded keys: `{loaded_keys}`")
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

        # ── Metric Grid ──
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card" style="border-color:{live_color}44; background: linear-gradient(135deg, {live_bg}, {live_bg}aa);">
                <div class="metric-title">{live['city']}</div>
                <div class="metric-value" style="color:{live_color};">{live['aqi']}</div>
                <div class="metric-sub" style="color:{live_color};">{live_label}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">CO</div>
                <div class="metric-value">{live["co"]:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Ozone</div>
                <div class="metric-value">{live["ozone"]:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">NO₂</div>
                <div class="metric-value">{live["no2"]:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">PM 2.5</div>
                <div class="metric-value">{live["pm25"]:.1f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Dominant pollutant: **{live['dominant_pollutant']}** · Last updated: {live['time']}")

        # ── Visual Analytics Row ──
        left, right = st.columns([1, 1], gap="medium")
        
        with left:
            st.markdown('<div class="section-header">Air Quality Gauge</div>', unsafe_allow_html=True)
            gauge_steps = [
                {"range": [0,   50],  "color": "#e6fbf3"},
                {"range": [50,  100], "color": "#fef3c7"},
                {"range": [100, 150], "color": "#ffedd5"},
                {"range": [150, 200], "color": "#fee2e2"},
                {"range": [200, 300], "color": "#f3e8ff"},
                {"range": [300, 500], "color": "#ffe4e6"},
            ]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=live["aqi"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"<b>{live_label}</b>", "font": {"size": 18, "color": live_color}},
                number={"font": {"size": 52, "color": live_color}},
                gauge={
                    "axis": {
                        "range": [0, 500],
                        "tickwidth": 1,
                        "tickcolor": "#94a3b8",
                        "tickfont": {"color": "#475569", "size": 11},
                    },
                    "bar": {"color": live_color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": gauge_steps,
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Outfit"},
                height=280,
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with right:
            st.markdown('<div class="section-header">Pollutants Breakdown</div>', unsafe_allow_html=True)
            raw_vals   = [live["co"], live["ozone"], live["no2"], live["pm25"]]
            fig_inputs = go.Figure()
            colors_map = ["#a5b4fc", "#86efac", "#fdba74", "#fca5a5"]

            for i, (label, val, color) in enumerate(zip(FEATURE_LABELS, raw_vals, colors_map)):
                fig_inputs.add_trace(go.Bar(
                    name=label,
                    x=[label],
                    y=[val],
                    marker_color=color,
                    text=f"{val:.1f}",
                    textposition="outside",
                    textfont={"color": "#334155"},
                ))

            fig_inputs.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Outfit", "color": "#334155"},
                height=280,
                margin=dict(t=30, b=20, l=10, r=10),
                showlegend=False,
                xaxis=dict(showgrid=False, tickfont={"size": 13, "color": "#475569"}),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                    title={"text": "Sub-Index Value", "font": {"color": "#475569"}},
                    tickfont={"color": "#475569"},
                ),
                bargap=0.35,
            )
            st.plotly_chart(fig_inputs, use_container_width=True)

        # ── AI Advisory Card ──
        st.markdown('<div class="section-header">AI Advisory — Live Conditions</div>', unsafe_allow_html=True)
        with st.spinner("Generating advisory…"):
            live_advisory = generate_advisory(
                live["aqi"], live_label, live["co"], live["ozone"], live["no2"], live["pm25"],
                city=live["city"], user_profile=user_profile,
            )
        st.markdown(f'<div class="info-box">{live_advisory}</div>', unsafe_allow_html=True)

# ── Tab: Forecast ─────────────────────────────────────────────────────────
elif st.session_state.active_tab == "Forecast":
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
                line=dict(color="#818cf8", width=3), marker=dict(size=8, color="#6366f1"),
            ))
            fig_fc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Outfit", "color": "#334155"},
                height=340, margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(showgrid=False, tickfont={"color": "#475569"}),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="Estimated AQI", tickfont={"color": "#475569"}),
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
elif st.session_state.active_tab == "Ask AI":
    st.markdown('<div class="section-header">Ask the AQI Assistant</div>', unsafe_allow_html=True)

    if live and not live.get("error"):
        q_city = live["city"]
        st.caption(f"Grounding conversation in live metrics for **{q_city}**")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Auto-reset chat history if city changes
        current_context_key = f"{q_city}"
        if "last_context_key" not in st.session_state:
            st.session_state.last_context_key = current_context_key

        if st.session_state.last_context_key != current_context_key:
            st.session_state.chat_history = []
            st.session_state.last_context_key = current_context_key

        # Clear history button
        col_title, col_clear = st.columns([5, 1])
        with col_clear:
            if st.button("Clear Chat", key="clear_chat", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Display chat messages from history on app rerun
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask a question about the air quality...", key="chat_user_input"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Generate assistant response
            q_aqi = live["aqi"]
            q_label = classify_aqi(live["aqi"])[0]
            q_co, q_ozone, q_no2, q_pm25 = live["co"], live["ozone"], live["no2"], live["pm25"]

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Extract the forecast summary if available
                    fdf = extract_forecast(live["raw"])
                    forecast_summary = None
                    if not fdf.empty:
                        pivot = daily_overall_forecast(fdf)
                        direction = trend_direction(pivot)
                        forecast_summary = f"{direction} trend, moving from {pivot['overall_aqi_est'].iloc[0]:.0f} to {pivot['overall_aqi_est'].iloc[-1]:.0f} over the forecast window."

                    response = answer_question(
                        prompt, q_aqi, q_label, q_co, q_ozone, q_no2, q_pm25,
                        city=q_city, forecast_summary=forecast_summary, user_profile=user_profile,
                        chat_history=st.session_state.chat_history[:-1] # pass history excluding latest prompt
                    )
                    st.markdown(response)
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.error("No valid live city data loaded to ground the AI assistant.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:0.78rem;margin-top:20px;">'
    'AQI Decision Intelligence Platform &nbsp;|&nbsp; Real-time Air Quality Insights'
    '</div>',
    unsafe_allow_html=True,
)
