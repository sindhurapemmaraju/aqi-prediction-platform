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

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AQI Analytics Platform",
    page_icon="🌿",
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

/* ── App background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    color: #e6edf3;
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

/* ── Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2533);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 24px 28px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
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
    font-size: 3.2rem;
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
    st.markdown("## 🌿 AQI Platform")
    st.markdown("**AI-Driven Analytics**")
    st.markdown("---")
    st.markdown("### 🎚️ Pollutant Inputs")
    st.caption("Adjust the sliders to simulate air quality conditions.")

    co    = st.slider("🟤 CO AQI Value",    min_value=0.0, max_value=50.0,  value=3.0,  step=0.1)
    ozone = st.slider("🔵 Ozone AQI Value", min_value=0.0, max_value=100.0, value=35.0, step=0.5)
    no2   = st.slider("🟠 NO₂ AQI Value",   min_value=0.0, max_value=100.0, value=20.0, step=0.5)
    pm25  = st.slider("🔴 PM 2.5 AQI Value",min_value=0.0, max_value=300.0, value=55.0, step=1.0)

    st.markdown("---")
    st.markdown("##### 📊 AQI Scale Reference")
    for ceil, lbl, col, _ in AQI_SCALE:
        st.markdown(
            f'<span style="color:{col};">●</span> **{lbl}** '
            f'<span style="color:#8b949e;font-size:0.8rem;">(≤{ceil})</span>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Model: Random Forest Regressor | Sklearn")

# ── Prediction ────────────────────────────────────────────────────────────────

input_arr = np.array([[co, ozone, no2, pm25]])
if scaler:
    input_arr = scaler.transform(input_arr)
predicted_aqi = float(model.predict(input_arr)[0])
aqi_label, aqi_color, aqi_bg = classify_aqi(predicted_aqi)

# ── Header ────────────────────────────────────────────────────────────────────

col_title, col_spacer = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">🌍 Air Quality Index Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-time AQI estimation using a trained Random Forest model</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Metric Row ───────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

for col, label, val, unit in [
    (c1, "CO AQI",     co,            ""),
    (c2, "Ozone AQI",  ozone,         ""),
    (c3, "NO₂ AQI",    no2,           ""),
    (c4, "PM 2.5 AQI", pm25,          ""),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{label}</div>
            <div class="metric-value" style="font-size:1.8rem;color:#c9d1d9;">{val:.1f}</div>
        </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card" style="border-color:{aqi_color}33;background:linear-gradient(135deg,{aqi_bg},{aqi_bg}aa);">
        <div class="metric-title">Predicted AQI</div>
        <div class="metric-value" style="color:{aqi_color};">{predicted_aqi:.1f}</div>
        <div class="metric-sub" style="color:{aqi_color};">{aqi_label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row ────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1], gap="large")

# ── Gauge Chart ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-header">📊 AQI Gauge</div>', unsafe_allow_html=True)

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
        value=predicted_aqi,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>{aqi_label}</b>", "font": {"size": 18, "color": aqi_color}},
        number={"font": {"size": 52, "color": aqi_color}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 500],
                "tickwidth": 1,
                "tickcolor": "#30363d",
                "tickfont": {"color": "#8b949e", "size": 11},
            },
            "bar": {"color": aqi_color, "thickness": 0.25},
            "bgcolor": "#0d1117",
            "borderwidth": 0,
            "steps": gauge_steps,
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.75,
                "value": predicted_aqi,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#161b22",
        font={"family": "Space Grotesk"},
        height=300,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── Feature Importance ───────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-header">🔬 Feature Importance</div>', unsafe_allow_html=True)

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
        height=300,
        margin=dict(t=20, b=20, l=10, r=60),
        xaxis=dict(
            showgrid=True, gridcolor="#21262d",
            title="Relative Importance", titlefont={"color": "#8b949e"},
            tickfont={"color": "#8b949e"},
        ),
        yaxis=dict(showgrid=False, tickfont={"size": 13}),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Pollutant Contribution Bar ─────────────────────────────────────────────────

st.markdown('<div class="section-header">📈 Current Input Contribution Overview</div>', unsafe_allow_html=True)

raw_vals   = [co, ozone, no2, pm25]
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
        title="AQI Sub-Index Value",
        titlefont={"color": "#8b949e"},
        tickfont={"color": "#8b949e"},
    ),
    bargap=0.35,
)
st.plotly_chart(fig_inputs, use_container_width=True)

# ── About the Data ────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📚 About the Data & AQI Formula</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>What is AQI?</b><br>
The <b>Air Quality Index (AQI)</b> is a standardised scale (0–500) used by the EPA to communicate how
polluted the air currently is, or how polluted it is forecast to become.<br><br>

<b>How is it computed?</b><br>
The overall AQI is determined by the <i>highest</i> sub-index among the measured pollutants:

<pre style="background:#0d1117;padding:10px;border-radius:8px;margin-top:8px;color:#79c0ff;">
AQI = max( I_CO, I_Ozone, I_NO₂, I_PM2.5 )

Each sub-index Iₚ is computed via EPA breakpoint interpolation:
  Iₚ = [(I_hi - I_lo) / (C_hi - C_lo)] × (Cₚ - C_lo) + I_lo
</pre>

<b>Dataset Features used in this model:</b><br>
• <b>CO AQI Value</b> — Carbon Monoxide sub-index<br>
• <b>Ozone AQI Value</b> — Ground-level Ozone sub-index<br>
• <b>NO₂ AQI Value</b> — Nitrogen Dioxide sub-index<br>
• <b>PM 2.5 AQI Value</b> — Fine Particulate Matter sub-index<br><br>

<b>Model:</b> Random Forest Regressor (200 trees) trained with StandardScaler normalisation.
The feature importance chart above shows the relative contribution of each pollutant
to the model's prediction.
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#484f58;font-size:0.78rem;">'
    'AI-Driven AQI Analytics Platform &nbsp;|&nbsp; Built with Streamlit + Plotly + Scikit-Learn'
    '</div>',
    unsafe_allow_html=True,
)
