# 🌍 AQI Decision Intelligence Platform

> A professional-grade, AI-powered air quality decision intelligence platform. It ingests real-time global monitoring data, projects multi-day trends, and generates conversational health advisories using state-of-the-art LLMs (Gemini 2.5, GPT-4o, and Claude) to help users make concrete, health-focused decisions.

**Who it's for:** Asthma patients planning outdoor workouts, parents tracking local conditions, athletes training outdoors, and anyone looking for clear, health-centric guidance instead of raw pollutant data tables.

---

## ✨ Features in this Version

| Dashboard Tab | Description |
|---|---|
| 🌐 **Live City AQI** | Real-time global AQI ingestion for any city worldwide, featuring a clean glassmorphic design, a unified **Air Quality Gauge**, a **Pollutants Breakdown** bar chart, and AI health advisories. |
| 📅 **Forecast** | Multi-day AQI trend visualization extracted from live monitoring forecasts, summarized with an automatic risk-direction trend indicator. |
| 🤖 **Ask AI** | An interactive conversational chat interface allowing users to ask natural-language questions grounded in the current city's live air quality metrics and forecast history. |

---

## 🏗️ Architecture & Technology Stack

The platform is designed with a lightweight, clean, and highly robust Python architecture:

```
aqi-prediction-platform/
├── app.py                  ← Streamlit UI controller (stateful navigation & CSS grid)
├── live_data.py            ← Real-time data client (AQICN Global API)
├── forecast_engine.py      ← Forecast parsing & trajectory analytics
├── ai_advisor.py           ← Multi-provider REST router (Gemini 2.5, OpenAI, Anthropic)
├── train_model.py          ← Model training script
├── requirements.txt        ← Project dependency manifest
├── .streamlit/
│   ├── config.toml         ← Theme configuration (forces clean light mode contrasts)
│   └── secrets.toml        ← Local API credentials (ignored by Git)
```

### 🛠️ Core Technologies Used:
* **Frontend UI**: **Streamlit** (Customized with glassmorphic CSS overlays, smooth scale hover transitions, and a stateful button-driven tab navigation that prevents client-side resets).
* **Machine Learning**: **Scikit-Learn** (Random Forest Regressor + MinMaxScaler to predict overall air quality indexes).
* **Visualizations**: **Plotly Graph Objects** (Responsive, transparent background gauges and horizontal/vertical charts synchronized with the official AQI color scheme).
* **LLM Integration**: **Multi-Provider REST Router** (Zero-dependency integration that directly communicates with OpenAI, Google Gemini, and Anthropic APIs via `requests` and handles key failover silently).

---

## ⚡ Silent Auto-Failover LLM Routing
To ensure the app always returns high-quality, conversational insights, the platform features a silent backend routing mechanism:
1. **Gemini 2.5 Flash** (Prioritized first due to free-tier availability and performance).
2. **OpenAI GPT-4o-Mini** (Queried second if Gemini keys are missing or restricted).
3. **Anthropic Claude 3.5 Sonnet** (Queried third).
4. **Rule-Based Fallback**: If no keys are configured, the app falls back to a preset local rule-based warning engine so it remains fully functional offline.

---

## 🚀 Setup & Installation (macOS)

### 1. Initialize Virtual Environment & Dependencies
```bash
# Clone the repository
cd aqi-prediction-platform

# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Random Forest Model
If you want to train or update the model parameters, run the training pipeline:
```bash
python train_model.py
```
This prints the MAE, MSE, and R² scores and generates `aqi_model.pkl` + `scaler.pkl` artifacts.

### 3. Add API Keys
Create a file at `.streamlit/secrets.toml` and configure your API keys:
```toml
# Get a free key at: https://aqicn.org/data-platform/token/
AQICN_TOKEN = "your_aqicn_token_here"

# Set one or more of these AI keys:
GEMINI_API_KEY = "your_google_ai_studio_key"
OPENAI_API_KEY = "your_openai_api_key"
```

### 4. Run the Platform
```bash
streamlit run app.py
```
Access the dashboard in your browser at: `http://localhost:8501`.

---

## 🌐 Deployment to Streamlit Cloud

To host the app online for free on **Streamlit Community Cloud**:
1. Push your code to your GitHub repository (your local `.streamlit/secrets.toml` is ignored via `.gitignore` to prevent leaking keys).
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **New App**, select your repository, branch (`main`), and path (`app.py`).
4. Click **Advanced Settings** and add your secrets:
   ```toml
   AQICN_TOKEN = "your_api_token"
   GEMINI_API_KEY = "your_gemini_key"
   ```
5. Click **Deploy**. Your app will compile and generate its own public shareable URL!
