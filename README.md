# 🌍 AQI Decision Intelligence Platform
> An AI-powered air quality decision tool: simulate pollutant scenarios, pull real-time
> city AQI, see a multi-day forecast, and ask a Gemini-powered assistant what to actually
> *do* about it — not just what the numbers are.

**Who it's for:** parents deciding whether kids can play outside, people with asthma
planning their day, runners timing outdoor workouts, and anyone who wants a straight
answer instead of a raw pollutant table.

---

## ✨ What's in this version

| Tab | What it does |
|---|---|
| 🎛️ Simulator | Original Random Forest AQI predictor — adjust pollutant sliders, get an instant prediction + AI advisory |
| 🌐 Live City AQI | Real-time data ingestion from the AQICN global monitoring network for any city |
| 📅 Forecast | Multi-day AQI trend pulled from the live feed's forecast data, with a risk-direction summary |
| 🤖 Ask AI | Free-form natural language Q&A grounded in current/live/forecast data, personalized by user profile (asthma, parent, elderly, athlete, general) |

Every tab includes a **Gemini-generated advisory** — a specific, actionable
recommendation, not just a category label. If no Gemini key is configured, the app
falls back to a rule-based advisory so it still works end-to-end.

---

## 📁 Folder Structure

```
aqi_platform/
├── app.py               ← Streamlit dashboard (4 tabs)
├── live_data.py          ← Real-time AQI ingestion (AQICN API)
├── forecast_engine.py     ← Multi-day forecast extraction & trend logic
├── ai_advisor.py           ← Gemini NL Q&A + advisory generation
├── train_model.py        ← Model training script
├── requirements.txt      ← Python dependencies
├── aqi_data.csv          ← Your dataset (auto-generated if missing)
├── aqi_model.pkl         ← Saved model  (created after training)
└── scaler.pkl            ← Saved scaler (created after training)
```

---

## 🚀 Setup Instructions (macOS)

### Step 1 – Create & activate a virtual environment
```bash
cd aqi_platform
python3 -m venv venv
source venv/bin/activate
```

### Step 2 – Install dependencies
```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install streamlit scikit-learn plotly pandas numpy joblib
```

### Step 3 – (Optional) Add your dataset
Place your CSV as `aqi_data.csv` in the project folder.
Expected columns (case-insensitive, whitespace-tolerant):
```
co aqi value, ozone aqi value, no2 aqi value, pm2.5 aqi value, aqi value
```
If the file is missing, the training script auto-generates synthetic demo data.

### Step 4 – Train the model
```bash
python train_model.py
```
This prints MAE / MSE / R² metrics and saves `aqi_model.pkl` + `scaler.pkl`.

### Step 5 – (Optional but recommended) Set API keys
For the live city AQI and Gemini AI advisories, set these as environment variables
(or add them to `.streamlit/secrets.toml`):

```bash
export AQICN_TOKEN="your_free_aqicn_token"      # https://aqicn.org/data-platform/token/
export GEMINI_API_KEY="your_gemini_api_key"     # https://aistudio.google.com/apikey
```

The app runs fully without these — the Simulator tab always works, and AI advisories
fall back to a rule-based response if no Gemini key is set.

### Step 6 – Launch the Streamlit app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎛️ How to Use the App
1. Use the **sidebar sliders** to set CO, Ozone, NO₂, and PM 2.5 values.
2. The **Predicted AQI** card updates instantly with a color-coded category.
3. The **Gauge Chart** gives a visual sense of severity.
4. The **Feature Importance** chart shows which pollutant drives predictions most.
5. Scroll down for the **About the Data** section explaining the AQI formula.

---

## 📊 AQI Categories
| Range    | Category                    | Color  |
|----------|-----------------------------|--------|
| 0–50     | Good                        | 🟢 Green  |
| 51–100   | Moderate                    | 🟡 Yellow |
| 101–150  | Unhealthy for Sensitive Groups | 🟠 Orange |
| 151–200  | Unhealthy                   | 🔴 Red    |
| 201–300  | Very Unhealthy              | 🟣 Purple |
| 301–500  | Hazardous                   | ☠️ Maroon |

---

## 🛑 Deactivate Environment (when done)
```bash
deactivate
```
