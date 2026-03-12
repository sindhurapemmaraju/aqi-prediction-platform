# 🌿 AI-Driven AQI Analytics Platform
> Predicts Air Quality Index using a Random Forest model with an interactive Streamlit dashboard.

---

## 📁 Folder Structure

```
aqi_platform/
├── app.py              ← Streamlit dashboard
├── train_model.py      ← Model training script
├── requirements.txt    ← Python dependencies
├── aqi_data.csv        ← Your dataset (auto-generated if missing)
├── aqi_model.pkl       ← Saved model  (created after training)
└── scaler.pkl          ← Saved scaler (created after training)
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

### Step 5 – Launch the Streamlit app
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
