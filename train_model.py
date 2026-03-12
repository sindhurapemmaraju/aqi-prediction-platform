"""
train_model.py  –  AQI Prediction Model Training Script
AI-Driven Analytics Platform | Data Science Project
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load & Clean ───────────────────────────────────────────────────────────

print("=" * 55)
print("  AQI Prediction Model – Training Pipeline")
print("=" * 55)

CSV_PATH = "aqi_data.csv"          # place your CSV in the same folder

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    # ── Synthetic demo data so the script runs out-of-the-box ────────────────
    print(f"\n[INFO] '{CSV_PATH}' not found → generating synthetic demo data.\n")
    np.random.seed(42)
    n = 2_000
    co    = np.random.uniform(0,  10, n)
    ozone = np.random.uniform(0,  80, n)
    no2   = np.random.uniform(0,  60, n)
    pm25  = np.random.uniform(0, 250, n)
    aqi   = (0.35 * pm25 + 0.25 * ozone + 0.20 * co * 10 + 0.15 * no2
             + np.random.normal(0, 8, n)).clip(0, 500)
    df = pd.DataFrame({
        "co aqi value":     co,
        "ozone aqi value":  ozone,
        "no2 aqi value":    no2,
        "pm2.5 aqi value":  pm25,
        "aqi value":        aqi,
    })
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Demo CSV saved as '{CSV_PATH}'.\n")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

print(f"[DATA] Raw shape : {df.shape}")

# Drop rows with any NaN
df.dropna(inplace=True)
print(f"[DATA] Clean shape: {df.shape}\n")

# ── 2. Features & Target ──────────────────────────────────────────────────────

FEATURES = ["co aqi value", "ozone aqi value", "no2 aqi value", "pm2.5 aqi value"]
TARGET   = "aqi value"

X = df[FEATURES].values
y = df[TARGET].values

# ── 3. Train / Test Split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[SPLIT] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── 4. Feature Scaling ────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train Random Forest ────────────────────────────────────────────────────

print("\n[TRAIN] Fitting Random Forest Regressor …")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)
print("[TRAIN] Done.\n")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("─" * 40)
print("  Model Performance Metrics")
print("─" * 40)
print(f"  MAE   : {mae:.4f}")
print(f"  MSE   : {mse:.4f}")
print(f"  RMSE  : {rmse:.4f}")
print(f"  R²    : {r2:.4f}")
print("─" * 40)

# Feature importances (for reference)
importances = model.feature_importances_
print("\n  Feature Importances:")
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    print(f"    {feat:<25} {imp:.4f}")

# ── 7. Save Artifacts ─────────────────────────────────────────────────────────

joblib.dump(model,  "aqi_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n[SAVED] aqi_model.pkl  ✓")
print("[SAVED] scaler.pkl     ✓")
print("\n[DONE]  Training complete. Run 'streamlit run app.py' to launch the app.")
