# ===============================
# 🚀 F1 Podium Predictor (Streamlit App)
# ===============================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 🎯 Load Model + Scaler
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "XGBoost.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="🏎️ F1 Podium Predictor", page_icon="🏁", layout="centered")

# -------------------------------
# 💅 Title + Description
# -------------------------------
st.title("🏁 Formula 1 Podium Predictor")
st.markdown("""
Welcome to your **F1 ML Predictor App** built for your DPEL Project!  
Enter race details below and see if the driver will finish on the **podium (Top 3)** 🏆  
""")

st.divider()

# -------------------------------
# 🧮 Input Form
# -------------------------------
st.subheader("📋 Race Details")

col1, col2 = st.columns(2)
with col1:
    grid = st.number_input("Starting Grid Position", min_value=1, max_value=20, value=5)
    quali_position = st.number_input("Qualifying Position", min_value=1, max_value=20, value=4)
    points = st.number_input("Previous Points", min_value=0, max_value=50, value=10)
    year = st.number_input("Season Year", min_value=2000, max_value=2025, value=2023)

with col2:
    round_ = st.number_input("Round Number", min_value=1, max_value=25, value=10)
    quali_best_sec = st.number_input("Quali Best Time (sec)", min_value=70.0, max_value=100.0, value=85.50)
    roll3_points_track = st.number_input("Rolling Avg Points (Track)", min_value=0.0, max_value=25.0, value=14.0)
    roll3_quali_pos_track = st.number_input("Rolling Avg Quali Pos (Track)", min_value=1.0, max_value=20.0, value=3.5)

# Hidden / defaulted features for model completeness
circuitName = "Default Circuit"
roll3_quali_sec_track = st.number_input("Rolling Avg Quali Time (sec, track)", min_value=70.0, max_value=100.0, value=85.8)
roll3_points_global = st.number_input("Rolling Avg Points (Global)", min_value=0.0, max_value=25.0, value=12.5)
roll3_delta_quali_finish_track = st.number_input("Rolling Delta Quali-Finish", min_value=-10.0, max_value=10.0, value=0.5)

# -------------------------------
# 🧠 Prepare Input Data
# -------------------------------
input_data = pd.DataFrame([{
    'grid': grid,
    'quali_position': quali_position,
    'points': points,
    'year': year,
    'round': round_,
    'quali_best_sec': quali_best_sec,
    'circuitName': circuitName,
    'roll3_points_track': roll3_points_track,
    'roll3_quali_pos_track': roll3_quali_pos_track,
    'roll3_quali_sec_track': roll3_quali_sec_track,
    'roll3_points_global': roll3_points_global,
    'roll3_delta_quali_finish_track': roll3_delta_quali_finish_track
}])

# Encode categorical
cat_cols = input_data.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    le = LabelEncoder()
    for col in cat_cols:
        input_data[col] = le.fit_transform(input_data[col].astype(str))

expected_features = list(scaler.feature_names_in_)

for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_features]

input_scaled = scaler.transform(input_data)

# -------------------------------
# 🏁 Prediction
# -------------------------------
if st.button("🏎️ Predict Podium Finish"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("🔮 Prediction Result")

    if pred == 1:
        st.success(f"🎉 The driver is **LIKELY to finish on the PODIUM!** 🏆 (Probability: {prob*100:.2f}%)")
        st.balloons()
    else:
        st.error(f"💤 The driver is **unlikely to reach the podium.** (Probability: {prob*100:.2f}%)")

    # Probability meter
    st.progress(float(prob))

    # Show dataframe
    st.dataframe(input_data.style.highlight_max(axis=1, color='lightgreen'))

# -------------------------------
# 📊 Footer
# -------------------------------
st.divider()
st.caption("Built with ❤️ using Streamlit & XGBoost | DPEL Project Phase 4")