# ===============================
# Phase 3 - Podium Prediction Using Saved XGBoost Model (Fixed Version)
# ===============================

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ===============================
# File Paths
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "XGBoost.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"
data_path = BASE_DIR / "data" / "processed" / "new_race_data.csv"

# ===============================
# Load Saved Model
# ===============================
model = joblib.load(model_path)
print(" Loaded XGBoost model successfully!")

# ===============================
# Load New Race Data
# ===============================
new_data = pd.read_csv(data_path)
print("\n New data shape:", new_data.shape)
print(new_data.head())

# Load scaler and get exact training feature order
scaler = joblib.load(scaler_path)
print(" Loaded saved scaler successfully!")

expected_features = list(scaler.feature_names_in_)
print("\nScaler expects:")
print(expected_features)

# ===============================
# Fix Missing Columns
# ===============================
missing = [col for col in expected_features if col not in new_data.columns]
if missing:
    print(f"\n Missing columns found in new data: {missing}")
    for col in missing:
        new_data[col] = 0  # Fill with neutral value

# Reorder columns to match training
new_data = new_data[expected_features]
print("\n Columns aligned successfully with training features!")

# ===============================
# Encode Categorical Columns (if any)
# ===============================
cat_cols = new_data.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    le = LabelEncoder()
    for col in cat_cols:
        new_data[col] = le.fit_transform(new_data[col].astype(str))

# ===============================
# Handle Missing Values
# ===============================
new_data = new_data.fillna(new_data.median())

print("\nFinal columns used for prediction:")
print(list(new_data.columns))

new_data_scaled = scaler.transform(new_data)

# ===============================
# Make Predictions
# ===============================
predictions = model.predict(new_data_scaled)
pred_probs = model.predict_proba(new_data_scaled)[:, 1]

new_data['Podium_Prediction'] = predictions
new_data['Podium_Probability'] = pred_probs
new_data['Podium_Label'] = new_data['Podium_Prediction'].apply(lambda x: "Yes (Top 3)" if x == 1 else "No")

# ===============================
# Save Predictions
# ===============================
prediction_dir = BASE_DIR / "data" / "predictions"
prediction_dir.mkdir(exist_ok=True)

output_path = prediction_dir / "Phase3_Predictions.csv"

new_data.to_csv(output_path, index=False)
print(f"\n Predictions saved to: {output_path}")

# ===============================
# Display Sample Predictions
# ===============================
print("\n Sample Predictions:")
print(new_data[['grid', 'quali_position', 'Podium_Label', 'Podium_Probability']].head())

# ===============================
# Optional: Plot Probability Distribution
# ===============================
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.hist(new_data['Podium_Probability'], bins=20, color='teal')
    plt.title("Predicted Podium Probability Distribution")
    plt.xlabel("Probability of Podium Finish")
    plt.ylabel("Number of Drivers")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plot skipped:", e)

print("\n Phase 3 prediction completed successfully!")
