# ===============================
# 🚀 F1 Podium Prediction - Phase 2 (Leakage-Free Edition)
# ===============================

# 📦 Import Libraries
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import joblib

# 🎯 File Path
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "processed" / "F1_merged_clean_features.csv"

# ===============================
# 1️⃣ Load Data
# ===============================
df = pd.read_csv(file_path)
print("✅ Data Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# ===============================
# 2️⃣ Create Target (Podium)
# ===============================
if 'finish_position' not in df.columns:
    raise ValueError("Column 'finish_position' not found!")

df['Podium'] = df['finish_position'].apply(lambda x: 1 if x <= 3 else 0)
print("\n✅ Podium target created successfully!\n")

# ===============================
# 3️⃣ Drop Leakage & Irrelevant Columns
# ===============================
leaky_cols = [
    'finish_position',             # target info
    'roll3_finish_pos_global',     # correlated with finish
    'roll3_finish_pos_track',      # correlated with finish
    'raceId', 'driverId', 'constructorId', 'circuitId', 'date'
]
df = df.drop(columns=[c for c in leaky_cols if c in df.columns], errors='ignore')

# ===============================
# 4️⃣ Encode Categorical Columns
# ===============================
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

# ===============================
# 5️⃣ Split Data (Features / Target)
# ===============================
X = df.drop(columns=['Podium'])
y = df['Podium']
X = X.fillna(X.median())

# ===============================
# 6️⃣ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 7️⃣ Scale Features
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 8️⃣ Define the 4 Sexy ML Models 😎
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42)
}

# ===============================
# 9️⃣ Train & Evaluate
# ===============================
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    results.append([name, acc, prec, rec, f1, roc])

    print(f"\n==================== {name} ====================")
    print(classification_report(y_test, y_pred, digits=3))

# ===============================
# 🔟 Compare Model Performances
# ===============================
results_df = pd.DataFrame(
    results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
)

print("\n\n🏁 Model Comparison:")
print(results_df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True))

# ===============================
# 🎨 Bar Chart Comparison
# ===============================
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["F1-Score"], color='teal')
plt.title("F1-Score Comparison of ML Models (Leakage-Free)")
plt.ylabel("F1-Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ===============================
# ✅ Done
# ===============================
print("\n🎉 Phase 2 completed successfully — now with realistic results 😎")

# ===============================
# 🔍 Phase 2.5: Feature Importance Visualization
# ===============================

import matplotlib.pyplot as plt
import numpy as np

print("\n🔍 Analyzing Feature Importances...\n")

# Helper to plot feature importances
def plot_feature_importance(model, model_name, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(8,5))
        plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f"Top {top_n} Important Features - {model_name}")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{model_name} does not provide feature importances.\n")

# Feature names
feature_names = X.columns

# Plot for Random Forest
plot_feature_importance(models["Random Forest"], "Random Forest", feature_names)

# Plot for XGBoost
plot_feature_importance(models["XGBoost"], "XGBoost", feature_names)

print("\n✅ Feature Importance Visualizations Completed!\n")

# ===============================
# 💾 Save Model Comparison + Plots
# ===============================

# Save directory
save_dir = BASE_DIR / "results"
save_dir.mkdir(exist_ok=True)

# 1️⃣ Save model performance table
results_path = save_dir / "model_comparison.csv"
results_df.to_csv(results_path, index=False)
print(f"✅ Saved model comparison results to: {results_path}")

# 2️⃣ Save bar chart comparison
chart_path = save_dir / "model_f1_scores.png"
plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["F1-Score"], color="teal")
plt.title("F1-Score Comparison of ML Models (Leakage-Free)")
plt.ylabel("F1-Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"✅ Saved model F1-score chart to: {chart_path}")

# 3️⃣ Save feature importance charts
def save_feature_importance(model, model_name, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(8,5))
        plt.barh(range(len(indices)), importances[indices], align="center", color="teal")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f"Top {top_n} Important Features - {model_name}")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        out_path = save_dir / f"feature_importance_{model_name.replace(' ','_')}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved feature importance for {model_name} → {out_path}")

# Save both feature-importance charts
save_feature_importance(models["Random Forest"], "Random Forest", X.columns)
save_feature_importance(models["XGBoost"], "XGBoost", X.columns)

print("\n🎯 All results and plots successfully exported to your Phase2_Results folder!")


# Directory to save models
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

# Save each model
for name, model in models.items():
    model_filename = model_dir / f"{name.replace(' ', '_')}.pkl"
    joblib.dump(model, model_filename)
    print(f"✅ Saved model: {model_filename}")

print("\n🎯 All models saved successfully! You can reload them later for predictions.")

# ===============================
# 💾 Save Scaler
# ===============================

scaler_path = model_dir / "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✅ Saved scaler: {scaler_path}")
