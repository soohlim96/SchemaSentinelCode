# ===============================
# RANDOM FOREST CLASSIFIER (Severity Prediction)
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -------------------------------
# Load Integrated Dataset
# (Person + Vehicle + Weather + Crash merged via Schema Sentinel)
# -------------------------------
df = pd.read_parquet("person_vehicle_weather.parquet")

# -------------------------------
# Target: severe injury = 1 (injured or killed)
# -------------------------------
df["severe"] = np.where(df["person_injury"].isin(["Injured", "Killed"]), 1, 0)

# -------------------------------
# Feature Subset (aligned with paper)
# -------------------------------
feature_cols = [
    "driver_license_status",
    "vehicle_type",
    "person_age",
    "person_sex",
    "weather_condition",
    "hour",                   # extracted crash hour
    "borough",
    "contributing_factor_1"
]

# Extract hour if needed
if "crash_time" in df.columns:
    df["hour"] = pd.to_datetime(df["crash_time"], format="%H:%M").dt.hour

df = df.dropna(subset=feature_cols + ["severe"])

X = df[feature_cols]
y = df["severe"]

# -------------------------------
# Train/Test Split (stratified due to imbalance)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------
# Preprocessing: OneHotEncoding for categorical features
# -------------------------------
categorical_features = [
    "driver_license_status",
    "vehicle_type",
    "person_sex",
    "weather_condition",
    "borough",
    "contributing_factor_1"
]

numeric_features = ["person_age", "hour"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# -------------------------------
# Model with Class Weighting (since severe outcomes are rare)
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

# -------------------------------
# Train the Model
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = pipeline.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Feature Importance (Top 20)
# -------------------------------
# Extract encoded feature names
encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
encoded_categories = encoder.get_feature_names_out(categorical_features)

all_feature_names = list(encoded_categories) + numeric_features

importances = pipeline.named_steps["classifier"].feature_importances_

feature_imp = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_imp.head(20))