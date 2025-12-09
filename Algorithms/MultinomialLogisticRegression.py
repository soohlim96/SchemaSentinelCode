import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Pandas display options
# ------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

# ------------------------------------------------
# 1. Define columns needed & load ONLY those
# ------------------------------------------------
file_path = r"C:\Users\SooLim\OneDrive\CS504 007\schema_sentinel_last5yrs.parquet"

cols_needed = [
    "collision_id",
    "driver_license_status",
    "crash_time",
    "collision_severity",
]

df_model = pd.read_parquet(file_path, columns=cols_needed)
print("Shape after loading selected columns:", df_model.shape)

# ------------------------------------------------
# 2. Drop rows with missing key fields
# ------------------------------------------------
df_model = df_model.dropna(
    subset=["collision_id", "driver_license_status", "crash_time", "collision_severity"]
)
print("Shape after dropping missing key fields:", df_model.shape)

# ------------------------------------------------
# 3. Convert crash_time to datetime and extract hour (0–23)
# ------------------------------------------------
df_model["crash_time"] = pd.to_datetime(df_model["crash_time"], errors="coerce")
df_model = df_model.dropna(subset=["crash_time"])

df_model["crash_hour"] = df_model["crash_time"].dt.hour

# ------------------------------------------------
# 4. Define 7 time-of-day ranges
# ------------------------------------------------
bins = [0, 4, 7, 10, 16, 19, 22, 24]

labels_pretty = [
    "Late Night (00:00–03:59)",
    "Early Morning (04:00–06:59)",
    "AM Peak (07:00–09:59)",
    "Midday (10:00–15:59)",
    "PM Peak (16:00–18:59)",
    "Evening (19:00–21:59)",
    "Late Evening (22:00–23:59)",
]

df_model["time_range_7"] = pd.cut(
    df_model["crash_hour"],
    bins=bins,
    right=False,
    labels=labels_pretty
)

df_model = df_model.dropna(subset=["time_range_7"])

# ------------------------------------------------
# 5. Deduplicate per collision_id
# ------------------------------------------------
df_model = df_model.sort_values(["collision_id", "crash_time"])
df_model = df_model.drop_duplicates(subset="collision_id", keep="first")

print("Number of unique collisions:", df_model["collision_id"].nunique())
print("Remaining duplicate collision_ids:",
      df_model["collision_id"].duplicated().sum())

# ------------------------------------------------
# 6. Keep only 3 severity categories
# ------------------------------------------------
valid_severity = ["No Injury Collision", "Injury Collision", "Fatal Collision"]
df_model = df_model[df_model["collision_severity"].isin(valid_severity)].copy()

print("\nCollision severity counts:")
print(df_model["collision_severity"].value_counts().to_string())

# ------------------------------------------------
# 7. Build X and y
# ------------------------------------------------
X = df_model[["driver_license_status", "time_range_7"]].copy()

X = pd.get_dummies(
    X,
    columns=["driver_license_status", "time_range_7"],
    drop_first=True      # baseline: Licensed + Late Night
)

y = df_model["collision_severity"]

print("\nPredictor columns used in the model:")
print(X.columns.tolist())
print("\nOutcome classes:")
print(y.unique())

# ------------------------------------------------
# 8. Fit multinomial logistic regression
# ------------------------------------------------
mlr = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced"
)

mlr.fit(X, y)

print("\nModel fit complete.")
print("Classes (row order of coef_):", mlr.classes_)
print("Coefficient matrix shape (n_classes, n_features):", mlr.coef_.shape)

# ------------------------------------------------
# 9. Coefficients + Odds Ratios
# ------------------------------------------------
coef_df = pd.DataFrame(
    mlr.coef_,
    index=mlr.classes_,
    columns=X.columns
)
odds_ratio_df = np.exp(coef_df)

print("\nCoefficients (log-odds):")
print(coef_df.round(3).to_string())
print("\nOdds Ratios (exp(coef)):")
print(odds_ratio_df.round(3).to_string())

# ------------------------------------------------
# 10. In-sample performance 
# ------------------------------------------------
y_pred = mlr.predict(X)
print("\nClassification report (in-sample, just for reference):")
print(classification_report(y, y_pred))

# ------------------------------------------------
# 11. Heatmap: Driver License Status (baseline = Licensed)
# ------------------------------------------------
license_cols = [c for c in odds_ratio_df.columns
                if c.startswith("driver_license_status_")]

or_license = odds_ratio_df[license_cols].copy()

# Clean column names: Permit, Unlicensed
or_license.columns = [
    col.replace("driver_license_status_", "")
       .replace("_", " ").title()
    for col in license_cols
]

plt.figure(figsize=(6, 3))
ax = sns.heatmap(or_license.round(2), annot=True, fmt=".2f",
                 cmap="magma", cbar=True)

# Main title, with some padding
plt.title("Odds Ratios for Collision Severity by Driver License Status",
          fontsize=14, pad=20)

# Baseline text: a bit above heatmap, under title
plt.text(
    0.5, 1.02,
    "Baseline Category: Licensed Drivers",
    ha="center", va="bottom",
    transform=ax.transAxes,
    fontsize=10
)

ax.set_ylabel("Collision Severity")
ax.set_xlabel("Driver License Status")

plt.tight_layout()
plt.show()

# ------------------------------------------------
# 12. Heatmap: Time of Day (baseline = Late Night)
# ------------------------------------------------
time_cols = [c for c in odds_ratio_df.columns
             if c.startswith("time_range_7_")]

or_time = odds_ratio_df[time_cols].copy()

# Nicer column names: drop prefix, add line break before parentheses
or_time.columns = [
    col.replace("time_range_7_", "")
       .replace(" (", "\n(")
    for col in time_cols
]

plt.figure(figsize=(12, 3))
ax2 = sns.heatmap(or_time.round(2), annot=True, fmt=".2f",
                  cmap="magma", cbar=True)

plt.title("Odds Ratios for Collision Severity by Time of Day",
          fontsize=14, pad=20)

plt.text(
    0.5, 1.02,
    "Baseline Category: Late Night (00:00–03:59)",
    ha="center", va="bottom",
    transform=ax2.transAxes,
    fontsize=10
)

ax2.set_ylabel("Collision Severity")
ax2.set_xlabel("Time of Day")

plt.tight_layout()
plt.show()