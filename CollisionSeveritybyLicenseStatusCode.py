import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------
# 1. Load data
# ------------------------------------------------
file_path = r"C:\Users\SooLim\OneDrive\CS504 007\schema_sentinel_last5yrs.parquet"
df = pd.read_parquet(file_path)

# ------------------------------------------------
# 2. Keep only rows with needed fields (driver + severity)
# ------------------------------------------------
df = df[
    df["driver_license_status"].notna()
    & df["collision_severity"].notna()
]

# Optional: drop 'Unknown' severity
if "Unknown" in df["collision_severity"].unique():
    df = df[df["collision_severity"] != "Unknown"]

# ------------------------------------------------
# 3. Ensure ONE ROW PER COLLISION (collision-level dataset)
# ------------------------------------------------
df = df.sort_values(["collision_id"])
df = df.drop_duplicates(subset=["collision_id"], keep="first")

# ------------------------------------------------
# 3a. Show counts
# ------------------------------------------------
license_counts = df["driver_license_status"].value_counts().sort_index()
print("Number of collisions by driver license status:")
print(license_counts)
print()

severity_counts = df["collision_severity"].value_counts().sort_index()
print("Number of collisions by severity outcome:")
print(severity_counts)
print()

# ------------------------------------------------
# 4. Crosstab: driver_license_status × collision_severity
# ------------------------------------------------
ct = pd.crosstab(
    df["driver_license_status"],
    df["collision_severity"]
)

severity_order = ["No Injury Collision", "Injury Collision", "Fatal Collision"]
severity_order = [s for s in severity_order if s in ct.columns]

ct = ct.reindex(columns=severity_order)

print("Collision counts by license status and severity:")
print(ct)
print()

ct_with_totals = ct.copy()
ct_with_totals["Total Collisions (row)"] = ct_with_totals.sum(axis=1)
ct_with_totals.loc["Total Across Groups"] = ct_with_totals.sum(axis=0)
print(ct_with_totals)
print()

# Row-wise proportions (one collision = one record)
ct_prop = ct.div(ct.sum(axis=1), axis=0)

# ------------------------------------------------
# 5. Heatmap of proportions (collision-level)
# ------------------------------------------------
plt.figure(figsize=(8, 5))

plt.imshow(ct_prop.values, aspect="auto")
plt.colorbar(label="Proportion of collisions")

plt.xticks(
    ticks=np.arange(len(ct_prop.columns)),
    labels=ct_prop.columns,
    rotation=45,
    ha="right"
)
plt.yticks(
    ticks=np.arange(len(ct_prop.index)),
    labels=ct_prop.index
)

plt.xlabel("Collision severity")
plt.ylabel("Driver license status")
plt.title("Percentage Distribution of Collision Severity Within Each License Status Group")

# Add percentage labels
for i in range(ct_prop.shape[0]):        # license categories
    for j in range(ct_prop.shape[1]):    # severity types
        value = ct_prop.iloc[i, j] * 100
        plt.text(
            j, i,
            f"{value:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=9
        )

plt.tight_layout()
plt.show()