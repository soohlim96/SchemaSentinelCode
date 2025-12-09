#Schema Sentinel - Add Injury and Collision Severity Factors
#-----------------------------------------------------------

import pandas as pd
import numpy as np
import os

# --------------------------
# CONFIGURATION
# --------------------------
BASE_PATH = "/"
INTEGRATED_FILE = f"{BASE_PATH}"
OUTPUT_FILE = f"{BASE_PATH}"

print("=" * 80)
print("SCHEMA SENTINEL - ADDING INJURY AND COLLISION SEVERITY FACTORS")
print("=" * 80)

# --------------------------
# STEP 1: LOAD DATASET
# --------------------------
print("\nSTEP 1: Loading integrated dataset...")
df = pd.read_parquet(INTEGRATED_FILE)
print(f"Loaded {len(df):,} records.")

# --------------------------
# STEP 2: CREATE PERSON-LEVEL INJURY SEVERITY
# --------------------------
print("\nSTEP 2: Creating person-level injury severity classification...")

df["person_injury"] = df["person_injury"].astype(str).str.title().fillna("Unspecified")

df["injury_severity"] = df["person_injury"].map({
    "Killed": "Fatality",
    "Injured": "Injury",
    "Unspecified": "No Injury"
})

# Fallback using binary variable if available
if "injury_occurred" in df.columns:
    df.loc[(df["injury_severity"].isna()) & (df["injury_occurred"] == 1), "injury_severity"] = "Injury"
    df.loc[(df["injury_severity"].isna()) & (df["injury_occurred"] == 0), "injury_severity"] = "No Injury"

print("Person-level injury_severity added.")
print(df["injury_severity"].value_counts(dropna=False))

# --------------------------
# STEP 3: CREATE COLLISION-LEVEL SEVERITY
# --------------------------
print("\nSTEP 3: Creating collision-level severity classification...")

if {"number_of_persons_injured", "number_of_persons_killed"}.issubset(df.columns):
    severity_df = (
        df.groupby("collision_id")
        .agg(
            persons_injured=("number_of_persons_injured", "max"),
            persons_killed=("number_of_persons_killed", "max")
        )
        .reset_index()
    )

    conditions = [
        severity_df["persons_killed"] > 0,
        (severity_df["persons_injured"] > 0) & (severity_df["persons_killed"] == 0),
        (severity_df["persons_injured"] == 0) & (severity_df["persons_killed"] == 0)
    ]
    choices = ["Fatal Collision", "Injury Collision", "No Injury Collision"]
    severity_df["collision_severity"] = np.select(conditions, choices, default="Unknown")

    df = df.merge(severity_df[["collision_id", "collision_severity"]], on="collision_id", how="left")
    print("Collision-level collision_severity added.")
    print(df["collision_severity"].value_counts(dropna=False))
else:
    print("Warning: Required crash severity columns not found. Skipping collision_severity creation.")

# --------------------------
# STEP 4: SAVE UPDATED DATASET
# --------------------------
print("\nSTEP 4: Saving dataset with severity factors...")

os.makedirs(f"{BASE_PATH}/conditioning", exist_ok=True)
df.to_parquet(OUTPUT_FILE, index=False)

print(f"Saved updated dataset: {OUTPUT_FILE}")

print("\n" + "=" * 80)
print("PROCESS COMPLETE - SEVERITY FACTORS ADDED")
print("=" * 80)