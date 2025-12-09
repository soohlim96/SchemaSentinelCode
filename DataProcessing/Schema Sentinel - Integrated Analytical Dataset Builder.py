Schema Sentinel - Integrated Analytical Dataset Builder
-------------------------------------------------------
Integrates POST-conditioned Person, Vehicle, Weather, and Crashes datasets into
a single dataset for team analysis and reporting.

Inputs
- person_POST_conditioning.parquet
- vehicles_POST_conditioning.parquet
- weather_POST_conditioning.parquet
- crashes_POST_conditioning.parquet

Output:
- schema_sentinel_integrated.parquet
- schema_sentinel_integrated.csv
"""

import pandas as pd
import os

# --------------------------
# CONFIG (update path to where you stored to files)
# --------------------------
BASE_PATH = ""
COND_PATH = f""
OUT_PATH = f""
os.makedirs(OUT_PATH, exist_ok=True)

print("=" * 80)
print("SCHEMA SENTINEL - INTEGRATION PIPELINE")
print("=" * 80)

# --------------------------
# STEP 1: LOAD CONDITIONED DATASETS
# --------------------------
print("\nSTEP 1: Loading POST-conditioned datasets...")

person = pd.read_parquet(f"{COND_PATH}/person_POST_conditioning.parquet")
vehicles = pd.read_parquet(f"{COND_PATH}/vehicles_POST_conditioning.parquet")
weather = pd.read_parquet(f"{COND_PATH}/weather_POST_conditioning.parquet")
crashes = pd.read_parquet(f"{COND_PATH}/crashes_POST_conditioning.parquet")


# ------------------------------------------------------------------
# Standardize collision_id data type across datasets
# ------------------------------------------------------------------
for name, df in [("person", person), ("vehicles", vehicles), ("crashes", crashes)]:
    if "collision_id" in df.columns:
        df["collision_id"] = df["collision_id"].astype(str).str.strip()

print("\nCollision ID types standardized to string for person, vehicle, and crashes.")


print(f"Person:   {len(person):,} records")
print(f"Vehicle:  {len(vehicles):,} records")
print(f"Weather:  {len(weather):,} records")
print(f"Crashes:  {len(crashes):,} records")

# Basic column checks
for name, df, cols in [
    ("Person", person, ["collision_id", "merge_date"]),
    ("Vehicle", vehicles, ["collision_id", "merge_date"]),
    ("Weather", weather, ["merge_date"]),
    ("Crashes", crashes, ["collision_id", "merge_date"])
]:
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"{name} dataset missing required column: {c}")

# --------------------------
# STEP 2: MERGE PERSON + VEHICLE
# --------------------------
print("\nSTEP 2: Merging Person + Vehicle on (collision_id, merge_date)...")

pv = person.merge(
    vehicles,
    on=["collision_id", "merge_date"],
    how="inner",
    suffixes=("_person", "_vehicle")
)

print(f"Person-Vehicle merged: {len(pv):,} records")

# --------------------------
# STEP 3: ADD WEATHER (BY DATE)
# --------------------------
print("\nSTEP 3: Adding Weather on merge_date...")

pvw = pv.merge(
    weather,
    on="merge_date",
    how="left"
)

print(f"After Weather merge: {len(pvw):,} records")

# --------------------------
# STEP 4: ADD CRASHES (BY collision_id)
# --------------------------
print("\nSTEP 4: Adding Crashes on collision_id...")

keep_crash_cols = [
    "collision_id",
    "crash_date",
    "crash_time",
    "borough",
    "zip_code",
    "latitude",
    "longitude",
    "on_street_name",
    "cross_street_name",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "contributing_factor_vehicle_1"
]
keep_crash_cols = [c for c in keep_crash_cols if c in crashes.columns]

full = pvw.merge(
    crashes[keep_crash_cols],
    on="collision_id",
    how="left"
)

print(f"After Crashes merge: {len(full):,} records")

# --------------------------
# STEP 5: VALIDATIONS
# --------------------------
print("\nSTEP 5: Validating integrated dataset...")

if "injury_occurred" not in full.columns:
    raise KeyError("injury_occurred not found in integrated dataset (from person conditioning).")

null_cid = full["collision_id"].isna().sum()
print(f"Null collision_id in final dataset: {null_cid:,}")
assert null_cid == 0, "collision_id should not be null in integrated dataset."

print("Basic integrity checks passed.")

# --------------------------
# STEP 6: SAVE OUTPUTS
# --------------------------
print("\nSTEP 6: Saving integrated dataset...")

parquet_path = f"{OUT_PATH}/schema_sentinel_integrated.parquet"
csv_path = f"{OUT_PATH}/schema_sentinel_integrated.csv"

full.to_parquet(parquet_path, index=False)
full.to_csv(csv_path, index=False)

print(f"Saved Parquet: {parquet_path}")
print(f"Saved CSV:     {csv_path}")

print("\n" + "=" * 80)
print("INTEGRATION PIPELINE COMPLETE")
print("=" * 80)








================================================================================
SCHEMA SENTINEL - INTEGRATION PIPELINE
================================================================================

STEP 1: Loading POST-conditioned datasets...

Collision ID types standardized to string for person, vehicle, and crashes.
Person:   5,807,949 records
Vehicle:  4,448,313 records
Weather:  4,865 records
Crashes:  2,219,657 records

STEP 2: Merging Person + Vehicle on (collision_id, merge_date)...
Person-Vehicle merged: 12,344,659 records

STEP 3: Adding Weather on merge_date...
After Weather merge: 12,344,659 records

STEP 4: Adding Crashes on collision_id...
After Crashes merge: 12,344,659 records

STEP 5: Validating integrated dataset...
Null collision_id in final dataset: 0
Basic integrity checks passed.

STEP 6: Saving integrated dataset...
Saved Parquet: /home/jovyan/shared-datasets/nyc-collisions/integrated/schema_sentinel_integrated.parquet
Saved CSV:     /home/jovyan/shared-datasets/nyc-collisions/integrated/schema_sentinel_integrated.csv

================================================================================
INTEGRATION PIPELINE COMPLETE