"""
Schema Sentinel - Crashes Dataset Conditioning
----------------------------------------------
"""

import pandas as pd
import os

# --------------------------
# CONFIGURATION
# --------------------------
RAW_DATA_PATH = ""
OUTPUT_PATH = ""
os.makedirs(OUTPUT_PATH, exist_ok=True)

INPUT_FILE = f"{RAW_DATA_PATH}/Motor_Vehicle_Collisions_-_Crashes_20251111.parquet"
OUTPUT_FILE = f"{OUTPUT_PATH}/crashes_POST_conditioning.parquet"

print("=" * 80)
print("CRASHES DATASET CONDITIONING - PIPELINE")
print("=" * 80)

# --------------------------
# STEP 1: LOAD & PROFILE
# --------------------------
print("\nSTEP 1: Loading crashes parquet file...")
crash_raw = pd.read_parquet(INPUT_FILE)

print(f"Records: {len(crash_raw):,}")
print(f"Columns: {len(crash_raw.columns)}")
print("\nSample of column names:")
print(crash_raw.columns.tolist()[:20])

# --------------------------
# STEP 2: AUTO-DETECT COLLISION ID COLUMN
# --------------------------
print("\nSTEP 2: Detecting collision ID column...")

id_col_candidates = [
    c for c in crash_raw.columns
    if c.strip().lower().replace(" ", "_") in ["collision_id", "collisionid"]
]

if id_col_candidates:
    id_col = id_col_candidates[0]
    crash_raw = crash_raw.rename(columns={id_col: "collision_id"})
    print(f"Detected and standardized collision ID column: {id_col} -> collision_id")
else:
    raise KeyError(
        f"No collision ID column found. Available columns: {list(crash_raw.columns)}"
    )

null_id = crash_raw["collision_id"].isna().sum()
dup_id = crash_raw["collision_id"].duplicated().sum()
print(f"Null collision_id: {null_id:,}")
print(f"Duplicate collision_id: {dup_id:,}")

# --------------------------
# STEP 3: STANDARDIZE DATES
# --------------------------
print("\nSTEP 3: Standardizing crash_date and merge_date...")

# Identify potential crash date column
date_col_candidates = [
    c for c in crash_raw.columns
    if c.strip().lower().replace(" ", "_") in ["crash_date", "crashdate"]
]
if date_col_candidates:
    src_date_col = date_col_candidates[0]
else:
    src_date_col = "CRASH DATE" if "CRASH DATE" in crash_raw.columns else None

if src_date_col is None:
    raise KeyError("Could not detect a crash date column in dataset.")

print(f"Detected crash date column: {src_date_col}")

crash_df = crash_raw.copy()
crash_df["crash_date"] = pd.to_datetime(crash_df[src_date_col], errors="coerce")
crash_df["merge_date"] = crash_df["crash_date"].dt.date

print(f"Date range: {crash_df['crash_date'].min()} -> {crash_df['crash_date'].max()}")

# --------------------------
# STEP 4: SELECT & RENAME ANALYTICAL COLUMNS
# --------------------------
print("\nSTEP 4: Selecting standardized analytical columns...")

def pick(candidates):
    """Return the first matching column from a list of candidates."""
    for name in candidates:
        if name in crash_df.columns:
            return name
    return None

cols_map = {
    "collision_id": "collision_id",
    "crash_date": "crash_date",
    "merge_date": "merge_date",
    "crash_time": pick(["CRASH TIME", "crash_time"]),
    "borough": pick(["BOROUGH", "borough"]),
    "zip_code": pick(["ZIP CODE", "zip_code"]),
    "latitude": pick(["LATITUDE", "latitude"]),
    "longitude": pick(["LONGITUDE", "longitude"]),
    "on_street_name": pick(["ON STREET NAME", "on_street_name"]),
    "cross_street_name": pick(["CROSS STREET NAME", "cross_street_name"]),
    "number_of_persons_injured": pick(["NUMBER OF PERSONS INJURED", "number_of_persons_injured"]),
    "number_of_persons_killed": pick(["NUMBER OF PERSONS KILLED", "number_of_persons_killed"]),
    "number_of_pedestrians_injured": pick(["NUMBER OF PEDESTRIANS INJURED", "number_of_pedestrians_injured"]),
    "number_of_cyclist_injured": pick(["NUMBER OF CYCLIST INJURED", "number_of_cyclist_injured"]),
    "number_of_motorist_injured": pick(["NUMBER OF MOTORIST INJURED", "number_of_motorist_injured"]),
    "contributing_factor_vehicle_1": pick(["CONTRIBUTING FACTOR VEHICLE 1", "contributing_factor_vehicle_1"]),
}

# Keep only existing columns
final_cols = [src for src in cols_map.values() if src is not None]

crashes_clean = crash_df[final_cols].copy()

# Standardize column names
rename_map = {
    cols_map["crash_time"]: "crash_time" if cols_map["crash_time"] else None,
    cols_map["borough"]: "borough" if cols_map["borough"] else None,
    cols_map["zip_code"]: "zip_code" if cols_map["zip_code"] else None,
    cols_map["latitude"]: "latitude" if cols_map["latitude"] else None,
    cols_map["longitude"]: "longitude" if cols_map["longitude"] else None,
    cols_map["on_street_name"]: "on_street_name" if cols_map["on_street_name"] else None,
    cols_map["cross_street_name"]: "cross_street_name" if cols_map["cross_street_name"] else None,
    cols_map["number_of_persons_injured"]: "number_of_persons_injured" if cols_map["number_of_persons_injured"] else None,
    cols_map["number_of_persons_killed"]: "number_of_persons_killed" if cols_map["number_of_persons_killed"] else None,
    cols_map["number_of_pedestrians_injured"]: "number_of_pedestrians_injured" if cols_map["number_of_pedestrians_injured"] else None,
    cols_map["number_of_cyclist_injured"]: "number_of_cyclist_injured" if cols_map["number_of_cyclist_injured"] else None,
    cols_map["number_of_motorist_injured"]: "number_of_motorist_injured" if cols_map["number_of_motorist_injured"] else None,
    cols_map["contributing_factor_vehicle_1"]: "contributing_factor_vehicle_1" if cols_map["contributing_factor_vehicle_1"] else None,
}

rename_map = {k: v for k, v in rename_map.items() if k and v}
crashes_clean = crashes_clean.rename(columns=rename_map)

print(f"Final crashes columns: {list(crashes_clean.columns)}")
print(f"Final records: {len(crashes_clean):,}")

# --------------------------
# STEP 5: QUALITY CHECKS
# --------------------------
print("\nSTEP 5: Running quality checks...")

null_ids = crashes_clean["collision_id"].isna().sum()
dups = crashes_clean["collision_id"].duplicated().sum()

print(f"Null collision_id: {null_ids:,}")
print(f"Duplicate collision_id: {dups:,}")

assert crashes_clean["merge_date"].isna().sum() == 0, "merge_date has nulls!"
print("merge_date valid: no nulls detected.")

print("Basic validation passed.")

# --------------------------
# STEP 6: SAVE CONDITIONED DATASET
# --------------------------
print("\nSTEP 6: Saving conditioned crashes dataset...")
crashes_clean.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")

print("\n" + "=" * 80)
print("CRASHES CONDITIONING COMPLETE")
print("=" * 80)

shared-datasets/pipeline_project_data/NYC_data/schema_sentinel_integrated.parquet