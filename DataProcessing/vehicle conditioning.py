import pandas as pd
import numpy as np
import os

# Set paths
RAW_DATA_PATH = ''
OUTPUT_PATH = ''

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*80)
print("VEHICLE DATASET CONDITIONING - COMPLETE PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\nSTEP 1: Loading raw vehicle data...")
vehicles_raw = pd.read_parquet(f'{RAW_DATA_PATH}/vehicles_full.parquet')

print(f"Raw vehicle data loaded: {len(vehicles_raw):,} records")
print(f"Columns: {len(vehicles_raw.columns)}")
print(f"\nFirst 5 rows:")
print(vehicles_raw[['collision_id', 'crash_date', 'vehicle_type']].head())

# Save pre-conditioning snapshot
vehicles_raw.to_parquet(f'{OUTPUT_PATH}/vehicles_PRE_conditioning.parquet', index=False)
print(f"\nPre-conditioning dataset saved: vehicles_PRE_conditioning.parquet")

# ============================================================================
# STEP 2: DATE STANDARDIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATE STANDARDIZATION")
print("="*80)

# Create working copy
vehicles_df = vehicles_raw.copy()

print("\nBEFORE:")
print(f"crash_date type: {vehicles_df['crash_date'].dtype}")
print(f"Sample dates: {vehicles_df['crash_date'].head(3).tolist()}")

# Convert to datetime
vehicles_df['crash_date'] = pd.to_datetime(vehicles_df['crash_date'])

# Create merge key
vehicles_df['merge_date'] = vehicles_df['crash_date'].dt.date

print("\nAFTER:")
print(f"crash_date type: {vehicles_df['crash_date'].dtype}")
print(f"merge_date created: {vehicles_df['merge_date'].head(3).tolist()}")

date_null_count = vehicles_df['crash_date'].isna().sum()
print(f"\nValidation: {date_null_count:,} null dates ({date_null_count/len(vehicles_df)*100:.2f}%)")

# ============================================================================
# STEP 3: CASE NORMALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: CASE NORMALIZATION")
print("="*80)

# Store original count
original_unique = vehicles_df['vehicle_type'].nunique()

print("\nBEFORE Case Normalization:")
print(f"Unique vehicle types: {original_unique:,}")
print("\nSample case variations (bus example):")
bus_variants = vehicles_df[vehicles_df['vehicle_type'].str.contains('bus', case=False, na=False)]['vehicle_type'].value_counts().head(10)
print(bus_variants)

# Apply case normalization
vehicles_df['vehicle_type'] = vehicles_df['vehicle_type'].str.title().str.strip()

# Store normalized count
normalized_unique = vehicles_df['vehicle_type'].nunique()

print("\nAFTER Case Normalization:")
print(f"Unique vehicle types: {normalized_unique:,}")
print("\nBus variants after normalization:")
bus_normalized = vehicles_df[vehicles_df['vehicle_type'].str.contains('Bus', case=False, na=False)]['vehicle_type'].value_counts().head(5)
print(bus_normalized)

reduction = original_unique - normalized_unique
print(f"\nReduction: {original_unique:,} -> {normalized_unique:,} (-{reduction:,} duplicates)")

# ============================================================================
# STEP 4: SEMANTIC CONSOLIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: SEMANTIC CONSOLIDATION")
print("="*80)

print("\nBEFORE Semantic Consolidation:")
print("\nSedans:")
sedan_before = vehicles_df[vehicles_df['vehicle_type'].str.contains('Sedan|Passenger Vehicle', case=False, na=False)]['vehicle_type'].value_counts().head(5)
print(sedan_before)

print("\nPickup Trucks:")
pickup_before = vehicles_df[vehicles_df['vehicle_type'].str.contains('Pick|Pk', case=False, na=False)]['vehicle_type'].value_counts()
print(pickup_before)

print("\nSUV/Wagon:")
suv_before = vehicles_df[vehicles_df['vehicle_type'].str.contains('Station Wagon|Sport Utility', case=False, na=False)]['vehicle_type'].value_counts()
print(suv_before)

# Define consolidation mapping
consolidation_map = {
    # Sedan group
    '4 Dr Sedan': 'Sedan',
    '2 Dr Sedan': 'Sedan',
    'Passenger Vehicle': 'Sedan',
    
    # Pickup group
    'Pick-Up Truck': 'Pickup Truck',
    'Pk': 'Pickup Truck',
    
    # SUV/Wagon group
    'Sport Utility / Station Wagon': 'SUV/Station Wagon',
    'Station Wagon/Sport Utility Vehicle': 'SUV/Station Wagon',
}

# Apply consolidation
vehicles_df['vehicle_type'] = vehicles_df['vehicle_type'].replace(consolidation_map)

print("\n" + "-"*80)
print("AFTER Semantic Consolidation:")
print("\nSedans:")
sedan_after = vehicles_df[vehicles_df['vehicle_type'] == 'Sedan']['vehicle_type'].value_counts()
print(f"Sedan: {sedan_after.sum():,}")

print("\nPickup Trucks:")
pickup_after = vehicles_df[vehicles_df['vehicle_type'] == 'Pickup Truck']['vehicle_type'].value_counts()
print(f"Pickup Truck: {pickup_after.sum():,}")

print("\nSUV/Wagon:")
suv_after = vehicles_df[vehicles_df['vehicle_type'] == 'SUV/Station Wagon']['vehicle_type'].value_counts()
print(f"SUV/Station Wagon: {suv_after.sum():,}")

print("\n" + "-"*80)
print("CONSOLIDATION IMPACT:")
total_consolidated = 0
for original, consolidated in consolidation_map.items():
    # Count in normalized data (after title case)
    count = (vehicles_raw['vehicle_type'].str.title().str.strip() == original).sum()
    if count > 0:
        print(f"  {original:40} -> {consolidated:20} ({count:>10,} records)")
        total_consolidated += count

print(f"\nTotal records consolidated: {total_consolidated:,}")

# ============================================================================
# STEP 5: VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: VALIDATION")
print("="*80)

# Check for data loss
assert len(vehicles_raw) == len(vehicles_df), "Record count changed!"
print(f"Record count preserved: {len(vehicles_df):,}")

# Check for new nulls
original_nulls = vehicles_raw['vehicle_type'].isna().sum()
new_nulls = vehicles_df['vehicle_type'].isna().sum()
assert original_nulls == new_nulls, "New null values introduced!"
print(f"Null values unchanged: {new_nulls:,}")

# Check unique types
final_unique = vehicles_df['vehicle_type'].nunique()
print(f"\nFinal unique vehicle types: {final_unique:,}")

# Top 20 vehicle types
print("\nTop 20 vehicle types after conditioning:")
print(vehicles_df['vehicle_type'].value_counts().head(20))

print("\nValidation: All checks passed!")

# ============================================================================
# STEP 6: SAVE POST-CONDITIONING
# ============================================================================
print("\n" + "="*80)
print("STEP 6: SAVE POST-CONDITIONING DATASET")
print("="*80)

# Save full dataset
vehicles_df.to_parquet(f'{OUTPUT_PATH}/vehicles_POST_conditioning.parquet', index=False)

print(f"\nPost-conditioning dataset saved: vehicles_POST_conditioning.parquet")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VEHICLE CONDITIONING SUMMARY")
print("="*80)
print(f"\nRecords: {len(vehicles_raw):,} -> {len(vehicles_df):,} (no change)")
print(f"Columns: {len(vehicles_raw.columns)} -> {len(vehicles_df.columns)} (added merge_date)")
print(f"\nVehicle type consolidation:")
print(f"  Original unique types: {original_unique:,}")
print(f"  After case normalization: {normalized_unique:,}")
print(f"  After semantic consolidation: {final_unique:,}")
print(f"  Total reduction: {original_unique - final_unique:,} types")
print(f"\nTop 3 categories:")
top3 = vehicles_df['vehicle_type'].value_counts().head(3)
for vtype, count in top3.items():
    pct = count / len(vehicles_df) * 100
    print(f"  {vtype}: {count:,} ({pct:.1f}%)")
print(f"\nTop 3 represent: {top3.sum() / len(vehicles_df) * 100:.1f}% of all records")

print("\n" + "="*80)
print("VEHICLE CONDITIONING COMPLETE!")
print("="*80)