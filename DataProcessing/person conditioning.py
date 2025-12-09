import pandas as pd
import numpy as np
import os

# Set paths
RAW_DATA_PATH = ''
OUTPUT_PATH = ''

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*80)
print("PERSON DATASET CONDITIONING - COMPLETE PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\nSTEP 1: Loading raw person data...")
person_raw = pd.read_parquet(f'{RAW_DATA_PATH}/person_full.parquet')

print(f"Raw person data loaded: {len(person_raw):,} records")
print(f"Columns: {len(person_raw.columns)}")
print(f"\nFirst 5 rows:")
print(person_raw[['collision_id', 'crash_date', 'person_injury']].head())

# Check person_injury distribution
print("\nOriginal person_injury distribution:")
print(person_raw['person_injury'].value_counts())

# Save pre-conditioning snapshot
person_raw.to_parquet(f'{OUTPUT_PATH}/person_PRE_conditioning.parquet', index=False)
print(f"\nPre-conditioning dataset saved: person_PRE_conditioning.parquet")

# ============================================================================
# STEP 2: DATE STANDARDIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATE STANDARDIZATION")
print("="*80)

# Create working copy
person_df = person_raw.copy()

print("\nBEFORE:")
print(f"crash_date type: {person_df['crash_date'].dtype}")
print(f"Sample dates: {person_df['crash_date'].head(3).tolist()}")

# Convert to datetime
person_df['crash_date'] = pd.to_datetime(person_df['crash_date'])

# Create merge key
person_df['merge_date'] = person_df['crash_date'].dt.date

print("\nAFTER:")
print(f"crash_date type: {person_df['crash_date'].dtype}")
print(f"merge_date created: {person_df['merge_date'].head(3).tolist()}")

date_null_count = person_df['crash_date'].isna().sum()
print(f"\nValidation: {date_null_count:,} null dates ({date_null_count/len(person_df)*100:.2f}%)")

# ============================================================================
# STEP 3: BINARY TARGET VARIABLE CREATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: BINARY TARGET VARIABLE CREATION")
print("="*80)

print("\nOriginal person_injury categories:")
injury_dist = person_df['person_injury'].value_counts()
print(injury_dist)
print("\nPercentages:")
print(person_df['person_injury'].value_counts(normalize=True) * 100)

# Define categorization function
def create_binary_target(injury_value):
    """
    Convert multi-category injury to binary
    1 = Injured or Killed
    0 = All other outcomes
    """
    if pd.isna(injury_value):
        return 0  # Treat missing as no injury
    elif injury_value in ['Injured', 'Killed']:
        return 1
    else:
        return 0

# Apply transformation
print("\nApplying binary transformation...")
person_df['injury_occurred'] = person_df['person_injury'].apply(create_binary_target)

# Show results
print("\n" + "-"*80)
print("Binary target variable created:")
print("\nDistribution:")
binary_dist = person_df['injury_occurred'].value_counts().sort_index()
print(binary_dist)

print("\nPercentages:")
binary_pct = person_df['injury_occurred'].value_counts(normalize=True).sort_index() * 100
print(binary_pct)

# Detailed breakdown
print("\n" + "-"*80)
print("Mapping from original to binary:")
mapping_table = person_df.groupby(['person_injury', 'injury_occurred']).size().reset_index(name='count')
mapping_table['percentage'] = mapping_table['count'] / len(person_df) * 100
mapping_table = mapping_table.sort_values(['injury_occurred', 'count'], ascending=[True, False])
print(mapping_table.to_string(index=False))

# Calculate class imbalance
no_injury = (person_df['injury_occurred'] == 0).sum()
injury = (person_df['injury_occurred'] == 1).sum()
ratio = no_injury / injury
print(f"\nClass imbalance ratio: {ratio:.1f}:1 (no injury : injury)")

# ============================================================================
# STEP 4: VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: VALIDATION")
print("="*80)

# Check for data loss
assert len(person_raw) == len(person_df), "Record count changed!"
print(f"Record count preserved: {len(person_df):,}")

# Check target variable
assert person_df['injury_occurred'].isna().sum() == 0, "Target variable has null values!"
print(f"Target variable complete: No null values")

# Verify binary values
assert set(person_df['injury_occurred'].unique()) == {0, 1}, "Target variable not binary!"
print(f"Target variable is binary: {sorted(person_df['injury_occurred'].unique())}")

# Check original injury field preserved
assert 'person_injury' in person_df.columns, "Original injury field was removed!"
print(f"Original person_injury field preserved")

# Summary statistics
print("\n" + "-"*80)
print("TARGET VARIABLE STATISTICS:")
print(f"\nTotal records: {len(person_df):,}")
print(f"No injury (0): {no_injury:,} ({no_injury/len(person_df)*100:.2f}%)")
print(f"Injury (1): {injury:,} ({injury/len(person_df)*100:.2f}%)")
print(f"Class imbalance: {ratio:.1f}:1")

print("\nValidation: All checks passed!")

# ============================================================================
# STEP 5: SAVE POST-CONDITIONING
# ============================================================================
print("\n" + "="*80)
print("STEP 5: SAVE POST-CONDITIONING DATASET")
print("="*80)

# Save full dataset
person_df.to_parquet(f'{OUTPUT_PATH}/person_POST_conditioning.parquet', index=False)

print(f"\nPost-conditioning dataset saved: person_POST_conditioning.parquet")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PERSON CONDITIONING SUMMARY")
print("="*80)
print(f"\nRecords: {len(person_raw):,} -> {len(person_df):,} (no change)")
print(f"Columns: {len(person_raw.columns)} -> {len(person_df.columns)} (added merge_date, injury_occurred)")
print(f"\nNew columns created:")
print(f"  - injury_occurred (binary target: 0/1)")
print(f"  - merge_date (date-only key)")
print(f"\nTarget variable distribution:")
print(f"  Class 0 (no injury): {no_injury:,} records ({no_injury/len(person_df)*100:.1f}%)")
print(f"  Class 1 (injury/death): {injury:,} records ({injury/len(person_df)*100:.1f}%)")
print(f"  Imbalance ratio: {ratio:.1f}:1")

print("\n" + "="*80)
print("PERSON CONDITIONING COMPLETE!")
print("="*80)