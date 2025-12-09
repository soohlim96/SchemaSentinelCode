import pandas as pd
import numpy as np
import os

# Set paths
RAW_DATA_PATH = ''
OUTPUT_PATH = ''

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*80)
print("WEATHER DATASET CONDITIONING - COMPLETE PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\nSTEP 1: Loading raw weather data...")
weather_raw = pd.read_csv(f'{RAW_DATA_PATH}/nyc weather data.csv')

print(f"Raw weather data loaded: {len(weather_raw):,} records")
print(f"Columns: {list(weather_raw.columns)}")

# Save pre-conditioning snapshot
weather_raw.to_csv(f'{OUTPUT_PATH}/weather_PRE_conditioning.csv', index=False)
print(f"Pre-conditioning dataset saved: weather_PRE_conditioning.csv")

# ============================================================================
# STEP 2: DATE STANDARDIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATE STANDARDIZATION")
print("="*80)

# Create working copy
weather_df = weather_raw.copy()

print("\nBEFORE:")
print(f"DATE column type: {weather_df['DATE'].dtype}")
print(f"Sample dates: {weather_df['DATE'].head(3).tolist()}")

# Convert DATE to datetime
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

# Rename for consistency
weather_df.rename(columns={'DATE': 'crash_date'}, inplace=True)

print("\nAFTER:")
print(f"crash_date column type: {weather_df['crash_date'].dtype}")
print(f"Sample dates: {weather_df['crash_date'].head(3).tolist()}")

# ============================================================================
# STEP 3: WEATHER CATEGORIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: WEATHER CATEGORIZATION")
print("="*80)

# Show raw weather indicators
print("\nRaw weather indicator columns:")
weather_codes = [col for col in weather_df.columns if col.startswith('WT')]
print(f"Weather codes available: {weather_codes}")

# Count days with each indicator
print("\nDays with each indicator (first 5):")
for code in weather_codes[:5]:
    count = weather_df[code].notna().sum()
    print(f"  {code}: {count} days")

# Define categorization function
def categorize_weather(row):
    """
    Hierarchical weather categorization
    Priority: Snow > Rain > Fog > Clear
    """
    # Priority 1: Snow
    if pd.notna(row.get('WT18')) or row.get('SNOW', 0) > 0:
        return 'Snow'
    
    # Priority 2: Rain
    elif pd.notna(row.get('WT16')) or row.get('PRCP', 0) > 0:
        return 'Rain'
    
    # Priority 3: Fog
    elif pd.notna(row.get('WT01')) or pd.notna(row.get('WT02')):
        return 'Fog'
    
    # Default: Clear
    else:
        return 'Clear'

# Apply categorization
print("\nApplying categorization...")
weather_df['weather_condition'] = weather_df.apply(categorize_weather, axis=1)

# Show results
print("\nWeather category distribution:")
print(weather_df['weather_condition'].value_counts())
print("\nPercentages:")
print(weather_df['weather_condition'].value_counts(normalize=True) * 100)

# ============================================================================
# STEP 4: CREATE MERGE KEY
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CREATE MERGE KEY")
print("="*80)

print("\nBEFORE:")
print(f"crash_date includes time: {weather_df['crash_date'].head(3).tolist()}")

# Create date-only merge key
weather_df['merge_date'] = weather_df['crash_date'].dt.date

print("\nAFTER:")
print(f"merge_date (date only): {weather_df['merge_date'].head(3).tolist()}")

# ============================================================================
# STEP 5: SAVE POST-CONDITIONING
# ============================================================================
print("\n" + "="*80)
print("STEP 5: SAVE POST-CONDITIONING DATASET")
print("="*80)

# Select final columns
weather_final = weather_df[[
    'crash_date',
    'merge_date',
    'weather_condition',
    'PRCP',
    'SNOW',
    'TMAX',
    'TMIN'
]]

# Save post-conditioning dataset
weather_final.to_csv(f'{OUTPUT_PATH}/weather_POST_conditioning.csv', index=False)
weather_final.to_parquet(f'{OUTPUT_PATH}/weather_POST_conditioning.parquet', index=False)

print(f"\nPost-conditioning dataset saved:")
print(f"  CSV: weather_POST_conditioning.csv")
print(f"  Parquet: weather_POST_conditioning.parquet")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("WEATHER CONDITIONING SUMMARY")
print("="*80)
print(f"\nRecords: {len(weather_raw):,} -> {len(weather_final):,} (no change)")
print(f"Columns: {len(weather_raw.columns)} -> {len(weather_final.columns)}")
print(f"\nNew columns created:")
print(f"  - weather_condition (categorical)")
print(f"  - merge_date (date-only key)")
print(f"\nDate range: {weather_final['crash_date'].min()} to {weather_final['crash_date'].max()}")
print(f"Total days: {len(weather_final)}")

print("\n" + "="*80)
print("WEATHER CONDITIONING COMPLETE!")
print("="*80)