import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

file_path = "schema_sentinel_last5yrs.parquet"

df = pd.read_parquet(file_path)
df.head()

# ---------------------------------
# 1. Create a clean crash_date column
# ---------------------------------
# Prefer crash_date_vehicle if available, otherwise fall back to crash_date_x or crash_date_person

if 'crash_date_vehicle' in df.columns:
    df['crash_date'] = pd.to_datetime(df['crash_date_vehicle'], errors='coerce')
elif 'crash_date_x' in df.columns:
    df['crash_date'] = pd.to_datetime(df['crash_date_x'], errors='coerce')
else:
    df['crash_date'] = pd.to_datetime(df['crash_date_person'], errors='coerce')

# Keep only rows with valid dates and PRCP
df_model = df[['collision_id', 'crash_date', 'PRCP']].dropna(subset=['crash_date', 'PRCP'])

# ---------------------------------
# 2. Go from person-level to crash-level
# ---------------------------------
# Your data is person-level, so we make one row per collision_id.

crash_df = df_model.drop_duplicates(subset=['collision_id'])

# ---------------------------------
# 3. Aggregate to DAILY crash counts + average PRCP
# ---------------------------------

daily = (
    crash_df
    .groupby('crash_date')
    .agg(
        crash_count=('collision_id', 'nunique'),  # number of unique crashes that day
        PRCP=('PRCP', 'mean')                     # mean precipitation that day
    )
    .reset_index()
)

print("Daily data preview:")
print(daily.head())

# ---------------------------------
# 4. Build design matrix (X) and response (y)
# ---------------------------------
# Simple model: crash_count ~ PRCP

X = daily[['PRCP']]
X = sm.add_constant(X)          # adds intercept column
y = daily['crash_count'].astype(float)

print("\nDesign matrix dtypes:")
print(X.dtypes)
print("\nResponse dtype:", y.dtype)

# Ensure numeric types only
X = X.astype(float)

# ---------------------------------
# 5. Fit Poisson regression (statsmodels)
# ---------------------------------

poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

print("\nPoisson Regression Summary:")
print(poisson_results.summary())

# ---------------------------------
# 6. Check for overdispersion
# ---------------------------------

deviance = poisson_results.deviance
df_resid = poisson_results.df_resid
dispersion = deviance / df_resid
print(f"\nDispersion statistic (deviance / df_resid): {dispersion:.3f}")
# Rule of thumb: if >> 1, there is overdispersion

# ---------------------------------
# 7. Add predictions back to daily data
# ---------------------------------

daily['predicted'] = poisson_results.predict(X)

print("\nDaily data with predictions preview:")
print(daily.head())

# ---------------------------------
# 8. Visualization 1:
#    Actual vs Predicted Crash Counts Over Time
# ---------------------------------

plt.figure(figsize=(14, 6))
plt.plot(daily['crash_date'], daily['crash_count'],
         label='Actual', linewidth=2)
plt.plot(daily['crash_date'], daily['predicted'],
         label='Predicted (Poisson)', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Number of Crashes")
plt.title("Poisson Regression: Actual vs Predicted Daily Crash Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------
# 9. Visualization 2:
#    Effect of PRCP on Expected Crash Counts
# ---------------------------------

# Create a range of PRCP values
prcp_range = np.linspace(daily['PRCP'].min(), daily['PRCP'].max(), 100)

# Build a dataframe for prediction: const + PRCP
viz_X = pd.DataFrame({
    'const': 1.0,
    'PRCP': prcp_range
})

viz_pred = poisson_results.predict(viz_X)

plt.figure(figsize=(10, 5))
plt.plot(prcp_range, viz_pred, linewidth=2)
plt.xlabel("Daily Precipitation (PRCP)")
plt.ylabel("Expected Number of Crashes")
plt.title("Poisson Regression: Effect of Precipitation on Expected Crashes")
plt.grid(True)
plt.tight_layout()
plt.show()
