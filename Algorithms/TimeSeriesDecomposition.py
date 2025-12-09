import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

file_path = "schema_sentinel_last5yrs.parquet"

df = pd.read_parquet(file_path)
df.head()

df['crash_date'] = pd.to_datetime(df['crash_date_vehicle'], errors='coerce')
df = df.dropna(subset=['crash_date'])

crash_df = df[['collision_id', 'crash_date']].drop_duplicates(subset=['collision_id'])

daily = (
    crash_df
    .groupby('crash_date')
    .agg(crash_count=('collision_id', 'nunique'))
    .reset_index()
)

daily_ts = daily.set_index('crash_date')
daily_ts = daily_ts.asfreq('D')   # enforce daily frequency

result = seasonal_decompose(daily_ts['crash_count'], model='additive', period=365)

plt.rcParams['figure.figsize'] = (12, 8)
result.plot()
plt.suptitle("Time Series Decomposition of NYC Daily Crash Counts", fontsize=14)
plt.tight_layout()
plt.show()