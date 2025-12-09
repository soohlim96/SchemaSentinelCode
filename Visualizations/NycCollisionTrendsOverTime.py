import pandas as pd
import matplotlib.pyplot as plt

file_path = "schema_sentinel_last5yrs.parquet"

df = pd.read_parquet(file_path)
df.head()

df['crash_date'] = pd.to_datetime(df['crash_date_vehicle'])
df['crash_date'].head()

df = df.dropna(subset=['crash_date'])

df['year'] = df['crash_date'].dt.year.astype(int)

yearly = df.groupby('year').size()
yearly

plt.figure(figsize=(12,6))
yearly.plot(kind='line', marker='o')
plt.title("NYC Collision Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Collisions")
plt.grid(True)
plt.show()

