# ===============================
# SEVERITY BY TIME OF DAY
# ===============================

import matplotlib.pyplot as plt
import seaborn as sns

df_time = df.copy()

# Extract hour
df_time["hour"] = pd.to_datetime(df_time["crash_time"], format="%H:%M").dt.hour

# Create time-of-day bins
bins = [0, 4, 7, 10, 16, 19, 22, 24]
labels = [
    "Late Night (00:00–03:59)",
    "Early Morning (04:00–06:59)",
    "AM Peak (07:00–09:59)",
    "Midday (10:00–15:59)",
    "PM Peak (16:00–18:59)",
    "Evening (19:00–21:59)",
    "Late Evening (22:00–23:59)"
]

df_time["time_of_day"] = pd.cut(df_time["hour"], bins=bins, labels=labels, right=False)

# Compute severity rate
time_summary = df_time.groupby("time_of_day").agg(
    collisions=("collision_id", "count"),
    severe_collisions=("severe", "sum")
).reset_index()

time_summary["severe_rate"] = time_summary["severe_collisions"] / time_summary["collisions"]
time_summary["severe_rate_pct"] = time_summary["severe_rate"] * 100

print(time_summary)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(data=time_summary, x="time_of_day", y="severe_rate_pct", color="steelblue")
plt.xticks(rotation=45, ha="right")
plt.title("Severity Risk (%) by Time of Day")
plt.ylabel("Severity Rate (%)")
plt.xlabel("")
plt.tight_layout()
plt.show()
