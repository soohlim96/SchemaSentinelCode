# ===============================
# SEVERITY BY WEATHER CONDITION
# ===============================

df_weather = df.copy()

weather_summary = df_weather.groupby("weather_condition").agg(
    collisions=("collision_id", "count"),
    severe_collisions=("severe", "sum")
).reset_index()

weather_summary["severe_rate"] = (
    weather_summary["severe_collisions"] / weather_summary["collisions"]
)
weather_summary["severe_rate_pct"] = weather_summary["severe_rate"] * 100

print(weather_summary)

# Plot
plt.figure(figsize=(8,5))
sns.barplot(data=weather_summary, x="weather_condition", y="severe_rate_pct", color="firebrick")
plt.title("Severity Risk (%) by Weather Condition")
plt.ylabel("Severity Rate (%)")
plt.xlabel("Weather")
plt.tight_layout()
plt.show()