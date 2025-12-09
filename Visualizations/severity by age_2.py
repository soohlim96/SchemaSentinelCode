# ===============================
# SEVERITY BY AGE GROUP
# ===============================

df_age = df.copy()

# Age bins based on your report
age_bins = [0, 18, 25, 35, 45, 55, 65, 120]
age_labels = ["<18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]

df_age["age_group"] = pd.cut(df_age["person_age"], bins=age_bins, labels=age_labels, right=False)

age_summary = df_age.groupby("age_group").agg(
    collisions=("collision_id", "count"),
    severe_collisions=("severe", "sum")
).reset_index()

age_summary["severe_rate"] = age_summary["severe_collisions"] / age_summary["collisions"]
age_summary["severe_rate_pct"] = age_summary["severe_rate"] * 100

print(age_summary)

# Plot
plt.figure(figsize=(8,5))
sns.barplot(data=age_summary, x="age_group", y="severe_rate_pct", color="darkgreen")
plt.title("Severity Risk (%) by Age Group")
plt.ylabel("Severity Rate (%)")
plt.xlabel("Age Group")
plt.tight_layout()
plt.show()
