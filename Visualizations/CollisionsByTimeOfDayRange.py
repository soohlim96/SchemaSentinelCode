import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ------------------------------------------------
# 1. Load data
# ------------------------------------------------
file_path = r"C:\Users\SooLim\OneDrive\CS504 007\schema_sentinel_last5yrs.parquet"
df = pd.read_parquet(file_path)

print("Initial shape (rows, columns):", df.shape)

# ------------------------------------------------
# 2. Build collision-level dataset (one row per collision_id)
# ------------------------------------------------
if "collision_id" not in df.columns:
    raise KeyError("collision_id column not found in dataframe.")

df_collision = (
    df
    .sort_values(["collision_id"])
    .drop_duplicates(subset=["collision_id"], keep="first")
    .copy()
)

print("After reducing to one row per collision_id:")
print("  Original rows:", df.shape[0])
print("  Collision-level rows:", df_collision.shape[0])

# Verify no duplicate collisions remain
dups_after = df_collision["collision_id"].duplicated().sum()
print("Duplicate collision_id values after dedupe:", dups_after)
assert not df_collision["collision_id"].duplicated().any(), \
    "There are still duplicate collision_id values in df_collision!"

# ------------------------------------------------
# 3. Parse crash_time into hour of day (0–23)
# ------------------------------------------------
if "crash_time" not in df_collision.columns:
    raise KeyError("crash_time column not found in dataframe.")

df_collision["crash_time_str"] = df_collision["crash_time"].astype(str).str.strip()

df_collision["crash_hour"] = pd.to_datetime(
    df_collision["crash_time_str"],
    errors="coerce"        # invalid times -> NaT
).dt.hour

before_drop = df_collision.shape[0]
df_collision = df_collision.dropna(subset=["crash_hour"]).copy()
df_collision["crash_hour"] = df_collision["crash_hour"].astype(int)
after_drop = df_collision.shape[0]

print("\nCrash hour parsing:")
print("  Rows before dropping NaN crash_hour:", before_drop)
print("  Rows after dropping NaN crash_hour:", after_drop)

# ------------------------------------------------
# 4. Define 7 time-of-day ranges & labels
# ------------------------------------------------
# Numeric bin edges in hours: [0,4), [4,7), [7,10), [10,16), [16,19), [19,22), [22,24)
bins = [0, 4, 7, 10, 16, 19, 22, 24]

labels_pretty = [
    "Late Night (00:00–03:59)",
    "Early Morning (04:00–06:59)",
    "AM Peak (07:00–09:59)",
    "Midday (10:00–15:59)",
    "PM Peak (16:00–18:59)",
    "Evening (19:00–21:59)",
    "Late Evening (22:00–23:59)",
]

df_collision["time_range_7"] = pd.cut(
    df_collision["crash_hour"],
    bins=bins,
    right=False,  # include left edge, exclude right
    labels=labels_pretty
)

# ------------------------------------------------
# 5. Counts per time range (and convert to thousands)
# ------------------------------------------------
group_counts = (
    df_collision["time_range_7"]
    .value_counts()
    .reindex(labels_pretty)
)

print("\nCollisions by time range (raw counts):")
print(group_counts)

counts_thousands = group_counts / 1000.0

# ------------------------------------------------
# 6. Bar chart in thousands + grid lines behind bars
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

x_pos = range(len(labels_pretty))

# Draw grid first, with low zorder, and send axes below so bars sit on top
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.xaxis.grid(False)

# Bars with higher zorder so they appear in front of the grid lines
ax.bar(x_pos, counts_thousands, zorder=3)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_pretty, rotation=45, ha="right")

ax.set_xlabel("Time of Day Range")
ax.set_ylabel("Number of Collisions (thousands)")
ax.set_title("Collisions by Time of Day Range")

# y-axis as whole numbers (e.g., 20, 40, 60)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))

plt.tight_layout()
plt.show()