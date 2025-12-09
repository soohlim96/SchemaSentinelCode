import pandas as pd

file_path = "schema_sentinel_last5yrs.parquet"

df = pd.read_parquet(file_path)
df.head()

#!pip install folium

import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

df['intersection'] = (
    df['on_street_name'].fillna('Unknown') 
    + " & " + 
    df['cross_street_name'].fillna('Unknown')
)

top_intersections = df['intersection'].value_counts().head(10)
top_intersections

plt.figure(figsize=(12,6))
top_intersections.sort_values().plot(kind='barh', color='steelblue')
plt.xlabel("Number of Collisions")
plt.ylabel("Intersection")
plt.title("Top 10 Most Common Collision Locations in NYC")
plt.tight_layout()
plt.show()

df_clean = df[
    (df['on_street_name'].notna()) & (df['cross_street_name'].notna())
]

df_clean = df_clean[
    (df_clean['on_street_name'].str.strip() != "") &
    (df_clean['cross_street_name'].str.strip() != "")
]

df_clean['intersection'] = (
    df_clean['on_street_name'] + " & " + df_clean['cross_street_name']
)

top_intersections = df_clean['intersection'].value_counts().head(10)
top_intersections

plt.figure(figsize=(12,6))
top_intersections.sort_values().plot(kind='barh', color='steelblue')
plt.xlabel("Number of Collisions")
plt.ylabel("Intersection")
plt.title("Top 10 Most Common Collision Locations in NYC (Cleaned Data)")
plt.tight_layout()
plt.show()

df_partial = df[
    (df['on_street_name'].notna()) &
    (df['on_street_name'].str.strip() != "")
].copy()

df_partial['cross_street_name'] = df_partial['cross_street_name'].fillna('Unknown')

df_partial['intersection'] = (
    df_partial['on_street_name'] + " & " + df_partial['cross_street_name']
)

top_intersections = df_partial['intersection'].value_counts().head(10)
print(top_intersections)

plt.figure(figsize=(12,6))
top_intersections.sort_values().plot(kind='barh', color='steelblue')
plt.xlabel("Number of Collisions")
plt.ylabel("Intersection")
plt.title("Top 10 Most Common Collision Locations in NYC (On-Street Known)")
plt.tight_layout()
plt.show()

df_geo = df_partial.dropna(subset=['latitude', 'longitude'])

df_geo = df_geo[
    (df_geo['latitude'] != 0) &
    (df_geo['longitude'] != 0)
]

m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

HeatMap(
    data=df_geo[['latitude', 'longitude']].values.tolist(),
    radius=8,     # size of each heat point
    blur=10,      # smoothness
    max_zoom=13
).add_to(m)

m