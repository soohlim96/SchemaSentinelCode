from google.colab import drive
drive.mount('/content/drive')

import polars as pl

file_path = "/content/drive/MyDrive/Colab Notebooks/CS-504/project/schema_sentinel_integrated_with_severity.parquet"

scan_df = pl.scan_parquet(file_path)  # LazyFrame, doesn't load all data

preview = scan_df.head(5).collect()
print(preview)
print(scan_df.schema)

date_col = "crash_date_x"

max_year_df = scan_df.select(
    pl.col(date_col).dt.year().max().alias("max_year")
).collect()

max_year = int(max_year_df["max_year"][0])
cutoff_year = max_year - 4   # last 5 years inclusive

print("Max year in data:", max_year)
print("Keeping records from year >=", cutoff_year)

filtered_lazy = scan_df.filter(
    pl.col(date_col).dt.year() >= cutoff_year
)

output_path = "/content/drive/MyDrive/Colab Notebooks/CS-504/project/schema_sentinel_last5yrs.parquet"

# This writes the filtered dataset to a new Parquet file without loading everything into RAM
filtered_lazy.sink_parquet(output_path)

print("Filtered 5-year dataset written to:", output_path)

df_5yr = pl.read_parquet(output_path)
print(df_5yr.shape)
print(df_5yr.head())

date_col = "crash_date_x"


def print_date_range(path, label):
    scan = pl.scan_parquet(path)

    date_range = scan.select([
        pl.col(date_col).min().alias("min_date"),
        pl.col(date_col).max().alias("max_date")
    ]).collect()

    print(date_range)

print_date_range(file_path, "Original Full Integrated Dataset")
print_date_range(output_path, "Filtered 5-Year Dataset")
