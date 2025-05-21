import h5py
import pandas as pd
import xarray as xr

# STEP 1: Set file path
file_path = r"C:\Users\chand\Downloads\Sat data\S5P_OFFL_L2__SO2____20250517T060501_20250517T074631_39340_03_020701_20250519T034329.nc"

# STEP 2: Inspect structure (optional)
print("Inspecting .nc file structure:")
with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)

# STEP 3: Open the SO2 product group
group_name = "PRODUCT"

try:
    ds = xr.open_dataset(file_path, engine="netcdf4", group=group_name)
    print("Variables found:", list(ds.variables))
except Exception as e:
    print("Error opening dataset:", e)
    exit()

# STEP 4: Define variables
so2_var = "sulfurdioxide_total_vertical_column"
qa_var = "qa_value"
time_var = "time_utc"

required_vars = [so2_var, qa_var, "latitude", "longitude", time_var]
missing = [v for v in required_vars if v not in ds.variables]
if missing:
    print(f"Missing variables: {missing}")
    exit()

# STEP 5: Convert to DataFrame
df = ds[required_vars].to_dataframe().reset_index()

# STEP 6: Filter:
# - Remove NaNs
# - Keep only positive SO₂
# - Keep only QA ≥ 0.5
df_filtered = df.dropna()
df_filtered = df_filtered[df_filtered[so2_var] >= 0]
df_filtered = df_filtered[df_filtered[qa_var] >= 0.5]

# STEP 7: Convert time to datetime format
df_filtered["datetime"] = pd.to_datetime(df_filtered[time_var], format='ISO8601')


# STEP 8: Export to CSV
output_csv = "so2.csv"
df_filtered.to_csv(output_csv, index=False)
print(f"\nFiltered SO₂ 1km data with time saved as: {output_csv}")
