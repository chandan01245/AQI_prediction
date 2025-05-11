import xarray as xr
import pandas as pd
import h5py

# STEP 1: Use a raw string for file path to avoid escape character issues
file_path = r"D:\S5P_OFFL_L2__NO2____20241231T070059_20241231T084229_37397_03_020800_20250104T044515.nc"

# STEP 2: Inspect file structure with h5py
print("Inspecting .nc file structure:")
with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)

# STEP 3: Open a specific group (usually 'PRODUCT' contains NO2, CO, etc.)
group_name = "PRODUCT"  # Common for Sentinel-5P files

try:
    ds = xr.open_dataset(file_path, engine="netcdf4", group=group_name)
    print("Variables found:", list(ds.variables))
except Exception as e:
    print("Error opening dataset:", e)
    exit()

# STEP 4: Choose a variable (update as per actual variables printed above)
var_name = "nitrogendioxide_tropospheric_column"  # Replace if different
if var_name not in ds.variables:
    print(f"Variable '{var_name}' not found. Available: {list(ds.variables)}")
    exit()

# STEP 5: Convert to DataFrame
data = ds[var_name].to_dataframe().reset_index()
data_clean = data.dropna()

# STEP 6: Export to CSV
output_csv = "output.csv"
data_clean.to_csv(output_csv, index=False)
print(f"\nCSV saved as: {output_csv}")
