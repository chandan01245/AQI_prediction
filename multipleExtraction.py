import os

import numpy as np
import pandas as pd
import xarray as xr

# === CONFIGURATION ===
input_folder = r"D:\chand\downloads\OMPS_NPP_NMSO2_PCA_L2_2-20250501_182148"
output_folder = r"C:\Users\chand\Documents\Coding\python\AQI_prediction\output"
group_name = "SCIENCE_DATA"
variable_name = "ColumnAmountSO2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === FUNCTION TO PROCESS SINGLE FILE ===
def process_file(file_path):
    try:
        # Load main variable from SCIENCE_DATA
        ds = xr.open_dataset(file_path, engine="netcdf4", group=group_name)
        if variable_name not in ds.variables:
            print(f"[SKIPPED] {os.path.basename(file_path)}: Variable not found.")
            return
        df = ds[variable_name].to_dataframe().reset_index().dropna()

        # === Load GEOLOCATION_DATA: lat/lon/time ===
        try:
            geo = xr.open_dataset(file_path, engine="netcdf4", group="GEOLOCATION_DATA")
            
            # Flatten and match lengths
            lat = geo['Latitude'].values.flatten()
            lon = geo['Longitude'].values.flatten()
            if 'Time' in geo.variables:
                time = pd.to_datetime(geo['Time'].values.flatten(), unit='s', origin='unix')
            else:
                time = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
            
            # Trim to same length if needed
            min_len = min(len(df), len(lat), len(lon), len(time))
            df = df.iloc[:min_len].copy()
            df['latitude'] = lat[:min_len]
            df['longitude'] = lon[:min_len]
            df['time'] = time[:min_len]

        except Exception as e:
            print(f"[WARNING] Could not load geolocation data: {e}")
            df['latitude'] = np.nan
            df['longitude'] = np.nan
            df['time'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

        # === Save CSV ===
        csv_name = os.path.splitext(os.path.basename(file_path))[0] + ".csv"
        csv_path = os.path.join(output_folder, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"[DONE] {csv_name}")

    except Exception as e:
        print(f"[ERROR] {os.path.basename(file_path)}: {e}")

# === LOOP OVER FILES ===
for filename in os.listdir(input_folder):
    if filename.endswith(".nc") or filename.endswith(".h5"):
        full_path = os.path.join(input_folder, filename)
        process_file(full_path)
