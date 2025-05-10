import tarfile
import xarray as xr
import pandas as pd
import os

def extract_tar_gz_full_netcdf_to_csv(tar_gz_path, output_folder):
    # Extract the .tar.gz file
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(output_folder)
    
    # Get the list of extracted files
    extracted_files = os.listdir(output_folder)
    
    # Process NetCDF files
    for file in extracted_files:
        file_path = os.path.join(output_folder, file)
        if file.endswith(".nc"):
            # Load NetCDF file using xarray
            ds = xr.open_dataset(file_path)
            
            # Convert entire dataset to DataFrame
            df = ds.to_dataframe().reset_index()
            
            # Save the DataFrame to a CSV file
            output_csv = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.csv")
            df.to_csv(output_csv, index=False)
            print(f"Extracted data from {file} saved to {output_csv}")

# Example usage
extract_tar_gz_full_netcdf_to_csv(
    tar_gz_path="path/to/your/data.tar.gz", 
    output_folder="output/"
)
