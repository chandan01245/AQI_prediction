import tarfile
import xarray as xr
import pandas as pd
import os

def extract_multiple_tar_gz_to_csv(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all .tar.gz files in the input folder
    for tar_gz_file in os.listdir(input_folder):
        if tar_gz_file.endswith(".tar.gz"):
            tar_gz_path = os.path.join(input_folder, tar_gz_file)
            print(f"Processing: {tar_gz_path}")
            
            # Create a temporary folder for extraction
            temp_extract_folder = os.path.join(output_folder, "temp_extracted")
            os.makedirs(temp_extract_folder, exist_ok=True)
            
            # Extract the .tar.gz file
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(temp_extract_folder)
            
            # Convert each extracted NetCDF file to CSV
            for file in os.listdir(temp_extract_folder):
                file_path = os.path.join(temp_extract_folder, file)
                if file.endswith(".nc"):
                    # Load NetCDF file using xarray
                    ds = xr.open_dataset(file_path)
                    
                    # Convert entire dataset to DataFrame
                    df = ds.to_dataframe().reset_index()
                    
                    # Save the DataFrame to a CSV file
                    output_csv = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.csv")
                    df.to_csv(output_csv, index=False)
                    print(f"Extracted data from {file} saved to {output_csv}")
            
            # Clean up the temporary extraction folder
            for file in os.listdir(temp_extract_folder):
                os.remove(os.path.join(temp_extract_folder, file))
            os.rmdir(temp_extract_folder)
    
    print("All .tar.gz files processed.")

# Example usage
extract_multiple_tar_gz_to_csv(
    input_folder="path/to/your/folder/containing/tar_gz_files", 
    output_folder="path/to/your/output_folder"
)
