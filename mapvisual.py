import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the CSV file
data_path = r"C:\Users\sreeh\Documents\GitHub\AQI_prediction\output\xgboost_predictions_optimized.csv"
data = pd.read_csv(data_path)

# Ensure NaN values are dropped in the relevant columns
data.dropna(subset=['latitude', 'longitude', 'predicted_SO2'], inplace=True)

# Convert data types to ensure no invalid values
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data['predicted_SO2'] = pd.to_numeric(data['predicted_SO2'], errors='coerce')

# Drop any rows with invalid values (NaNs after conversion)
data.dropna(subset=['latitude', 'longitude', 'predicted_SO2'], inplace=True)

# Ensure positive values for SO2 (optional)
data['predicted_SO2'] = data['predicted_SO2'].clip(lower=0)

# Create base map (centered on the average coordinates)
base_map = folium.Map(
    location=[data['latitude'].mean(), data['longitude'].mean()],
    zoom_start=5
)

# Prepare heat data (only valid rows)
heat_data = data[['latitude', 'longitude', 'predicted_SO2']].values.tolist()

# Ensure heat data is not empty
if not heat_data:
    raise ValueError("No valid data available for the heatmap. Please check your input file.")

# Debug print to check data format
print("Heat Data (First 5 Rows):", heat_data[:5])

# Generate heatmap without specifying gradient (for testing)
HeatMap(heat_data, radius=10, max_zoom=13).add_to(base_map)

# Save the map as an HTML file
output_path = r"C:\Users\sreeh\Documents\GitHub\AQI_prediction\predicted_so2_map.html"
base_map.save(output_path)
print(f"Map saved to: {output_path}")
