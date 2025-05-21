import pandas as pd
import plotly.express as px

# Load the data
df = pd.read_csv(r"C:\Users\chand\Documents\Coding\python\AQI_prediction\output\xgboost_predictions_optimized.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Option 1: Use absolute values for size
df['size'] = df['predicted_SO2'].abs()

# Option 2 (safer): Set all negative predictions to zero before sizing
# df['size'] = df['predicted_SO2'].clip(lower=0)

# Optional hover info
df['hover'] = (
    "Time: " + df['timestamp'].astype(str) +
    "<br>Actual: " + df['actual_SO2'].astype(str) +
    "<br>Predicted: " + df['predicted_SO2'].astype(str)
)

# Plot using updated Plotly API
fig = px.scatter_map(
    df,
    lat="latitude",
    lon="longitude",
    color="predicted_SO2",
    size="size",
    hover_name="hover",
    animation_frame=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
    color_continuous_scale="YlOrRd",
    size_max=15,
    zoom=4,
    title="Predicted SO₂ Concentration Over Time"
)


fig.show()

