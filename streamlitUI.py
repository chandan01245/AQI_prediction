import streamlit as st
import pandas as pd
from datetime import datetime, time, timedelta
import os
import pydeck as pdk # Import pydeck for advanced mapping
import streamlit.components.v1 as components # Import for embedding HTML

# Import the prediction function from your refactored script
# Ensure predict_futures02.py is in the same directory or accessible via Python path
try:
    from predict_futures02 import get_future_predictions
except ImportError:
    st.error("Error: Could not import 'get_future_predictions' from 'predict_futures02.py'.")
    st.stop() # Stop the app if the function cannot be imported

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide", page_title="SO2 Future Prediction")

st.title("SO2 Future Level Prediction")
st.markdown("Enter a date and time range to predict SO2 levels at the last known location.")

# --- Date and Time Input ---
col1, col2 = st.columns(2)

# Set default start date to today, default end date to 1 day from now
default_start_date = datetime.now().date()
default_end_date = (datetime.now() + timedelta(days=1)).date()

with col1:
    st.header("Start Date and Time")
    start_date = st.date_input("Select start date", value=default_start_date)
    # Add step to time input for minute granularity
    start_time = st.time_input("Select start time", value=time(0, 0), step=timedelta(minutes=15))

with col2:
    st.header("End Date and Time")
    end_date = st.date_input("Select end date", value=default_end_date)
    # Add step to time input for minute granularity
    end_time = st.time_input("Select end time", value=time(0, 0), step=timedelta(minutes=15))

# Combine date and time inputs
start_datetime = datetime.combine(start_date, start_time)
end_datetime = datetime.combine(end_date, end_time)

# --- Prediction Button ---
st.markdown("---")
if st.button("Generate Predictions"):
    if start_datetime >= end_datetime:
        st.error("Error: Start date and time must be before end date and time.")
    else:
        with st.spinner("Generating predictions... This may take a moment."):
            # Call the prediction function
            predictions_df = get_future_predictions(start_datetime, end_datetime)

            if not predictions_df.empty:
                st.success("Predictions generated successfully!")

                st.subheader("Predicted SO2 Levels (Raw Data)")
                st.dataframe(predictions_df)

                # --- Map Visualization (using PyDeck) ---
                st.subheader("Predicted SO2 Levels on Map (PyDeck)")
                # Ensure latitude and longitude are present and not NaN
                map_data_for_pydeck = predictions_df[['latitude', 'longitude', 'predicted_SO2']].dropna()

                if not map_data_for_pydeck.empty:
                    # Calculate average SO2 for the color-coding
                    avg_so2 = map_data_for_pydeck['predicted_SO2'].mean()
                    
                    # Define color based on average SO2 (example: green for low, red for high)
                    # You might need to adjust these thresholds based on your actual SO2 value range
                    if avg_so2 < 0.05:
                        color = [0, 255, 0, 160] # Green
                    elif avg_so2 < 0.15:
                        color = [255, 255, 0, 160] # Yellow
                    else:
                        color = [255, 0, 0, 160] # Red

                    # Get the single latitude and longitude for the map point
                    map_lat = map_data_for_pydeck['latitude'].iloc[0]
                    map_lon = map_data_for_pydeck['longitude'].iloc[0]

                    # Create a DataFrame for the single map point
                    single_map_point = pd.DataFrame([{
                        'latitude': map_lat,
                        'longitude': map_lon,
                        'avg_predicted_so2': avg_so2
                    }])

                    # Define a layer for the map
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        single_map_point,
                        get_position=["longitude", "latitude"],
                        get_color=color, # Use the dynamic color
                        get_radius=5000, # Radius in meters, adjust as needed
                        pickable=True,
                        tooltip={"text": "Average Predicted SO2: {avg_predicted_so2:.4f}"}
                    )

                    # Set the view state (camera position)
                    view_state = pdk.ViewState(
                        latitude=map_lat,
                        longitude=map_lon,
                        zoom=8, # Adjust zoom level as needed
                        pitch=0
                    )

                    # Create the PyDeck chart
                    r = pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        map_style="mapbox://styles/mapbox/light-v9" # You can choose other map styles
                    )
                    st.pydeck_chart(r)

                    st.info(f"Average Predicted SO2 for this period: **{avg_so2:.4f}**")

                else:
                    st.warning("No valid latitude/longitude data found for PyDeck map visualization. This might happen if the historical data is empty or missing location info.")
                
                # --- Embed predicted_so2_map.html ---
                st.subheader("Predicted SO2 Levels on Map (HTML Embed)")
                html_file_path = os.path.join("", "predicted_so2_map_UPDATED.html")
                if os.path.exists(html_file_path):
                    with open(html_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                    st.success(f"Embedded map from: {html_file_path}")
                else:
                    st.warning(f"Map HTML file not found at: {html_file_path}. Please ensure it exists.")


                # --- Time Series Plot ---
                st.subheader("Predicted SO2 Levels Over Time")
                st.line_chart(predictions_df.set_index('timestamp')['predicted_SO2'])

            else:
                st.warning("Could not generate predictions. This might be due to issues with the prediction script or the underlying data. Please check your console for more details.")

