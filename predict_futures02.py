import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import warnings # Import warnings module

# Suppress specific UserWarning from sklearn about feature names
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# === CONFIG ===
csv_folder = r"C:\Dev\AQI_prediction\output"
original_data_file = os.path.join(csv_folder, "xgboost_predictions_optimized.csv")
target_column = "actual_SO2"
sequence_length_predict = 10

# Load model and scaler globally to avoid reloading on every function call in Streamlit
# This assumes the model and scaler files are stable and exist.
try:
    MODEL = joblib.load(os.path.join(csv_folder, "xgboost_model.pkl"))
    SCALER = joblib.load(os.path.join(csv_folder, "scaler.pkl"))
    print("Global: Loaded saved model and scaler for prediction function.")
except FileNotFoundError as e:
    print(f"Global Error: Model or scaler files not found in '{csv_folder}'. Please ensure mainxgboost.py has been run: {e}")
    MODEL = None
    SCALER = None
except Exception as e:
    print(f"Global Error: Could not load model or scaler: {e}")
    MODEL = None
    SCALER = None

def get_future_predictions(start_datetime_user, end_datetime_user):
    """
    Generates future SO2 predictions for a given date range.

    Args:
        start_datetime_user (datetime): The start datetime for predictions.
        end_datetime_user (datetime): The end datetime for predictions.

    Returns:
        pd.DataFrame: A DataFrame containing 'timestamp', 'latitude', 'longitude',
                      and 'predicted_SO2' for the specified range.
                      Returns an empty DataFrame if prediction fails.
    """
    if MODEL is None or SCALER is None:
        print("Prediction function: Model or scaler not loaded. Cannot make predictions.")
        return pd.DataFrame()

    # === LOAD ORIGINAL DATA ===
    try:
        data = pd.read_csv(original_data_file)
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data = data.sort_values('timestamp')
        # print(f"Shape of data after loading: {data.shape}") # Removed for cleaner Streamlit output

        # Rename 'ColumnAmountSO2' to 'actual_SO2' if it exists (for compatibility)
        if 'ColumnAmountSO2' in data.columns:
            data.rename(columns={'ColumnAmountSO2': 'actual_SO2'}, inplace=True)

        data = data[pd.notna(data['actual_SO2'])]
        # print(f"Shape of data after removing NaN in 'actual_SO2': {data.shape}") # Removed for cleaner Streamlit output

        # Ensure 'nTimes', 'nXtrack', and 'predicted_SO2' are loaded and handled correctly
        # If they are not in the CSV, they will be NaN, and we'll need to fill them
        if 'nTimes' not in data.columns:
            data['nTimes'] = np.nan
            # print("Warning: 'nTimes' column not found in the loaded data. Filling with NaN.")
        if 'nXtrack' not in data.columns:
            data['nXtrack'] = np.nan
            # print("Warning: 'nXtrack' column not found in the loaded data. Filling with NaN.")
        if 'predicted_SO2' not in data.columns:
            data['predicted_SO2'] = np.nan
            # print("Warning: 'predicted_SO2' column not found in the loaded data. Filling with NaN.")

    except FileNotFoundError:
        print(f"Error: Original data file not found at {original_data_file}. Cannot make predictions.")
        return pd.DataFrame()
    except KeyError as e:
        print(f"Error: Column not found in the original data file: {e}. Cannot make predictions.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading or preprocessing data: {e}. Cannot make predictions.")
        return pd.DataFrame()

    # === FEATURE ENGINEERING ===
    data['lag_1'] = data['actual_SO2'].shift(1)
    data['lag_2'] = data['actual_SO2'].shift(2)
    data['lag_3'] = data['actual_SO2'].shift(3)
    data['lag_4'] = data['actual_SO2'].shift(4)
    data['lag_5'] = data['actual_SO2'].shift(5)
    data['lag_6'] = data['actual_SO2'].shift(6)
    data['lag_7'] = data['actual_SO2'].shift(7)
    data['lag_8'] = data['actual_SO2'].shift(8)
    data['lag_9'] = data['actual_SO2'].shift(9)
    data['lag_10'] = data['actual_SO2'].shift(10)
    data['rolling_mean'] = data['actual_SO2'].rolling(window=sequence_length_predict).mean()
    data['exp_smooth'] = data['actual_SO2'].ewm(span=sequence_length_predict, adjust=False).mean()
    # print(f"Shape of data after feature engineering: {data.shape}") # Removed for cleaner Streamlit output

    # Fill NaN values (including those from 'nTimes', 'nXtrack' if they were missing)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    # print(f"Shape of data after fillna: {data.shape}") # Removed for cleaner Streamlit output

    # Drop rows where the engineered features still have NaN
    columns_to_check_for_na = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', 'rolling_mean', 'exp_smooth']
    data.dropna(subset=columns_to_check_for_na, inplace=True)
    # print(f"Shape of data after dropping NaN in engineered features: {data.shape}") # Removed for cleaner Streamlit output

    # Check if the DataFrame is empty after NaN removal
    if data.empty:
        print("Error: DataFrame is empty after handling NaN. Cannot make predictions.")
        return pd.DataFrame()

    # === PREDICT FOR EACH TIME POINT IN THE RANGE ===
    all_future_predictions_df = pd.DataFrame(columns=['timestamp', 'latitude', 'longitude', 'predicted_SO2'])

    # Ensure start_datetime_user and end_datetime_user are within a reasonable range
    if start_datetime_user >= end_datetime_user:
        print("Error: Start date must be before end date.")
        return pd.DataFrame()

    future_timestamps = pd.date_range(start=start_datetime_user, end=end_datetime_user, freq='H')

    if future_timestamps.empty:
        print("No timestamps generated for the given range.")
        return pd.DataFrame()

    last_known_row = data.iloc[-1].copy()
    # EXACT feature columns that were used for training (17 features)
    feature_columns_trained_predict = ['predicted_SO2', 'nTimes', 'nXtrack', 'latitude', 'longitude',
                                        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
                                        'lag_9', 'lag_10', 'rolling_mean', 'exp_smooth']

    # Get the last known latitude and longitude for mapping
    last_known_latitude = last_known_row['latitude'] if 'latitude' in last_known_row.index else np.nan
    last_known_longitude = last_known_row['longitude'] if 'longitude' in last_known_row.index else np.nan

    for future_time in future_timestamps:
        # Ensure all trained feature columns are present in the last known row
        for col in feature_columns_trained_predict:
            if col not in last_known_row.index:
                print(f"Error: Column '{col}' not found in the latest data for prediction. Mismatch between training and prediction features.")
                return pd.DataFrame()

        current_features = last_known_row[feature_columns_trained_predict].values.reshape(1, -1)
        scaled_features = SCALER.transform(current_features)
        prediction_scaled = MODEL.predict(scaled_features)[0]

        # Clip prediction at 0 as SO2 cannot be negative
        prediction_original_scale = max(0, prediction_scaled)

        # Append prediction to DataFrame
        new_row = pd.DataFrame([{
            'timestamp': future_time,
            'latitude': last_known_latitude,
            'longitude': last_known_longitude,
            'predicted_SO2': prediction_original_scale
        }])
        all_future_predictions_df = pd.concat([all_future_predictions_df, new_row], ignore_index=True)

        # Naive update of features for the next prediction
        # Shift lag features
        for i in range(10, 1, -1):
            last_known_row[f'lag_{i}'] = last_known_row[f'lag_{i-1}']
        last_known_row['lag_1'] = last_known_row['actual_SO2'] # The most recent 'actual' value becomes lag_1

        # Update 'actual_SO2' and 'predicted_SO2' with the new (clipped) prediction
        last_known_row['actual_SO2'] = prediction_original_scale
        last_known_row['predicted_SO2'] = prediction_original_scale # For consistency, assuming predicted is the new actual

        # Simplified re-calculation of rolling mean and exp smooth
        current_values_for_mean_smooth = [last_known_row['actual_SO2']] + [last_known_row[f'lag_{i}'] for i in range(1, sequence_length_predict)]
        temp_series_for_mean_smooth = pd.Series(current_values_for_mean_smooth)
        last_known_row['rolling_mean'] = temp_series_for_mean_smooth.mean()

        alpha = 2 / (sequence_length_predict + 1) # Common EWMA alpha
        last_known_row['exp_smooth'] = alpha * last_known_row['actual_SO2'] + (1 - alpha) * last_known_row['exp_smooth']

    return all_future_predictions_df

# The direct execution part is removed, as it will be called by Streamlit
# if __name__ == '__main__':
#     # Example usage for testing the function directly
#     start_date_str = "2026-01-13 00:00"
#     end_date_str = "2026-01-15 23:00"
#     start_dt = pd.to_datetime(start_date_str)
#     end_dt = pd.to_datetime(end_date_str)
#     predictions_df = get_future_predictions(start_dt, end_dt)
#     if not predictions_df.empty:
#         print("\nGenerated Predictions:")
#         print(predictions_df.head())
#         # You can save this DataFrame if needed for debugging
#         # predictions_df.to_csv(os.path.join(csv_folder, "future_predictions_range_test.csv"), index=False)
