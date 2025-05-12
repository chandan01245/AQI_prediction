import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# === CONFIG ===
csv_folder = r"C:\Users\sreeh\Documents\GitHub\AQI_prediction\output"
prediction_file = os.path.join(csv_folder, "xgboost_predictions_optimized.csv")

# === LOAD SAVED MODEL AND SCALER ===
print("Loading saved model and scaler...")
with open("output/xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("output/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

print("Loaded saved model and scaler.")

# === LOAD SAVED PREDICTIONS FILE ===
data = pd.read_csv(prediction_file)

# === REGENERATE FEATURE COLUMNS ===
sequence_length = 10
for i in range(1, sequence_length + 1):
    data[f'lag_{i}'] = data['actual_SO2'].shift(i)

# Rolling mean and exponential smoothing
data['rolling_mean'] = data['actual_SO2'].rolling(window=sequence_length).mean()
data['exp_smooth'] = data['actual_SO2'].ewm(span=sequence_length).mean()

# === HANDLE MISSING VALUES ===
data.fillna(method='bfill', inplace=True)

feature_columns = [col for col in data.columns if col.startswith("lag_") or col in ["rolling_mean", "exp_smooth"]]

if not feature_columns:
    raise ValueError("No feature columns could be created. Please check your data.")

# === ASK FOR FUTURE DATE ===
user_input_date = input("Enter the future date (YYYY-MM-DD) to predict SO2: ")

try:
    future_date = pd.to_datetime(user_input_date)
    print(f"Generating prediction for {future_date}.")
except ValueError:
    print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")

# === EXTRACT LAST ROW AND PREPARE FEATURES ===
last_row = data.iloc[-1]
last_row_features = last_row[feature_columns].values.reshape(1, -1)

# === SCALE FEATURES ===
last_row_features_scaled = scaler.transform(last_row_features)

# === PREDICT FOR FUTURE DATE ===
future_prediction = model.predict(last_row_features_scaled)

print(f"Predicted SO2 for {future_date}: {future_prediction[0]}")

# === SAVE FUTURE PREDICTION ===
future_result_df = pd.DataFrame({
    'timestamp': [future_date],
    'predicted_SO2': future_prediction
})

future_output_path = os.path.join(csv_folder, "future_predictions.csv")
future_result_df.to_csv(future_output_path, index=False, mode='a', header=not os.path.exists(future_output_path))

print(f"Future prediction saved to: {future_output_path}")