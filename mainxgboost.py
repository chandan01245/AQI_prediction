import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from datetime import datetime
import joblib  # For saving model and scaler

# === CONFIG ===
csv_folder = r"C:\Users\sreeh\Documents\GitHub\AQI_prediction\output"
target_column = "ColumnAmountSO2"
sequence_length = 10

# === LOAD & CONCATENATE CSV FILES ===
all_files = glob.glob(os.path.join(csv_folder, "*.csv"))
df_list = [pd.read_csv(file) for file in all_files]
data = pd.concat(df_list, ignore_index=True)

# === ENSURE 'time' COLUMN IS CONVERTED TO DATETIME ===
data['time'] = pd.to_datetime(data['time'], errors='coerce')

# === FEATURE ENGINEERING WITH ADAPTIVE SEQUENCE LENGTH ===
max_sequence_length = 10
available_rows = len(data)
sequence_length = min(max_sequence_length, available_rows // 2)

if sequence_length < 1:
    raise ValueError("Not enough data for feature engineering. Please provide more data.")

# Creating lag features
for i in range(1, sequence_length + 1):
    data[f'lag_{i}'] = data[target_column].shift(i)

# Rolling mean and exponential smoothing
data['rolling_mean'] = data[target_column].rolling(window=sequence_length).mean()
data['exp_smooth'] = data[target_column].ewm(span=sequence_length).mean()

# === HANDLE MISSING VALUES (FILL INSTEAD OF DROPPING) ===
data.fillna(method='bfill', inplace=True)

# === HANDLE TIME COLUMN ===
if 'time' not in data.columns:
    raise ValueError("Missing 'time' column in your data.")
data.sort_values('time', inplace=True)

# === REMOVE NON-NUMERIC COLUMNS BEFORE SCALING ===
X = data.drop(columns=['time', target_column])

# Ensure only numeric columns are in X
X = X.select_dtypes(include=[np.number])

if X.empty:
    raise ValueError("No valid features left for scaling after preprocessing.")

# === SCALE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = data[target_column].values

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# === XGBOOST MODEL WITH RANDOMIZED SEARCH ===
model = XGBRegressor(objective='reg:squarederror', random_state=42)
param_distributions = {
    'n_estimators': range(50, 300, 50),
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': range(3, 10),
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
ts_split = TimeSeriesSplit(n_splits=5)
random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=ts_split, n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)

# === BEST MODEL ===
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# === OUTPUT PREDICTIONS TO CSV ===
result_df = pd.DataFrame({
    'timestamp': data['time'].iloc[-len(y_test):].values,
    'latitude': data['latitude'].iloc[-len(y_test):].values if 'latitude' in data.columns else np.nan,
    'longitude': data['longitude'].iloc[-len(y_test):].values if 'longitude' in data.columns else np.nan,
    'actual_SO2': y_test,
    'predicted_SO2': y_pred
})

output_path = os.path.join(csv_folder, "xgboost_predictions_optimized.csv")
result_df.to_csv(output_path, index=False)
print(f"\nSaved predictions to: {output_path}")

# === SAVE THE MODEL AND SCALER ===
model_file = os.path.join(csv_folder, "xgboost_model.pkl")
scaler_file = os.path.join(csv_folder, "scaler.pkl")

# Save the trained model and scaler using joblib
joblib.dump(best_model, model_file)
joblib.dump(scaler, scaler_file)

print(f"Model and scaler saved to: {model_file}, {scaler_file}")

# === CALL VISUALIZATION SCRIPT ===
from visualisation import visualize_predictions
visualize_predictions(output_path)

# === PREDICTING FOR A FUTURE DATE ===

# Get user input for future date
user_input_date = input("Enter the future date (YYYY-MM-DD) to predict SO2: ")

# Convert the user input to datetime
try:
    future_date = pd.to_datetime(user_input_date)
    print(f"Predictions will be generated for {future_date}")
except ValueError:
    print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")

# Extract the most recent row for future prediction
last_row = data.iloc[-1].copy()  # Most recent data row
last_row_features = last_row.drop(columns=['time', target_column])  # Drop time and target columns

# Prepare the feature set for prediction
last_row_features = last_row_features.values.reshape(1, -1)
last_row_features_scaled = scaler.transform(last_row_features)

# Predict the SO2 value for the future date
future_prediction = best_model.predict(last_row_features_scaled)

print(f"Predicted SO2 for {future_date}: {future_prediction[0]}")

# OPTIONAL: Save future prediction to a new CSV
future_result_df = pd.DataFrame({
    'timestamp': [future_date],
    'predicted_SO2': future_prediction
})

future_output_path = os.path.join(csv_folder, "future_predictions.csv")
future_result_df.to_csv(future_output_path, index=False)
print(f"Future prediction saved to: {future_output_path}")
