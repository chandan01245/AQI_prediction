import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit


# === CONFIG ===
csv_folder = r"C:\Users\chand\Documents\Coding\python\AQI_prediction\output"
target_column = "ColumnAmountSO2"
sequence_length = 10

# === LOAD & CONCATENATE CSV FILES ===
all_files = glob.glob(os.path.join(csv_folder, "*.csv"))
df_list = [pd.read_csv(file) for file in all_files]
data = pd.concat(df_list, ignore_index=True)

# === DEBUGGING: CHECK DATA SIZE ===
print(f"Initial data size: {data.shape}")

# === FEATURE ENGINEERING WITH ADAPTIVE SEQUENCE LENGTH ===
max_sequence_length = 10  # Set max value for sequence length
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

# === DEBUGGING: CHECK FOR MISSING VALUES ===
print(f"Data before filling NA: {data.shape}")
print(data.isna().sum())

# === HANDLE MISSING VALUES (FILL INSTEAD OF DROPPING) ===
data.fillna(method='bfill', inplace=True)

# === CHECK FOR EMPTY DATAFRAME ===
if data.empty:
    raise ValueError(f"Dataframe is empty after feature engineering with sequence length {sequence_length}.")

print(f"Final data size after feature engineering: {data.shape}")

# === HANDLE TIME COLUMN ===
if 'time' not in data.columns:
    raise ValueError("Missing 'time' column in your data.")
data['time'] = pd.to_datetime(data['time'])
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

# === METRICS ===
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.6f}")
print(f"R² Score: {r2:.6f}")
#
# === OUTPUT PREDICTIONS TO CSV ===
result_df = pd.DataFrame({
    'timestamp': data['time'].iloc[-len(y_test):].values,
    'actual_SO2': y_test,
    'predicted_SO2': y_pred
})

output_path = os.path.join(csv_folder, "xgboost_predictions_optimized.csv")
result_df.to_csv(output_path, index=False)
print(f"\nSaved predictions to: {output_path}")

# === CALL VISUALIZATION SCRIPT ===
from visualisation import visualize_predictions
visualize_predictions(output_path)
