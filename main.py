import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === CONFIG ===
csv_folder = r"C:\Users\chand\Documents\Coding\python\AQI_prediction\output"
target_column = "ColumnAmountSO2"
sequence_length = 10

# === LOAD & CONCATENATE CSV FILES ===
all_files = glob.glob(os.path.join(csv_folder, "*.csv"))
df_list = [pd.read_csv(file) for file in all_files]
data = pd.concat(df_list, ignore_index=True)

# === HANDLE TIME COLUMN ===
if 'time' not in data.columns:
    raise ValueError("Missing 'time' column in your data.")
data['time'] = pd.to_datetime(data['time'])
data.sort_values('time', inplace=True)

# === DROP NA AND KEEP RELEVANT COLUMNS ===
df = data[['time', target_column]].dropna().reset_index(drop=True)

# === SCALE TARGET ===
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df[[target_column]])

# === CREATE SEQUENCES ===
X, y, timestamps = [], [], []

for i in range(len(scaled_values) - sequence_length):
    X.append(scaled_values[i:i + sequence_length])
    y.append(scaled_values[i + sequence_length])
    timestamps.append(df['time'].iloc[i + sequence_length])

X, y = np.array(X), np.array(y)
timestamps = np.array(timestamps)

# === TRAIN/TEST SPLIT ===
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
test_timestamps = timestamps[split_index:]

# === LSTM MODEL ===
model = Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# === TRAIN ===
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# === PREDICT & INVERSE SCALE ===
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# === METRICS ===
mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"\nMSE: {mse:.6f}")
print(f"R² Score: {r2:.6f}")

# === OUTPUT PREDICTIONS TO CSV ===
result_df = pd.DataFrame({
    'timestamp': test_timestamps,
    'actual_SO2': y_test_inv.flatten(),
    'predicted_SO2': y_pred_inv.flatten()
})

output_path = os.path.join(csv_folder, "lstm_predictions.csv")
result_df.to_csv(output_path, index=False)
print(f"\nSaved predictions to: {output_path}")
