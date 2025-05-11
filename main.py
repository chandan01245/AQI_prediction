# lstm_model.py
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === CONFIG ===
csv_folder = r"C:\Users\sreeh\Documents\GitHub\AQI_prediction\output"
target_column = "ColumnAmountSO2"
sequence_length = 10
epochs = 10
batch_size = 32

def load_data(csv_folder, target_column):
    """Load and concatenate all CSV files in the specified folder."""
    all_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    df_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(df_list, ignore_index=True)
    
    if 'time' not in data.columns:
        raise ValueError("Missing 'time' column in your data.")
    
    data['time'] = pd.to_datetime(data['time'])
    data.sort_values('time', inplace=True)
    
    df = data[['time', target_column]].dropna().reset_index(drop=True)
    return df

def create_sequences(data, sequence_length):
    """Create sequences for LSTM model."""
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(data[[target_column]])
    
    X, y, timestamps = [], [], []
    for i in range(len(scaled_values) - sequence_length):
        X.append(scaled_values[i:i + sequence_length])
        y.append(scaled_values[i + sequence_length])
        timestamps.append(data['time'].iloc[i + sequence_length])
    
    return np.array(X), np.array(y), np.array(timestamps), scaler

def train_lstm_model(X_train, y_train):
    """Train the LSTM model with EarlyStopping and ReduceLROnPlateau."""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              verbose=1,
              validation_split=0.2,
              callbacks=[early_stopping, reduce_lr])
    
    return model

def save_predictions(model, X_test, y_test, scaler, timestamps):
    """Make predictions, inverse scale them, and save to CSV."""
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print(f"\nMSE: {mse:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual_SO2': y_test_inv.flatten(),
        'predicted_SO2': y_pred_inv.flatten()
    })
    
    output_path = os.path.join(csv_folder, "lstm_predictions.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    return output_path

# === EXECUTION ===
data = load_data(csv_folder, target_column)
X, y, timestamps, scaler = create_sequences(data, sequence_length)

# === TRAIN/TEST SPLIT ===
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
test_timestamps = timestamps[split_index:]

# === RESHAPE FOR LSTM ===
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === TRAIN MODEL ===
model = train_lstm_model(X_train, y_train)

# === SAVE PREDICTIONS ===
output_path = save_predictions(model, X_test, y_test, scaler, test_timestamps)

# === CALL VISUALIZATION SCRIPT ===
from visualisation import visualize_predictions
visualize_predictions(output_path)
