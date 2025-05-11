import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# === CONFIG ===
csv_folder = r"C:\Users\sreeh\Documents\GitHub\AQI_prediction\output"
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

# === FEATURE ENGINEERING ===
for i in range(1, sequence_length + 1):
    df[f'lag_{i}'] = df[target_column].shift(i)
df.dropna(inplace=True)

# === TRAIN/TEST SPLIT ===
X = df.drop(columns=['time', target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# === XGBOOST MODEL WITH GRID SEARCH ===
model = XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# === BEST MODEL ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# === METRICS ===
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.6f}")
print(f"R² Score: {r2:.6f}")

# === OUTPUT PREDICTIONS TO CSV ===
result_df = pd.DataFrame({
    'timestamp': df['time'].iloc[-len(y_test):].values,
    'actual_SO2': y_test.values,
    'predicted_SO2': y_pred
})

output_path = os.path.join(csv_folder, "xgboost_predictions.csv")
result_df.to_csv(output_path, index=False)
print(f"\nSaved predictions to: {output_path}")

# === CALL VISUALIZATION SCRIPT ===
from visualisation import visualize_predictions
visualize_predictions(output_path)
