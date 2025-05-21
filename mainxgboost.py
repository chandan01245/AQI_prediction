import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
import joblib

# --- LOAD SATELLITE DATA ---
data = pd.read_csv('so2.csv')
data['datetime'] = pd.to_datetime(data['time']).dt.tz_localize(None)
data = data[['datetime', 'sulfurdioxide_total_vertical_column']]
data.rename(columns={'sulfurdioxide_total_vertical_column': 'satellite_SO2'}, inplace=True)
print(f"Satellite data size: {data.shape}")

# --- LOAD GROUND DATA ---
so2_ground_df = pd.read_csv('so2[ground data].csv')
so2_ground_df['datetime'] = pd.to_datetime(so2_ground_df['datetimeUtc']).dt.tz_localize(None)
so2_ground_df = so2_ground_df[['datetime', 'value']]
so2_ground_df.rename(columns={'value': 'ground_SO2'}, inplace=True)
print(f"Ground data size: {so2_ground_df.shape}")

# --- MERGE ON datetime (nearest join with 1 minute tolerance) ---
data = data.sort_values('datetime')
so2_ground_df = so2_ground_df.sort_values('datetime')

merged = pd.merge_asof(
    data,
    so2_ground_df,
    on='datetime',
    direction='nearest',
    tolerance=pd.Timedelta('1min')
)

# Drop rows where no match found (NaN in ground_SO2)
merged.dropna(inplace=True)

print(f"Merged data size: {merged.shape}")

# Save merged dataset
merged_csv_path = "merged_data.csv"
merged.to_csv(merged_csv_path, index=False)
print(f"Saved merged dataset to: {merged_csv_path}")

# --- FEATURE ENGINEERING ---
target_column = "satellite_SO2"
max_sequence_length = 10
available_rows = len(merged)
sequence_length = min(max_sequence_length, available_rows // 2)

if sequence_length < 1:
    raise ValueError("Not enough data for feature engineering. Please provide more data.")

for i in range(1, sequence_length + 1):
    merged[f'lag_{i}'] = merged[target_column].shift(i)

merged['rolling_mean'] = merged[target_column].rolling(window=sequence_length).mean()
merged['exp_smooth'] = merged[target_column].ewm(span=sequence_length).mean()

merged.fillna(method='bfill', inplace=True)

print(f"Final data size after feature engineering: {merged.shape}")

# --- PREPARE FEATURES AND LABEL ---
X = merged.drop(columns=['datetime', target_column])
X = X.select_dtypes(include=[np.number])
y = merged[target_column].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# --- XGBOOST MODEL + RANDOMIZED SEARCH ---
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

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# --- OUTPUT PREDICTIONS TO CSV ---
result_df = pd.DataFrame({
    'datetime': merged['datetime'].iloc[-len(y_test):].values,
    'actual_SO2': y_test,
    'predicted_SO2': y_pred
})

result_df.to_csv("xgboost_predictions_optimized.csv", index=False)
print(f"\nSaved predictions to: xgboost_predictions_optimized.csv")

# --- SAVE MODEL AND SCALER ---
joblib.dump(best_model, "xgboost_best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"\nSaved model to: xgboost_best_model.pkl")
print(f"Saved scaler to: scaler.pkl")
