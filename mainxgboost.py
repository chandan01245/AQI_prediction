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
csv_folder = r"output"
target_column = "ColumnAmountSO2"
sequence_length = 10

# Define the name of the output CSV file
OUTPUT_PREDICTIONS_CSV = "xgboost_predictions_optimized.csv"

# === LOAD & CONCATENATE CSV FILES ===
# Exclude the output file itself from the input list
all_files = [f for f in glob.glob(os.path.join(csv_folder, "*.csv"))
             if os.path.basename(f) != OUTPUT_PREDICTIONS_CSV]

if not all_files:
    raise FileNotFoundError(f"No input CSV files found in '{csv_folder}' (excluding '{OUTPUT_PREDICTIONS_CSV}'). Please ensure 'multipleExtraction.py' has been run.")

df_list = [pd.read_csv(file) for file in all_files]
data = pd.concat(df_list, ignore_index=True)

# === DATETIME HANDLING ===
data['time'] = pd.to_datetime(data['time'], errors='coerce')
data.sort_values('time', inplace=True)

# === FEATURE ENGINEERING ===
# Ensure target column is numeric before engineering features
data[target_column] = pd.to_numeric(data[target_column], errors='coerce')

max_sequence_length = 10
sequence_length = min(max_sequence_length, len(data) // 2)
if sequence_length < 1:
    raise ValueError("Not enough data for feature engineering.")

for i in range(1, sequence_length + 1):
    data[f'lag_{i}'] = data[target_column].shift(i)

data['rolling_mean'] = data[target_column].rolling(window=sequence_length).mean()
data['exp_smooth'] = data[target_column].ewm(span=sequence_length).mean()

# Apply ffill then bfill to engineered features to handle leading/trailing NaNs
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)


# === PREPARE FEATURES & TARGET ===
# Drop the original 'time' column and the target column itself from features
X = data.drop(columns=['time', target_column], errors='ignore')
X = X.select_dtypes(include=[np.number]) # Select only numeric columns

# Print the columns of X before initial NaN check - THIS IS THE CRITICAL LIST
print("Features before scaling in mainxgboost.py:", X.columns.tolist())

if X.empty:
    raise ValueError("No numeric features found after preprocessing.")

y = data[target_column].values

# === ROBUSTLY HANDLE PROBLEMATIC TARGET VALUES AND ALIGN FEATURES ===
# Ensure all relevant columns are numeric before combining and cleaning
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Combine X and y into a single DataFrame for synchronized cleaning
combined_df = X.copy()
combined_df[target_column] = y

# Drop rows where the target_column is NaN, Inf, or too large
initial_valid_mask = ~(np.isnan(combined_df[target_column]) |
                       np.isinf(combined_df[target_column]) |
                       (np.abs(combined_df[target_column]) > 1e10))
combined_df = combined_df[initial_valid_mask]

# Drop rows where any feature column is NaN or Inf. This is crucial for XGBoost.
# Replace infinities with NaN first so dropna can remove them.
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df.dropna(inplace=True)

X_cleaned = combined_df.drop(columns=[target_column])
y_cleaned = combined_df[target_column].values

print(f"Shape of features after cleaning: {X_cleaned.shape}")
print(f"Shape of target after cleaning: {y_cleaned.shape}")

if y_cleaned.size == 0 or X_cleaned.empty:
    raise ValueError("No valid data left after cleaning features and target. Cannot train model.")

# Ensure X_cleaned is a DataFrame with consistent column order (matching original X's columns)
# This is important if original X had a specific order that the scaler expects.
# Use the columns from the X that was generated *before* the combined_df cleaning.
X_cleaned = X_cleaned[X.columns.tolist()]

# === SCALING ===
scaler = StandardScaler()
# Print the shape of X_cleaned right before fitting the scaler
print(f"Shape of X_cleaned before scaler.fit_transform: {X_cleaned.shape}")
X_scaled = scaler.fit_transform(X_cleaned)

# === TRAIN/TEST SPLIT ===
# Ensure split is done on the cleaned data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cleaned, test_size=0.2, shuffle=False)

if X_train.size == 0 or y_train.size == 0:
    raise ValueError("Training data is empty after split. Check data size and test_size.")

# === MODEL TRAINING ===
model = XGBRegressor(objective='reg:squarederror', random_state=42)
param_distributions = {
    'n_estimators': range(50, 300, 50),
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': range(3, 10),
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
ts_split = TimeSeriesSplit(n_splits=5)

# Add error handling for RandomizedSearchCV fit
try:
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=ts_split, n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
except Exception as e:
    print(f"Error during RandomizedSearchCV fit: {e}")
    print("This might be due to insufficient data for TimeSeriesSplit or remaining data issues.")
    exit() # Exit if training fails

# === PREDICTIONS & OUTPUT ===
# Ensure best_model is defined before predicting
if 'best_model' not in locals():
    print("Error: best_model was not trained successfully. Cannot make predictions.")
    exit()

y_pred = best_model.predict(X_test)

# Align predictions with original data's time and other features
# We need to get the original 'data' indices that correspond to X_test (from combined_df)
final_data_indices = combined_df.index[-len(y_test):] # These are the original indices from 'data'

result_df = pd.DataFrame({
    'timestamp': data['time'].loc[final_data_indices].values,
    'nTimes': data['nTimes'].loc[final_data_indices].values if 'nTimes' in data.columns else np.nan,
    'nXtrack': data['nXtrack'].loc[final_data_indices].values if 'nXtrack' in data.columns else np.nan,
    'latitude': data['latitude'].loc[final_data_indices].values if 'latitude' in data.columns else np.nan,
    'longitude': data['longitude'].loc[final_data_indices].values if 'longitude' in data.columns else np.nan,
    'actual_SO2': y_test, # This is the actual ColumnAmountSO2 from the test set
    'predicted_SO2': y_pred
})
output_path = os.path.join(csv_folder, OUTPUT_PREDICTIONS_CSV)
result_df.to_csv(output_path, index=False)
print(f"Saved predictions to: {output_path}")

# === SAVE MODEL AND SCALER CORRECTLY ===
model_file = os.path.join(csv_folder, "xgboost_model.pkl")
scaler_file = os.path.join(csv_folder, "scaler.pkl")
joblib.dump(best_model, model_file)
joblib.dump(scaler, scaler_file)
print(f"Model and scaler saved to: {model_file}, {scaler_file}")

# === CALL VISUALIZATION SCRIPT ===
try:
    from visualisation import visualize_predictions
    visualize_predictions(output_path)
except ImportError:
    print("Warning: 'visualisation.py' not found, skipping visualization.")

# === OPTIONAL: FUTURE PREDICTION (Moved to predict_futures02.py) ===
