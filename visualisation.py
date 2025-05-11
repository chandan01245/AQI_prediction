# visualize_predictions.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_predictions(prediction_file):
    # Load data
    data = pd.read_csv(prediction_file)
    
    # Convert timestamp column
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print(data['timestamp'].head())

    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(data['timestamp'], data['actual_SO2'], label='Actual SO2', color='blue', linewidth=1.5)
    plt.plot(data['timestamp'], data['predicted_SO2'], label='Predicted SO2', color='red', linewidth=1.5)
    plt.title('Actual vs Predicted SO2 Levels')
    plt.xlabel('Timestamp')
    plt.ylabel('SO2 Levels')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save plot
    output_plot_path = os.path.join(os.path.dirname(prediction_file), "lstm_predictions_visualization.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"\nSaved visualization to: {output_plot_path}")
