import matplotlib.pyplot as plt
import numpy as np

# Data
sizes = np.array([33, 43, 53, 63])
r2 = np.array([0.75, 0.75, 0.88, 0.83])
mae = np.array([1.42, 1.12, 0.72, 0.94])
rmse = np.array([2.04, 1.46, 0.91, 1.26])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each metric
plt.plot(sizes, r2, 'o-', label='R²', color='blue', linewidth=2, markersize=10)
plt.plot(sizes, mae, 's-', label='MAE', color='green', linewidth=2, markersize=10)
plt.plot(sizes, rmse, '^-', label='RMSE', color='red', linewidth=2, markersize=10)

# Add vertical line at optimal point
plt.axvline(x=53, color='gray', linestyle='--', alpha=0.5)

# Customize the plot
plt.xlabel('Dataset Size', fontsize=20)
plt.ylabel('Metric Value', fontsize=20)
plt.title('Model Performance Metrics vs Dataset Size', fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=18)

# Increase tick label size
plt.tick_params(axis='both', which='major', labelsize=17)  # Added this line

# Add annotations for the optimal point
plt.annotate(f'Optimal point\nR²={0.88}\nMAE={0.72}\nRMSE={0.91}', 
            xy=(53, 0.88), xytext=(50, 1.5),
            fontsize=16,  # Added this line to increase text size
            arrowprops=dict(facecolor='black', shrink=0.05),
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

# Set axis limits with some padding
plt.ylim(0.5, 2.2)

# Show the plot
plt.tight_layout()
plt.savefig('optimal_dataset_plot.png', dpi=350, bbox_inches='tight')
plt.show()
