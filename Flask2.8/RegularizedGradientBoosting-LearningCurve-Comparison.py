import matplotlib.pyplot as plt
import numpy as np

# Sample data (approximating what's in your plot)
training_examples = [5, 10, 20, 30, 40, 50]

# Approximate MSE values from your plot
original_regularized_gb = [13.0, 10.8, 7.8, 6.2, 5.0, 4.9]
further_regularized_gb = [12.5, 9.5, 5.9, 5.7, 5.0, 4.8]
gb_early_stopping = [13.2, 11.0, 6.4, 6.2, 5.3, 5.6]
balanced_gb = [9.3, 8.5, 7.3, 5.7, 5.0, 4.8]

plt.figure(figsize=(14, 10))

# Increase all font sizes
plt.rcParams.update({
    'font.size': 16,          # Base font size 
    'axes.titlesize': 22,     # Title font size
    'axes.labelsize': 20,     # Axis label font size
    'xtick.labelsize': 18,    # X-axis tick label size
    'ytick.labelsize': 18,    # Y-axis tick label size
    'legend.fontsize': 18,    # Legend font size
})

# Plot with larger markers and line width
plt.plot(training_examples, original_regularized_gb, 'o-', color='blue', linewidth=2.5, markersize=10, label='Original Regularized GB')
plt.plot(training_examples, further_regularized_gb, 'o-', color='orange', linewidth=2.5, markersize=10, label='Further Regularized GB')
plt.plot(training_examples, gb_early_stopping, 'o-', color='green', linewidth=2.5, markersize=10, label='GB with Early Stopping')
plt.plot(training_examples, balanced_gb, 'o-', color='red', linewidth=2.5, markersize=10, label='Balanced GB')

# Add title and labels with larger fonts
plt.title('Comparison of Learning Curves (MSE)', fontsize=24, pad=20)
plt.xlabel('Training examples', fontsize=22, labelpad=15)
plt.ylabel('Cross-validation MSE', fontsize=22, labelpad=15)

# Create a larger legend and place it outside the plot for better readability
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
           ncol=2, fontsize=18, frameon=True, framealpha=1, 
           edgecolor='black', borderpad=1)

# Adjust the layout to make room for the legend
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Save the figure with high resolution
plt.savefig('learning_curve_mse_comparison_larger_font.png', dpi=300, bbox_inches='tight')

plt.show()