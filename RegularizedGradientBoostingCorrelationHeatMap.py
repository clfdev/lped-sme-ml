import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Increase font size globally for all matplotlib plots
plt.rcParams.update({
    'font.size': 14,            # Base font size
    'axes.titlesize': 20,       # Title font size
    'axes.labelsize': 18,       # Axis label font size
    'xtick.labelsize': 16,      # x-axis tick label size
    'ytick.labelsize': 16,      # y-axis tick label size
    'legend.fontsize': 16,      # Legend font size
    'figure.titlesize': 22,     # Figure title size
})

# Load the dataset
file_path = 'Dataset3.xlsx'
data = pd.read_excel(file_path)

# Create the features used in the original regularized gradient boosting model
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Calculate Lennard-Jones potential
epsilon = 1
sigma = 1
data['Lennard_Jones potential'] = 4 * epsilon * (
    (sigma * data['Inverse interatomic distance']) ** 12 - 
    (sigma * data['Inverse interatomic distance']) ** 6
)

# Select features used in the original regularized gradient boosting model
selected_features = ['Interatomic distance', 'Atomic charge A (MK)', 'Atomic charge B (MK)', 
                    'Lennard_Jones potential', 'LPED']

# Create dataframe with selected features
data_gb = data[selected_features]

# Generate correlation matrix
corr_gb = data_gb.corr()
print("\nCorrelation Matrix for Original Regularized Gradient Boosting Features:")
print(corr_gb)

# Create enhanced heatmap
plt.figure(figsize=(12, 10))

# Generate mask for upper triangle (optional, for cleaner visualization)
# mask = np.triu(np.ones_like(corr_gb, dtype=bool))

# Create heatmap with larger annotations and line widths
heatmap = sns.heatmap(
    corr_gb, 
    annot=True,                   # Show values on cells
    cmap='coolwarm',              # Color map
    fmt='.2f',                    # Format for values (.2f = 2 decimal places)
    linewidths=1,                 # Width of cell borders
    # mask=mask,                  # Optional: mask for upper triangle
    annot_kws={"size": 16},       # Font size of annotations inside cells
    square=True,                  # Make cells square-shaped
    cbar_kws={"shrink": 0.8}      # Resize colorbar
)

# Customize title and labels
plt.title('Correlation Matrix: Original Regularized Gradient Boosting Features', fontsize=22, pad=20)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Save the enhanced heatmap
plt.savefig('gradient_boosting_correlation_heatmap.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Optional: Create a feature importance heatmap based on the model
# Define the target and features
X = data_gb.drop('LPED', axis=1)
y = data_gb['LPED']

# Create and train the original regularized gradient boosting model
regularized_gb = make_pipeline(
    StandardScaler(), 
    GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
)

# Fit the model
regularized_gb.fit(X, y)

# Extract feature importances
feature_importances = regularized_gb.named_steps['gradientboostingregressor'].feature_importances_

# Create dataframe for feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Create feature importance visualization
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')

# Add value labels to the bars
for i, v in enumerate(importance_df['Importance']):
    ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=14)

plt.title('Feature Importance: Original Regularized Gradient Boosting', fontsize=22, pad=20)
plt.xlabel('Importance', fontsize=18)
plt.ylabel('Feature', fontsize=18)
plt.xlim(0, max(importance_df['Importance']) * 1.15)  # Add some space for the value labels
plt.tight_layout()
plt.savefig('gradient_boosting_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()