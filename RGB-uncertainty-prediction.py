import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.patches import Rectangle

# Load the dataset
file_path = 'Dataset3.xlsx'
data = pd.read_excel(file_path)

# Create inverse interatomic distance
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create Lennard-Jones potential
epsilon = 1
sigma = 1
data['Lennard_Jones potential'] = 4 * epsilon * (
    (sigma * data['Inverse interatomic distance']) ** 12 - 
    (sigma * data['Inverse interatomic distance']) ** 6
)

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'])

# Separate features and target
X = data_cleaned.drop(columns=['LPED'])
y = data_cleaned['LPED']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Define the Regularized Gradient Boosting model
def create_model():
    return make_pipeline(
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

# Function to estimate prediction uncertainty using cross-validation
def estimate_prediction_uncertainty(X_train, y_train, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = []
    
    for train_idx, val_idx in kf.split(X_train):
        # Split data for this fold
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        
        # Create and train model
        model = create_model()
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on test set
        fold_predictions = model.predict(X_test)
        predictions.append(fold_predictions)
    
    # Convert to numpy array for easier calculation
    predictions = np.array(predictions)
    
    # Calculate mean and standard deviation for each test point
    mean_prediction = np.mean(predictions, axis=0)
    prediction_std = np.std(predictions, axis=0)
    
    # Calculate 95% confidence intervals
    lower_bound = mean_prediction - 1.96 * prediction_std
    upper_bound = mean_prediction + 1.96 * prediction_std
    
    return mean_prediction, lower_bound, upper_bound, prediction_std

# Estimate prediction uncertainty
mean_pred, lower_bound, upper_bound, pred_std = estimate_prediction_uncertainty(X_train, y_train, X_test)

# Create a final model trained on all training data for comparison
final_model = create_model()
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, final_pred)
rmse = np.sqrt(mean_squared_error(y_test, final_pred))

# Create a DataFrame to display results
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': final_pred,
    'Mean CV Prediction': mean_pred,
    'Lower Bound': lower_bound,
    'Upper Bound': upper_bound,
    'Uncertainty (Std Dev)': pred_std,
    'Uncertainty (%)': (pred_std / abs(mean_pred) * 100)
})

print("Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} kcal mol⁻¹ Bohr⁻³")
print("\nPrediction Results with Uncertainty:")
print(results_df)

# Calculate average uncertainty
avg_uncertainty = np.mean(pred_std)
avg_uncertainty_percent = np.mean(pred_std / abs(mean_pred) * 100)
print(f"\nAverage Prediction Uncertainty: {avg_uncertainty:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Average Uncertainty as Percentage: {avg_uncertainty_percent:.2f}%")

# Visualize predictions with uncertainty
plt.figure(figsize=(12, 6))

# Sort by actual values for better visualization
sort_idx = np.argsort(y_test.values)
x = np.arange(len(y_test))

plt.errorbar(x, mean_pred[sort_idx], yerr=1.96*pred_std[sort_idx], 
             fmt='o', ecolor='black', linewidth=1.5, capsize=4, label='95% Confidence Interval')
plt.scatter(x, y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, mean_pred[sort_idx], color='blue', label='Mean Prediction')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Uncertainty\nR² = {r2:.4f}, Avg. Uncertainty = {avg_uncertainty_percent:.2f}%', 
          fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_uncertainty.png', dpi=300)

# Create histogram of uncertainty distribution
plt.figure(figsize=(10, 6))
plt.hist(pred_std, bins=10, alpha=0.7, edgecolor='black')
plt.axvline(avg_uncertainty, color='red', linestyle='--', 
            label=f'Mean Uncertainty: {avg_uncertainty:.4f}')
plt.xlabel('Prediction Standard Deviation (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Prediction Uncertainty', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uncertainty_distribution.png', dpi=300)


# Create scatter plot of uncertainty vs prediction value
plt.figure(figsize=(10, 6))

# Calculate the interquartile range of uncertainties
q1 = np.percentile(pred_std, 25)  # 25th percentile
q3 = np.percentile(pred_std, 75)  # 75th percentile

# Get the x-axis limits
x_min = min(abs(mean_pred))
x_max = max(abs(mean_pred))

# Create a rectangle patch
iqr_rect = Rectangle((x_min, q1), x_max - x_min, q3 - q1, 
                     facecolor='lightgray', alpha=0.4, 
                     edgecolor='gray', linestyle='--', linewidth=1.5,
                     label='Interquartile Range (25-75%)')

# Add the rectangle to the plot
plt.gca().add_patch(iqr_rect)

# Add horizontal lines at Q1 and Q3
plt.axhline(y=q1, color='gray', linestyle='--', alpha=0.8, linewidth=1.0)
plt.axhline(y=q3, color='gray', linestyle='--', alpha=0.8, linewidth=1.0)

# Add text labels for Q1 and Q3
plt.text(x_max + 0.1, q1, f'Q1: {q1:.3f}', verticalalignment='center')
plt.text(x_max + 0.1, q3, f'Q3: {q3:.3f}', verticalalignment='center')

# Plot the scatter points on top of the rectangle
plt.scatter(abs(mean_pred), pred_std, alpha=0.7, zorder=10)

plt.xlabel('|Predicted LPED| (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.ylabel('Prediction Uncertainty (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title('Uncertainty vs. Magnitude of Prediction', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('uncertainty_vs_prediction.png', dpi=300)


# Display summary statistics of uncertainty
uncertainty_df = pd.DataFrame({
    'Statistic': ['Min', 'Q1 (25%)', 'Median', 'Mean', 'Q3 (75%)', 'Max'],
    'Absolute Value (kcal mol⁻¹ Bohr⁻³)': [
        np.min(pred_std),
        np.percentile(pred_std, 25),
        np.median(pred_std),
        np.mean(pred_std),
        np.percentile(pred_std, 75),
        np.max(pred_std)
    ],
    'Relative Value (%)': [
        np.min(pred_std / abs(mean_pred) * 100),
        np.percentile(pred_std / abs(mean_pred) * 100, 25),
        np.median(pred_std / abs(mean_pred) * 100),
        np.mean(pred_std / abs(mean_pred) * 100),
        np.percentile(pred_std / abs(mean_pred) * 100, 75),
        np.max(pred_std / abs(mean_pred) * 100)
    ]
})

print("\nUncertainty Summary Statistics:")
print(uncertainty_df)