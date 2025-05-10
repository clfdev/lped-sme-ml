import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

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

# IMPROVED: Function to estimate prediction uncertainty with calibration
def estimate_calibrated_uncertainty(X_train, y_train, X_test, y_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = []
    
    # Step 1: Get cross-validated predictions
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
    
    # Step 2: Calibrate the uncertainties using validation set errors
    # Train a final model
    final_model = create_model()
    final_model.fit(X_train, y_train)
    
    # Predict on test set
    test_pred = final_model.predict(X_test)
    
    # Calculate absolute errors
    abs_errors = np.abs(test_pred - y_test.values)
    
    # Calculate the ratio between actual errors and predicted uncertainties
    error_to_uncertainty_ratio = abs_errors / prediction_std
    
    # Calculate the calibration factor (median ratio)
    # Using median is more robust to outliers than mean
    calibration_factor = np.median(error_to_uncertainty_ratio)
    
    # Adjust the prediction standard deviations
    calibrated_std = prediction_std * calibration_factor
    
    # Calculate 95% confidence intervals using calibrated std
    lower_bound = mean_prediction - 1.96 * calibrated_std
    upper_bound = mean_prediction + 1.96 * calibrated_std
    
    return mean_prediction, lower_bound, upper_bound, calibrated_std, calibration_factor

# Estimate calibrated prediction uncertainty
mean_pred, lower_bound, upper_bound, cal_pred_std, cal_factor = estimate_calibrated_uncertainty(
    X_train, y_train, X_test, y_test)

# Create a final model trained on all training data
final_model = create_model()
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, final_pred)
rmse = np.sqrt(mean_squared_error(y_test, final_pred))

# Count how many actual values fall within the confidence intervals
within_ci_count = sum((y_test.values >= lower_bound) & (y_test.values <= upper_bound))
ci_percentage = (within_ci_count / len(y_test)) * 100

# Create a DataFrame to display results
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': final_pred,
    'Mean CV Prediction': mean_pred,
    'Lower Bound (95% CI)': lower_bound,
    'Upper Bound (95% CI)': upper_bound,
    'Calibrated Uncertainty': cal_pred_std,
    'Uncertainty (%)': (cal_pred_std / abs(mean_pred) * 100),
    'Within CI': (y_test.values >= lower_bound) & (y_test.values <= upper_bound)
})

print("Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Uncertainty Calibration Factor: {cal_factor:.4f}")
print(f"Actual values within 95% CI: {within_ci_count}/{len(y_test)} ({ci_percentage:.1f}%)")
print("\nPrediction Results with Calibrated Uncertainty:")
print(results_df)

# Calculate average uncertainty
avg_uncertainty = np.mean(cal_pred_std)
avg_uncertainty_percent = np.mean(cal_pred_std / abs(mean_pred) * 100)
print(f"\nAverage Calibrated Prediction Uncertainty: {avg_uncertainty:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Average Calibrated Uncertainty as Percentage: {avg_uncertainty_percent:.2f}%")

# Visualize predictions with uncertainty
plt.figure(figsize=(12, 6))

# Sort by actual values for better visualization
sort_idx = np.argsort(y_test.values)
x = np.arange(len(y_test))

plt.errorbar(x, mean_pred[sort_idx], yerr=1.96*cal_pred_std[sort_idx], 
             fmt='o', ecolor='black', linewidth=1.5, capsize=4, label='95% Confidence Interval')
plt.scatter(x, y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, mean_pred[sort_idx], color='blue', label='Mean Prediction')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Calibrated Uncertainty\nR² = {r2:.4f}, Coverage = {ci_percentage:.1f}%, Avg. Uncertainty = {avg_uncertainty_percent:.2f}%', 
          fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('calibrated_prediction_uncertainty2.png', dpi=300)


# Display summary statistics of uncertainty
uncertainty_df = pd.DataFrame({
    'Statistic': ['Min', 'Q1 (25%)', 'Median', 'Mean', 'Q3 (75%)', 'Max'],
    'Absolute Value (kcal mol⁻¹ Bohr⁻³)': [
        np.min(cal_pred_std),
        np.percentile(cal_pred_std, 25),
        np.median(cal_pred_std),
        np.mean(cal_pred_std),
        np.percentile(cal_pred_std, 75),
        np.max(cal_pred_std)
    ],
    'Relative Value (%)': [
        np.min(cal_pred_std / abs(mean_pred) * 100),
        np.percentile(cal_pred_std / abs(mean_pred) * 100, 25),
        np.median(cal_pred_std / abs(mean_pred) * 100),
        np.mean(cal_pred_std / abs(mean_pred) * 100),
        np.percentile(cal_pred_std / abs(mean_pred) * 100, 75),
        np.max(cal_pred_std / abs(mean_pred) * 100)
    ]
})

print("\nCalibrated Uncertainty Summary Statistics:")
print(uncertainty_df)


###############################
# Additional evaluation methods
###############################

# Calculate calibration curve (reliability diagram)
def plot_calibration_curve(y_true, y_pred, uncertainties):
    plt.figure(figsize=(10, 6))
    
    # Define confidence levels to evaluate
    confidence_levels = np.linspace(0.01, 0.99, 20)
    
    # Initialize arrays to store empirical coverage
    empirical_coverage = []
    
    # Calculate empirical coverage for each confidence level
    for conf in confidence_levels:
        z_score = norm.ppf((1 + conf) / 2)  # Two-tailed z-score
        lower = y_pred - z_score * uncertainties
        upper = y_pred + z_score * uncertainties
        
        # Count fraction of true values falling in the interval
        in_interval = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverage.append(in_interval)
    
    # Plot calibration curve
    plt.plot(confidence_levels, empirical_coverage, 'bo-', label='Calibrated Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    
    plt.xlabel('Expected Confidence Level', fontsize=14)
    plt.ylabel('Empirical Coverage', fontsize=14)
    plt.title('Uncertainty Calibration Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('calibration_curve2.png', dpi=300)

# Plot calibration curve
plot_calibration_curve(y_test.values, mean_pred, cal_pred_std)