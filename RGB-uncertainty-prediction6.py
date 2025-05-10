import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

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

def bootstrap_uncertainty_estimation(X_train, y_train, X_test, n_bootstraps=200, alpha=0.05):
    """
    Estimate uncertainty using bootstrap resampling
    
    Args:
    - X_train: Training features
    - y_train: Training target
    - X_test: Test features
    - n_bootstraps: Number of bootstrap iterations
    - alpha: Significance level for confidence interval
    
    Returns:
    - mean_pred: Mean predictions
    - lower_bound: Lower bound of prediction interval
    - upper_bound: Upper bound of prediction interval
    - uncertainty: Prediction interval width
    """
    # Prepare bootstrap predictions
    bootstrap_preds = np.zeros((n_bootstraps, len(X_test)))
    
    # Perform bootstrap resampling
    for i in range(n_bootstraps):
        # Resample training data with replacement
        indices = np.random.randint(0, len(X_train), len(X_train))
        X_train_bootstrap = X_train.iloc[indices]
        y_train_bootstrap = y_train.iloc[indices]
        
        # Fit model on bootstrap sample
        model = create_model()
        model.fit(X_train_bootstrap, y_train_bootstrap)
        
        # Predict on test set
        bootstrap_preds[i, :] = model.predict(X_test)
    
    # Compute statistics
    mean_pred = np.mean(bootstrap_preds, axis=0)
    
    # Compute prediction intervals
    lower_bound = np.percentile(bootstrap_preds, alpha/2 * 100, axis=0)
    upper_bound = np.percentile(bootstrap_preds, (1 - alpha/2) * 100, axis=0)
    
    # Compute uncertainty as interval width
    uncertainty = upper_bound - lower_bound
    
    return mean_pred, lower_bound, upper_bound, uncertainty

# Estimate uncertainty
mean_pred, lower_bound, upper_bound, calibrated_uncertainty = bootstrap_uncertainty_estimation(X_train, y_train, X_test)

# Evaluate model
r2 = r2_score(y_test, mean_pred)
rmse = np.sqrt(mean_squared_error(y_test, mean_pred))

within_ci_count = sum((y_test.values >= lower_bound) & (y_test.values <= upper_bound))
ci_percentage = (within_ci_count / len(y_test)) * 100

# Visualize predictions with uncertainty
plt.figure(figsize=(12, 6))
sort_idx = np.argsort(y_test.values)
x = np.arange(len(y_test))

plt.errorbar(x, mean_pred[sort_idx], yerr=calibrated_uncertainty[sort_idx], 
             fmt='o', ecolor='black', linewidth=1.5, capsize=4, label='Bootstrap Prediction Interval')
plt.scatter(x, y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, mean_pred[sort_idx], color='blue', label='Mean Prediction')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Bootstrap Uncertainty\nR² = {r2:.4f}, Coverage = {ci_percentage:.1f}%', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bootstrap_prediction_uncertainty.png', dpi=300)

# Calibration curve
def plot_bootstrap_calibration_curve(y_true, y_pred, lower_bound, upper_bound):
    plt.figure(figsize=(10, 6))
    confidence_levels = np.linspace(0.01, 0.99, 20)
    empirical_coverage = []
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    
    for conf in confidence_levels:
        # Compute interval based on confidence level
        # This uses percentile method similar to the uncertainty estimation
        lower = np.percentile(y_pred, (1 - conf) / 2 * 100)
        upper = np.percentile(y_pred, (1 + conf) / 2 * 100)
        
        # Check how many true values fall within these bounds
        in_interval = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverage.append(in_interval)
    
    plt.plot(confidence_levels, empirical_coverage, 'bo-', label='Bootstrap Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Expected Confidence Level', fontsize=14)
    plt.ylabel('Empirical Coverage', fontsize=14)
    plt.title('Bootstrap Uncertainty Calibration Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('bootstrap_calibration_curve.png', dpi=300)
    plt.close()

# Generate calibration curve
plot_bootstrap_calibration_curve(y_test.values, mean_pred, lower_bound, upper_bound)

# Print performance metrics
print("Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Actual values within Prediction Interval: {within_ci_count}/{len(y_test)} ({ci_percentage:.1f}%)")