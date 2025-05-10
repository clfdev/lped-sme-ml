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

def residual_based_uncertainty_calibration(X_train, y_train, X_test, y_test):
    """
    Estimate uncertainty using residual-based calibration
    
    Args:
    - X_train: Training features
    - y_train: Training target
    - X_test: Test features
    - y_test: Test target
    
    Returns:
    - mean_pred: Mean predictions
    - lower_bound: Lower bound of prediction interval
    - upper_bound: Upper bound of prediction interval
    - uncertainty: Prediction interval width
    """
    # Create and fit the model
    model = create_model()
    model.fit(X_train, y_train)
    
    # Predict on training set to analyze residuals
    train_pred = model.predict(X_train)
    train_residuals = np.abs(train_pred - y_train)
    
    # Predict on test set
    mean_pred = model.predict(X_test)
    
    # Compute residual-based scaling factor
    # Use median absolute deviation (MAD) for robust scaling
    residual_scale = 1.4826 * np.median(train_residuals)
    
    # Estimate uncertainty using residual statistics
    # Compute uncertainty as a function of prediction magnitude and residual scale
    base_uncertainty = np.std(train_residuals)
    
    # Adaptive uncertainty estimation
    # Scales uncertainty based on prediction magnitude and residual characteristics
    adaptive_scaling = np.abs(mean_pred) / np.mean(np.abs(train_pred))
    uncertainty = base_uncertainty * adaptive_scaling * residual_scale
    
    # Compute prediction intervals
    lower_bound = mean_pred - uncertainty
    upper_bound = mean_pred + uncertainty
    
    return mean_pred, lower_bound, upper_bound, uncertainty

# Estimate uncertainty
mean_pred, lower_bound, upper_bound, calibrated_uncertainty = residual_based_uncertainty_calibration(X_train, y_train, X_test, y_test)

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
             fmt='o', ecolor='black', linewidth=1.5, capsize=4, label='Residual-based Prediction Interval')
plt.scatter(x, y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, mean_pred[sort_idx], color='blue', label='Mean Prediction')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Residual-based Uncertainty\nR² = {r2:.4f}, Coverage = {ci_percentage:.1f}%', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residual_prediction_uncertainty.png', dpi=300)

# Calibration curve
def plot_residual_calibration_curve(y_true, y_pred, lower_bound, upper_bound):
    plt.figure(figsize=(10, 6))
    confidence_levels = np.linspace(0.01, 0.99, 20)
    empirical_coverage = []
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    
    for conf in confidence_levels:
        # Compute theoretical prediction interval
        z_score = np.abs(norm.ppf((1 - conf) / 2))
        theoretical_lower = y_pred - z_score * (upper_bound - y_pred)
        theoretical_upper = y_pred + z_score * (upper_bound - y_pred)
        
        # Check how many true values fall within these bounds
        in_interval = np.mean((y_true >= theoretical_lower) & (y_true <= theoretical_upper))
        empirical_coverage.append(in_interval)
    
    plt.plot(confidence_levels, empirical_coverage, 'bo-', label='Residual-based Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Expected Confidence Level', fontsize=14)
    plt.ylabel('Empirical Coverage', fontsize=14)
    plt.title('Residual-based Uncertainty Calibration Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('residual_calibration_curve.png', dpi=300)
    plt.close()

# Import norm for z-score calculation
from scipy.stats import norm

# Generate calibration curve
plot_residual_calibration_curve(y_test.values, mean_pred, lower_bound, upper_bound)

# Print performance metrics
print("Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Actual values within Prediction Interval: {within_ci_count}/{len(y_test)} ({ci_percentage:.1f}%)")