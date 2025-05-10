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

def quantile_uncertainty_calibration(X_train, y_train, X_test, y_test, quantiles=[0.05, 0.95]):
    """
    Calibrate uncertainty using quantile regression
    
    Args:
    - X_train: Training features
    - y_train: Training target
    - X_test: Test features
    - y_test: Test target
    - quantiles: List of quantiles to estimate (default: 5th and 95th percentiles)
    
    Returns:
    - mean_pred: Mean predictions
    - lower_bound: Lower bound of prediction interval
    - upper_bound: Upper bound of prediction interval
    - uncertainty: Prediction interval width
    """
    # Create models for different quantiles
    models = {}
    for q in quantiles:
        model = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                loss='quantile',
                alpha=q,
                random_state=42
            )
        )
        model.fit(X_train, y_train)
        models[q] = model
    
    # Predict mean and quantiles
    mean_model = create_model()
    mean_model.fit(X_train, y_train)
    
    mean_pred = mean_model.predict(X_test)
    lower_pred = models[quantiles[0]].predict(X_test)
    upper_pred = models[quantiles[1]].predict(X_test)
    
    # Compute uncertainty as interval width
    uncertainty = upper_pred - lower_pred
    
    return mean_pred, lower_pred, upper_pred, uncertainty

# Estimate uncertainty
mean_pred, lower_bound, upper_bound, calibrated_uncertainty = quantile_uncertainty_calibration(X_train, y_train, X_test, y_test)

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
             fmt='o', ecolor='black', linewidth=1.5, capsize=4, label='Quantile Prediction Interval')
plt.scatter(x, y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, mean_pred[sort_idx], color='blue', label='Mean Prediction')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Quantile Uncertainty Calibration\nR² = {r2:.4f}, Coverage = {ci_percentage:.1f}%', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quantile_calibrated_prediction_uncertainty.png', dpi=300)

# Calibration curve
def plot_quantile_calibration_curve(y_true, y_pred, lower_bound, upper_bound):
    plt.figure(figsize=(10, 6))
    confidence_levels = np.linspace(0.01, 0.99, 20)
    empirical_coverage = []
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    
    for conf in confidence_levels:
        # Compute interval based on confidence level
        in_interval = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        empirical_coverage.append(in_interval)
    
    plt.plot(confidence_levels, empirical_coverage, 'bo-', label='Quantile Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Expected Confidence Level', fontsize=14)
    plt.ylabel('Empirical Coverage', fontsize=14)
    plt.title('Quantile Uncertainty Calibration Curve', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('quantile_calibration_curve.png', dpi=300)
    plt.close()

# Generate calibration curve
plot_quantile_calibration_curve(y_test.values, mean_pred, lower_bound, upper_bound)

# Print performance metrics
print("Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Actual values within Prediction Interval: {within_ci_count}/{len(y_test)} ({ci_percentage:.1f}%)")