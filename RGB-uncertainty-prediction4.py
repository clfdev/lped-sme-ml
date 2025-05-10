import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression

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

def estimate_nonlinear_uncertainty(X_train, y_train, X_test, y_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Collect cross-validation predictions and errors for the training set
    train_cv_preds = np.zeros_like(y_train, dtype=float)
    train_cv_errors = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = create_model()
        model.fit(X_train_fold, y_train_fold)
        
        val_pred = model.predict(X_val_fold)
        train_cv_preds[val_idx] = val_pred
        train_cv_errors.extend(np.abs(val_pred - y_val_fold))
    
    # Fit a single model on entire training data
    full_model = create_model()
    full_model.fit(X_train, y_train)
    
    # Predict on test set
    test_pred = full_model.predict(X_test)
    
    # Step 2: Train Isotonic Regression on absolute errors
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(train_cv_preds, train_cv_errors)
    
    # Predict uncertainty for test set
    calibrated_uncertainty = ir.predict(test_pred)
    
    # Step 3: Compute prediction intervals
    lower_bound = test_pred - calibrated_uncertainty
    upper_bound = test_pred + calibrated_uncertainty
    
    return test_pred, lower_bound, upper_bound, calibrated_uncertainty

# Estimate nonlinear calibrated uncertainty
mean_pred, lower_bound, upper_bound, calibrated_uncertainty = estimate_nonlinear_uncertainty(X_train, y_train, X_test, y_test)

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
             fmt='o', ecolor='black', linewidth=1.5, capsize=4, label='Nonlinear Prediction Interval')
plt.scatter(x, y_test.values[sort_idx], color='red', label='Actual Values')
plt.scatter(x, mean_pred[sort_idx], color='blue', label='Mean Prediction')

plt.xlabel('Test Sample Index (sorted by actual value)', fontsize=14)
plt.ylabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=14)
plt.title(f'LPED Predictions with Nonlinear Uncertainty Calibration\nR² = {r2:.4f}, Coverage = {ci_percentage:.1f}%', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nonlinear_calibrated_prediction_uncertainty.png', dpi=300)

print("Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} kcal mol⁻¹ Bohr⁻³")
print(f"Actual values within Prediction Interval: {within_ci_count}/{len(y_test)} ({ci_percentage:.1f}%)")


###############################
# Additional evaluation methods
###############################

def plot_nonlinear_calibration_curve(y_true, y_pred, uncertainties):
    # Add print statements for debugging
    print("Plotting Nonlinear Calibration Curve")
    print(f"y_true shape: {np.array(y_true).shape}")
    print(f"y_pred shape: {np.array(y_pred).shape}")
    print(f"uncertainties shape: {np.array(uncertainties).shape}")
    
    try:
        plt.figure(figsize=(10, 6))
        confidence_levels = np.linspace(0.01, 0.99, 20)
        empirical_coverage = []
        
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        uncertainties = np.asarray(uncertainties)
        
        for conf in confidence_levels:
            # Compute lower and upper bounds for each prediction
            lower = y_pred - uncertainties * (conf / 2)
            upper = y_pred + uncertainties * (conf / 2)
            
            # Check how many true values fall within these bounds
            in_interval = np.mean((y_true >= lower) & (y_true <= upper))
            empirical_coverage.append(in_interval)
        
        plt.plot(confidence_levels, empirical_coverage, 'bo-', label='Calibrated Model')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Expected Confidence Level', fontsize=14)
        plt.ylabel('Empirical Coverage', fontsize=14)
        plt.title('Nonlinear Uncertainty Calibration Curve', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Use full path to ensure plot is saved
        import os
        full_path = os.path.join(os.getcwd(), 'nonlinear_calibration_curve.png')
        plt.savefig(full_path, dpi=300)
        print(f"Plot saved to {full_path}")
        
        plt.close()  # Close the plot to free up memory
    except Exception as e:
        print(f"Error generating calibration curve plot: {e}")
        import traceback
        traceback.print_exc()
# Add this at the end of the script
plot_nonlinear_calibration_curve(y_test.values, mean_pred, calibrated_uncertainty)