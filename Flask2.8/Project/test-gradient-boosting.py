import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn import metrics

# Load the new dataset with the engineered features
file_path = 'Dataset3.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Create a column called Inverse interatomic distance
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create a column called Lennard_Jones interatomic distance
# Lennard-Jones potential parameters
epsilon = 1
sigma = 1

# Calculate Lennard-Jones potential
data['Lennard_Jones interatomic distance'] = 4 * epsilon * (
    (sigma * data['Inverse interatomic distance']) ** 12 - 
    (sigma * data['Inverse interatomic distance']) ** 6
)

# Drop the index column, categorical variable, and the original interatomic distance column
data_cleaned = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'])

# Data analysis 
pd.set_option('display.max_columns', None)
data_cleaned_describe = data_cleaned.describe()
print(data_cleaned_describe)

# Separate target (Y) and features (X)
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

# Split the dataset into train and test sets with a 0.20 proportion with the best random_state value
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=4)

# Define the Regularized Gradient Boosting model with optimal parameters
best_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

# Create a pipeline with standardization
model_pipeline = make_pipeline(StandardScaler(), best_model)

# Train the model
model_pipeline.fit(X_train, Y_train)

# Make predictions
Y_pred_train = model_pipeline.predict(X_train)
Y_pred_test = model_pipeline.predict(X_test)

def get_metrics(Y_actual, Y_pred):
    dict_metrics = {
        'r2': metrics.r2_score(Y_actual, Y_pred),
        'mae': metrics.mean_absolute_error(Y_actual, Y_pred),
        'mse': np.mean((Y_actual - Y_pred)**2),
        'rmse': np.sqrt(np.mean((Y_actual - Y_pred)**2))
    }
    return dict_metrics

# Get metrics for training and test sets
train_metrics = get_metrics(Y_train, Y_pred_train)
test_metrics = get_metrics(Y_test, Y_pred_test)

print("\nTraining Set Metrics:")
print(train_metrics)

print("\nTest Set Metrics:")
print(test_metrics)

# Calculate overfitting indicators
mse_ratio = test_metrics['mse'] / train_metrics['mse'] if train_metrics['mse'] > 0 else float('inf')
r2_gap = train_metrics['r2'] - test_metrics['r2']

print("\nOverfitting Indicators:")
print(f"MSE Ratio (Test/Train): {mse_ratio:.2f}x")
print(f"R² Gap (Train - Test): {r2_gap:.4f}")

# Extract feature importances from the model
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_pipeline.named_steps['gradientboostingregressor'].feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance in Regularized Gradient Boosting Model', fontsize=16)
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

# Visualize actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred_test, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.title(f'Actual vs. Predicted Values (Test Set, R² = {test_metrics["r2"]:.4f})', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()

# Calculate and plot residuals
residuals = Y_test - Y_pred_test

plt.figure(figsize=(12, 5))

# Residuals vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(Y_pred_test, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs Predicted Values', fontsize=14)
plt.grid(True, alpha=0.3)

# Histogram of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=10, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Residual Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of Residuals', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300)
plt.show()