import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn import metrics

# Load the dataset
file_path = 'Dataset.xlsx'
data = pd.read_excel(file_path)

# Create inverse interatomic distance column
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create Lennard-Jones potential column
# Lennard-Jones potential parameters
epsilon = 1
sigma = 1
data['Lennard_Jones potential'] = 4 * epsilon * ((sigma * data['Inverse interatomic distance']) ** 12 - 
                                                 (sigma * data['Inverse interatomic distance']) ** 6)

# Create two dataframes - one with inverse distance and one with Lennard-Jones
# For inverse distance model
data_inverse = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Lennard_Jones potential'], 
                       errors='ignore')

# For Lennard-Jones model
data_lj = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'], 
                  errors='ignore')

print("=== Comparing Inverse Distance vs. Lennard-Jones Potential as Features ===\n")

#---------------------
# CORRELATION MATRICES
#---------------------
print("CORRELATION MATRICES:\n")

# Correlation matrix for Inverse Distance model
corr_inverse = data_inverse.corr()
print("Correlation Matrix with Inverse Distance:")
print(corr_inverse)

# Correlation matrix for Lennard-Jones model
corr_lj = data_lj.corr()
print("\nCorrelation Matrix with Lennard-Jones Potential:")
print(corr_lj)

# Create correlation heatmaps
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.heatmap(corr_inverse, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix: Inverse Distance Model', fontsize=14)

plt.subplot(1, 2, 2)
sns.heatmap(corr_lj, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix: Lennard-Jones Model', fontsize=14)

plt.tight_layout()
plt.savefig('correlation_matrices_comparison.png', dpi=300, bbox_inches='tight')

#---------------------
# MODEL PERFORMANCE
#---------------------
print("\nMODEL PERFORMANCE COMPARISON:\n")

# Function to evaluate model performance
def evaluate_model(data_df, model_name):
    # Separate features and target
    X = data_df.drop(columns=['LPED'])
    y = data_df['LPED']
    
    # Split data with consistent random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics_dict = {
        'r2': metrics.r2_score(y_test, y_pred),
        'mae': metrics.mean_absolute_error(y_test, y_pred),
        'mse': np.mean((y_test - y_pred)**2),
        'rmse': np.sqrt(np.mean((y_test - y_pred)**2))
    }
    
    # Get coefficients
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    
    # Calculate feature importance using permutation importance
    perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    # Get regression equation
    equation = f"LPED = {model.intercept_:.4f}"
    for feature, coef in zip(X.columns, model.coef_):
        if coef >= 0:
            equation += f" + {coef:.4f} × {feature}"
        else:
            equation += f" - {abs(coef):.4f} × {feature}"
    
    return {
        'metrics': metrics_dict,
        'coefficients': coefficients,
        'feature_importance': feature_importance,
        'equation': equation,
        'model': model,
        'scaler': scaler,
        'X_columns': X.columns,
        'X_test': X_test,
        'y_test': y_test
    }

# Evaluate both models
inverse_results = evaluate_model(data_inverse, "Inverse Distance")
lj_results = evaluate_model(data_lj, "Lennard-Jones Potential")

# Print metrics comparison
print("Metrics for Inverse Distance Model:")
print(inverse_results['metrics'])
print("\nMetrics for Lennard-Jones Potential Model:")
print(lj_results['metrics'])

# Print coefficients (feature importance from linear regression)
print("\nFeature Importance by Coefficient (Inverse Distance Model):")
print(inverse_results['coefficients'])
print("\nFeature Importance by Coefficient (Lennard-Jones Potential Model):")
print(lj_results['coefficients'])

# Print permutation importance
print("\nFeature Importance by Permutation (Inverse Distance Model):")
print(inverse_results['feature_importance'])
print("\nFeature Importance by Permutation (Lennard-Jones Potential Model):")
print(lj_results['feature_importance'])

# Print regression equations
print("\nRegression Equation (Inverse Distance Model):")
print(inverse_results['equation'])
print("\nRegression Equation (Lennard-Jones Potential Model):")
print(lj_results['equation'])

#---------------------
# VISUALIZATION
#---------------------

# Feature importance visualization
plt.figure(figsize=(16, 10))

# Coefficient importance
plt.subplot(2, 2, 1)
inverse_results['coefficients'].plot(x='Feature', y='Coefficient', kind='bar', ax=plt.gca())
plt.title('Feature Importance by Coefficient\n(Inverse Distance Model)', fontsize=12)
plt.tight_layout()

plt.subplot(2, 2, 2)
lj_results['coefficients'].plot(x='Feature', y='Coefficient', kind='bar', ax=plt.gca())
plt.title('Feature Importance by Coefficient\n(Lennard-Jones Potential Model)', fontsize=12)
plt.tight_layout()

# Permutation importance
plt.subplot(2, 2, 3)
inverse_results['feature_importance'].plot(x='Feature', y='Importance', kind='bar', ax=plt.gca())
plt.title('Feature Importance by Permutation\n(Inverse Distance Model)', fontsize=12)
plt.tight_layout()

plt.subplot(2, 2, 4)
lj_results['feature_importance'].plot(x='Feature', y='Importance', kind='bar', ax=plt.gca())
plt.title('Feature Importance by Permutation\n(Lennard-Jones Potential Model)', fontsize=12)
plt.tight_layout()

plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')

# Performance metrics comparison
metrics_comparison = pd.DataFrame({
    'Metric': ['R² Score', 'MAE', 'MSE', 'RMSE'],
    'Inverse Distance': [
        inverse_results['metrics']['r2'],
        inverse_results['metrics']['mae'],
        inverse_results['metrics']['mse'],
        inverse_results['metrics']['rmse']
    ],
    'Lennard-Jones': [
        lj_results['metrics']['r2'],
        lj_results['metrics']['mae'],
        lj_results['metrics']['mse'],
        lj_results['metrics']['rmse']
    ]
})

# Calculate percentage improvement
metrics_comparison['Improvement (%)'] = [
    ((lj_results['metrics']['r2'] - inverse_results['metrics']['r2']) / inverse_results['metrics']['r2']) * 100,
    ((inverse_results['metrics']['mae'] - lj_results['metrics']['mae']) / inverse_results['metrics']['mae']) * 100,
    ((inverse_results['metrics']['mse'] - lj_results['metrics']['mse']) / inverse_results['metrics']['mse']) * 100,
    ((inverse_results['metrics']['rmse'] - lj_results['metrics']['rmse']) / inverse_results['metrics']['rmse']) * 100
]

print("\nModel Performance Comparison:")
print(metrics_comparison)

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics_comparison['Metric']))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, metrics_comparison['Inverse Distance'], width, label='Inverse Distance')
bars2 = ax.bar(x + width/2, metrics_comparison['Lennard-Jones'], width, label='Lennard-Jones')

ax.set_ylabel('Value')
ax.set_title('Performance Metric Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_comparison['Metric'])
ax.legend()

# Add value labels above the bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight')

print("\nAnalysis completed. Visualization images saved as:")
print("- correlation_matrices_comparison.png")
print("- feature_importance_comparison.png")
print("- performance_metrics_comparison.png")