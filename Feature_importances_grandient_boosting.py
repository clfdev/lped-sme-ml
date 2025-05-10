import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib as mpl

# Increase font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# Load the dataset
file_path = 'Dataset3.xlsx'
data = pd.read_excel(file_path)

# Create inverse interatomic distance column
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create Lennard-Jones potential column
epsilon = 1
sigma = 1
data['Lennard_Jones potential'] = 4 * epsilon * ((sigma * data['Inverse interatomic distance']) ** 12 - 
                                                (sigma * data['Inverse interatomic distance']) ** 6)

# For Lennard-Jones model - drop columns we don't need
data_lj = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'], 
                  errors='ignore')

# Separate features and target
X = data_lj.drop(columns=['LPED'])
y = data_lj['LPED']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the gradient boosting model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Get built-in feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Get the total importance to calculate percentage
total_importance = feature_importances['Importance'].sum()
feature_importances['Percentage'] = (feature_importances['Importance'] / total_importance * 100).round(2)

# Calculate cumulative importance
feature_importances['Cumulative Percentage'] = feature_importances['Percentage'].cumsum().round(2)

print("Built-in Feature Importances Analysis:")
print(feature_importances)

# Identify the features that account for 80% of the importance
threshold_80 = feature_importances[feature_importances['Cumulative Percentage'] <= 80]
key_features = len(threshold_80) + 1  # +1 because we need the next feature to cross 80%

print(f"\nNumber of features accounting for 80% of total importance: {key_features} out of {len(feature_importances)}")
print(f"These key features are: {', '.join(feature_importances['Feature'].iloc[:key_features].tolist())}")

# Calculate some statistics
mean_importance = feature_importances['Importance'].mean()
median_importance = feature_importances['Importance'].median()
std_importance = feature_importances['Importance'].std()

print(f"\nImportance Statistics:")
print(f"Mean Importance: {mean_importance:.6f}")
print(f"Median Importance: {median_importance:.6f}")
print(f"Standard Deviation: {std_importance:.6f}")
print(f"Importance Variation Coefficient: {(std_importance/mean_importance):.6f}")

# Plot feature importances
plt.figure(figsize=(14, 10))

# Plot feature importances as bar chart
ax1 = plt.subplot(1, 1, 1)
bars = ax1.bar(feature_importances['Feature'], feature_importances['Importance'])

# Add percentage labels on top of bars with larger font size
for bar, percentage in zip(bars, feature_importances['Percentage']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{percentage}%', ha='center', va='bottom', rotation=0, fontsize=14)

plt.title('Feature Importance Analysis (Lennard-Jones Model)', fontsize=22, pad=20)
plt.ylabel('Importance Score', fontsize=18, labelpad=15)
plt.xlabel('Features', fontsize=18, labelpad=15)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a line for cumulative percentage (right y-axis)
ax2 = ax1.twinx()
ax2.plot(feature_importances['Feature'], feature_importances['Cumulative Percentage'], 
         color='red', marker='o', linestyle='-', linewidth=3, markersize=10)
ax2.set_ylabel('Cumulative Percentage (%)', color='red', fontsize=18, labelpad=15)
ax2.tick_params(axis='y', colors='red', labelsize=16)
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax2.text(0, 81, '80% Threshold', color='red', ha='left', va='bottom', fontsize=16, fontweight='bold')

# Improve overall spacing
plt.subplots_adjust(bottom=0.25, right=0.85, top=0.9)

plt.savefig('lennard_jones_feature_importance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate a summary report
print("\n============= FEATURE IMPORTANCE ANALYSIS SUMMARY =============")
print(f"Model: Gradient Boosting Regressor with Lennard-Jones Potential")
print(f"Total features analyzed: {len(feature_importances)}")
print(f"Top 3 most important features:")
for i in range(min(3, len(feature_importances))):
    print(f"  {i+1}. {feature_importances['Feature'].iloc[i]} ({feature_importances['Percentage'].iloc[i]}%)")
    
print(f"\nFeatures contributing to 80% of model decisions: {key_features}")
print(f"Features contributing less than 1% each: {len(feature_importances[feature_importances['Percentage'] < 1])}")

# Calculate feature correlation with target
feature_target_corr = pd.DataFrame({
    'Feature': X.columns,
    'Correlation with Target': [np.corrcoef(X[col], y)[0, 1] for col in X.columns]
}).sort_values(by='Correlation with Target', key=abs, ascending=False)

print("\nFeature Correlation with Target (LPED):")
print(feature_target_corr)

# Analyze relationship between correlation and importance
merged_analysis = pd.merge(
    feature_importances, 
    feature_target_corr,
    on='Feature'
)
merged_analysis['Abs Correlation'] = merged_analysis['Correlation with Target'].abs()

print("\nRelationship between Correlation and Importance:")
print(merged_analysis[['Feature', 'Importance', 'Correlation with Target', 'Abs Correlation']])

correlation_btw_importance_and_corr = np.corrcoef(
    merged_analysis['Importance'], 
    merged_analysis['Abs Correlation']
)[0, 1]

print(f"\nCorrelation between feature importance and absolute correlation with target: {correlation_btw_importance_and_corr:.4f}")
print("This indicates how strongly the model's feature importance aligns with linear relationships to the target.")