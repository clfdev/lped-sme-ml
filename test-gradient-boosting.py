import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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

best_model = GradientBoostingRegressor(random_state=42)
best_model.fit(X_train, Y_train)

def get_metrics(Y_test, Y_pred):
    dict_metrics = {
        'r2': metrics.r2_score(Y_test, Y_pred),
        'mae': metrics.mean_absolute_error(Y_test, Y_pred),
        'mse': np.mean((Y_test - Y_pred)**2),
        'rmse': metrics.root_mean_squared_error(Y_test, Y_pred)
    }
    return dict_metrics

Y_pred = best_model.predict(X_test)
print(get_metrics(Y_test, Y_pred))

# Get feature names and feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
})
# Sort by importance
feature_importances = feature_importances.sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Print the shapes of the split datasets
print("\nDataset Split Information:")
print(f"Total dataset size: {len(data_cleaned)}")
print(f"Training set size: {len(X_train)} samples ({(1-0.20)*100}%)")
print(f"Test set size: {len(X_test)} samples ({0.20*100}%)")

# Print and export the data splits with index
print("\nTraining Data:")
train_data = pd.concat([X_train, Y_train], axis=1)
print(train_data)
# Export training data to Excel with index
train_data.to_excel('training_data2.xlsx', index=True, index_label='Sample Index')

print("\nTest Data:")
test_data = pd.concat([X_test, Y_test], axis=1)
print(test_data)
# Export test data to Excel with index
test_data.to_excel('test_data2.xlsx', index=True, index_label='Sample Index')

# Export both sets to different sheets in a single Excel file with index
with pd.ExcelWriter('split_datasets2.xlsx') as writer:
    train_data.to_excel(writer, sheet_name='Training Data', index=True, index_label='Sample Index')
    test_data.to_excel(writer, sheet_name='Test Data', index=True, index_label='Sample Index')

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting Regressor Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()