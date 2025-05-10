import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn import metrics

# Load the new dataset with the engineered features
file_path = 'Dataset.xlsx'  # Update with your file path
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

# Store the 'LPED' column separately
lped_column = data_cleaned['LPED']

# Drop 'LPED' column temporarily
data_cleaned = data_cleaned.drop(columns=['LPED'])

# Reassign 'LPED' as the last column
data_cleaned['LPED'] = lped_column

# Data analysis 
data_cleaned_describe = data.describe()
print("Dataset Description:")
print(data_cleaned_describe)
print("\n")

# Separate target (Y) and features (X)
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

def get_metrics(Y_test, Y_pred):
    dict_metrics = {
        'r2': metrics.r2_score(Y_test, Y_pred),
        'mae': metrics.mean_absolute_error(Y_test, Y_pred),
        'mape': metrics.mean_absolute_percentage_error(Y_test, Y_pred),
        'mse': np.mean((Y_test - Y_pred)**2),
        'rmse': metrics.root_mean_squared_error(Y_test, Y_pred)
    }
    return dict_metrics

# Dictionary to store results for each random state
results = {}

# Test different random states
for random_state in range(10):  # 0 to 9
    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=random_state)
    
    # Train model
    model = SVR()
    model.fit(X_train, Y_train)
    
    # Make predictions
    Y_pred = model.predict(X_test)
    
    # Get metrics
    metrics_dict = get_metrics(Y_test, Y_pred)
    results[random_state] = metrics_dict

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T
print("Results for different random states:")
print(results_df)
print("\n")

# Find the best random state based on R2 score
best_random_state = results_df['r2'].idxmax()
print(f"Best random state based on R2 score: {best_random_state}")
print(f"Best metrics:")
print(results_df.loc[best_random_state])

# Visualize R2 scores across different random states
plt.figure(figsize=(10, 6))
plt.plot(results_df.index, results_df['r2'], marker='o')
plt.title('R2 Score vs Random State')
plt.xlabel('Random State')
plt.ylabel('R2 Score')
plt.grid(True)
plt.show()