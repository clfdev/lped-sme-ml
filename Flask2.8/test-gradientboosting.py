import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
print(data_cleaned_describe)

# Separate target (Y) and features (X)
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

# Split the dataset into train and test sets with a 0.20 proportion with the best random_state value
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)

best_model3=GradientBoostingRegressor(random_state=42)
best_model3.fit(X_train, Y_train)

def get_metrics(Y_test, Y_pred):
    dict_metrics = {
        'r2': metrics.r2_score(Y_test, Y_pred),
        'mae': metrics.mean_absolute_error(Y_test, Y_pred),
        'mape': metrics.mean_absolute_percentage_error(Y_test, Y_pred),
        'rmse': np.sqrt(mean_squared_error(Y_test, Y_pred))
    }
    return dict_metrics

Y_pred = best_model3.predict(X_test)
print(get_metrics(Y_test,Y_pred))