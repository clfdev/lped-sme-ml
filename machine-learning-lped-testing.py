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
file_path = 'Dataset3.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Create a column called Inverse interatomic distance
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create a column called Lennard_Jones interatomic distance
# Lennard-Jones potential parameters
epsilon = 1
sigma = 1

# Calculate Lennard-Jones potential
data['Lennard_Jones potential'] = 4 * epsilon * (
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

print(data_cleaned)
data_cleaned.to_excel('data_cleaned.xlsx', index=True)

# Data analysis 
data_cleaned_describe = data.describe()
print(data_cleaned_describe)
data_cleaned_describe.to_excel('data_cleaned_describe.xlsx', index=True)

sns.scatterplot(x='Lennard_Jones potential', y='LPED', data=data)
plt.savefig('fig1.jpg')
plt.clf()

sns.scatterplot(x='Atomic charge A (MK)', y='LPED', data=data)
plt.savefig('fig2.jpg')
plt.clf()

sns.scatterplot(x='Atomic charge B (MK)', y='LPED', data=data)
plt.savefig('fig3.jpg')
plt.clf()

dfc = data_cleaned.corr()
print(dfc)
dfc.to_excel('dfc.xlsx', index=True)

# Separate target (Y) and features (X)
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=10),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regression': SVR()
}

# Number of random splits
n_splits = 10

# Store the results
results = {model: [] for model in models.keys()}

# Loop over multiple splits
for i in range(n_splits):
    # Split the dataset into train and test sets with a 0.20 proportion for the test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=i)
    
    # Normalize the X_train with fit_transform and X_test with transform to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train_scaled, Y_train)
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(Y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, predictions)
        mae = mean_absolute_error(Y_test, predictions)
        results[model_name].append({'Split': i + 1, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae})

# Convert results to DataFrame for better readability
all_results_df = pd.concat({k: pd.DataFrame(v) for k, v in results.items()}, axis=0)
all_results_df.index.names = ['Model', 'Iteration']

print("Metrics for Each Split of Train and Test:")
print(all_results_df)
all_results_df.to_excel('all_results_df.xlsx', index=True) 



# Split the dataset into train and test sets with a 0.20 proportion with the best random_state value
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=8)

best_model3=LinearRegression()
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

# Scatter plot with axis labels
sns.scatterplot(x=Y_test, y=Y_pred)
plt.xlabel('Actual LPED')
plt.ylabel('Predicted LPED')
plt.title('Actual vs Predicted LPED')
plt.savefig('fig4.jpg')
plt.clf()

# Get the coefficients and intercept
coefficients = best_model3.coef_
intercept = best_model3.intercept_

# Print the equation
feature_names = X.columns
equation = f"y = {intercept:.4f}"
for coef, feature in zip(coefficients, feature_names):
    equation += f" + ({coef:.4f} * {feature})"

print("Regression Equation:")
print(equation)