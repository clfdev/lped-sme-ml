import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy.stats import pearsonr

# Load the dataset from the Excel file
file_path = 'Secondary-testing-metrics.xlsx'  # Replace with your file path

# Read the Excel file
data = pd.read_excel(file_path)

# Assuming the columns are labeled as 'Predicted' and 'Real'
predicted_values = data['Predicted_LPED']
real_values = data['Real_LPED']

# Calculate R (Pearson correlation coefficient)
R, _ = pearsonr(predicted_values, real_values)

# Calculate R² (coefficient of determination)
R2 = r2_score(real_values, predicted_values)

# Calculate Mean Squared Error (MSE)
MSE = mean_squared_error(real_values, predicted_values)

# Calculate Mean Absolute Error (MAE)
MAE = mean_absolute_error(real_values, predicted_values)

# Calculate Root Mean Squared Error (RMSE)
RMSE = np.sqrt(MSE)

# Print the calculated values
print(f"R (Correlation Coefficient): {R}")
print(f"R² (Coefficient of Determination): {R2}")
print(f"Mean Squared Error (MSE): {MSE}")
print(f"Mean Absolute Error (MAE): {MAE}")
print(f"Root Mean Squared Error (RMSE): {RMSE}")
