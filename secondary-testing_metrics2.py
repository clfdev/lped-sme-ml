import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

# Data
data = {
    'Predicted_LPED': [-1.87, -4.47, -6.29, -1.65, -5.10, -2.67, -6.73, -2.77, -2.45, -5.77, -2.18],
    'Real_LPED': [-1.80, -0.39, -5.03, -1.37, -0.96, -1.87, -4.07, -1.59, -1.87, -4.07, -1.59]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate R² (coefficient of determination)
R2 = r2_score(df['Real_LPED'], df['Predicted_LPED'])
print(f"R² (Python): {R2}")


# Real and predicted values
real = np.array([-1.80, -0.39, -5.03, -1.37, -0.96, -1.87, -4.07, -1.59, -1.87, -4.07, -1.59])
predicted = np.array([-1.87, -4.47, -6.29, -1.65, -5.10, -2.67, -6.73, -2.77, -2.45, -5.77, -2.18])

# Mean of real values
mean_real = np.mean(real)

# Calculate SS_res and SS_tot
SS_res = np.sum((real - predicted) ** 2)
SS_tot = np.sum((real - mean_real) ** 2)

# Calculate R²
R2_manual = 1 - (SS_res / SS_tot)
print(f"R² (Manual Calculation): {R2_manual}")