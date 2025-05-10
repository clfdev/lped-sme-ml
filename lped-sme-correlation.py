import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the data from Excel file with explicit engine
data = pd.read_excel('SME-LPED.xlsx', engine='openpyxl')

# Print detailed information about the DataFrame
print("DataFrame Information:")
print(data.info())
print("\nDataFrame Content:")
print(data)
print("\nColumn Names (with data types):")
for col in data.columns:
    print(f"{col}: {data[col].dtype}")

# Check if there are any NaN values
print("\nNaN Values:")
print(data.isna().sum())

# Try to identify LPED and SME columns
# Method 1: First check if they exist by name (case-insensitive)
lped_col = None
sme_col = None

for col in data.columns:
    if 'lped' in col.lower():
        lped_col = col
    elif 'sme' in col.lower():
        sme_col = col

# If not found by name, look at numeric columns and print their statistics
if lped_col is None or sme_col is None:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print("\nNumeric Columns Statistics:")
    for col in numeric_cols:
        col_stats = data[col].describe()
        print(f"\nColumn: {col}")
        print(f"Min: {col_stats['min']}")
        print(f"Max: {col_stats['max']}")
        print(f"Mean: {col_stats['mean']}")
        print(f"Standard Deviation: {col_stats['std']}")
    
    # Try to guess which columns might be LPED and SME
    print("\nBased on typical values:")
    print("LPED is typically negative with values around -1 to -15 kcal mol⁻¹ Bohr⁻³")
    print("SME is typically a similar magnitude but can be more variable")
    
    # Let the user select columns explicitly
    print("\nPlease choose columns based on the above information.")
    print("For now, we'll try with the first two numeric columns assuming:")
    if len(numeric_cols) >= 2:
        lped_col = numeric_cols[0]
        sme_col = numeric_cols[1]
        print(f"- LPED might be column '{lped_col}'")
        print(f"- SME might be column '{sme_col}'")

# Explicitly print selected columns for analysis
print(f"\nAnalyzing relationship between:")
print(f"- Independent variable (LPED): {lped_col}")
print(f"- Dependent variable (SME): {sme_col}")

try:
    # Try reading the first few values from each column
    print("\nSample values from LPED column:")
    print(data[lped_col].head())
    print("\nSample values from SME column:")
    print(data[sme_col].head())
    
    # Check for any string values that might be causing conversion issues
    lped_data = data[lped_col]
    sme_data = data[sme_col]
    
    # Try to convert to numeric if not already
    lped_numeric = pd.to_numeric(lped_data, errors='coerce')
    sme_numeric = pd.to_numeric(sme_data, errors='coerce')
    
    # Check for NaN values after conversion
    if lped_numeric.isna().any() or sme_numeric.isna().any():
        print("\nWarning: Some values couldn't be converted to numbers!")
        print(f"NaN values in LPED after conversion: {lped_numeric.isna().sum()}")
        print(f"NaN values in SME after conversion: {sme_numeric.isna().sum()}")
    
    # Drop any rows with NaN values
    valid_indices = ~(lped_numeric.isna() | sme_numeric.isna())
    lped_clean = lped_numeric[valid_indices]
    sme_clean = sme_numeric[valid_indices]
    
    print(f"\nData points after cleaning: {len(lped_clean)}")
    
    # Create the correct regression: SME = f(LPED)
    X = lped_clean  # Independent variable (LPED)
    y = sme_clean   # Dependent variable (SME)
    
    # Add a constant to the independent variable
    X_with_const = sm.add_constant(X)
    
    # Fit the regression model
    model = sm.OLS(y, X_with_const).fit()
    
    # Print the regression summary
    print("\nRegression Summary:")
    print(model.summary())
    
    # Extract key statistics
    slope = model.params[1]
    intercept = model.params[0]
    r_squared = model.rsquared
    p_value = model.pvalues[1]
    f_stat = model.fvalue
    f_pvalue = model.f_pvalue
    
    # Print key results in a more readable format
    print("\nKey Results:")
    print(f"Regression Equation: SME = {slope:.4f} × LPED + {intercept:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"p-value for slope = {p_value:.8f}")
    print(f"F-statistic = {f_stat:.4f}, p-value = {f_pvalue:.8f}")
    
    # Create a scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6)
    plt.plot(X, intercept + slope * X, color='red', linewidth=2)
    plt.title('Relationship between LPED and SME', fontsize=14)
    plt.xlabel('LPED (kcal mol⁻¹ Bohr⁻³)', fontsize=12)
    plt.ylabel('SME (kcal mol⁻¹)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add regression equation and R² to the plot
    equation_text = f"SME = {slope:.4f} × LPED + {intercept:.4f}"
    r2_text = f"R² = {r_squared:.4f}"
    p_text = f"p-value = {p_value:.8f}"
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.90, r2_text, transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, p_text, transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('lped_sme_regression.png', dpi=300)
    plt.show()
    
except Exception as e:
    print(f"\nError during analysis: {e}")
