import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

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

# Define features to check for outliers
features = ['Atomic charge A (MK)', 'Atomic charge B (MK)', 'Lennard_Jones potential']

# Create vertical box plots for each feature
for i, feature in enumerate(features):
    plt.subplot(3, 1, i + 1)  # Arrange subplots vertically
    sns.boxplot(x=data_cleaned[feature])
    plt.title(feature)

    # Add padding between plots
    if i < len(features) - 1:
        plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between plots

plt.tight_layout()
plt.show()

# Using IQR method for outlier detection
for feature in features:
    Q1 = data_cleaned[feature].quantile(0.25)
    Q3 = data_cleaned[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data_cleaned[(data_cleaned[feature] < lower_bound) | (data_cleaned[feature] > upper_bound)]
    print(f"Outliers in {feature} using IQR method:")
    print(outliers[[feature]])

 
# Plot histograms for all variables with 2 plots per row, adding a KDE line
num_plots = len(data_cleaned.columns)
num_cols = 2  # Number of plots per row
num_rows = math.ceil(num_plots / num_cols)

plt.figure(figsize=(15, 5 * num_rows))  # Adjust figure size to accommodate more rows
for i, feature in enumerate(data_cleaned.columns, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.histplot(data_cleaned[feature], bins=60, kde=True, edgecolor='black')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between plots
plt.show()

