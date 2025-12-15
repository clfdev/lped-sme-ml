import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Dataset3.xlsx'
data = pd.read_excel(file_path)

# Create inverse interatomic distance column
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create Lennard-Jones potential column
# Lennard-Jones potential parameters
epsilon = 1
sigma = 1
data['Lennard_Jones potential'] = 4 * epsilon * ((sigma * data['Inverse interatomic distance']) ** 12 - 
                                                 (sigma * data['Inverse interatomic distance']) ** 6)

# For Lennard-Jones model - drop columns we don't need
data_lj = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'], 
                  errors='ignore')

# Generate correlation matrix for Lennard-Jones model
corr_lj = data_lj.corr()
print("\nCorrelation Matrix with Lennard-Jones Potential:")
print(corr_lj)

# Create the heatmap
plt.figure(figsize=(10, 8))
ax=sns.heatmap(corr_lj, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matriz de correlação: Modelo Lennard-Jones', fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center')
plt.tight_layout()

# Save the heatmap
plt.savefig('lennard_jones_correlation_heatmap.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()