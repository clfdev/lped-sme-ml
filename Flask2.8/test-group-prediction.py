import numpy as np
import pandas as pd
import pickle

# Read CSV with specified delimiter
df = pd.read_csv('Dataset.csv', delimiter=';')
print(df.head())

# Create DataFrame (although this is unnecessary here since it's already a DataFrame)
df = pd.DataFrame(df)
print(df.head())

# Identify categorical features
categorical_features = df.select_dtypes(include=['object', 'category']).columns
print("Categorical features:", categorical_features) 
df = df.drop(columns=categorical_features)
print(df.head())

# Remove '(MK)' from column names
df.rename(columns=lambda x: x.replace(' (MK)', ''), inplace=True)
print(df.columns)
print(df.head())

# Load the saved model pipeline
with open('./model/model.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

# Preprocessor and model from the pipeline
preprocessor = model_pipeline.named_steps['preprocessor']
lr_model = model_pipeline.named_steps['model']
scaler = preprocessor.named_steps['scaler']

# Get the feature names after preprocessing
feature_names = preprocessor.named_steps['scaler'].get_feature_names_out()
empty_data = pd.DataFrame(columns=feature_names)

def predict_lped_and_sme(interatomic_distance, atomic_charge_a, atomic_charge_b):
    inverse_interatomic_distance = 1 / interatomic_distance
    lennard_jones_interatomic_distance = 4 * 1 * (
        (1 * inverse_interatomic_distance) ** 12 - (1 * inverse_interatomic_distance) ** 6
    )
    input_data = pd.DataFrame({
        'Lennard_Jones interatomic distance': [lennard_jones_interatomic_distance],
        'Atomic charge A (MK)': [atomic_charge_a],
        'Atomic charge B (MK)': [atomic_charge_b]
    }, columns=feature_names)  # Use the extracted feature names
    
    input_data_scaled = scaler.transform(input_data)
    lped_prediction = lr_model.predict(input_data_scaled)[0]
    sme_prediction = 1.2112 * lped_prediction + 0.6007

    return round(lped_prediction, 2), round(sme_prediction, 2)

# Rename columns if needed to match the expected names
df.rename(columns={'Interatomic distance': 'Interatomic distance', 
         'Atomic charge A': 'Atomic charge A', 
         'Atomic charge B': 'Atomic charge B'}, inplace=True)

# Apply the predict function to each row and add new columns
df['Predicted_LPED'], df['Predicted_SME'] = zip(*df.apply(
          lambda row: predict_lped_and_sme(row['Interatomic distance'], 
                 row['Atomic charge A'], 
                 row['Atomic charge B']), 
                axis=1))

print(df.head())