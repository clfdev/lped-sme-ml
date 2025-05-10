from flask import Flask, request, render_template, send_file, session, url_for
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


app = Flask(__name__)
app.secret_key = 'my_secret_key'


# Load the new dataset with the engineered features
file_path = 'Dataset.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

pipeline = Pipeline([
    ('inverse_distance', 
     lambda data: data.assign(**{'Inverse interatomic distance': 1 / data.get('Interatomic distance', 1)})),
    ('lennard_jones_distance', 
     lambda data: data.assign(**{'Lennard_Jones interatomic distance': 4 * 1 * (
         (1 * data['Inverse interatomic distance']) ** 12 -
         (1 * data['Inverse interatomic distance']) ** 6
     )})),
])

# Apply the pipeline transformations directly without storing intermediate results
for _, step in pipeline.steps:
    data = step(data)

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'])

# Separate target (Y) and features (X)
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

# Split the dataset into train and test sets with a 0.25 proportion for the test set (Iteration 0, Split 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Pipeline preprocessor normalization
# Normalize the X_train with fit_transform and X_test with transform to avoid data leakage
preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
]).set_output(transform='pandas')

# Define the overall pipeline with preprocessor and model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Fit the model pipeline with training data
model_pipeline.fit(X_train, Y_train)

# Predict using the model pipeline with test data
Y_pred = model_pipeline.predict(X_test)


# Main page
@app.route('/', methods=['GET'])
def main_page():
    return render_template('main.html')

# Preprocessor and model from the pipeline
preprocessor = model_pipeline.named_steps['preprocessor']
lr_model = model_pipeline.named_steps['model']
scaler = preprocessor.named_steps['scaler']

# Get the feature names after preprocessing
feature_names = preprocessor.named_steps['scaler'].get_feature_names_out()
empty_data = pd.DataFrame(columns=feature_names)

# Function to preprocess input and make individual predictions
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


# Individual LPED-SME predictions page
@app.route('/individual_predictions', methods=['GET', 'POST'])
def individual_predictions():
    if request.method == 'POST':
        interatomic_distance = float(request.form['interatomic_distance'])
        atomic_charge_a = float(request.form['atomic_charge_a'])
        atomic_charge_b = float(request.form['atomic_charge_b'])
        predicted_lped, predicted_sme = predict_lped_and_sme(interatomic_distance, atomic_charge_a, atomic_charge_b)
        return render_template('individual_predictions.html', lped=predicted_lped, sme=predicted_sme)
    return render_template('individual_predictions.html', lped=None, sme=None)


# GROUP PREDICTIONS
# Group LPED-SME predictions
@app.route('/group_predictions', methods=['GET', 'POST'])
def group_predictions():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file, delimiter=';')
            df = pd.DataFrame(df)
            df.rename(columns=lambda x: x.replace(' (MK)', ''), inplace=True)
            print("Model loaded successfully.")
          
            # Check if the required columns exist in the DataFrame
            expected_columns = ['Interatomic distance', 'Atomic charge A', 'Atomic charge B']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if not missing_columns:
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
                # Store the DataFrame in the session
                session['predictions'] = df.to_csv(index=False)
                download_link = url_for('download_predictions') 
                                
                # **Pass predictions here, both in success and error cases:**
                return render_template('group_predictions.html', predictions=df, 
                                       download_link=download_link, error=None)
            else:
                error_message = f"Missing Columns: {missing_columns}"
                # **Pass predictions here, both in success and error cases:**
                return render_template('group_predictions.html', predictions=df, 
                                       download_link=None, error=error_message)
        else: print("Error loading model") 
    return render_template('group_predictions.html', 
                           predictions=None, download_link=None, error=None)


# Create the download route
@app.route('/download_predictions', methods=['GET'])
def download_predictions():
    # Retrieve the DataFrame from the session
    predictions_csv = session.get('predictions')

    if predictions_csv:
        # Create the buffer and write the CSV data
        buffer = BytesIO()
        buffer.write(predictions_csv.encode())
        buffer.seek(0)

        # Send the CSV file for download
        return send_file(
            buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name='predictions.csv'
        )
    else:
        # Handle the case where the data is not available (e.g., redirect to an error page)
        return 'Predictions data not found. Please upload a CSV file.'

if __name__ == '__main__':
    app.run(debug=True, port=5002)