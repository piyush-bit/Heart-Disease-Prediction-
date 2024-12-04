import pandas as pd
from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load your trained model
try:
    heart_data = pd.read_csv('heart.csv')  # Ensure the correct path to the CSV file
except FileNotFoundError as e:
    raise RuntimeError("The file 'heart.csv' could not be found.") from e

# Prepare the data
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=7)

# Train the model
model = LogisticRegression(max_iter=1000)  # Increase from default (usually 100)
model.fit(x_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')  # Ensure `index.html` is in the templates/ directory

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Match these features with the HTML form fields
            form_features = ['age', 'sex', 'chestPain', 'restBp', 'chol', 'fastingBloodSugar', 
                             'restEcg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_data = [float(request.form[feature]) for feature in form_features]
            
            # Reshape input data for prediction
            reshaped_data = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(reshaped_data)
            
            # Return the prediction result
            result = "The person is suffering from heart disease." if prediction[0] == 1 else "The person is not suffering from heart disease."
            return render_template('index.html', prediction_result=result)
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return render_template('index.html', prediction_result="An error occurred. Please check your input.")
    return render_template('index.html', prediction_result="")

if name == 'main':
    app.run(host='0.0.0.0', port=5000)  # No debug mode for production