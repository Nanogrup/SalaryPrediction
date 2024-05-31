from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from machine_learning import train_and_save_model

app = Flask(__name__)
model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler, accuracy = None, None, None, None, None, None

# Function to delete the model file when app is finished
def delete_model_file():
    model_file = 'salary_model.pkl'
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"Model file '{model_file}' deleted.")

# Load the trained model, label encoder, and scaler
print("Training model...")
error_dist_div, actual_vs_predicted_div = train_and_save_model()
print("Model training completed.")
with open('salary_model.pkl', 'rb') as file:
    model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler, accuracy  = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/salary_prediction')
def salary_prediction():
    return render_template('salary_prediction.html')

@app.route('/data_analysis')
def data_analysis():
    return render_template('data_analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form
    print(label_encoder_gender.classes_)
    title = request.form['title']
    age = float(request.form['age'])
    experience = float(request.form['experience'])
    gender = request.form['gender']
    education_level = request.form['education_level']
    
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[age, gender, education_level, title, experience]], 
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
    
    # Apply the same encoding and scaling as in training
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
    input_data['Education Level'] = label_encoder_education.transform(input_data['Education Level'])
    input_data['Job Title'] = label_encoder_title.transform(input_data['Job Title'])

    # During prediction
    print("Column names of input data during prediction:", input_data.columns.tolist())
    print("Feature names seen at fit time:", model.feature_names_in_)
    print(input_data)
    # Predict the salary
    prediction = model.predict(input_data)
    inverse_normalization_prediction = scaler.inverse_transform(prediction.reshape(-1, 1));
    print(inverse_normalization_prediction[0][0])
    prediction_parts = str(inverse_normalization_prediction[0][0]).split('.')

    user_inputs = {
        'title': title,
        'age': age,
        'experience': experience,
        'gender': gender,
        'education_level': education_level
    }


    return render_template('result.html', prediction=prediction_parts[0], accuracy= round(accuracy * 100, 2),
                            error_dist_div=error_dist_div, actual_vs_predicted_div=actual_vs_predicted_div,
                            user_inputs = user_inputs)

@app.route('/job_titles')
def job_titles():
    df = pd.read_csv('data/salaryData.csv')
    titles = df['Job Title'].dropna().unique().tolist()
    return jsonify(titles)

# Register a function to run when the Flask application is shutting down
@app.teardown_appcontext
def cleanup(exception=None):
    delete_model_file()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
