from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from machine_learning import train_and_save_model
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import mpld3
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.express as px
import plotly.utils
app = Flask(__name__)
model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler, accuracy = None, None, None, None, None, None

# Function to delete the model file when app is finished
def delete_model_file():
    model_file = 'salary_model.pkl'
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"Model file '{model_file}' deleted.")

def save_plot_as_image(fig, filename):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_string = base64.b64encode(img_buffer.read()).decode('utf-8').replace('\n', '')
    with open(f'static/{filename}.png', 'wb') as img_file:
        img_file.write(base64.b64decode(img_string))


def cinsiyetXmaas(data):
    fig = go.Figure()
    fig.add_trace(go.Violin(x=data['Gender'], y=data['Salary'], box_visible=True, meanline_visible=True))
    fig.update_layout(title='Salary Comparison by Gender', xaxis_title='Gender', yaxis_title='Salary')
    return fig.to_html(full_html=False)


def yasDeneyimXmaas(data):
    # Geçersiz değerleri ortalama ile değiştirme
    mean_experience = data['Years_of_Experience'].mean()
    data['Years_of_Experience'].fillna(mean_experience, inplace=True)

    # Scatter plot oluşturma
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Age'], y=data['Salary'], mode='markers', marker=dict(size=data['Years_of_Experience'], color=data['Years_of_Experience'], showscale=True)))
    fig.update_layout(title='Salary Analysis by Age and Experience', xaxis_title='Age', yaxis_title='Salary')
    
    # JSON formatına dönüştürerek çıktıyı al
    return fig.to_html(full_html=False)



def egitimSeviyesiXmaas(data):
    fig = go.Figure()
    fig.add_trace(go.Box(x=data['Education_Level'], y=data['Salary']))
    fig.update_layout(title='Salary Distribution by Education Level', xaxis_title='Education_Level', yaxis_title='Salary')
    return fig.to_html(full_html=False)

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['int', 'float'])
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index.values, y=corr.columns.values, colorscale='Viridis'))
    fig.update_layout(title='Correlation Matrix')
    return fig.to_html(full_html=False)

def plot_top_20_job_titles_salary(df):
    top_20_job_titles = df['Job_Title'].value_counts().index[:20]
    df_top_20 = df[df['Job_Title'].isin(top_20_job_titles)]
    fig = go.Figure()
    for job_title in top_20_job_titles:
        df_job = df_top_20[df_top_20['Job_Title'] == job_title]
        fig.add_trace(go.Box(y=df_job['Salary'], name=job_title))
    fig.update_layout(title='Salary Breakdown by 20 Most Common Job Titles', yaxis_title='Salary')
    return fig.to_html(full_html=False)

def plot_salary_relationships(df, columns, target='Salary'):
    plots = []
    for col in columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[col], y=df[target], mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x=df[col], y=model.predict(df[[col]]), mode='lines', name='Predicted'))
        fig.update_layout(title=f'Salary vs {col.capitalize()} with Regression Line', xaxis_title=col.capitalize(), yaxis_title='Salary')
        plots.append(fig.to_html(full_html=False))
    return plots

def plot_salary_vs_columns(df, columns, width=2200, height=2600):
    plots = []
    for col in columns:
        fig = go.Figure()
        fig.add_trace(go.Box(x=df[col], y=df['Salary'], name='Salary'))
        fig.update_layout(title=f'Salary vs {col.capitalize()}', xaxis_title=col.capitalize(), yaxis_title='Salary', width=width, height=height)
        plots.append(fig.to_html(full_html=False))
    return plots

# Load the trained model, label encoder, and scaler
if os.path.exists('salary_model.pkl'):
    with open('salary_model.pkl', 'rb') as file:
        model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler,accuracy = pickle.load(file)
else:
    print("Training model...")
    train_and_save_model()
    print("Model training completed.")
    with open('salary_model.pkl', 'rb') as file:
        model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler,accuracy = pickle.load(file)
# Load the trained model, label encoder, and scaler
print("Training model...")
error_dist_div, actual_vs_predicted_div = train_and_save_model()
print("Model training completed.")
with open('salary_model.pkl', 'rb') as file:
    model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler, accuracy = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/salary_prediction')
def salary_prediction():
    return render_template('salary_prediction.html')

@app.route('/data_analysis')
def data_analysis():
    df = pd.read_csv('data/Salary_Data.csv')
    gender_salary_plot = cinsiyetXmaas(df)
    age_experience_salary_plot = yasDeneyimXmaas(df)
    education_salary_plot = egitimSeviyesiXmaas(df)
    correlation_matrix_plot = plot_correlation_matrix(df)
    job_title_salary_plot = plot_top_20_job_titles_salary(df)

    return render_template('data_analysis.html', 
                           gender_salary_plot=gender_salary_plot,
                           age_experience_salary_plot=age_experience_salary_plot,
                           education_salary_plot=education_salary_plot,
                           correlation_matrix_plot=correlation_matrix_plot,
                           job_title_salary_plot=job_title_salary_plot)


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
                              columns=['Age', 'Gender', 'Education_Level', 'Job_Title', 'Years_of_Experience'])
    
    # Apply the same encoding and scaling as in training
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
    input_data['Education_Level'] = label_encoder_education.transform(input_data['Education_Level'])
    input_data['Job_Title'] = label_encoder_title.transform(input_data['Job_Title'])

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
    df = pd.read_csv('data/Salary_Data.csv')
    titles = df['Job_Title'].dropna().unique().tolist()
    return jsonify(titles)

# Register a function to run when the Flask application is shutting down
@app.teardown_appcontext
def cleanup(exception=None):
    delete_model_file()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
