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
import mpld3
import plotly.express as px
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
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='Gender', y='Salary', data=data, ax=ax)  
    ax.set_title('Salary Comparison by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Salary')
    save_plot_as_image(fig, 'gender_salary_plot')
    plt.close(fig)

def yasDeneyimXmaas(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Age', y='Salary', hue='Years of Experience', data=data, palette='inferno', s=100, ax=ax)
    ax.set_title('Salary Analysis by Age and Experience')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary')
    save_plot_as_image(fig, 'age_experience_salary_plot')
    plt.close(fig)

def egitimSeviyesiXmaas(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Education Level', y='Salary', data=data, ax=ax)
    ax.set_title('Salary Distribution by Education Level')
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Salary')
    save_plot_as_image(fig, 'education_salary_plot')
    plt.close(fig)

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['int', 'float'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    ax.set_title('Correlation Matrix')
    save_plot_as_image(fig, 'correlation_matrix_plot')
    plt.close(fig)

def plot_top_20_job_titles_salary(df):
    """
    En yaygın 20 iş unvanı için maaş dağılımını gösteren box plot oluşturur.

    Parametre:
    df (pandas.DataFrame): Job Title ve Salary kolonlarına sahip veri çerçevesi.
    """

    # En yaygın 20 iş unvanı için veri çerçevesi oluşturma
    top_20_job_titles = df['Job Title'].value_counts().index[:20]
    df_top_20 = df[df['Job Title'].isin(top_20_job_titles)]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(y='Job Title', x='Salary', data=df_top_20, ax=ax)

    # Grafik başlığı ve eksen etiketlerini beyaz renge ayarlama
    ax.set_title('En Yaygın 20 İş Unvanına Göre Maaş Dağılımı', color='white')
    ax.set_xlabel('Maaş', color='white')
    ax.set_ylabel('İş Unvanı', color='white')

    # X ve Y eksen renklerini beyaz olarak ayarlama
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Kenarlık renklerini beyaz olarak ayarlama
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Etiket renklerini beyaz olarak ayarlama
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    # Arka plan şeffaflığı
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Y eksen etiketini sola kaydırma
    ax.yaxis.set_label_position('left')
    ax.set_ylabel('Job Title', labelpad=20)

    # Grafik düzeni ayarla
    plt.tight_layout()

    # Boxplot'u HTML'e dönüştürme
    return mpld3.fig_to_html(fig)

def plot_salary_relationships(df, columns, target='Salary'):
    """
    Verilen sütunlar ile maaş arasındaki ilişkiyi gösteren regresyon çizgili grafikleri oluşturur.

    Parametreler:
    df (pandas.DataFrame): Veriyi içeren veri çerçevesi.
    columns (list): Maaş ile ilişkilendirilmek istenen sütunların listesi.
    target (str): İlişki kurulacak hedef sütun, varsayılan 'Salary'.
    """
    plots = []
    for col in columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lmplot(x=col, y=target, data=df, aspect=1.5)  # Burada ax parametresi kaldırıldı
        plt.title(f'Salary vs {col.capitalize()} with Regression Line')
        plt.xlabel(col.capitalize())
        plt.ylabel(target.capitalize())
        plot_html = mpld3.fig_to_html(fig)
        plots.append(plot_html)
    return plots


def plot_salary_vs_columns(df, columns, width=2200, height=2600):
    """
    Verilen sütunlara göre maaşın kutu grafiği ile dağılımını gösteren Plotly grafiği oluşturur.

    Parametreler:
    df (pandas.DataFrame): Veriyi içeren veri çerçevesi.
    columns (list): Maaş ile ilişkilendirilmek istenen sütunların listesi.
    width (int): Grafik genişliği, varsayılan 2200.
    height (int): Grafik yüksekliği, varsayılan 2600.
    """
    plots = []
    for col in columns:
        fig = px.box(df, x=col, y='Salary', title=f'Salary vs {col.capitalize()}',
                     width=width, height=height)
        plot_html = fig.to_html(full_html=False)
        plots.append(plot_html)
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
    df = pd.read_csv('data/salaryData.csv')
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
