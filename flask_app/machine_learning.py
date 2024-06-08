import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def train_and_save_model():
    # read data
    df = pd.read_csv("data/Salary_Data.csv")

    # Fill missing values with mean 
    # only for numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist() 
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Fill missing values with mode
    # for categorical columns
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Education_Level'].fillna(df['Education_Level'].mode()[0], inplace=True)
    df['Job_Title'].fillna(df['Job_Title'].mode()[0], inplace=True)

    # encode categorical columns 
    data = df.copy()  
    label_encoder_gender = LabelEncoder()
    label_encoder_education = LabelEncoder()
    label_encoder_title = LabelEncoder()
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
    data['Education_Level'] = label_encoder_education.fit_transform(data['Education_Level'])
    data['Job_Title'] = label_encoder_title.fit_transform(data['Job_Title'])

    # normalization for numerical columns
    scaler_age = MinMaxScaler()
    scaler_experience = MinMaxScaler()
    scaler_salary = MinMaxScaler()
    data['Age'] = scaler_age.fit_transform(data['Age'].values.reshape(-1, 1))
    data['Years_of_Experience'] = scaler_experience.fit_transform(data['Years_of_Experience'].values.reshape(-1, 1))
    data['Salary'] = scaler_salary.fit_transform(data['Salary'].values.reshape(-1, 1))
    # Split features and output 
    X = data.drop('Salary', axis=1)
    y = data['Salary']

    # Splitting data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = model.score(X_test, y_test)  

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Accuracy: {accuracy}")

    # Save the trained model, encoders, and scaler
    with open('salary_model.pkl', 'wb') as file:
        pickle.dump((model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler_salary, accuracy), file)

    # Generate the error distribution plot
    errors = y_test - y_pred
    error_fig = px.histogram(errors, nbins=50, title="Error Distribution",
                             template="plotly_dark")
    error_fig.update_layout(
        xaxis_title="Prediction Error",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    error_div = error_fig.to_html(full_html=False)

    # Generate the actual vs predicted values plot
    actual_vs_predicted_fig = go.Figure()
    actual_vs_predicted_fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers', name='Actual-Predicted',
        marker=dict(color='rgb(0, 204, 150)')
    ))
    actual_vs_predicted_fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
        mode='lines', name='Ideal', line=dict(dash='dash', color='rgb(255, 102, 102)')
    ))
    actual_vs_predicted_fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    actual_vs_predicted_div = actual_vs_predicted_fig.to_html(full_html=False)

    return error_div, actual_vs_predicted_div
