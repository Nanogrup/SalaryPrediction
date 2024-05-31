import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def train_and_save_model():
    # read data
    df = pd.read_csv("data/salaryData.csv")

    # Fill missing values with mean 
    # only for numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist() 
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Fill missing values with mode
    # for categorical columns
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Education Level'].fillna(df['Education Level'].mode()[0], inplace=True)
    df['Job Title'].fillna(df['Job Title'].mode()[0], inplace=True)

    # encode categorical columns 
    data = df.copy()  
    label_encoder_gender = LabelEncoder()
    label_encoder_education = LabelEncoder()
    label_encoder_title = LabelEncoder()
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
    data['Education Level'] = label_encoder_education.fit_transform(data['Education Level'])
    data['Job Title'] = label_encoder_title.fit_transform(data['Job Title'])

    # normalization for numerical columns
    scaler_age = MinMaxScaler()
    scaler_experience = MinMaxScaler()
    scaler_salary = MinMaxScaler()
    data['Age'] = scaler_age.fit_transform(data['Age'].values.reshape(-1, 1))
    data['Years of Experience'] = scaler_experience.fit_transform(data['Years of Experience'].values.reshape(-1, 1))
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

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save the trained model, encoders, and scaler
    with open('salary_model.pkl', 'wb') as file:
        pickle.dump((model, label_encoder_gender, label_encoder_education, label_encoder_title, scaler_salary), file)
