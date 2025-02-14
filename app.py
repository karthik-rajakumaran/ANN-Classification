import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## Load the trained model, scaler and encoder
model = tf.keras.models.load_model('model.h5')

## Loader encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title of the web app
st.title("Customer Churn prediction")

# Create two columns
col1, col2 = st.columns(2)

#User input
# Create input fields based on the dictionary keys
with col1:
    credit_score = st.number_input("Credit Score", min_value=0, value=600)
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", min_value=0, max_value=100)
    tenure = st.slider("Tenure (years)", min_value=0, max_value=10) 

with col2:
    balance = st.number_input("Balance", format="%.2f")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
    has_cr_card = st.selectbox("Has Credit Card?", options=[0, 1])
    is_active_member = st.selectbox("Is Active Member?", options=[0, 1])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, format="%.2f")

#prepare the input data 
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#one hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Concat one hot encoded data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scaling the input data
input_scaled = scaler.transform(input_data)

##Predict churn
predictions = model.predict(input_scaled)
prediction_proba = predictions[0][0]


st.write(prediction_proba)
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')