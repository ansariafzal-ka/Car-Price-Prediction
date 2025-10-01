import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder

features = ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'brand_name', 'car_age']

st.header('🚗Car Price Predictor')
st.write("""
📊This web app predicts the price of a car based on its features, the details filled is processed and passed into a machine learning model which then predicts the price of the car.
""")

st.subheader('Car Details')

col1, col2 = st.columns(2)
with col1:
    fuel = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'Other'])
with col2:
    seller_type = st.selectbox('Seller Type', ['Individual', 'other_dealer'])

col3, col4 = st.columns(2)
with col3:
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
with col4:
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
col5, col6 = st.columns(2)
with col5:
    brand_name = st.selectbox('Brand Name', [
        'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford', 'Toyota', 
        'Chevrolet', 'Renault', 'Volkswagen', 'Skoda', 'Nissan', 'Audi', 
        'BMW', 'Fiat', 'Datsun', 'Mercedes-Benz', 'Mitsubishi', 'Jaguar', 
        'Land', 'Volvo', 'Ambassador', 'Jeep', 'OpelCorsa', 'MG', 'Force', 
        'Daewoo', 'Isuzu', 'Kia'
    ])
with col6:
    car_age = st.number_input('Car Age (Years)', min_value=0, max_value=50, value=5)

km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=50000, step=1000)


def preprocess_features(km_driven, fuel, seller_type, transmission, owner, brand_name, car_age):
    
    input_data = {
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'brand_name': [brand_name],
        'car_age': [car_age]
    }
    
    input_df = pd.DataFrame(input_data)
    
    input_df['km_driven'] = np.log(input_df['km_driven'])
    
    owner_categories = [['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']]
    ordinal_encoder = OrdinalEncoder(categories=owner_categories)
    input_df['owner_encoded'] = ordinal_encoder.fit_transform(input_df[['owner']])
    input_df = input_df.drop('owner', axis=1)
    
    input_df = pd.get_dummies(input_df, columns=['fuel', 'seller_type', 'transmission', 'brand_name'], drop_first=True, dtype=int)
    
    expected_columns = [
        'km_driven', 'car_age', 'owner_encoded', 'fuel_Petrol', 'fuel_Other',
        'seller_type_other_dealer', 'transmission_Manual', 'brand_name_Audi',
        'brand_name_BMW', 'brand_name_Chevrolet', 'brand_name_Daewoo',
        'brand_name_Datsun', 'brand_name_Fiat', 'brand_name_Force', 'brand_name_Ford',
        'brand_name_Honda', 'brand_name_Hyundai', 'brand_name_Isuzu', 'brand_name_Jaguar',
        'brand_name_Jeep', 'brand_name_Kia', 'brand_name_Land', 'brand_name_MG',
        'brand_name_Mahindra', 'brand_name_Maruti', 'brand_name_Mercedes-Benz',
        'brand_name_Mitsubishi', 'brand_name_Nissan', 'brand_name_OpelCorsa',
        'brand_name_Renault', 'brand_name_Skoda', 'brand_name_Tata', 'brand_name_Toyota',
        'brand_name_Volkswagen', 'brand_name_Volvo'
    ]
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expected_columns]
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    
    scaled_features = scaler.transform(input_df.values)
    
    return scaled_features

if st.button('Predict Price'):
    try:
        processed_input = preprocess_features(km_driven, fuel, seller_type, transmission, owner, brand_name, car_age)
        
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        prediction = model.predict(processed_input)[0]
        predicted_price = np.exp(prediction)
        
        st.success(f"Predicted Car Price: ₹{predicted_price:,.2f}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")