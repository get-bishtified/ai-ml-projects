import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("car_price_model.pkl")
le_brand = joblib.load("le_brand.pkl")
le_fuel = joblib.load("le_fuel.pkl")
le_trans = joblib.load("le_trans.pkl")

st.set_page_config(page_title="🚘 Used Car Price Predictor")
st.title("🚗 Used Car Price Predictor")
st.markdown("Enter your car details and get an estimated resale price!")

# User inputs
brand = st.selectbox("Brand", le_brand.classes_)
year = st.number_input("Year", min_value=2000, max_value=2024, value=2015)
mileage = st.number_input("Mileage (in km)", min_value=1000, max_value=300000, step=1000, value=50000)
fuel = st.selectbox("Fuel Type", le_fuel.classes_)
trans = st.selectbox("Transmission", le_trans.classes_)

if st.button("Predict Price"):
    input_data = [[
        le_brand.transform([brand])[0],
        year,
        mileage,
        le_fuel.transform([fuel])[0],
        le_trans.transform([trans])[0]
    ]]

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Resale Price: ₹{prediction:,.2f}")