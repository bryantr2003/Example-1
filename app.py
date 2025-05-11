
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load model
@st.cache_resource
def load_model():
    return joblib.load("linreg_model.pkl")

model = load_model()

# Define expected features
features = ['Gr Liv Area', 'Garage Cars', 'Garage Area', '1st Flr SF', 'Year Remod/Add']

# Title
st.title("🏡 Ames Housing Price Predictor")

# Sidebar inputs
st.sidebar.header("Enter House Features")
gr_liv_area = st.sidebar.number_input("Gr Liv Area (sq ft)", min_value=200, max_value=4000, value=1500)
garage_cars = st.sidebar.number_input("Garage Cars", min_value=0, max_value=4, value=2)
garage_area = st.sidebar.number_input("Garage Area (sq ft)", min_value=0, max_value=1000, value=500)
first_flr = st.sidebar.number_input("1st Floor SF", min_value=200, max_value=3000, value=1200)
year_remod = st.sidebar.number_input("Year Remodeled", min_value=1950, max_value=2025, value=2000)

# Make prediction
input_data = [[gr_liv_area, garage_cars, garage_area, first_flr, year_remod]]
if st.button("🔍 Predict Sale Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Sale Price: ${prediction:,.2f}")
