import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page title
st.set_page_config(page_title="Seattle Weather Predictor", layout="centered")

# Load the saved model and encoder
@st.cache_resource
def load_assets():
    with open('seattle_weather_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

try:
    model, encoder = load_assets()
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'seattle_weather_best_model.pkl' and 'encoder.pkl' are in the same folder.")
    st.stop()

# App Header
st.title("🌧️ Seattle Weather Prediction App")
st.markdown("Enter the weather parameters below to predict the condition.")

# Sidebar for Input
st.sidebar.header("Input Weather Parameters")
precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 60.0, 5.0)
temp_max = st.sidebar.slider("Max Temperature (°C)", -5.0, 45.0, 15.0)
temp_min = st.sidebar.slider("Min Temperature (°C)", -10.0, 30.0, 8.0)
wind = st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0)

# Prediction Logic
if st.button("Predict Weather"):
    # Prepare data for model
    features = np.array([[precipitation, temp_max, temp_min, wind]])
    
    # Make prediction
    prediction_numeric = model.predict(features)
    prediction_text = encoder.inverse_transform(prediction_numeric)[0]
    
    # Visual feedback based on result
    st.subheader(f"Prediction: {prediction_text.capitalize()}")
    
    if prediction_text == "sun":
        st.write("☀️ It looks like a clear day!")
    elif prediction_text == "rain":
        st.write("☔ Don't forget your umbrella!")
    elif prediction_text == "snow":
        st.write("❄️ Stay warm, it's snowing!")
    elif prediction_text == "fog":
        st.write("🌫️ Visibility might be low.")
    else:
        st.write("☁️ Expect some light drizzle.")

# Show data summary
if st.checkbox("Show historical data statistics"):
    st.write("The model was trained on Seattle weather data featuring precipitation, temperature extremes, and wind speeds.")