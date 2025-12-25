import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Seattle Weather Predictor", layout="centered")

@st.cache_resource
def load_assets():
    # Loading the best model and the label encoder
    with open('seattle_weather_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

try:
    model, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

st.title("🌧️ Seattle Weather Prediction App")
st.markdown("Enter weather details below to predict the conditions in Seattle.")

# User inputs
col1, col2 = st.columns(2)
with col1:
    precipitation = st.number_input("Precipitation (mm)", 0.0, 100.0, 0.0)
    temp_max = st.number_input("Max Temperature (°C)", -20.0, 50.0, 15.0)
with col2:
    temp_min = st.number_input("Min Temperature (°C)", -30.0, 40.0, 5.0)
    wind = st.number_input("Wind Speed (m/s)", 0.0, 25.0, 3.0)

if st.button("Predict Weather"):
    # Features must match the training order: precipitation, temp_max, temp_min, wind
    features = np.array([[precipitation, temp_max, temp_min, wind]])
    prediction = model.predict(features)
    weather_type = encoder.inverse_transform(prediction)[0]
    
    st.success(f"The predicted weather is: **{weather_type.upper()}**")
    
    # Visual cues
    if weather_type == 'sun': st.write("☀️ Clear skies ahead!")
    elif weather_type == 'rain': st.write("☔ Grab an umbrella!")
    elif weather_type == 'snow': st.write("❄️ Stay warm, it's snowing!")
