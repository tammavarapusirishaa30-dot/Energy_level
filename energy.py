import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.title('Energy Level Prediction App')
st.write('Predict your energy level based on sleep hours and break time.')

# --- Model Training (re-included for standalone Streamlit app) ---
# Load data
df = pd.read_csv('/content/energy_level.csv')

# Define features (X) and target (y)
x = df[['sleep_hours', 'break_time']]
y = df[['energy_level']]

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(x, y)
# --- End Model Training ---

st.header('Input your details:')

# Input widgets using sliders
sleep_hours = st.slider('Sleep Hours', min_value=0.0, max_value=12.0, value=7.0, step=0.1)
break_time = st.slider('Break Time (hours)', min_value=0.0, max_value=4.0, value=1.0, step=0.1)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'sleep_hours': [sleep_hours],
    'break_time': [break_time]
})

# Make prediction
predicted_energy_level = model.predict(input_data)[0][0]

st.subheader('Predicted Energy Level:')
st.write(f'Your predicted energy level is: **{predicted_energy_level:.2f}**')
