
import streamlit as st
import joblib
import numpy as np

# Load model & RFE
model = joblib.load('RandomForestRegressor_model.pkl')
rfe = joblib.load('rfe_selector.pkl')
selected_features = joblib.load('selected_features.pkl')

st.title("🏡 House Price Prediction (Random Forest + RFE)")

# Create input fields dynamically
inputs = []
st.subheader("Enter feature values:")
for feature in selected_features:
    val = st.number_input(f"{feature}", value=0.0)
    inputs.append(val)

# Predict
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"Predicted Value: {prediction[0]:.2f}")
