import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
# Make sure the paths match where you saved the files
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please run the saving code first.")
    st.stop()

# Get the list of top features used during training
# This list should match the 'top_features' list from your notebook
# You might want to manually copy this list from your notebook or save it as a file
# For now, I'll use a placeholder. REPLACE THIS WITH YOUR ACTUAL top_features LIST
top_features = ['g__Lactobacillus', 'g__Cryptobacterium', 'g__Bifidobacterium', 'g__Scardovia', 'g__Alloscardovia', 'g__Cardiobacterium', 'g__Bulleidia', 'c__Betaproteobacteria', 'g__Prevotella_6', 'g__Bergeyella', 'p__Proteobacteria', 'p__Saccharibacteria', 'g__Kingella', 'c__Flavobacteriia', 'c__Coriobacteriia', 'g__Mycoplasma', 'g__Corynebacterium', 'c__Clostridia', 'g__Haemophilus', 'g__Porphyromonas'] # Replace with your actual top_features list


st.title("Oral Microbiome and Depression Prediction")

st.write("Enter the abundance values for the following microbial features to predict depression:")

# Create input fields for each of the top features
feature_inputs = {}
for feature in top_features:
    # You might want to add more specific input types or hints based on the feature data
    feature_inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0, format="%.6f")

# Create a button to make predictions
if st.button("Predict Depression"):
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([feature_inputs])

    # Ensure the input DataFrame has the correct order of columns
    input_df = input_df[top_features]

    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Make a prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Display the prediction result
    if prediction[0] == 1:
        st.write("Prediction: **Depression**")
        st.write(f"Confidence: {prediction_proba[0][1]:.2f}")
    else:
        st.write("Prediction: **No Depression**")
        st.write(f"Confidence: {prediction_proba[0][0]:.2f}")
