import streamlit as st
import numpy as np
import tensorflow as tf

# Assuming predict_next_note and your model are loaded here

st.title("Predict the Next Note")

# Input: Notes as a list and temperature
notes = st.text_input("Enter notes as a list (e.g., [60, 62, 64]):", "[]")
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)

# Predict button
if st.button("Predict"):
    # Convert notes input to numpy array
    notes_array = np.array(eval(notes))

    # Make prediction
    prediction = predict_next_note(notes_array, model, temperature)
    
    # Display prediction
    st.write("Prediction:")
    st.write(f"Pitch: {prediction[0]}")
    st.write(f"Step: {prediction[1]}")
    st.write(f"Duration: {prediction[2]}")
