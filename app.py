import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("machine_temperature_rnn.keras")

st.title("Machine Temperature Predictor")

st.write("Enter the temperature and vibration values from the previous two timestamps.")

st.subheader("Previous Timestamp 1")

temperature1 = st.number_input(
    "Temperature 1 (°C)",
    value=81.0
)

vibration1 = st.number_input(
    "Vibration 1",
    value=3.5
)

st.subheader("Previous Timestamp 2")

temperature2 = st.number_input(
    "Temperature 2 (°C)",
    value=83.0
)

vibration2 = st.number_input(
    "Vibration 2",
    value=3.6
)

if st.button("Predict Next Temperature"):

    input_data = np.array([
        [
            [temperature1, vibration1],
            [temperature2, vibration2]
        ]
    ])

    prediction = model.predict(input_data, verbose=0)

    predicted_temperature = prediction[0][0]

    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
