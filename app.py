
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model


# Load trained model
model = load_model("machine_temperature_rnn.keras")


# Dataset min and max values
TEMP_MIN = 60.0
TEMP_MAX = 83.0

VIBRATION_MIN = 2.1
VIBRATION_MAX = 3.6


# Scaling function
def scale_temperature(value):
    return (value - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)


def scale_vibration(value):
    return (value - VIBRATION_MIN) / (VIBRATION_MAX - VIBRATION_MIN)


# Reverse scaling for temperature
def inverse_scale_temperature(value):
    return value * (TEMP_MAX - TEMP_MIN) + TEMP_MIN


# Page title
st.title("Machine Temperature Predictor")

st.write(
    "Enter the temperature and vibration values "
    "from the previous two timestamps."
)


# Previous Timestamp 1
st.subheader("Previous Timestamp 1")

temperature1 = st.number_input(
    "Temperature 1 (°C)",
    min_value=60.0,
    max_value=83.0,
    value=81.0
)

vibration1 = st.number_input(
    "Vibration 1",
    min_value=2.1,
    max_value=3.6,
    value=3.5
)


# Previous Timestamp 2
st.subheader("Previous Timestamp 2")

temperature2 = st.number_input(
    "Temperature 2 (°C)",
    min_value=60.0,
    max_value=83.0,
    value=83.0
)

vibration2 = st.number_input(
    "Vibration 2",
    min_value=2.1,
    max_value=3.6,
    value=3.6
)


# Prediction
if st.button("Predict Next Temperature"):

    # Scale the input values
    input_data = np.array([
        [
            scale_temperature(temperature1),
            scale_vibration(vibration1)
        ],
        [
            scale_temperature(temperature2),
            scale_vibration(vibration2)
        ]
    ])

    # Reshape for RNN
    # (samples, time steps, features)
    input_data = input_data.reshape(1, 2, 2)

    # Predict scaled temperature
    prediction_scaled = model.predict(
        input_data,
        verbose=0
    )

    # Convert prediction back to °C
    predicted_temperature = inverse_scale_temperature(
        float(prediction_scaled[0][0])
    )

    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
