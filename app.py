
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# Load trained model and scalers
model = load_model("machine_temperature_rnn.keras")

X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")


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
    value=81.0
)

vibration1 = st.number_input(
    "Vibration 1",
    value=3.5
)


# Previous Timestamp 2
st.subheader("Previous Timestamp 2")

temperature2 = st.number_input(
    "Temperature 2 (°C)",
    value=83.0
)

vibration2 = st.number_input(
    "Vibration 2",
    value=3.6
)


# Prediction
if st.button("Predict Next Temperature"):

    # Create input with two timestamps
    new_input = np.array([
        [temperature1, vibration1],
        [temperature2, vibration2]
    ])

    # Scale using the same X scaler used during training
    new_input_scaled = X_scaler.transform(new_input)

    # Reshape to RNN format:
    # (samples, time steps, features)
    new_input_scaled = new_input_scaled.reshape(1, 2, 2)

    # Predict scaled temperature
    prediction_scaled = model.predict(
        new_input_scaled,
        verbose=0
    )

    # Convert prediction back to original temperature
    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )

    predicted_temperature = float(prediction[0][0])

    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
