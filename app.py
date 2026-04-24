import streamlit as st
import numpy as np
import pickle

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Machine Failure Prediction")

st.title("Machine Failure Prediction System")
st.write("Enter sensor values to predict failure probability")

# INPUTS
air_temp = st.number_input("Air Temperature (K)")
process_temp = st.number_input("Process Temperature (K)")
rpm = st.number_input("RPM")
torque = st.number_input("Torque")
tool_wear = st.number_input("Tool Wear")

type_L = st.selectbox("Type L", [0,1])
type_M = st.selectbox("Type M", [0,1])

# INPUT ARRAY (MATCH TRAINING ORDER!)
input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear, type_L, type_M]])

input_scaled = scaler.transform(input_data)

# PREDICT
if st.button("Predict"):
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.subheader("Result")

    if pred == 1:
        st.error("MACHINE FAILURE RISK")
    else:
        st.success("MACHINE HEALTHY")

    st.write(f"Failure Probability: {prob*100:.2f}%")

    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.7:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")
