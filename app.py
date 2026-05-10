import streamlit as st
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="AI Wireless Signal Classifier")

st.title("📡 AI Wireless Signal Classifier")

st.write("Enter signal values for prediction")

sample = st.text_input("Signal input (comma separated)", "1,2,3,4")

if st.button("Predict"):
    try:
        values = np.array([float(x) for x in sample.split(",")])
        st.success("Signal received successfully!")
        st.write("Input shape:", values.shape)
    except:
        st.error("Please enter valid numbers only")