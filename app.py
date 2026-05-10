import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import load_model

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(

    page_title="AI Wireless Signal Classifier",

    page_icon="📡",

    layout="wide"

)

# -----------------------------------
# Dark Theme CSS
# -----------------------------------

st.markdown("""

<style>

body {
    background-color: #0E1117;
    color: white;
}

.main {
    background-color: #0E1117;
}

.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

</style>

""", unsafe_allow_html=True)

# -----------------------------------
# Load CNN Model
# -----------------------------------

model = load_model("signal_model.h5")

# -----------------------------------
# Labels
# -----------------------------------

labels = ["BPSK", "QPSK", "QAM", "OFDM"]

# -----------------------------------
# Signal Generators
# -----------------------------------

def generate_bpsk(samples):

    bits = np.random.randint(0, 2, samples)

    signal = 2 * bits - 1

    return signal


def generate_qpsk(samples):

    bits = np.random.randint(0, 4, samples)

    phase = bits * (np.pi / 2)

    signal = np.cos(phase)

    return signal


def generate_qam(samples):

    levels = [-3, -1, 1, 3]

    I = np.random.choice(levels, samples)

    Q = np.random.choice(levels, samples)

    signal = I + Q

    signal = signal / np.max(np.abs(signal))

    return signal


def generate_ofdm(samples):

    carriers = 8

    signal = np.zeros(samples)

    for i in range(carriers):

        freq = np.random.randint(1, 10)

        signal += np.sin(
            2 * np.pi * freq * np.arange(samples) / samples
        )

    signal = signal / np.max(np.abs(signal))

    return signal

# -----------------------------------
# Add Noise
# -----------------------------------

def add_noise(signal, noise_level=0.2):

    noise = noise_level * np.random.randn(len(signal))

    return signal + noise

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.title("📡 Signal Controls")

signal_type = st.sidebar.selectbox(

    "Choose Signal Type",

    ["BPSK", "QPSK", "QAM", "OFDM"]

)

noise_level = st.sidebar.slider(

    "Noise Level",

    0.0,
    1.0,
    0.2

)

st.sidebar.markdown("---")

st.sidebar.info(

    "Deep Learning Based Wireless Signal Analysis"

)

# -----------------------------------
# Main Title
# -----------------------------------

st.title("📡 AI Wireless Signal Classification Dashboard")

st.write(

    "CNN-Based Real-Time Communication Signal Analysis"

)

# -----------------------------------
# Generate Signal
# -----------------------------------

if st.button("🚀 Generate and Analyze"):

    # Generate selected signal
    if signal_type == "BPSK":

        sample = generate_bpsk(128)

    elif signal_type == "QPSK":

        sample = generate_qpsk(128)

    elif signal_type == "QAM":

        sample = generate_qam(128)

    else:

        sample = generate_ofdm(128)

    # Add realistic noise
    sample = add_noise(sample, noise_level)

    # CNN input
    input_signal = sample.reshape(1, 128, 1)

    # Predict
    prediction = model.predict(input_signal)

    result = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # -----------------------------------
    # Metrics
    # -----------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Predicted Signal",

            labels[result]

        )

    with col2:

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )

    # -----------------------------------
    # Live Animated Signal
    # -----------------------------------

    st.subheader("🎬 Live Signal Animation")

    chart_placeholder = st.empty()

    for i in range(1, len(sample)+1):

        fig, ax = plt.subplots(figsize=(12,4))

        ax.plot(sample[:i])

        ax.set_title("Real-Time Signal Waveform")

        ax.set_xlabel("Time")

        ax.set_ylabel("Amplitude")

        ax.grid(True)

        chart_placeholder.pyplot(fig)

        time.sleep(0.02)

    # -----------------------------------
    # FFT Spectrum
    # -----------------------------------

    st.subheader("🌈 FFT Spectrum")

    fft = np.fft.fft(sample)

    fig2, ax2 = plt.subplots(figsize=(12,4))

    ax2.plot(np.abs(fft))

    ax2.set_title("Frequency Spectrum")

    ax2.set_xlabel("Frequency")

    ax2.set_ylabel("Magnitude")

    ax2.grid(True)

    st.pyplot(fig2)

    # -----------------------------------
    # Constellation Diagram
    # -----------------------------------

    st.subheader("🎯 Constellation Diagram")

    I = sample[:64]

    Q = sample[64:]

    fig3, ax3 = plt.subplots(figsize=(6,6))

    ax3.scatter(I, Q)

    ax3.set_title("Constellation Plot")

    ax3.set_xlabel("In-Phase (I)")

    ax3.set_ylabel("Quadrature (Q)")

    ax3.grid(True)

    st.pyplot(fig3)

    # -----------------------------------
    # Signal Information
    # -----------------------------------

    st.subheader("📚 Signal Information")

    if signal_type == "BPSK":

        st.write("""
        • Binary Phase Shift Keying  
        • Uses 2 phases  
        • Basic digital modulation  
        """)

    elif signal_type == "QPSK":

        st.write("""
        • Quadrature Phase Shift Keying  
        • Uses 4 phases  
        • Used in wireless communication  
        """)

    elif signal_type == "QAM":

        st.write("""
        • Quadrature Amplitude Modulation  
        • Used in WiFi and 5G  
        """)

    else:

        st.write("""
        • Orthogonal Frequency Division Multiplexing  
        • Multi-carrier communication system  
        • Used in LTE and WiFi  
        """)