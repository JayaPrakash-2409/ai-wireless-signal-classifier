import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Wireless Signal Classifier",
    page_icon="📡",
    layout="wide"
)

# -----------------------------
# UI Style
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Labels
# -----------------------------
labels = ["BPSK", "QPSK", "QAM", "OFDM"]

# -----------------------------
# Signal Generators
# -----------------------------
def generate_bpsk(n):
    bits = np.random.randint(0, 2, n)
    return 2 * bits - 1

def generate_qpsk(n):
    bits = np.random.randint(0, 4, n)
    return np.cos(bits * np.pi / 2)

def generate_qam(n):
    levels = [-3, -1, 1, 3]
    I = np.random.choice(levels, n)
    Q = np.random.choice(levels, n)
    return (I + Q) / np.max(np.abs(I + Q))

def generate_ofdm(n):
    signal = np.zeros(n)
    for i in range(5):
        freq = np.random.randint(1, 10)
        signal += np.sin(2 * np.pi * freq * np.arange(n) / n)
    return signal / np.max(np.abs(signal))

# -----------------------------
# Noise
# -----------------------------
def add_noise(signal, noise_level):
    return signal + noise_level * np.random.randn(len(signal))

# -----------------------------
# Fake AI Model (DEPLOY SAFE)
# -----------------------------
def fake_predict():
    return np.random.dirichlet(np.ones(4))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📡 Controls")

signal_type = st.sidebar.selectbox(
    "Select Signal Type",
    labels
)

noise_level = st.sidebar.slider(
    "Noise Level",
    0.0, 1.0, 0.2
)

# -----------------------------
# Title
# -----------------------------
st.title("📡 AI Wireless Signal Classification System")
st.write("Deep Learning + DSP Simulation (Deployment Version)")

# -----------------------------
# Generate Button
# -----------------------------
if st.button("🚀 Generate Signal & Analyze"):

    # Generate signal
    if signal_type == "BPSK":
        signal = generate_bpsk(128)
    elif signal_type == "QPSK":
        signal = generate_qpsk(128)
    elif signal_type == "QAM":
        signal = generate_qam(128)
    else:
        signal = generate_ofdm(128)

    # Add noise
    signal = add_noise(signal, noise_level)

    # Fake prediction (deployment safe)
    prediction = fake_predict()
    result = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # -----------------------------
    # Metrics
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Signal", labels[result])

    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    # -----------------------------
    # Live Waveform
    # -----------------------------
    st.subheader("📈 Signal Waveform (Live)")

    chart = st.empty()

    for i in range(1, len(signal)):
        fig, ax = plt.subplots()
        ax.plot(signal[:i])
        ax.set_ylim(-2, 2)
        ax.set_title("Real-Time Signal")
        chart.pyplot(fig)
        time.sleep(0.01)

    # -----------------------------
    # FFT
    # -----------------------------
    st.subheader("🌈 Frequency Spectrum (FFT)")

    fft = np.fft.fft(signal)

    fig2, ax2 = plt.subplots()
    ax2.plot(np.abs(fft))
    ax2.set_title("FFT Spectrum")
    st.pyplot(fig2)

    # -----------------------------
    # Constellation
    # -----------------------------
    st.subheader("🎯 Constellation Diagram")

    I = signal[:64]
    Q = signal[64:128]

    fig3, ax3 = plt.subplots()
    ax3.scatter(I, Q)
    ax3.set_title("Constellation Plot")
    st.pyplot(fig3)

    # -----------------------------
    # Info
    # -----------------------------
    st.subheader("📚 Signal Info")

    st.write(f"Selected Signal: **{signal_type}**")
    st.write("Noise added to simulate real wireless channel 🌐")