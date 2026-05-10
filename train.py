import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------------
# Add AWGN Noise
# -----------------------------------

def add_noise(signal, noise_level=0.2):

    noise = noise_level * np.random.randn(len(signal))

    noisy_signal = signal + noise

    return noisy_signal

# -----------------------------------
# Generate BPSK Signal
# -----------------------------------

def generate_bpsk(samples):

    bits = np.random.randint(0, 2, samples)

    signal = 2 * bits - 1

    return signal

# -----------------------------------
# Generate QPSK Signal
# -----------------------------------

def generate_qpsk(samples):

    bits = np.random.randint(0, 4, samples)

    phase = bits * (np.pi / 2)

    signal = np.cos(phase)

    return signal

# -----------------------------------
# Generate QAM Signal
# -----------------------------------

def generate_qam(samples):

    levels = [-3, -1, 1, 3]

    I = np.random.choice(levels, samples)

    Q = np.random.choice(levels, samples)

    signal = I + Q

    signal = signal / np.max(np.abs(signal))

    return signal

# -----------------------------------
# Generate OFDM Signal
# -----------------------------------

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
# Create Dataset
# -----------------------------------

X = []
y = []

for i in range(2000):

    # BPSK
    bpsk = add_noise(generate_bpsk(128))

    X.append(bpsk)

    y.append(0)

    # QPSK
    qpsk = add_noise(generate_qpsk(128))

    X.append(qpsk)

    y.append(1)

    # QAM
    qam = add_noise(generate_qam(128))

    X.append(qam)

    y.append(2)

    # OFDM
    ofdm = add_noise(generate_ofdm(128))

    X.append(ofdm)

    y.append(3)

# -----------------------------------
# Convert Arrays
# -----------------------------------

X = np.array(X)

y = np.array(y)

# Save original labels
y_original = y.copy()

# CNN input shape
X = X.reshape(-1, 128, 1)

# One-hot encoding
y = to_categorical(y, 4)

# -----------------------------------
# Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test, y_train_original, y_test_original = train_test_split(

    X,
    y,
    y_original,

    test_size=0.2,

    random_state=42

)

# -----------------------------------
# Build CNN Model
# -----------------------------------

model = Sequential([

    Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=(128,1)
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu'
    ),

    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(4, activation='softmax')

])

# -----------------------------------
# Compile Model
# -----------------------------------

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

# -----------------------------------
# Train Model
# -----------------------------------

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    batch_size=32,

    validation_data=(X_test, y_test)

)

# -----------------------------------
# Save Model
# -----------------------------------

model.save("signal_model.h5")

# -----------------------------------
# Accuracy Graph
# -----------------------------------

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

plt.grid(True)

plt.show()

# -----------------------------------
# Loss Graph
# -----------------------------------

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.grid(True)

plt.show()

# -----------------------------------
# Predictions
# -----------------------------------

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(

    y_test_original,

    predicted_labels

)

labels = ["BPSK", "QPSK", "QAM", "OFDM"]

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=labels

)

disp.plot(cmap='Blues')

plt.title("Confusion Matrix")

plt.show()

# -----------------------------------
# Classification Report
# -----------------------------------

print("\nClassification Report:\n")

print(

    classification_report(

        y_test_original,

        predicted_labels,

        target_names=labels

    )

)

print("Advanced CNN Training Completed!")