import streamlit as st
import numpy as np
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------

st.set_page_config(
    page_title="AI Image Classifier",
    layout="centered"
)

# -----------------------------------------------------------
# MODEL FILE PATHS
# -----------------------------------------------------------

MNIST_MODEL_PATH = "mnist_model.keras"

CIFAR_MODEL_PATH = "cifar10_cnn_model.keras"

# -----------------------------------------------------------
# CIFAR CLASS NAMES
# -----------------------------------------------------------

CIFAR_CLASSES = [
    'Airplane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

# -----------------------------------------------------------
# LOAD SAVED MODELS
# -----------------------------------------------------------

@st.cache_resource
def load_models():

    mnist_model = keras.models.load_model(
        MNIST_MODEL_PATH
    )

    cifar_model = keras.models.load_model(
        CIFAR_MODEL_PATH
    )

    return mnist_model, cifar_model


with st.spinner("Loading saved models..."):

    mnist_model, cifar_model = load_models()

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------

st.title("🧠 AI Image Classifier")

st.write(
    "Upload an image and choose a model "
    "to classify it."
)

# -----------------------------------------------------------
# MODEL SELECTION
# -----------------------------------------------------------

model_type = st.selectbox(
    "Choose Model",
    [
        "MNIST Digit Classifier",
        "CIFAR-10 Classifier"
    ]
)

# -----------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    if st.button("Predict"):

        # ---------------------------------------------------
        # MNIST PREDICTION
        # ---------------------------------------------------

        if model_type == "MNIST Digit Classifier":

            img = image.convert('L').resize((28, 28))

            arr = np.array(img).astype('float32') / 255.0

            arr = arr.reshape(1, 784)

            pred = mnist_model.predict(
                arr,
                verbose=0
            )[0]

            label = str(np.argmax(pred))

            confidence = float(np.max(pred)) * 100

            st.success(f"Prediction: {label}")

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

            st.subheader("All Probabilities")

            for i in range(10):

                st.write(
                    f"{i}: {pred[i] * 100:.2f}%"
                )

        # ---------------------------------------------------
        # CIFAR PREDICTION
        # ---------------------------------------------------

        else:

            img = image.convert('RGB').resize((32, 32))

            arr = np.array(img).astype('float32') / 255.0

            arr = arr.reshape(1, 32, 32, 3)

            pred = cifar_model.predict(
                arr,
                verbose=0
            )[0]

            idx = int(np.argmax(pred))

            label = CIFAR_CLASSES[idx]

            confidence = float(np.max(pred)) * 100

            st.success(f"Prediction: {label}")

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

            st.subheader("All Probabilities")

            for i in range(10):

                st.write(
                    f"{CIFAR_CLASSES[i]}: "
                    f"{pred[i] * 100:.2f}%"
                )