import streamlit as st
import numpy as np
from tensorflow import keras  # pyright: ignore[reportMissingModuleSource]
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Neural Network Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0a0a0f;
        color: #e8e8f0;
    }

    /* Header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #7878a0;
        font-weight: 300;
        letter-spacing: 0.05em;
    }
    .accent {
        color: #6c63ff;
    }

    /* Model Cards */
    .model-card {
        background: #13131f;
        border: 1px solid #2a2a40;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        transition: border-color 0.2s;
    }
    .model-card:hover {
        border-color: #6c63ff;
    }
    .model-card-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #6c63ff;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .model-card-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .model-card-desc {
        font-size: 0.82rem;
        color: #6060a0;
        line-height: 1.5;
    }
    .model-badge {
        display: inline-block;
        background: #1e1e35;
        border: 1px solid #3a3a60;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        color: #a0a0d0;
        margin-top: 0.6rem;
        font-family: 'Space Mono', monospace;
    }

    /* Prediction Result */
    .prediction-box {
        background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
        border: 1px solid #6c63ff;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #6c63ff;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .pred-value {
        font-family: 'Space Mono', monospace;
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .pred-confidence {
        font-size: 1rem;
        color: #a0a0d0;
    }
    .pred-confidence span {
        color: #6c63ff;
        font-weight: 700;
    }

    /* Confidence Bars */
    .conf-bar-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #9090c0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.8rem;
    }
    .bar-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.45rem;
        gap: 0.6rem;
    }
    .bar-name {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #a0a0d0;
        min-width: 80px;
        text-align: right;
    }
    .bar-track {
        flex: 1;
        height: 6px;
        background: #1e1e35;
        border-radius: 3px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 3px;
        background: #6c63ff;
        transition: width 0.6s ease;
    }
    .bar-fill-top {
        background: linear-gradient(90deg, #6c63ff, #a78bfa);
    }
    .bar-pct {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #7070a0;
        min-width: 40px;
    }

    /* Upload area */
    .upload-hint {
        font-size: 0.78rem;
        color: #50508a;
        text-align: center;
        margin-top: 0.4rem;
    }

    /* Info box */
    .info-box {
        background: #0e0e1c;
        border: 1px solid #2a2a40;
        border-left: 3px solid #6c63ff;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-top: 0.8rem;
        font-size: 0.82rem;
        color: #7878a8;
        line-height: 1.6;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #1e1e35;
        margin: 1.5rem 0;
    }

    /* Streamlit override */
    div[data-testid="stFileUploader"] > div {
        background: #13131f !important;
        border: 1.5px dashed #3a3a60 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploader"]:hover > div {
        border-color: #6c63ff !important;
    }

    button[kind="primary"], .stButton > button {
        background: #6c63ff !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 400 !important;
    }
    .stButton > button:hover {
        background: #7c73ff !important;
        transform: translateY(-1px);
    }

    /* Hide streamlit default elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD MODELS (cached)
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    mnist = keras.models.load_model("mnist_model.keras")
    cifar = keras.models.load_model("cifar10_cnn_model.keras")
    return mnist, cifar

# -----------------------------------------------------------
# PREPROCESSING FUNCTIONS
# -----------------------------------------------------------
def preprocess_mnist(image: Image.Image) -> np.ndarray:
    """
    MNIST expects: 28x28, grayscale, white digit on BLACK background, values 0-1
    Most uploaded images are black on white → we invert them
    """
    img = image.convert("L")           # Convert to grayscale
    img = img.resize((28, 28))         # Resize to 28x28
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize 0-1

    # If the image is mostly white (dark digit on white bg), invert it
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    return img_array.reshape(1, 28, 28, 1)

def preprocess_cifar(image: Image.Image) -> np.ndarray:
    """CIFAR-10 expects: 32x32, RGB, values 0-1"""
    img = image.convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array.reshape(1, 32, 32, 3)

# -----------------------------------------------------------
# LABELS
# -----------------------------------------------------------
MNIST_LABELS = [str(i) for i in range(10)]
CIFAR_LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# -----------------------------------------------------------
# CONFIDENCE BARS HTML
# -----------------------------------------------------------
def confidence_bars_html(probs, labels, top_n=5):
    sorted_idx = np.argsort(probs)[::-1][:top_n]
    bars = ""
    for i, idx in enumerate(sorted_idx):
        pct = probs[idx] * 100
        fill_class = "bar-fill bar-fill-top" if i == 0 else "bar-fill"
        bars += f"""
        <div class="bar-row">
            <div class="bar-name">{labels[idx]}</div>
            <div class="bar-track">
                <div class="{fill_class}" style="width:{min(pct,100):.1f}%"></div>
            </div>
            <div class="bar-pct">{pct:.1f}%</div>
        </div>
        """
    return f"""
    <div>
        <div class="conf-bar-label">Top {top_n} Confidence Scores</div>
        {bars}
    </div>
    """

# -----------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------
def main():
    # Load models
    try:
        mnist_model, cifar_model = load_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        models_loaded = False

    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🧠 Neural <span class="accent">Classifier</span></div>
        <div class="hero-subtitle">Image Classification using Deep Learning &nbsp;·&nbsp; Minor Project · 6th Sem</div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------
    # LAYOUT: Two columns
    # -----------------------------------------------------------
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        # Model Selection
        st.markdown("""
        <div class="model-card">
            <div class="model-card-title">Select Model</div>
            <div class="model-card-name">Choose a Neural Network</div>
            <div class="model-card-desc">Two CNN models trained on different datasets with different classification tasks.</div>
        </div>
        """, unsafe_allow_html=True)

        model_choice = st.radio(
            "",
            ["MNIST — Handwritten Digit Recognition", "CIFAR-10 — Object Classification"],
            label_visibility="collapsed"
        )

        is_mnist = "MNIST" in model_choice

        # Model info card
        if is_mnist:
            st.markdown("""
            <div class="model-card">
                <div class="model-card-title">Model Info</div>
                <div class="model-card-name">MNIST CNN</div>
                <div class="model-card-desc">
                    Trained on 60,000 handwritten digit images.<br>
                    Classifies digits 0–9. Accuracy: <b style="color:#a78bfa">97.92%</b>
                </div>
                <span class="model-badge">28×28 Grayscale Input</span>
                <span class="model-badge" style="margin-left:6px">10 Classes</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="info-box">📌 Upload a clear handwritten digit image. Works best with <b>white digit on dark background</b>, or plain <b>black on white</b> — both work!</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-card">
                <div class="model-card-title">Model Info</div>
                <div class="model-card-name">CIFAR-10 CNN</div>
                <div class="model-card-desc">
                    Trained on 50,000 color images across 10 object categories.<br>
                    Accuracy: <b style="color:#a78bfa">82.5%</b>
                </div>
                <span class="model-badge">32×32 RGB Input</span>
                <span class="model-badge" style="margin-left:6px">10 Classes</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="info-box">📌 Upload a clear image of: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, or Truck.</div>', unsafe_allow_html=True)

        # Upload
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )
        st.markdown('<div class="upload-hint">Supported: JPG, PNG, BMP, WEBP</div>', unsafe_allow_html=True)

    with col_right:
        if uploaded_file is not None and models_loaded:
            image = Image.open(uploaded_file)

            # Display uploaded image
            st.markdown('<div class="model-card-title" style="margin-bottom:0.6rem;">Uploaded Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

            # Predict
            try:
                if is_mnist:
                    processed = preprocess_mnist(image)
                    predictions = mnist_model.predict(processed, verbose=0)[0]
                    labels = MNIST_LABELS
                else:
                    processed = preprocess_cifar(image)
                    predictions = cifar_model.predict(processed, verbose=0)[0]
                    labels = CIFAR_LABELS

                pred_idx = np.argmax(predictions)
                confidence = predictions[pred_idx] * 100
                pred_label = labels[pred_idx]

                # Result box
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="pred-label">Prediction Result</div>
                    <div class="pred-value">{pred_label}</div>
                    <div class="pred-confidence">Confidence: <span>{confidence:.2f}%</span></div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bars
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(confidence_bars_html(predictions, labels), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        elif not models_loaded:
            st.markdown("""
            <div class="model-card" style="text-align:center; padding: 3rem 1rem;">
                <div class="pred-label">Status</div>
                <div class="model-card-name">⚠️ Models Not Found</div>
                <div class="model-card-desc">Make sure mnist_model.keras and cifar10_cnn_model.keras are in the same folder as app2.py</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-card" style="text-align:center; padding: 3.5rem 1rem;">
                <div class="pred-value" style="font-size:2.5rem;">↑</div>
                <div class="model-card-title" style="margin-top:0.5rem;">Upload an image to begin</div>
                <div class="model-card-desc" style="margin-top:0.5rem;">Select a model on the left, then upload an image to see the neural network classify it in real time.</div>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------------
    # FOOTER — Architecture Info
    # -----------------------------------------------------------
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="model-card" style="padding:1rem;">
            <div class="model-card-title">MNIST Architecture</div>
            <div class="model-card-desc" style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#6868a8; line-height:1.8">
            Conv2D(32) → MaxPool<br>Conv2D(64) → MaxPool<br>Flatten → Dense(128)<br>Dropout(0.5) → Dense(10)<br>Activation: Softmax
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="model-card" style="padding:1rem;">
            <div class="model-card-title">CIFAR-10 Architecture</div>
            <div class="model-card-desc" style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#6868a8; line-height:1.8">
            Conv2D(32) → BatchNorm → Pool<br>Conv2D(64) → BatchNorm → Pool<br>Conv2D(128) → Flatten<br>Dense(256) → Dropout(0.5)<br>Dense(10) → Softmax
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="model-card" style="padding:1rem;">
            <div class="model-card-title">Training Info</div>
            <div class="model-card-desc" style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#6868a8; line-height:1.8">
            Platform: Google Colab<br>Optimizer: Adam<br>Loss: Categorical CE<br>MNIST Acc: 97.92%<br>CIFAR Acc: 82.5%
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()