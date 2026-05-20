import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Network Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (no unsafe HTML in prediction area)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e0e1a;
    color: #e0e0e0;
}
.main { background-color: #0e0e1a; }

h1, h2, h3 { font-family: 'Space Mono', monospace; color: #6c63ff; }

.model-card {
    background: #1a1a2e;
    border: 1px solid #6c63ff44;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.accuracy-badge {
    background: #6c63ff22;
    border: 1px solid #6c63ff;
    color: #6c63ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.result-box {
    background: #1a1a2e;
    border: 2px solid #6c63ff;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
}
.pred-label {
    font-size: 48px;
    font-weight: 700;
    color: #ffffff;
    font-family: 'Space Mono', monospace;
}
.conf-text {
    font-size: 22px;
    color: #6c63ff;
    font-weight: 600;
}
.footer-bar {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 16px 24px;
    margin-top: 32px;
    font-size: 13px;
    color: #888;
    border: 1px solid #6c63ff22;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS  (cached so loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_mnist():
    return tf.keras.models.load_model("mnist_model.keras")

@st.cache_resource
def load_cifar():
    return tf.keras.models.load_model("cifar10_cnn_model.keras")

mnist_model  = load_mnist()
cifar_model  = load_cifar()

MNIST_LABELS = [str(i) for i in range(10)]
CIFAR_LABELS = ["Airplane","Automobile","Bird","Cat","Deer",
                 "Dog","Frog","Horse","Ship","Truck"]

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_mnist(image: Image.Image):
    """
    Handles both Dense model  → expects (1, 784)
    and CNN model             → expects (1, 28, 28, 1)
    Auto-detects from model input shape.
    Also inverts image if background is white (normal paper).
    """
    img = image.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0

    # Invert if mostly white background (normal handwritten on paper)
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Check what shape the model expects
    input_shape = mnist_model.input_shape  # e.g. (None, 784) or (None,28,28,1)

    if len(input_shape) == 2:
        # Dense / Flatten model → flatten to (1, 784)
        arr = arr.flatten().reshape(1, 784)
    else:
        # CNN model → reshape to (1, 28, 28, 1)
        arr = arr.reshape(1, 28, 28, 1)

    return arr


def preprocess_cifar(image: Image.Image):
    img = image.convert("RGB").resize((32, 32))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 32, 32, 3)

# ─────────────────────────────────────────────
# CONFIDENCE BAR CHART  (native Streamlit — no HTML)
# ─────────────────────────────────────────────
def show_confidence_bars(probs, labels, top_n=5):
    st.markdown("#### 📊 Top Confidence Scores")
    top_indices = np.argsort(probs)[::-1][:top_n]

    for rank, idx in enumerate(top_indices):
        label = labels[idx]
        prob  = float(probs[idx])
        col1, col2 = st.columns([3, 1])
        with col1:
            # st.progress expects 0.0–1.0
            st.write(f"**{label}**")
            st.progress(prob)
        with col2:
            st.metric(label="", value=f"{prob*100:.1f}%")

# ─────────────────────────────────────────────
# PREDICTION RESULT BOX
# ─────────────────────────────────────────────
def show_result(pred_label, confidence):
    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:16px; color:#aaa; margin-bottom:8px;">PREDICTION</div>
        <div class="pred-label">{pred_label}</div>
        <div class="conf-text">Confidence: {confidence*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🧠 Neural Network Image Classifier")
st.markdown("**6th Semester Minor Project** &nbsp;|&nbsp; Deep Learning · TensorFlow · Streamlit")
st.divider()

# Model info cards
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="model-card">
        <b style="color:#6c63ff;font-size:18px;">📝 MNIST — Digit Recognizer</b><br><br>
        Dataset: 70,000 images &nbsp;|&nbsp; Classes: 10 (0–9)<br>
        Input: 28×28 Grayscale<br>
        Architecture: Dense Neural Network<br><br>
        <span class="accuracy-badge">✅ Accuracy: 97.92%</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <b style="color:#6c63ff;font-size:18px;">🖼️ CIFAR-10 — Object Classifier</b><br><br>
        Dataset: 60,000 images &nbsp;|&nbsp; Classes: 10<br>
        Input: 32×32 RGB Color<br>
        Architecture: CNN + BatchNorm + Dropout<br><br>
        <span class="accuracy-badge">✅ Accuracy: 82.5%</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 MNIST — Digit Recognition", "🖼️ CIFAR-10 — Object Classification"])

# ── TAB 1: MNIST ─────────────────────────────
with tab1:
    st.subheader("Upload a Handwritten Digit (0–9)")
    st.info("💡 Tip: Write a big digit on plain white paper, take a photo, upload here. Works best with clear digits!")

    uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], key="mnist_upload")

    if uploaded:
        image = Image.open(uploaded)
        c1, c2 = st.columns([1, 2])

        with c1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with c2:
            with st.spinner("Analyzing..."):
                try:
                    arr   = preprocess_mnist(image)
                    preds = mnist_model.predict(arr, verbose=0)[0]
                    pred_class = int(np.argmax(preds))
                    confidence = float(np.max(preds))

                    show_result(MNIST_LABELS[pred_class], confidence)
                    show_confidence_bars(preds, MNIST_LABELS, top_n=5)

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.info("Make sure `mnist_model.keras` is in the same folder as `app2.py`.")

# ── TAB 2: CIFAR-10 ──────────────────────────
with tab2:
    st.subheader("Upload an Image to Classify")
    st.info("💡 Categories: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck")

    uploaded2 = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], key="cifar_upload")

    if uploaded2:
        image2 = Image.open(uploaded2)
        c1, c2 = st.columns([1, 2])

        with c1:
            st.image(image2, caption="Uploaded Image", use_container_width=True)

        with c2:
            with st.spinner("Analyzing..."):
                try:
                    arr2   = preprocess_cifar(image2)
                    preds2 = cifar_model.predict(arr2, verbose=0)[0]
                    pred_class2 = int(np.argmax(preds2))
                    confidence2 = float(np.max(preds2))

                    show_result(CIFAR_LABELS[pred_class2], confidence2)
                    show_confidence_bars(preds2, CIFAR_LABELS, top_n=5)

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.info("Make sure `cifar10_cnn_model.keras` is in the same folder as `app2.py`.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    🧠 <b>Architecture Summary</b> &nbsp;|&nbsp;
    MNIST: Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax) &nbsp;&nbsp;·&nbsp;&nbsp;
    CIFAR-10: Conv2D×3 → BatchNorm → MaxPool → Dropout → Dense(10, Softmax) &nbsp;&nbsp;·&nbsp;&nbsp;
    Built with TensorFlow + Streamlit
</div>
""", unsafe_allow_html=True)