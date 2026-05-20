import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
from scipy import ndimage

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Network Image Classifier",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:#0e0e1a; color:#e0e0e0; }
.main { background-color:#0e0e1a; }
h1,h2,h3 { font-family:'Space Mono',monospace; color:#6c63ff; }
.model-card { background:#1a1a2e; border:1px solid #6c63ff44; border-radius:12px; padding:20px; margin-bottom:16px; }
.accuracy-badge { background:#6c63ff22; border:1px solid #6c63ff; color:#6c63ff; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }
.result-box { background:#1a1a2e; border:2px solid #6c63ff; border-radius:16px; padding:28px; text-align:center; margin:16px 0; }
.pred-label { font-size:64px; font-weight:700; color:#ffffff; font-family:'Space Mono',monospace; }
.conf-text { font-size:22px; color:#6c63ff; font-weight:600; }
.footer-bar { background:#1a1a2e; border-radius:12px; padding:16px 24px; margin-top:32px; font-size:13px; color:#888; border:1px solid #6c63ff22; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_mnist():
    return tf.keras.models.load_model("mnist_model.keras")

@st.cache_resource
def load_cifar():
    return tf.keras.models.load_model("cifar10_cnn_model.keras")

mnist_model = load_mnist()
cifar_model = load_cifar()

MNIST_LABELS = [str(i) for i in range(10)]
CIFAR_LABELS = ["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

# ─────────────────────────────────────────────
# MNIST PREPROCESSING — Proper MNIST-style pipeline
# ─────────────────────────────────────────────
def preprocess_mnist(image: Image.Image):
    """
    Mimics the exact preprocessing MNIST dataset used:
    1. Grayscale + Invert (digit=white, bg=black)
    2. Find bounding box → crop digit
    3. Resize to 20x20 (keeping aspect ratio)
    4. Pad to 28x28 centered
    5. Shift to center of mass (key MNIST trick)
    6. Reshape to match model input shape
    """
    # Step 1: Grayscale
    img = image.convert("L")

    # Step 2: Slight blur to remove noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(img, dtype=np.float32) / 255.0

    # Step 3: Invert if white background
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Step 4: Threshold — remove noise pixels
    arr = np.where(arr > 0.2, arr, 0.0)

    # Step 5: Find bounding box of the digit
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to digit only
        digit = arr[rmin:rmax+1, cmin:cmax+1]

        # Step 6: Resize to 20x20 (MNIST puts digit in 20x20 box)
        digit_img = Image.fromarray((digit * 255).astype(np.uint8))
        # Keep aspect ratio — fit within 20x20
        h, w = digit.shape
        if h > w:
            new_h, new_w = 20, max(1, int(20 * w / h))
        else:
            new_h, new_w = max(1, int(20 * h / w)), 20
        digit_img = digit_img.resize((new_w, new_h), Image.LANCZOS)

        # Step 7: Pad to 28x28 centered
        padded = np.zeros((28, 28), dtype=np.float32)
        top  = (28 - new_h) // 2
        left = (28 - new_w) // 2
        digit_arr = np.array(digit_img, dtype=np.float32) / 255.0
        padded[top:top+new_h, left:left+new_w] = digit_arr
        arr = padded
    else:
        arr = np.zeros((28, 28), dtype=np.float32)

    # Step 8: Center of mass shift — THE KEY MNIST TRICK
    cy, cx = ndimage.center_of_mass(arr)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = 14.0 - cy
        shift_x = 14.0 - cx
        arr = ndimage.shift(arr, [shift_y, shift_x], mode='constant', cval=0)

    # Step 9: Normalize to [0,1]
    if arr.max() > 0:
        arr = arr / arr.max()

    # Step 10: Reshape for model
    input_shape = mnist_model.input_shape
    if len(input_shape) == 2:
        return arr.flatten().reshape(1, 784)
    else:
        return arr.reshape(1, 28, 28, 1)


def preprocess_cifar(image: Image.Image):
    img = image.convert("RGB").resize((32, 32))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 32, 32, 3)

# ─────────────────────────────────────────────
# CONFIDENCE BARS — Native Streamlit
# ─────────────────────────────────────────────
def show_confidence_bars(probs, labels, top_n=5):
    st.markdown("#### 📊 Top Confidence Scores")
    top_indices = np.argsort(probs)[::-1][:top_n]
    for idx in top_indices:
        label = labels[idx]
        prob  = float(probs[idx])
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**{label}**")
            st.progress(min(prob, 1.0))
        with c2:
            st.metric(label="", value=f"{prob*100:.1f}%")

def show_result(pred_label, confidence):
    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:16px;color:#aaa;margin-bottom:8px;">PREDICTION</div>
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

with tab1:
    st.subheader("Upload a Handwritten Digit (0–9)")
    st.info("💡 **Best results:** Write ONE digit clearly on plain white paper with dark pen/marker. Take photo in good lighting. Digit should be large and fill most of the image.")

    uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], key="mnist")

    if uploaded:
        image = Image.open(uploaded)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(image, caption="Your Image", use_container_width=True)
            # Show preprocessed image so user can verify
            try:
                img_gray = image.convert("L").filter(ImageFilter.GaussianBlur(1))
                arr_preview = np.array(img_gray, dtype=np.float32) / 255.0
                if arr_preview.mean() > 0.5:
                    arr_preview = 1.0 - arr_preview
                preview_img = Image.fromarray((arr_preview * 255).astype(np.uint8))
                st.image(preview_img, caption="Preprocessed (what model sees)", use_container_width=True)
            except:
                pass
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
                    st.error(f"❌ Error: {e}")

with tab2:
    st.subheader("Upload an Image to Classify")
    st.info("💡 Categories: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck")

    uploaded2 = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], key="cifar")

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
                    st.error(f"❌ Error: {e}")

# FOOTER
st.markdown("""
<div class="footer-bar">
    🧠 <b>Architecture Summary</b> &nbsp;|&nbsp;
    MNIST: Input(784) → Dense(128,ReLU) → Dense(64,ReLU) → Dense(10,Softmax) &nbsp;·&nbsp;
    CIFAR-10: Conv2D×3 → BatchNorm → MaxPool → Dropout → Dense(10,Softmax) &nbsp;·&nbsp;
    Built with TensorFlow + Streamlit
</div>
""", unsafe_allow_html=True)