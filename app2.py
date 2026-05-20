import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf

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
# PURE NUMPY — Center of mass & shift (no scipy!)
# ─────────────────────────────────────────────
def get_center_of_mass(arr):
    total = arr.sum()
    if total == 0:
        return 14.0, 14.0
    rows = np.arange(arr.shape[0])
    cols = np.arange(arr.shape[1])
    cy = float(rows @ arr.sum(axis=1)) / total
    cx = float(cols @ arr.sum(axis=0)) / total
    return cy, cx

def shift_image(arr, shift_y, shift_x):
    """Shift a 28x28 array by (shift_y, shift_x) pixels using slicing."""
    result = np.zeros((28, 28), dtype=np.float32)
    sy = int(round(shift_y))
    sx = int(round(shift_x))

    src_r0 = max(0, -sy);  src_r1 = min(28, 28 - sy)
    dst_r0 = max(0,  sy);  dst_r1 = min(28, 28 + sy)
    src_c0 = max(0, -sx);  src_c1 = min(28, 28 - sx)
    dst_c0 = max(0,  sx);  dst_c1 = min(28, 28 + sx)

    if (src_r1 > src_r0) and (src_c1 > src_c0):
        result[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return result

# ─────────────────────────────────────────────
# MNIST PREPROCESSING — MNIST-style pipeline
# ─────────────────────────────────────────────
def preprocess_mnist(image: Image.Image):
    # 1. Grayscale + slight blur
    img = image.convert("L").filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(img, dtype=np.float32) / 255.0

    # 2. Invert if white background (normal paper)
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # 3. Threshold — remove weak noise pixels
    arr = np.where(arr > 0.2, arr, 0.0)

    # 4. Bounding box crop
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        digit = arr[rmin:rmax+1, cmin:cmax+1]

        # 5. Resize to fit 20×20 (keeping aspect ratio)
        h, w = digit.shape
        if h > w:
            new_h, new_w = 20, max(1, int(20 * w / h))
        else:
            new_h, new_w = max(1, int(20 * h / w)), 20

        digit_img = Image.fromarray((digit * 255).astype(np.uint8))
        digit_img = digit_img.resize((new_w, new_h), Image.LANCZOS)

        # 6. Place in center of 28×28 canvas
        padded = np.zeros((28, 28), dtype=np.float32)
        top  = (28 - new_h) // 2
        left = (28 - new_w) // 2
        d = np.array(digit_img, dtype=np.float32) / 255.0
        padded[top:top+new_h, left:left+new_w] = d
        arr = padded
    else:
        arr = np.zeros((28, 28), dtype=np.float32)

    # 7. Center of mass shift (pure numpy)
    cy, cx = get_center_of_mass(arr)
    arr = shift_image(arr, 14.0 - cy, 14.0 - cx)

    # 8. Normalize
    if arr.max() > 0:
        arr = arr / arr.max()

    # 9. Reshape for model input shape
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
# UI HELPERS
# ─────────────────────────────────────────────
def show_confidence_bars(probs, labels, top_n=5):
    st.markdown("#### 📊 Top Confidence Scores")
    for idx in np.argsort(probs)[::-1][:top_n]:
        prob = float(probs[idx])
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**{labels[idx]}**")
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
    st.info("💡 Write ONE big digit on plain white paper with dark pen. Fill most of the image. Good lighting!")

    uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], key="mnist")
    if uploaded:
        image = Image.open(uploaded)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(image, caption="Your Image", use_container_width=True)
            # Show preprocessed preview
            try:
                arr_prev = preprocess_mnist(image)
                if len(arr_prev.shape) == 2 or arr_prev.shape[-1] == 1:
                    prev_28 = arr_prev.reshape(28, 28)
                else:
                    prev_28 = arr_prev.reshape(28, 28)
                prev_img = Image.fromarray((prev_28 * 255).astype(np.uint8))
                st.image(prev_img, caption="What model sees (28×28)", use_container_width=True)
            except:
                pass
        with c2:
            with st.spinner("Analyzing..."):
                try:
                    arr   = preprocess_mnist(image)
                    preds = mnist_model.predict(arr, verbose=0)[0]
                    show_result(MNIST_LABELS[int(np.argmax(preds))], float(np.max(preds)))
                    show_confidence_bars(preds, MNIST_LABELS)
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
                    show_result(CIFAR_LABELS[int(np.argmax(preds2))], float(np.max(preds2)))
                    show_confidence_bars(preds2, CIFAR_LABELS)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

st.markdown("""
<div class="footer-bar">
    🧠 <b>Architecture Summary</b> &nbsp;|&nbsp;
    MNIST: Input(784) → Dense(128,ReLU) → Dense(64,ReLU) → Dense(10,Softmax) &nbsp;·&nbsp;
    CIFAR-10: Conv2D×3 → BatchNorm → MaxPool → Dropout → Dense(10,Softmax) &nbsp;·&nbsp;
    Built with TensorFlow + Streamlit
</div>
""", unsafe_allow_html=True)
