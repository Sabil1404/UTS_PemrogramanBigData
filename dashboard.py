import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/deteksi.pt")  # Model deteksi
    classifier = tf.keras.models.load_model("model/klasifikasi.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🐾 Intelligent Vision", page_icon="🔍", layout="wide")

# ==========================
# CUSTOM STYLE
# ==========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #E0EAFC 0%, #CFDEF3 100%); /* Soft blue gradient */
}
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #1E3A8A; /* Deep blue */
    font-family: 'Poppins', sans-serif;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #374151; /* Dark gray */
    font-family: 'Open Sans', sans-serif;
    margin-bottom: 40px;
}
.upload-box {
    border: 2px solid #1E40AF;
    padding: 15px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}
.stImage img {
    border-radius: 12px;
    width: 60% !important; /* adjust image size smaller */
    margin: 0 auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown('<div class="title">🐾 Intelligent Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deteksi dan klasifikasikan gambar hewan dengan AI secara instan.</div>', unsafe_allow_html=True)

# ==========================
# SIDEBAR MENU
# ==========================
menu = st.sidebar.selectbox("Pilih Mode:", ["🔍 Deteksi Objek (YOLO)", "📸 Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📸 Unggah gambar kamu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Gambar yang diunggah", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if menu == "🔍 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        try:
            results = yolo_model(img)
            result_img = results[0].plot(labels=True)
            st.image(result_img, caption="Hasil Deteksi", use_container_width=False)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat deteksi: {e}")

    elif menu == "📸 Klasifikasi Gambar":
        st.subheader("🔬 Hasil Klasifikasi")
        try:
            img_resized = img.resize((128, 128))
            im
