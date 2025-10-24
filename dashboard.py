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
    yolo_model = YOLO("model/deteksi.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/klasifikasi.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.set_page_config(page_title="Aplikasi Deteksi & Klasifikasi Gambar", page_icon=":camera:", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
        }
        .description {
            font-size: 18px;
            color: #555;
            text-align: center;
        }
        .upload-box {
            border: 2px solid #4CAF50;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .sidebar .sidebar-content {
            background-color: #f1f1f1;
            border-radius: 10px;
            padding: 10px;
        }
        .stImage img {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header with title and description
st.markdown('<div class="title">🧠 Image Classification & Object Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Unggah gambar dan pilih mode untuk mendeteksi objek atau mengklasifikasikan gambar.</div>', unsafe_allow_html=True)

# Sidebar for choosing mode
menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# Upload gambar
uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

# Display image in a neat box with border
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.markdown('<div class="upload-box">'
