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
