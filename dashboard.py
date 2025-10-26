import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/deteksi.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/klasifikasi.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🐾 Intelligent Vision", page_icon="🔍", layout="wide")

# ==========================
# CUSTOM CSS STYLE
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
