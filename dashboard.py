import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as kimage
import pandas as pd
import datetime
import plotly.express as px
import os

# ------------------------
# Helper: show image with max width
# ------------------------
def show_image_limited(img_obj, caption=None, max_width=400):
    try:
        if isinstance(img_obj, np.ndarray):
            h, w = img_obj.shape[:2]
            display_w = min(w, max_width)
            st.image(img_obj, caption=caption, width=display_w)
        elif isinstance(img_obj, Image.Image):
            w = img_obj.width
            display_w = min(w, max_width)
            st.image(img_obj, caption=caption, width=display_w)
        else:
            st.image(img_obj, caption=caption)
    except Exception:
        st.image(img_obj, caption=caption)

# ------------------------
# Dummy predict (fallback)
# ------------------------
def dummy_predict(img: Image.Image):
    classes = ["Anjing", "Ayam", "Kupu-Kupu"]
    idx = np.random.randint(0, len(classes))
    conf = float(np.round(np.random.uniform(0.5, 0.98), 3))
    bbox = (10, 10, img.width - 10, img.height - 10)
    return classes[idx], conf, bbox

# ------------------------
# Lazy-load models (optional)
# ------------------------
@st.cache_resource
def get_yolo_model():
    from ultralytics import YOLO
    return YOLO("model/deteksi.pt")

@st.cache_resource
def get_classifier():
    import tensorflow as tf
    return tf.keras.models.load_model("model/klasifikasi.h5")

# Try load models but don't crash app if unavailable
try:
    yolo_model = get_yolo_model()
except Exception:
    yolo_model = None

try:
    classifier = get_classifier()
except Exception:
    classifier = None

# ------------------------
# App config & CSS
# ------------------------
st.set_page_config(page_title="SeeBil", page_icon="👁️", layout="wide")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #d8c7ff 0%, #ffe6f7 100%);
}
.title {
    text-align: center;
    font-size: 48px;
    font-family: 'Poppins', sans-serif;
    color: #4B0082;
    font-weight: 700;
    margin-bottom: 6px;
}
.subtitle {
    font-size: 17px;
    color: #333;
    text-align: center;
    margin-bottom: 20px;
    font-family: 'Poppins', sans-serif;
}
.upload-box {
    border: 2px solid #836FFF;
    padding: 14px;
    border-radius: 10px;
    background-color: #fbf8ff;
    text-align: center;
}
.sidebar .sidebar-content {
    background-color: #f8f8ff;
    border-radius: 12px;
    padding: 12px;
}
.footer-note {
    text-align: center;
    color: #666;
    font-size: 13px;
