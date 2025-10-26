import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as kimage
import pandas as pd
import datetime
import plotly.express as px
import os

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
# Lazy-load models
# ------------------------
@st.cache_resource
def get_yolo_model():
    """Load YOLO model on demand. Raises if model file missing."""
    from ultralytics import YOLO
    return YOLO("model/deteksi.pt")

@st.cache_resource
def get_classifier():
    """Load classifier model on demand. Raises if model file missing."""
    import tensorflow as tf
    return tf.keras.models.load_model("model/klasifikasi.h5")

# ------------------------
# App config & CSS
# ------------------------
st.set_page_config(page_title="✨ Intelligent Vision", page_icon="🔍", layout="wide")
st.markdown("""
    <style>
        .title { text-align:center; font-size:48px; color:navy; font-weight:700; margin-bottom:6px; }
        .subtitle { text-align:center; color:#666; margin-bottom:24px; }
        .upload-box { border:2px solid #FF5733; padding:16px; border-radius:10px; background:#fbfbfb; }
        .small-muted { color:#777; font-size:13px; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">✨ Intelligent Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload gambar satu-per-satu — simpan metadata & lihat Treemap kelas</div>', unsafe_allow_html=True)

# ------------------------
# Sidebar: mode & controls
# ------------------------
mode = st.sidebar.selectbox("Mode:", ["🔍 Object Detection (YOLO)", "📸 Image Classification", "⚙️ Demo (No model)"])
persist_csv = st.sidebar.checkbox("Simpan ke CSV setiap upload", value=False)
csv_path = st.sidebar.text_input("Path CSV (jika centang Simpan)", value="data/records.csv")

# Tombol hapus data session
if st.sidebar.button("🗑️ Hapus Semua Record"):
    st.session_state["records"] = []
    st.sidebar.success("Semua record dihapus dari session.")

# ------------------------
# Session storage init
# ------------------------
if "records" not in st.session_state:
    st.session_state["records"] = []  # list of dicts

# ------------------------
# Upload UI
# ------------------------
uploaded_file = st.file_uploader("Upload gambar (satu-satu)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Gagal membuka gambar: {e}")
        st.stop()

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------
    # Run prediction depending on mode
    # ------------------------
    use_dummy = (mode == "⚙️ Demo (No model)")
    pred_class = None
    conf = None
    bbox = (None, None, None, None)

    if not use_dummy:
        if mode.startswith("🔍"):
            # YOLO detection
            try:
                yolo = get_yolo_model()
                results = yolo(np.array(img))
                boxes = results[0].boxes
                # if there are detections
                if boxes and len(boxes.cls) > 0:
                    # take detection with highest confidence
                    confs = [float(c) for c in boxes.conf]
                    best_idx = int(np.argmax(confs))
                    class_id = int(boxes.cls[best_idx])
                    pred_class = results[0].names.get(class_id, str(class_id))
                    conf = float(boxes.conf[best_idx].item())
                    xyxy = boxes.xyxy[best_idx].tolist()  # [xmin,ymin,xmax,ymax]
                    bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                else:
                    pred_class = "NoObject"
