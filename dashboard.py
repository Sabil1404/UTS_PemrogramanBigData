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
# Set page configuration
st.set_page_config(page_title="✨ Intelligent Vision", page_icon="🔍", layout="wide")

# Custom CSS for styling the app
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 42px;
            font-family: 'Arial', sans-serif;
            color: #FF5733;  /* Orange color for excitement */
            font-weight: bold;
            text-shadow: 2px 2px 15px rgba(0, 0, 0, 0.5);
        }
        .welcome-message {
            font-size: 22px;
            font-family: 'Verdana', sans-serif;
            color: #666;
            text-align: center;
            margin-bottom: 40px;
        }
        .description {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-box {
            border: 2px solid #FF5733;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-s
