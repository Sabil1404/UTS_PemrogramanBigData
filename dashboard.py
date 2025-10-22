import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_models():
    logger.debug("Mencoba memuat model YOLO...")
    try:
        yolo_model = YOLO("model/Azzahra_Salsabil_Lubis_Laporan4.pt")
        logger.debug("YOLO model berhasil dimuat")
    except Exception as e:
        logger.error(f"Error saat memuat model YOLO: {e}")
        raise

    logger.debug("Mencoba memuat model Keras...")
    try:
        classifier = tf.keras.models.load_model("model/Azzahra_Salsabil_Lubis_Laporan2.h5")
        logger.debug("Model Keras berhasil dimuat")
    except Exception as e:
        logger.error(f"Error saat memuat model Keras: {e}")
        raise

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)

        st.write("Probabilitas:", np.max(prediction))


