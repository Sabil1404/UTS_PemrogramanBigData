import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import logging
import

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    logger.debug("Mencoba memuat model YOLO...")
    try:
        yolo_model = YOLO("model/deteksi.pt")
        logger.debug("YOLO model berhasil dimuat")
    except Exception as e:
        logger.error(f"Error saat memuat model YOLO: {e}")
        raise

    logger.debug("Mencoba memuat model Keras...")
    try:
        classifier = tf.keras.models.load_model("model/klasifikasi.h5")
        logger.debug("Model Keras berhasil dimuat")
    except Exception as e:
        logger.error(f"Error saat memuat model Keras: {e}")
        raise

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
# Judul aplikasi dan deskripsi
st.title("🌟 Aplikasi Deteksi Objek & Klasifikasi Gambar")
st.markdown("""
    Aplikasi ini memungkinkan Anda untuk mengunggah gambar dan melakukan deteksi objek menggunakan YOLO 
    atau klasifikasi gambar dengan model yang sudah terlatih. Silakan pilih mode yang Anda inginkan dari menu samping.
""")

# Sidebar untuk memilih mode
mode = st.sidebar.selectbox("Pilih Mode Deteksi & Klasifikasi", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# Upload gambar
uploaded_file = st.file_uploader("📸 Unggah Gambar Anda", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diupload
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # Mengonversi gambar menjadi numpy array untuk YOLO
    img_array = np.array(img)

    # Deteksi Objek menggunakan YOLO
    if mode == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        # Deteksi objek
        results = yolo_model(img_array)  # Menggunakan numpy array untuk YOLO
        result_img = results[0].plot()  # Hasil deteksi objek (gambar dengan box)
        st.image(result_img, caption="Gambar dengan Deteksi", use_container_width=True)

    # Klasifikasi Gambar
    elif mode == "Klasifikasi Gambar":
        st.subheader("🔬 Hasil Klasifikasi Gambar")
        # Preprocessing gambar agar sesuai dengan model
        img_resized = img.resize((224, 224))  # Sesuaikan ukuran gambar dengan input model
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Membuat batch size 1
        img_array = img_array / 255.0  # Normalisasi gambar

        # Prediksi kelas gambar
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)  # Menentukan kelas dengan probabilitas tertinggi

        # Menampilkan hasil prediksi dan probabilitas
        st.write("### Kelas Prediksi:", class_index)
        st.write("Probabilitas Prediksi: {:.2f}%".format(np.max(prediction) * 100))

# Footer dengan informasi kontak atau dokumentasi
st.markdown("""
    ---
    Jika Anda memiliki pertanyaan atau butuh bantuan, kunjungi [Dokumentasi Aplikasi](#).
    """)

