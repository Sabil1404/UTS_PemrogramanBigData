import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import logging
import time

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
        yolo_model = YOLO("model/deteksi.pt")  # Pastikan path model YOLO benar
        logger.debug("YOLO model berhasil dimuat")
    except Exception as e:
        logger.error(f"Error saat memuat model YOLO: {e}")
        raise

    logger.debug("Mencoba memuat model Keras...")
    try:
        classifier = tf.keras.models.load_model("model/klasifikasi.h5")  # Pastikan path model Keras benar
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
        try:
            img_tensor = img_array / 255.0  # Normalisasi

        # YOLO melakukan deteksi objek
        results = yolo_model(img_tensor)  # Menggunakan tensor untuk YOLO
        
        # Menampilkan gambar dengan bounding box dan label
        result_img = results[0].plot(labels=True)  # Menambahkan label pada bounding box
        st.image(result_img, caption="Gambar dengan Deteksi", use_container_width=True)

        # Menampilkan informasi objek yang terdeteksi
        if results[0].boxes.xywh.shape[0] > 0:  # Jika ada objek yang terdeteksi
            for i in range(len(results[0].boxes.cls)):
                class_id = int(results[0].boxes.cls[i])  # Mendapatkan ID kelas
                class_name = results.names[class_id]  # Mendapatkan nama kelas
                confidence = results[0].boxes.conf[i].item()  # Mendapatkan confidence
                st.write(f"Objek Terdeteksi: {class_name} (Confidence: {confidence*100:.2f}%)")
        else:
            st.write("Tidak ada objek yang terdeteksi.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat mendeteksi objek dengan YOLO: {e}")


   # Klasifikasi Gambar
# Klasifikasi Gambar
    elif mode == "Klasifikasi Gambar":
        st.subheader("🔬 Hasil Klasifikasi Gambar")
        try:
            # Preprocessing gambar agar sesuai dengan model klasifikasi
            img_resized = img.resize((128, 128))  # Sesuaikan ukuran gambar dengan input model
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)  # Membuat batch size 1
            img_array = img_array / 255.0  # Normalisasi gambar

            # Prediksi kelas gambar
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)  # Menentukan kelas dengan probabilitas tertinggi

            # Menampilkan hasil prediksi dan probabilitas
            class_labels = ['Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'Cauliflower', 'Carrot', 'Capsicum', 'Cabbage', 'Broccoli', 
                            'Brinjal', 'Bottle_Gourd', 'Bitter_Gourd', 'Bean']
            class_name = class_labels[class_index]

            st.write("### Kelas Prediksi:", class_name)
            st.write("Probabilitas Prediksi: {:.2f}%".format(np.max(prediction) * 100))
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mengklasifikasi gambar: {e}")

# Footer dengan informasi kontak atau dokumentasi
st.markdown("""
    ---
    Jika Anda memiliki pertanyaan atau butuh bantuan, kunjungi [Dokumentasi Aplikasi](#).
    """)














