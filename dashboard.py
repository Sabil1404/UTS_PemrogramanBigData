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
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        try:
            # Deteksi objek menggunakan YOLO
            results = yolo_model(img)  # Menggunakan gambar yang diupload
            result_img = results[0].plot(labels=True)  # Menambahkan label pada bounding box
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
            
            # Menampilkan informasi objek yang terdeteksi
            if len(results[0].boxes.cls) > 0:  # Jika ada objek yang terdeteksi
                for i in range(len(results[0].boxes.cls)):
                    class_id = int(results[0].boxes.cls[i])  # Mendapatkan ID kelas
                    class_name = results[0].names[class_id]  # Mendapatkan nama kelas dengan benar
                    confidence = results[0].boxes.conf[i].item()  # Mendapatkan confidence
                    st.write(f"Objek Terdeteksi: {class_name.capitalize()} (Confidence: {confidence*100:.2f}%)")
            else:
                st.write("Tidak ada objek yang terdeteksi.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mendeteksi objek dengan YOLO: {e}")

    elif menu == "Klasifikasi Gambar":
        st.subheader("🔬 Hasil Klasifikasi Gambar")
        with st.spinner("Sedang mengklasifikasikan gambar..."):
            try:
                # Preprocessing
                img_resized = img.resize((128, 128))  # Sesuaikan ukuran dengan model
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)  # Membuat batch size 1
                img_array = img_array / 255.0

                # Prediksi
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)

                # Kelas yang diinginkan
                class_labels = ['Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'Cauliflower', 
                                'Carrot', 'Capsicum', 'Cabbage', 'Broccoli', 'Brinjal', 'Bottle_Gourd', 'Bitter_Gourd', 'Bean']
                
                class_name = class_labels[class_index]  # Mendapatkan nama kelas dari index
                st.write("### Hasil Prediksi:", class_name)
                st.write("Probabilitas Prediksi: {:.2f}%".format(np.max(prediction) * 100))
            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengklasifikasikan gambar: {e}")

# Footer dengan informasi kontak atau dokumentasi
st.markdown(""" 
    --- 
    Jika Anda memiliki pertanyaan atau butuh bantuan, kunjungi [Dokumentasi Aplikasi](#). 
""")
