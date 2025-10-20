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
    # Model deteksi objek (YOLO)
    yolo_model = YOLO("model/Azzahra Salsabil Lubis_Laporan 4.pt")

    # Model klasifikasi (TensorFlow)
    classifier = tf.keras.models.load_model("model/Azzahra_Salsabil_Lubis_Laporan2.h5")

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")  # pastikan gambar RGB
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        st.write("🔍 Sedang mendeteksi objek...")
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi Objek", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        st.write("🧩 Sedang mengklasifikasi gambar...")

        # Preprocessing gambar sesuai model
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        st.write("### Hasil Prediksi:")
        st.success(f"Label: {class_index}")
        st.info(f"Akurasi: {confidence:.2%}")
