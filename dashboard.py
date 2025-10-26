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
    font-family: 'Poppins', sans-serif;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #374151; /* Dark gray */
    font-family: 'Open Sans', sans-serif;
    margin-bottom: 40px;
}
.upload-box {
    border: 2px solid #1E40AF;
    padding: 15px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}
.stImage img {
    border-radius: 12px;
    width: 60% !important; /* Ukuran gambar lebih kecil */
    margin: 0 auto;
    display: block;
}
.sidebar .sidebar-content {
    background-color: #f8fafc;
    border-radius: 15px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown('<div class="title">🐾 Intelligent Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deteksi dan klasifikasikan gambar hewan secara cerdas dengan AI.</div>', unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
menu = st.sidebar.selectbox("Pilih Mode:", ["🔍 Deteksi Objek (YOLO)", "📸 Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📸 Unggah gambar kamu", type=["jpg", "jpeg", "png"])

# ==========================
# MAIN CONTENT
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Gambar yang diunggah", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # MODE DETEKSI
    if menu == "🔍 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        try:
            results = yolo_model(img)
            result_img = results[0].plot(labels=True)
            st.image(result_img, caption="Hasil Deteksi", use_container_width=False)

            if len(results[0].boxes.cls) > 0:
                for i in range(len(results[0].boxes.cls)):
                    class_id = int(results[0].boxes.cls[i])
                    class_name = results[0].names[class_id]
                    confidence = results[0].boxes.conf[i].item()
                    st.write(f"**Objek Terdeteksi:** {class_name.capitalize()} (Kepercayaan: {confidence*100:.2f}%)")
            else:
                st.info("Tidak ada objek yang terdeteksi.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat deteksi: {e}")

    # MODE KLASIFIKASI
    elif menu == "📸 Klasifikasi Gambar":
        st.subheader("🔬 Hasil Klasifikasi")
        try:
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            class_labels = ['Anjing', 'Ayam', 'Kupu-Kupu']
            class_name = class_labels[class_index]

            st.success(f"Hasil Prediksi: **{class_name}**")
            st.write(f"Akurasi: {np.max(prediction)*100:.2f}%")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

else:
    st.info("📥 Silakan unggah gambar terlebih dahulu untuk memulai.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
---
🌐 **Intelligent Vision App** — Powered by YOLOv8 & TensorFlow  
""")
