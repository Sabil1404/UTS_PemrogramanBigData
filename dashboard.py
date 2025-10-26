import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os
import datetime

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
    background: linear-gradient(135deg, #E0EAFC 0%, #CFDEF3 100%);
}
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #1E3A8A;
    font-family: 'Poppins', sans-serif;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #374151;
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
    width: 60% !important;
    margin: 0 auto;
    display: block;
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

if "records" not in st.session_state:
    st.session_state["records"] = []  # simpan hasil upload

# ==========================
# MAIN CONTENT
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Gambar yang diunggah", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    pred_class = "Tidak diketahui"
    confidence = 0.0

    # MODE DETEKSI
    if menu == "🔍 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        try:
            results = yolo_model(img)
            result_img = results[0].plot(labels=True)
            st.image(result_img, caption="Hasil Deteksi", use_container_width=False)

            if len(results[0].boxes.cls) > 0:
                best_idx = int(np.argmax(results[0].boxes.conf))
                class_id = int(results[0].boxes.cls[best_idx])
                pred_class = results[0].names[class_id]
                confidence = results[0].boxes.conf[best_idx].item()
                st.success(f"Objek terdeteksi: **{pred_class}** ({confidence*100:.2f}%)")
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
            pred_class = class_labels[class_index]
            confidence = np.max(prediction)

            st.success(f"Hasil Prediksi: **{pred_class}**")
            st.write(f"Akurasi: {confidence*100:.2f}%")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

    # Simpan hasil ke session
    record = {
        "nama_file": uploaded_file.name,
        "tanggal_upload": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": menu.replace("📸 ", "").replace("🔍 ", ""),
        "hasil_prediksi": pred_class,
        "confidence": round(float(confidence), 4)
    }
    st.session_state["records"].append(record)
    st.success("✅ Hasil prediksi disimpan!")

# ==========================
# VISUALISASI
# ==========================
if len(st.session_state["records"]) > 0:
    st.markdown("---")
    st.subheader("📊 Visualisasi Hasil Prediksi")

    df = pd.DataFrame(st.session_state["records"])
    st.dataframe(df)

    # Hitung jumlah tiap kelas
    agg = df.groupby("hasil_prediksi").agg(
        jumlah=("nama_file", "count"),
        rata_confidence=("confidence", "mean")
    ).reset_index()

    col1, col2 = st.columns(2)

    # Visualisasi 1: Bar chart jumlah kelas
    with col1:
        fig_bar = px.bar(
            agg,
            x="hasil_prediksi",
            y="jumlah",
            color="hasil_prediksi",
            title="Jumlah Gambar per Kelas",
            text_auto=True
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Visualisasi 2: Treemap confidence rata-rata
    with col2:
        fig_tree = px.treemap(
            agg,
            path=["hasil_prediksi"],
            values="jumlah",
            color="rata_confidence",
            color_continuous_scale="RdYlGn",
            title="Treemap Confidence Rata-Rata"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

else:
    st.info("Belum ada gambar yang diunggah — silakan upload beberapa untuk melihat visualisasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
---
🌐 **Intelligent Vision App** — Powered by YOLOv8 & TensorFlow  
""")
