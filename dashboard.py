import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Konfigurasi dan Cek Model
# ==========================

# --- Ganti nama file model kamu tanpa spasi & tanda kutip ---
YOLO_FILE = ("model/Azzahra Salsabil Lubis_Laporan 4.pt")       # Contoh: rename file jadi "azzahra_laporan4.pt"
H5_FILE   = ("model/Azzahra_Salsabil_Lubis_Laporan2.h5")       # Contoh: rename file jadi "azzahra_laporan2.h5"

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Cek apakah file model ada
    missing_files = []
    if not os.path.exists(YOLO_FILE):
        missing_files.append(YOLO_FILE)
    if not os.path.exists(H5_FILE):
        missing_files.append(H5_FILE)

    if missing_files:
        st.error("⚠️ File model tidak ditemukan:\n" + "\n".join(missing_files))
        st.info("Pastikan file `.pt` dan `.h5` sudah ada di folder `model/` dan namanya sesuai.")
        return None, None

    # Load model YOLO
    try:
        yolo_model = YOLO(YOLO_FILE)
        st.success("✅ YOLO model berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        yolo_model = None

    # Load model klasifikasi (.h5)
    try:
        classifier = tf.keras.models.load_model(H5_FILE)
        st.success("✅ Model klasifikasi berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi (.h5): {e}")
        classifier = None

    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==========================
# UI Streamlit
# ==========================
st.title("🧠 Image Classification & Object Detection App")
st.caption("Dibuat oleh Az'zahra Salsabil Lubis")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Gambar yang diunggah", use_container_width=True)

    # ==========================
    # MODE DETEKSI OBJEK
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is None:
            st.warning("Model deteksi belum dimuat. Periksa file `.pt` di folder model.")
        else:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()  # hasil deteksi
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    # ==========================
    # MODE KLASIFIKASI GAMBAR
    # ==========================
    elif menu == "Klasifikasi Gambar":
        if classifier is None:
            st.warning("Model klasifikasi belum dimuat. Periksa file `.h5` di folder model.")
        else:
            with st.spinner("🧠 Sedang memproses gambar..."):
                # Preprocessing
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Prediksi
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Tampilkan hasil
                st.write("### 🏷️ Kelas Prediksi:", class_index)
                st.write(f"**Probabilitas:** {confidence:.2f}")
