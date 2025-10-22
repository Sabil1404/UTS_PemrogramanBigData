import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import os
import logging

# -------------------------------------------------
# DEBUG READY: kode ini menampilkan info di logs
# -------------------------------------------------
st.set_page_config(page_title="Debug App", layout="wide")

# Tampilkan versi file yg sedang dijalankan (ini akan muncul di logs)
st.write("=== DASHBOARD.PY STARTED ===")
st.write("Jika Anda melihat baris ini di logs Streamlit, ini adalah file yang sedang dijalankan.")

# Setup logging untuk debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# =================================
# Fungsi untuk mencoba load model dengan penanganan error
# =================================
def load_models_safe():
    yolo_model = None
    classifier = None

    # Memuat YOLO model
    try:
        model_path_yolo = "model/Azzahra Salsabil Lubis_Laporan 4.pt"
        st.write(f"Mencoba memuat model YOLO dari: {model_path_yolo}")
        yolo_model = YOLO(model_path_yolo)
        logger.info(f"YOLO model berhasil dimuat dari {model_path_yolo}")
        st.write("YOLO model berhasil dimuat")
    except Exception as e:
        logger.error(f"Error saat memuat YOLO: {str(e)}", exc_info=True)
        st.error(f"YOLO load ERROR: {str(e)}")

    # Memuat Keras model
    try:
        model_path_keras = "model/Azzahra_Salsabil_Lubis_Laporan 2.h5"
        st.write(f"Mencoba memuat model Keras dari: {model_path_keras}")
        classifier = tf.keras.models.load_model(model_path_keras)
        logger.info(f"Model Keras berhasil dimuat dari {model_path_keras}")
        st.write("Keras model berhasil dimuat")
    except Exception as e:
        logger.error(f"Error saat memuat model Keras: {str(e)}", exc_info=True)
        st.error(f"Keras load ERROR: {str(e)}")

    return yolo_model, classifier

# =================================
# Cek apakah file model ada di direktori
# =================================
def check_model_files():
    try:
        st.write("Menampilkan direktori saat ini:")
        st.write(os.listdir())  # Menampilkan file di direktori kerja saat ini
        if "model" in os.listdir():
            st.write("Isi folder model:")
            st.write(os.listdir("model"))
        else:
            st.write("Tidak ada folder 'model' di direktori saat ini.")
    except Exception as e:
        logger.error(f"Gagal mengakses direktori: {str(e)}", exc_info=True)
        st.error(f"Gagal mengakses direktori: {str(e)}")

# =================================
# Memuat model
# =================================
yolo_model, classifier = load_models_safe()

# =================================
# Tampilan UI untuk menampilkan hasil dan status model
# =================================
st.title("DEBUG: Image App")

# Tampilkan status apakah model berhasil dimuat
st.write("Model YOLO berhasil dimuat:", bool(yolo_model))
st.write("Model Keras berhasil dimuat:", bool(classifier))

st.write("---- Daftar isi direktori (mungkin membantu) ----")
check_model_files()

st.write("Lihat *Logs* di Streamlit Cloud untuk output debug (Manage app → Logs).")
