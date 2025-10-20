import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -------------------------------------------------
# DEBUG READY: kode ini menampilkan info di logs
# -------------------------------------------------
st.set_page_config(page_title="Debug App", layout="wide")

# Tampilkan versi file yg sedang dijalankan (ini akan muncul di logs)
print("=== DASHBOARD.PY STARTED ===")
print("If you see this line in Streamlit logs, this is the current file being executed.")

# =================================
# Try load models but with try/except
# =================================
def load_models_safe():
    try:
        print("Trying to load YOLO model from: model/Azzahra Salsabil Lubis_Laporan 4.pt")
        yolo_model = YOLO("model/Azzahra Salsabil Lubis_Laporan 4.pt")
        print("YOLO loaded OK")
    except Exception as e:
        yolo_model = None
        print("YOLO load ERROR:", repr(e))
        st.error("YOLO load ERROR (lihat logs).")

    try:
        print("Trying to load Keras model from: model/Azzahra_Salsabil_Lubis_Laporan2.h5")
        classifier = tf.keras.models.load_model("model/Azzahra_Salsabil_Lubis_Laporan2.h5")
        print("Keras model loaded OK")
    except Exception as e:
        classifier = None
        print("Keras load ERROR:", repr(e))
        st.error("Keras load ERROR (lihat logs).")

    return yolo_model, classifier

yolo_model, classifier = load_models_safe()

# =================================
# Simple UI to test if models exist
# =================================
st.title("DEBUG: Image App")

st.write("YOLO model loaded:", bool(yolo_model))
st.write("Classifier loaded:", bool(classifier))

st.write("---- Directory listing (may help) ----")
# print directory listing to logs (not requiring os in UI)
try:
    import os
    print("CWD:", os.getcwd())
    print("Files here:", os.listdir())
    if "model" in os.listdir():
        print("MODEL folder contents:", os.listdir("model"))
    else:
        print("No model folder in current dir.")
except Exception as e:
    print("Could not list directories:", repr(e))

st.write("Lihat *Logs* di Streamlit Cloud untuk output debug (Manage app → Logs).")
