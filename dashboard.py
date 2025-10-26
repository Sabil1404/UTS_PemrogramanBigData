import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px

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
st.set_page_config(page_title="✨ Intelligent Vision", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .title { text-align: center; font-size: 42px; color: navy; font-weight: bold; }
        .subtitle { text-align: center; color: #666; margin-bottom: 20px; }
        .upload-box { border: 2px solid #FF5733; padding: 12px; border-radius: 10px; background-color: #f9f9f9; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">✨ Intelligent Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Image tools + satu visualisasi interaktif (Scatter)</div>', unsafe_allow_html=True)

# Sidebar
menu = st.sidebar.selectbox("Select Mode:", ["🔍 Object Detection (YOLO)", "📸 Image Classification", "📊 Visualisasi: Scatter"])

# ==========================
# MODE 1 & 2: IMAGE PANEL
# ==========================
if menu in ["🔍 Object Detection (YOLO)", "📸 Image Classification"]:
    uploaded_file = st.file_uploader("📸 Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if menu == "🔍 Object Detection (YOLO)":
            st.subheader("🔍 Object Detection Results")
            try:
                results = yolo_model(img)
                result_img = results[0].plot(labels=True)
                st.image(result_img, caption="Detection Results", use_container_width=True)

                if len(results[0].boxes.cls) > 0:
                    for i in range(len(results[0].boxes.cls)):
                        class_id = int(results[0].boxes.cls[i])
                        class_name = results[0].names[class_id]
                        confidence = results[0].boxes.conf[i].item()
                        st.write(f"Detected Object: {class_name.capitalize()} (Confidence: {confidence*100:.2f}%)")
                else:
                    st.write("No objects detected.")
            except Exception as e:
                st.error(f"Error while detecting objects with YOLO: {e}")

        elif menu == "📸 Image Classification":
            st.subheader("🔬 Image Classification Results")
            try:
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)

                class_labels = [
                    'Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'Cauliflower',
                    'Carrot', 'Capsicum', 'Cabbage', 'Broccoli', 'Brinjal', 'Bottle_Gourd', 'Bitter_Gourd', 'Bean'
                ]

                class_name = class_labels[class_index]
                st.write("### Prediction Result:", class_name)
                st.write("Prediction Probability: {:.2f}%".format(np.max(prediction) * 100))
            except Exception as e:
                st.error(f"Error while classifying image: {e}")

# ==========================
# MODE 3: VISUALISASI SCATTER
# ==========================
elif menu == "📊 Visualisasi: Scatter":
    st.header("📊 Scatter Plot (satu saja)")
    st.write("Upload CSV (opsional). Kalau tidak, contoh dataset Iris akan dipakai.")

    uploaded_csv = st.file_uploader("Upload CSV untuk divisualisasi (opsional)", type=["csv"], key="csv_scatter")

    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            df = pd.DataFrame()
    else:
        df = px.data.iris()

    if df.empty:
        st.warning("Data kosong — upload CSV yang valid.")
    else:
        st.write("Preview data:", df.head())

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Butuh minimal 2 kolom numerik untuk scatter plot.")
        else:
            x = st.selectbox("Pilih sumbu X", numeric_cols, index=0)
            y = st.selectbox("Pilih sumbu Y", numeric_cols, index=1)
            color = st.selectbox("Warna (opsional)", [None] + df.columns.tolist())
            size_col = st.selectbox("Ukuran titik (opsional)", [None] + numeric_cols)

            df_plot = df.sample(5000, random_state=42) if len(df) > 5000 else df

            try:
                fig = px.scatter(df_plot, x=x, y=y,
                                 color=(color if color else None),
                                 size=(size_col if size_col else None),
                                 hover_data=df_plot.columns)
                fig.update_layout(title=f"Scatter: {x} vs {y}", legend_title_text=(color if color else ""))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Gagal membuat scatter plot: {e}")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("Butuh opsi lain nanti? Bilang aja — tapi ini satu visual dulu sesuai permintaan.")
