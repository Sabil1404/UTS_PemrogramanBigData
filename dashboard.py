import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd
import datetime

# NOTE:
# - Model imports (ultralytics, tensorflow) dilakukan hanya saat diperlukan (lazy load)
#   supaya Streamlit UI cepat muncul saat startup.
# - Pastikan file model ada di path "model/deteksi.pt" dan "model/klasifikasi.h5"
#   jika kamu memakai fitur deteksi/klasifikasi.

# ==========================
# Helpers: lazy-load models
# ==========================
@st.cache_resource
def get_yolo_model():
    """Load YOLO model on demand."""
    from ultralytics import YOLO
    return YOLO("model/deteksi.pt")


@st.cache_resource
def get_classifier():
    """Load classifier model on demand."""
    import tensorflow as tf
    return tf.keras.models.load_model("model/klasifikasi.h5")


# ==========================
# UI / Page config & CSS
# ==========================
st.set_page_config(page_title="✨ Intelligent Vision", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
        .title { text-align: center; font-size: 50px; font-family: 'Arial', sans-serif; color: navy; font-weight: bold; text-shadow: 3px 3px 5px rgba(0,0,0,0.3); }
        .welcome-message { font-size: 22px; font-family: 'Verdana', sans-serif; color: #666; text-align: center; margin-bottom: 40px; }
        .upload-box { border: 2px solid #FF5733; padding: 20px; border-radius: 10px; background-color: #f9f9f9; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); }
        .stImage img { border-radius: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">✨ Welcome to Intelligent Vision!</div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-message">Start exploring image classification and object detection with advanced AI models. Choose your action below and upload your image!</div>', unsafe_allow_html=True)

# Sidebar: choose mode
menu = st.sidebar.selectbox("Select Mode:", ["🔍 Object Detection (YOLO)", "📸 Image Classification"])

# ==========================
# Image upload UI
# ==========================
uploaded_file = st.file_uploader("📸 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Gagal membuka gambar: {e}")
        st.stop()

    # Show uploaded image inside a box
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------
    # Object Detection (YOLO)
    # --------------------------
    if menu == "🔍 Object Detection (YOLO)":
        st.subheader("🔍 Object Detection Results")

        try:
            # Load YOLO model lazily (cached)
            yolo_model = get_yolo_model()

            # Run detection - ultralytics accepts numpy array or PIL image
            results = yolo_model(np.array(img))

            # Plot annotated image (result[0].plot returns numpy array)
            result_img = results[0].plot(labels=True)
            st.image(result_img, caption="Detection Results", use_column_width=True)

            # Display detected objects info
            boxes = results[0].boxes
            if boxes and len(boxes.cls) > 0:
                for i in range(len(boxes.cls)):
                    class_id = int(boxes.cls[i])
                    class_name = results[0].names.get(class_id, str(class_id))
                    confidence = float(boxes.conf[i].item())
                    st.write(f"Detected Object: **{class_name.capitalize()}** (Confidence: {confidence*100:.2f}%)")
            else:
                st.info("No objects detected.")
        except FileNotFoundError as fnf:
            st.error(f"Model file not found: {fnf}")
        except Exception as e:
            st.error(f"Error while detecting objects with YOLO: {e}")

    # --------------------------
    # Image Classification
    # --------------------------
    elif menu == "📸 Image Classification":
        st.subheader("🔬 Image Classification Results")
        with st.spinner("Classifying image..."):
            try:
                # Load classifier lazily
                classifier = get_classifier()

                # Preprocessing (match training input: 128x128 asumsi)
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Prediction
                prediction = classifier.predict(img_array)
                class_index = int(np.argmax(prediction))

                # Define class labels (pastikan urutan sesuai modelmu)
                class_labels = [
                    "Tomato", "Radish", "Pumpkin", "Potato", "Papaya", "Cucumber",
                    "Cauliflower", "Carrot", "Capsicum", "Cabbage", "Broccoli",
                    "Brinjal", "Bottle_Gourd", "Bitter_Gourd", "Bean"
                ]

                # Safe guard: index range
                if 0 <= class_index < len(class_labels):
                    class_name = class_labels[class_index]
                else:
                    class_name = f"Class_{class_index}"

                prob = float(np.max(prediction))
                st.markdown(f"### Prediction Result: **{class_name}**")
                st.write(f"Prediction Probability: **{prob*100:.2f}%**")
            except FileNotFoundError as fnf:
                st.error(f"Model file not found: {fnf}")
            except Exception as e:
                st.error(f"Error while classifying image: {e}")

# If no file uploaded, show friendly hint
else:
    st.info("Upload sebuah gambar untuk memulai deteksi atau klasifikasi.")

# Footer
st.markdown(""" 
    ---
    If you need any help or want more information, visit the [Documentation](#). 
""")
