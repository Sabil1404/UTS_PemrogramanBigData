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
# Set page configuration
st.set_page_config(page_title="✨ Intelligent Vision", page_icon="🔍", layout="wide")

# Custom CSS for styling the app
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 50px;
            font-family: 'Arial', sans-serif;
            color: navy;  /* Navy color */
            font-weight: bold;
            text-shadow: 3px 3px 5px rgba(0, 0, 0, 0.3);  /* Soft shadow effect */
        }
        .welcome-message {
            font-size: 22px;
            font-family: 'Verdana', sans-serif;
            color: #666;
            text-align: center;
            margin-bottom: 40px;
        }
        .description {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-box {
            border: 2px solid #FF5733;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content {
            background-color: #f1f1f1;
            border-radius: 15px;
            padding: 20px;
        }
        .stImage img {
            border-radius: 15px;
        }
        .sidebar .sidebar-title {
            font-size: 18px;
            font-weight: bold;
            color: #FF5733;
        }
        .sidebar .stSelectbox, .sidebar .stFileUploader {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header with title and a welcome message
st.markdown('<div class="title">✨ Welcome to Intelligent Vision!</div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-message">Start exploring image classification and object detection with advanced AI models. Choose your action below and upload your image!</div>', unsafe_allow_html=True)

# Sidebar for choosing mode with custom links
menu = st.sidebar.selectbox("Select Mode:", ["🔍 Object Detection (YOLO)", "📸 Image Classification"])

# Upload image
uploaded_file = st.file_uploader("📸 Upload your image", type=["jpg", "jpeg", "png"])

# Display image in a neat box with border
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if menu == "🔍 Object Detection (YOLO)":
        st.subheader("🔍 Object Detection Results")
        try:
            # Perform object detection using YOLO
            results = yolo_model(img)  # Using the uploaded image
            result_img = results[0].plot(labels=True)  # Add labels on the bounding box
            st.image(result_img, caption="Detection Results", use_container_width=True)
            
            # Display information about detected objects
            if len(results[0].boxes.cls) > 0:  # If there are detected objects
                for i in range(len(results[0].boxes.cls)):
                    class_id = int(results[0].boxes.cls[i])  # Get class ID
                    class_name = results[0].names[class_id]  # Get class name
                    confidence = results[0].boxes.conf[i].item()  # Get confidence
                    st.write(f"Detected Object: {class_name.capitalize()} (Confidence: {confidence*100:.2f}%)")
            else:
                st.write("No objects detected.")
        except Exception as e:
            st.error(f"Error while detecting objects with YOLO: {e}")

    elif menu == "📸 Image Classification":
        st.subheader("🔬 Image Classification Results")
        with st.spinner("Classifying image..."):
            try:
                # Preprocessing
                img_resized = img.resize((128, 128))  # Resize image according to the model's input size
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)  # Create batch size of 1
                img_array = img_array / 255.0

                # Prediction
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)

                # Define class labels
                class_labels = ['Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'C]()_
