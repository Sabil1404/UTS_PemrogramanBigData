import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Memuat model YOLO dan model klasifikasi
    yolo_model = YOLO("model/deteksi.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/klasifikasi.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.set_page_config(page_title="🧠 Image Classification & Object Detection App", page_icon=":camera:", layout="wide")

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
st.markdown('<div class="description">Upload an image and select the mode to detect objects or classify images.</div>', unsafe_allow_html=True)

# Sidebar for choosing mode
menu = st.sidebar.selectbox("Choose Mode:", ["Object Detection (YOLO)", "Image Classification"])

# Upload gambar
uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

# Display image in a neat box with border
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if menu == "Object Detection (YOLO)":
        st.subheader("🔍 Object Detection Results")
        try:
            # Perform object detection using YOLO
            results = yolo_model(img)
            result_img = results[0].plot(labels=True)  # Add label to bounding box
            st.image(result_img, caption="Detection Results", use_container_width=True)
            
            # Display detected objects with confidence score
            detected_objects = []
            confidence_scores = []
            if len(results[0].boxes.cls) > 0:
                for i in range(len(results[0].boxes.cls)):
                    class_id = int(results[0].boxes.cls[i])
                    class_name = results[0].names[class_id]  # Get class name correctly
                    confidence = results[0].boxes.conf[i].item()
                    detected_objects.append(class_name)
                    confidence_scores.append(confidence)
                    st.write(f"Detected Object: {class_name.capitalize()} (Confidence: {confidence*100:.2f}%)")
            else:
                st.write("No objects detected.")

            # Plot detected objects and their confidence
            if detected_objects:
                fig, ax = plt.subplots()
                ax.bar(detected_objects, confidence_scores, color='skyblue')
                ax.set_title('Confidence per Detected Object')
                ax.set_xlabel('Objects')
                ax.set_ylabel('Confidence')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error detecting objects with YOLO: {e}")

    elif menu == "Image Classification":
        st.subheader("🔬 Image Classification Results")
        with st.spinner("Classifying image..."):
            try:
                # Preprocessing
                img_resized = img.resize((128, 128))  # Resize image to fit the model input
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)  # Create batch size of 1
                img_array = img_array / 255.0

                # Prediction
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)

                # Class labels
                class_labels = ['Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'Cauliflower', 
                                'Carrot', 'Capsicum', 'Cabbage', 'Broccoli', 'Brinjal', 'Bottle_Gourd', 'Bitter_Gourd', 'Bean']

                class_name = class_labels[class_index]  # Get class name from index
                st.write("### Predicted Class:", class_name)
                st.write("Prediction Probability: {:.2f}%".format(np.max(prediction) * 100))
            except Exception as e:
                st.error(f"Error classifying image: {e}")

# Footer with documentation link
st.markdown(""" 
    --- 
    If you have any questions or need help, visit the [App Documentation](#). 
""")
