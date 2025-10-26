import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

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
                class_labels = ['Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'Cauliflower', 
                                'Carrot', 'Capsicum', 'Cabbage', 'Broccoli', 'Brinjal', 'Bottle_Gourd', 'Bitter_Gourd', 'Bean']
                
                class_name = class_labels[class_index]  # Get class name from index
                st.write("### Prediction Result:", class_name)
                st.write("Prediction Probability: {:.2f}%".format(np.max(prediction) * 100))
            except Exception as e:
                st.error(f"Error while classifying image: {e}")

# Footer with information
st.markdown(""" 
    --- 
    If you need any help or want more information, visit the [Documentation](#). 
""")

# SINGLE VISUAL: SCATTER
# --------------------------
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

        # Pilihan kolom numeric untuk sumbu
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Butuh minimal 2 kolom numerik untuk scatter plot.")
        else:
            x = st.selectbox("Pilih sumbu X", numeric_cols, index=0)
            y = st.selectbox("Pilih sumbu Y", numeric_cols, index=1)
            color = st.selectbox("Warna (opsional)", [None] + df.columns.tolist())
            size_col = st.selectbox("Ukuran titik (opsional)", [None] + numeric_cols)

            # Optional sampling jika dataset sangat besar
            if len(df) > 5000:
                st.info("Dataset besar — men-sample 5000 baris untuk menjaga responsif.")
                df_plot = df.sample(5000, random_state=42)
            else:
                df_plot = df

            try:
                fig = px.scatter(df_plot, x=x, y=y,
                                 color=(color if color else None),
                                 size=(size_col if size_col else None),
                                 hover_data=df_plot.columns)
                fig.update_layout(title=f"Scatter: {x} vs {y}", legend_title_text=(color if color else ""))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Gagal membuat scatter plot: {e}")

# Footer
st.markdown("---")
st.markdown("Butuh opsi lain nanti? Bilang aja — tapi ini satu visual dulu sesuai permintaan.")
