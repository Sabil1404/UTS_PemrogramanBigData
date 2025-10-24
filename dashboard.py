import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt

# Fungsi untuk deteksi objek dengan MobileNet SSD
def object_detection(image, model, threshold=0.5):
    # Mengubah gambar menjadi format yang diterima model
    img_resized = cv2.resize(image, (300, 300))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(img_rgb, 1/255.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)
    
    # Mendapatkan hasil deteksi
    output = model.forward()
    
    # Menampilkan objek yang terdeteksi dengan bounding box dan confidence
    (h, w) = img_resized.shape[:2]
    boxes, confidences, class_ids = [], [], []
    
    for i in range(output.shape[2]):
        confidence = output[0, 0, i, 2]
        if confidence > threshold:
            box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            boxes.append([x, y, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(int(output[0, 0, i, 1]))
    
    return boxes, confidences, class_ids

# Model MobileNet SSD
net = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco.pb')

# Judul Aplikasi
st.title("Object Detection with MobileNet SSD")
st.markdown("""
    Upload an image to perform object detection with MobileNet SSD.
    You can adjust the confidence threshold to filter out less confident detections.
""")

# Unggah Gambar
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Slider untuk Threshold Confidence
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Menampilkan hasil jika gambar diunggah
if uploaded_file is not None:
    # Membaca gambar
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Menampilkan gambar
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Deteksi objek
    boxes, confidences, class_ids = object_detection(image_np, net, confidence_threshold)

    # Menggambar bounding boxes pada gambar
    for i in range(len(boxes)):
        (x, y, x2, y2) = boxes[i]
        label = str(class_ids[i])  # Gantilah dengan label sesuai model Anda
        confidence = confidences[i]
        
        # Gambar bounding box
        cv2.rectangle(image_np, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, f"{label} {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Menampilkan gambar dengan bounding box
    st.image(image_np, caption="Detected Image", use_container_width=True)
