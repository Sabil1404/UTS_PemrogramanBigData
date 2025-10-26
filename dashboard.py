import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as kimage
import pandas as pd
import datetime
import plotly.express as px
import os

# ------------------------
# Dummy predict (fallback)
# ------------------------
def dummy_predict(img: Image.Image):
    classes = ["Anjing", "Ayam", "Kupu-Kupu"]
    idx = np.random.randint(0, len(classes))
    conf = float(np.round(np.random.uniform(0.5, 0.98), 3))
    bbox = (10, 10, img.width - 10, img.height - 10)
    return classes[idx], conf, bbox

# ------------------------
# Lazy-load models
# ------------------------
@st.cache_resource
def get_yolo_model():
    from ultralytics import YOLO
    return YOLO("model/deteksi.pt")

@st.cache_resource
def get_classifier():
    import tensorflow as tf
    return tf.keras.models.load_model("model/klasifikasi.h5")

# ------------------------
# App config & styling
# ------------------------
st.set_page_config(page_title="SeeBil", page_icon="👀", layout="wide")
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 50px;
            font-weight: 800;
            background: linear-gradient(90deg, #FF5733, #FFC300);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }
        .subtitle {
            text-align: center;
            font-size: 17px;
            color: #444;
            font-style: italic;
            margin-bottom: 25px;
        }
        .upload-box {
            border: 2px dashed #00BFFF;
            border-radius: 12px;
            padding: 15px;
            background-color: #F8FBFF;
        }
        .stImage > img {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Header Section
# ------------------------
st.markdown('<div class="title">👀 SeeBil</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">"See closer, understand deeper — intelligence through your vision.</div>', unsafe_allow_html=True)

# ------------------------
# Sidebar: mode & controls
# ------------------------
mode = st.sidebar.selectbox("Mode:", ["🔍 Object Detection (YOLO)", "📸 Image Classification", "⚙️ Demo (No model)"])
persist_csv = st.sidebar.checkbox("Simpan ke CSV setiap upload", value=False)
csv_path = st.sidebar.text_input("Path CSV (jika Simpan aktif)", value="data/records.csv")

if st.sidebar.button("🗑️ Hapus Semua Record"):
    st.session_state["records"] = []
    st.sidebar.success("Semua record dihapus dari session.")

if "records" not in st.session_state:
    st.session_state["records"] = []

# ------------------------
# Upload Section
# ------------------------
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Gagal membuka gambar: {e}")
        st.stop()

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Gambar yang diunggah", width=400)
    st.markdown('</div>', unsafe_allow_html=True)

    use_dummy = (mode == "⚙️ Demo (No model)")
    pred_class, conf, bbox = None, None, (None, None, None, None)

    if not use_dummy:
        if mode.startswith("🔍"):
            try:
                yolo = get_yolo_model()
                results = yolo(np.array(img))
                boxes = results[0].boxes
                if boxes and len(boxes.cls) > 0:
                    confs = [float(c) for c in boxes.conf]
                    best_idx = int(np.argmax(confs))
                    class_id = int(boxes.cls[best_idx])
                    pred_class = results[0].names.get(class_id, str(class_id))
                    conf = float(boxes.conf[best_idx].item())
                    xyxy = boxes.xyxy[best_idx].tolist()
                    bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    result_img = results[0].plot(labels=True)
                    st.image(result_img, caption="Hasil Deteksi", width=400)
                else:
                    pred_class, conf = "NoObject", 0.0
            except Exception:
                st.warning("Model YOLO gagal dijalankan — gunakan dummy predict.")
                pred_class, conf, bbox = dummy_predict(img)
        else:
            try:
                clf = get_classifier()
                img_resized = img.resize((128, 128))
                arr = kimage.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0
                pred = clf.predict(arr)
                idx = int(np.argmax(pred))
                class_labels = [
                    "Tomato", "Radish", "Pumpkin", "Potato", "Papaya", "Cucumber",
                    "Cauliflower", "Carrot", "Capsicum", "Cabbage", "Broccoli",
                    "Brinjal", "Bottle_Gourd", "Bitter_Gourd", "Bean"
                ]
                pred_class = class_labels[idx] if 0 <= idx < len(class_labels) else f"Class_{idx}"
                conf = float(np.max(pred))
            except Exception:
                st.warning("Model klasifikasi gagal dijalankan — gunakan dummy predict.")
                pred_class, conf, bbox = dummy_predict(img)
    else:
        pred_class, conf, bbox = dummy_predict(img)

    st.markdown(f"**Prediksi:** `{pred_class}` — Confidence: **{conf*100:.1f}%**")

    record = {
        "filename": getattr(uploaded_file, "name", f"img_{len(st.session_state['records'])+1}.jpg"),
        "uploaded_at": datetime.datetime.now().isoformat(timespec='seconds'),
        "pred_class": pred_class,
        "confidence": round(float(conf), 4)
    }
    st.session_state["records"].append(record)
    st.success("Metadata gambar disimpan (session only).")

    if persist_csv:
        try:
            df_temp = pd.DataFrame(st.session_state["records"])
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            df_temp.to_csv(csv_path, index=False)
            st.info(f"Data juga disimpan ke `{csv_path}`")
        except Exception as e:
            st.error(f"Gagal simpan CSV: {e}")

# ------------------------
# Treemap Visualization
# ------------------------
if len(st.session_state["records"]) > 0:
    df = pd.DataFrame(st.session_state["records"])
    agg = (
        df.groupby("pred_class")
        .agg(count=("filename", "count"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )

    fig = px.treemap(
        agg,
        path=["pred_class"],
        values="count",
        color="avg_confidence",
        color_continuous_scale="Blues",
        color_continuous_midpoint=agg["avg_confidence"].mean(),
        hover_data={"count": True, "avg_confidence": True}
    )
    fig.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        title="Treemap, ukuran Data 📊"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Belum ada gambar diunggah — upload satu gambar untuk mulai.")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("👀 SeeBil — A smart glance, powered by your vision ✨")






