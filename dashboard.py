import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as kimage
import pandas as pd
import datetime
import plotly.express as px
import os

# ------------------------
# Helper: show image with max width
# ------------------------
def show_image_limited(img_obj, caption=None, max_width=400):
    try:
        if isinstance(img_obj, np.ndarray):
            h, w = img_obj.shape[:2]
            display_w = min(w, max_width)
            st.image(img_obj, caption=caption, width=display_w)
        elif isinstance(img_obj, Image.Image):
            w = img_obj.width
            display_w = min(w, max_width)
            st.image(img_obj, caption=caption, width=display_w)
        else:
            st.image(img_obj, caption=caption)
    except Exception:
        st.image(img_obj, caption=caption)

# ------------------------
# Dummy predict (fallback)
# ------------------------
def dummy_predict(img: Image.Image):
    classes = ['Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'Cauliflower',
               'Carrot', 'Capsicum', 'Cabbage', 'Broccoli', 'Brinjal', 'Bottle_Gourd', 'Bitter_Gourd', 'Bean']
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

try:
    yolo_model = get_yolo_model()
except Exception:
    yolo_model = None

try:
    classifier = get_classifier()
except Exception:
    classifier = None

# ------------------------
# App config & CSS
# ------------------------
st.set_page_config(page_title="SeeBil", page_icon="👁️", layout="wide")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #d8c7ff 0%, #ffe6f7 100%);
}
.title {
    text-align: center;
    font-size: 48px;
    font-family: 'Poppins', sans-serif;
    color: #4B0082;
    font-weight: 700;
    margin-bottom: 6px;
}
.subtitle {
    font-size: 17px;
    color: #333;
    text-align: center;
    margin-bottom: 20px;
    font-family: 'Poppins', sans-serif;
}
.upload-box {
    border: 2px solid #836FFF;
    padding: 14px;
    border-radius: 10px;
    background-color: #fbf8ff;
    text-align: center;
}
.footer-note {
    text-align: center;
    color: #666;
    font-size: 13px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">SeeBil</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Melihat lebih dekat, memahami lebih dalam — sentuhan cerdas dari pandanganmu.</div>', unsafe_allow_html=True)

# ------------------------
# Sidebar controls
# ------------------------
menu = st.sidebar.selectbox("Pilih Mode:", ["🔍 Deteksi Objek (YOLO)", "📸 Klasifikasi Gambar", "⚙️ Demo (Tanpa model)"])
persist_csv = st.sidebar.checkbox("Simpan ke CSV setiap upload", value=False)
csv_path = st.sidebar.text_input("Path CSV (jika Simpan diaktifkan)", value="data/records.csv")
max_display_width = st.sidebar.number_input("Max width gambar (px)", min_value=150, max_value=1200, value=380, step=50)

if st.sidebar.button("🗑️ Hapus Semua Record"):
    st.session_state["records"] = []
    st.sidebar.success("Semua record session dihapus.")

if "records" not in st.session_state:
    st.session_state["records"] = []

# ------------------------
# Upload Area
# ------------------------
uploaded_file = st.file_uploader("📸 Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Gagal membuka gambar: {e}")
        img = None

    if img is not None:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        show_image_limited(img, caption="Gambar yang diunggah", max_width=int(max_display_width))
        st.markdown('</div>', unsafe_allow_html=True)

        pred_class = "Tidak diketahui"
        confidence = 0.0

        # ------------------------
        # YOLO MODE
        # ------------------------
        if menu == "🔍 Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            if yolo_model is None:
                st.warning("Model YOLO tidak tersedia.")
                pred_class, confidence, _ = dummy_predict(img)
            else:
                try:
                    results = yolo_model(np.array(img))
                    boxes = results[0].boxes
                    if boxes and len(boxes.cls) > 0:
                        best_idx = int(np.argmax(boxes.conf))
                        class_id = int(boxes.cls[best_idx])
                        pred_class = results[0].names[class_id]
                        confidence = float(boxes.conf[best_idx].item())
                        st.success(f"Objek: **{pred_class}** — Confidence: {confidence*100:.2f}%")
                    else:
                        st.info("Tidak ada objek terdeteksi.")
                except Exception as e:
                    st.error(f"Kesalahan deteksi: {e}")
                    pred_class, confidence, _ = dummy_predict(img)

        # ------------------------
        # CLASSIFICATION MODE (15 kelas sayuran)
        # ------------------------
        elif menu == "📸 Klasifikasi Gambar":
            st.subheader("🌿 Hasil Klasifikasi Gambar")
            class_labels = [
                'Tomato', 'Radish', 'Pumpkin', 'Potato', 'Papaya', 'Cucumber', 'Cauliflower',
                'Carrot', 'Capsicum', 'Cabbage', 'Broccoli', 'Brinjal',
                'Bottle_Gourd', 'Bitter_Gourd', 'Bean'
            ]
            if classifier is None:
                st.warning("Model klasifikasi tidak tersedia.")
                pred_class, confidence, _ = dummy_predict(img)
            else:
                try:
                    img_resized = img.resize((128, 128))
                    arr = kimage.img_to_array(img_resized)
                    arr = np.expand_dims(arr, axis=0) / 255.0
                    pred = classifier.predict(arr)
                    idx = int(np.argmax(pred))
                    pred_class = class_labels[idx] if 0 <= idx < len(class_labels) else f"Class_{idx}"
                    confidence = float(np.max(pred))
                    st.success(f"Hasil: **{pred_class}**")
                    st.write(f"Akurasi: {confidence*100:.2f}%")
                except Exception as e:
                    st.error(f"Kesalahan klasifikasi: {e}")
                    pred_class, confidence, _ = dummy_predict(img)

        # ------------------------
        # DEMO MODE
        # ------------------------
        else:
            pred_class, confidence, _ = dummy_predict(img)
            st.info(f"(Demo) Prediksi: **{pred_class}** — Confidence: {confidence*100:.1f}%")

        # ------------------------
        # Simpan hasil
        # ------------------------
        record = {
            "filename": getattr(uploaded_file, "name", f"img_{len(st.session_state['records'])+1}.jpg"),
            "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": menu,
            "pred_class": pred_class,
            "confidence": round(float(confidence), 4)
        }
        st.session_state["records"].append(record)
        st.success("Hasil prediksi disimpan di session.")

        if persist_csv:
            try:
                df_temp = pd.DataFrame(st.session_state["records"])
                os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
                df_temp.to_csv(csv_path, index=False)
                st.info(f"Data juga disimpan ke `{csv_path}`")
            except Exception as e:
                st.error(f"Gagal simpan CSV: {e}")

# ------------------------
# Treemap Interaktif
# ------------------------
if len(st.session_state["records"]) > 0:
    df = pd.DataFrame(st.session_state["records"])
    if "pred_class" not in df.columns or df["pred_class"].isnull().all():
        st.warning("Tidak ada data prediksi yang valid.")
    else:
        df = df.dropna(subset=["pred_class"])
        agg = (
            df.groupby("pred_class", dropna=True)
            .agg(count=("filename", "count"), avg_confidence=("confidence", "mean"))
            .reset_index()
        )
        if not agg.empty:
            st.markdown("---")
            st.subheader("🌳 Treemap Interaktif — Confidence per Kelas")
            fig = px.treemap(
                agg,
                path=["pred_class"],
                values="count",
                color="avg_confidence",
                color_continuous_scale="YlGnBu",
                hover_data={"count": True, "avg_confidence": True}
            )
            fig.update_traces(textinfo="label+value+percent entry")
            fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), coloraxis_colorbar=dict(title="Avg Confidence"))
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Belum ada gambar yang diunggah — unggah untuk melihat treemap.")

# Footer
st.markdown('<div class="footer-note">✨ Crafted with a subtle personal touch — SeeBil</div>', unsafe_allow_html=True)
