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
    """Load YOLO model on demand. Raises if model file missing."""
    from ultralytics import YOLO
    return YOLO("model/deteksi.pt")

@st.cache_resource
def get_classifier():
    """Load classifier model on demand. Raises if model file missing."""
    import tensorflow as tf
    return tf.keras.models.load_model("model/klasifikasi.h5")

# ------------------------
# App config & CSS
# ------------------------
st.set_page_config(page_title="✨ Intelligent Vision", page_icon="🔍", layout="wide")
st.markdown("""
    <style>
        .title { text-align:center; font-size:48px; color:navy; font-weight:700; margin-bottom:6px; }
        .subtitle { text-align:center; color:#666; margin-bottom:24px; }
        .upload-box { border:2px solid #FF5733; padding:16px; border-radius:10px; background:#fbfbfb; }
        .small-muted { color:#777; font-size:13px; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">✨ Intelligent Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload gambar satu-per-satu — simpan metadata & lihat Treemap kelas</div>', unsafe_allow_html=True)

# ------------------------
# Sidebar: mode & controls
# ------------------------
mode = st.sidebar.selectbox("Mode:", ["🔍 Object Detection (YOLO)", "📸 Image Classification", "⚙️ Demo (No model)"])
persist_csv = st.sidebar.checkbox("Simpan ke CSV setiap upload", value=False)
csv_path = st.sidebar.text_input("Path CSV (jika centang Simpan)", value="data/records.csv")

# Tombol hapus data session
if st.sidebar.button("🗑️ Hapus Semua Record"):
    st.session_state["records"] = []
    st.sidebar.success("Semua record dihapus dari session.")

# ------------------------
# Session storage init
# ------------------------
if "records" not in st.session_state:
    st.session_state["records"] = []  # list of dicts

# ------------------------
# Upload UI
# ------------------------
uploaded_file = st.file_uploader("Upload gambar (satu-satu)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Gagal membuka gambar: {e}")
        st.stop()

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------
    # Run prediction depending on mode
    # ------------------------
    use_dummy = (mode == "⚙️ Demo (No model)")
    pred_class = None
    conf = None
    bbox = (None, None, None, None)

    if not use_dummy:
        if mode.startswith("🔍"):
            # YOLO detection
            try:
                yolo = get_yolo_model()
                results = yolo(np.array(img))
                boxes = results[0].boxes
                # if there are detections
                if boxes and len(boxes.cls) > 0:
                    # take detection with highest confidence
                    confs = [float(c) for c in boxes.conf]
                    best_idx = int(np.argmax(confs))
                    class_id = int(boxes.cls[best_idx])
                    pred_class = results[0].names.get(class_id, str(class_id))
                    conf = float(boxes.conf[best_idx].item())
                    xyxy = boxes.xyxy[best_idx].tolist()  # [xmin,ymin,xmax,ymax]
                    bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                else:
                    pred_class = "NoObject"
                    conf = 0.0
                    bbox = (0, 0, 0, 0)
                # show annotated image if available
                try:
                    result_img = results[0].plot(labels=True)
                    st.image(result_img, caption="Detection Results (annotated)", use_container_width=True)
                except Exception:
                    pass
            except Exception:
                st.warning("Gagal memuat/menjalankan YOLO model — memakai dummy predict.")
                pred_class, conf, bbox = dummy_predict(img)
        else:
            # Classification mode
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
                bbox = (0, 0, 0, 0)
            except Exception:
                st.warning("Gagal memuat/menjalankan classifier — memakai dummy predict.")
                pred_class, conf, bbox = dummy_predict(img)
    else:
        pred_class, conf, bbox = dummy_predict(img)

    # Display result summary
    st.markdown(f"**Prediksi:** `{pred_class}` — Confidence: **{conf*100:.1f}%**")

    # ------------------------
    # Save record to session (and optionally CSV)
    # ------------------------
    record = {
        "filename": getattr(uploaded_file, "name", f"img_{len(st.session_state['records'])+1}.jpg"),
        "uploaded_at": datetime.datetime.now().isoformat(timespec='seconds'),
        "pred_class": pred_class,
        "confidence": round(float(conf), 4),
        "xmin": int(bbox[0]) if bbox[0] is not None else None,
        "ymin": int(bbox[1]) if bbox[1] is not None else None,
        "xmax": int(bbox[2]) if bbox[2] is not None else None,
        "ymax": int(bbox[3]) if bbox[3] is not None else None
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
# If we have records -> show table + treemap
# ------------------------
if len(st.session_state["records"]) > 0:
    df = pd.DataFrame(st.session_state["records"])
    st.subheader("Ringkasan data (preview terakhir)")
    st.dataframe(df.tail(10))

    # Aggregate: count & avg confidence per class
    agg = (
        df.groupby("pred_class")
        .agg(count=("filename", "count"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )

    # Build treemap
    fig = px.treemap(
        agg,
        path=["pred_class"],
        values="count",
        color="avg_confidence",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=agg["avg_confidence"].mean(),
        hover_data={"count": True, "avg_confidence": True}
    )
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), title="Treemap: Distribusi kelas (size=count, color=avg confidence)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Belum ada gambar diupload — upload satu gambar untuk mulai mengumpulkan metadata.")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.markdown("Catatan: jika ingin deploy, gunakan lazy-load model agar UI tidak lama saat startup. Pilih mode 'Demo' untuk uji coba tanpa model.")

