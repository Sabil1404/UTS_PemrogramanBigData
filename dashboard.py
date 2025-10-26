import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import squarify
import pandas as pd
import datetime
import plotly.express as px
import os

# ==========================
# Load Models (lazy-cached inside load_models)
# ==========================
@st.cache_resource
def load_models():
    # If models not present or you want to use demo, handle exceptions in runtime
    yolo_model = YOLO("model/deteksi.pt")
    classifier = tf.keras.models.load_model("model/klasifikasi.h5")
    return yolo_model, classifier

# Try to load models; if fails, we'll still let app run and show warnings later
try:
    yolo_model, classifier = load_models()
except Exception:
    yolo_model, classifier = None, None

# ==========================
# App config & CSS
# ==========================
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
    text-shadow: 1px 2px 8px rgba(75, 0, 130, 0.12);
    margin-bottom: 6px;
}
.subtitle {
    font-size: 17px;
    color: #333;
    text-align: center;
    margin-bottom: 28px;
    font-family: 'Poppins', sans-serif;
    letter-spacing: 0.2px;
}
.upload-box {
    border: 2px solid #836FFF;
    padding: 16px;
    border-radius: 12px;
    background-color: #fbf8ff;
    box-shadow: 0px 6px 18px rgba(131, 111, 255, 0.08);
    text-align: center;
}
.stImage img {
    border-radius: 10px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
.sidebar .sidebar-content {
    background-color: #f8f8ff;
    border-radius: 12px;
    padding: 12px;
}
.footer-note {
    text-align: center;
    color: #5b5b6b;
    font-size: 13px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header (SeeBil + subtitle in Indonesian)
# ==========================
st.markdown('<div class="title">SeeBil</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Melihat lebih dekat, memahami lebih dalam — sentuhan cerdas dari pandanganmu.</div>', unsafe_allow_html=True)

# --------------------------
# Sidebar controls
# --------------------------
menu = st.sidebar.selectbox("Pilih Mode:", ["🔍 Deteksi Objek (YOLO)", "📸 Klasifikasi Gambar", "⚙️ Demo (Tanpa model)"])
persist_csv = st.sidebar.checkbox("Simpan ke CSV setiap upload", value=False)
csv_path = st.sidebar.text_input("Path CSV (jika centang Simpan)", value="data/records.csv")
max_display_width = st.sidebar.number_input("Max width gambar (px)", min_value=150, max_value=1200, value=380, step=50)

if st.sidebar.button("🗑️ Hapus Semua Record"):
    st.session_state["records"] = []
    st.sidebar.success("Semua record session dihapus.")

# session init
if "records" not in st.session_state:
    st.session_state["records"] = []

# helper: show image with max width
def show_image_limited(img_obj, caption=None, max_width=380):
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

# --------------------------
# Upload area
# --------------------------
uploaded_file = st.file_uploader("📸 Unggah gambar (satu-per-satu)", type=["jpg", "jpeg", "png"])
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

        # prediction
        pred_class = "Tidak diketahui"
        confidence = 0.0

        # choose behavior based on menu
        if menu == "🔍 Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            if yolo_model is None:
                st.warning("Model YOLO tidak tersedia. Pilih mode 'Demo' jika ingin coba tanpa model.")
                # fallback to simple demo behavior
                pred_class = "NoModel"
                confidence = 0.0
            else:
                try:
                    results = yolo_model(np.array(img))
                    # show annotated image (limited width)
                    try:
                        result_img = results[0].plot(labels=True)
                        show_image_limited(result_img, caption="Hasil Deteksi (annotated)", max_width=int(max_display_width))
                    except Exception:
                        pass

                    boxes = results[0].boxes
                    if boxes and len(boxes.cls) > 0:
                        best_idx = int(np.argmax(boxes.conf))
                        class_id = int(boxes.cls[best_idx])
                        pred_class = results[0].names[class_id]
                        confidence = float(boxes.conf[best_idx].item())
                        st.success(f"Objek terdeteksi: **{pred_class}** — Confidence: {confidence*100:.2f}%")

                        # make small treemap of detections (confidence-weighted)
                        detected_objs = [results[0].names[int(cid)] for cid in boxes.cls.tolist()]
                        confs = [float(c) for c in boxes.conf.tolist()]
                        sizes = [c * 100 for c in confs]
                        labels = [f"{o}\n{c*100:.1f}%" for o, c in zip(detected_objs, confs)]

                        if len(sizes) > 0:
                            plt.figure(figsize=(6, 3.5))
                            squarify.plot(sizes=sizes, label=labels, alpha=0.85,
                                          color=px.colors.qualitative.Pastel)
                            plt.axis("off")
                            st.subheader("📊 Treemap (Deteksi)")
                            st.pyplot(plt)
                    else:
                        st.info("Tidak ada objek yang terdeteksi.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mendeteksi: {e}")

        elif menu == "📸 Klasifikasi Gambar":
            st.subheader("🔬 Hasil Klasifikasi")
            if classifier is None:
                st.warning("Model klasifikasi tidak tersedia. Pilih mode 'Demo' jika ingin coba tanpa model.")
                pred_class = "NoModel"
                confidence = 0.0
            else:
                try:
                    img_resized = img.resize((128, 128))
                    arr = image.img_to_array(img_resized)
                    arr = np.expand_dims(arr, axis=0) / 255.0
                    pred = classifier.predict(arr)
                    idx = int(np.argmax(pred))
                    class_labels = ['Anjing', 'Ayam', 'Kupu-Kupu']
                    pred_class = class_labels[idx] if 0 <= idx < len(class_labels) else f"Class_{idx}"
                    confidence = float(np.max(pred))
                    st.success(f"Hasil Prediksi: **{pred_class}**")
                    st.write(f"Akurasi: {confidence*100:.2f}%")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

        else:  # Demo mode
            # simple demo fallback: random pick
            demo_classes = ["Anjing", "Ayam", "Kupu-Kupu"]
            pred_class = np.random.choice(demo_classes)
            confidence = float(np.round(np.random.uniform(0.5, 0.98), 3))
            st.info(f"(Demo) Prediksi: **{pred_class}** — Confidence: {confidence*100:.1f}%")

        # Save record to session and optional CSV
        record = {
            "nama_file": getattr(uploaded_file, "name", f"img_{len(st.session_state['records'])+1}.jpg"),
            "tanggal_upload": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": menu,
            "hasil_prediksi": pred_class,
            "confidence": round(float(confidence), 4)
        }
        st.session_state["records"].append(record)
        st.success("Hasil prediksi disimpan (session).")

        if persist_csv:
            try:
                df_temp = pd.DataFrame(st.session_state["records"])
                os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
                df_temp.to_csv(csv_path, index=False)
                st.info(f"Data juga disimpan ke `{csv_path}`")
            except Exception as e:
                st.error(f"Gagal simpan CSV: {e}")

# --------------------------
# Treemap Visualization (only) aggregated across session records
# --------------------------
if len(st.session_state["records"]) > 0:
    st.markdown("---")
    st.subheader("🌳 Treemap Confidence Rata-Rata per Kelas")

    df = pd.DataFrame(st.session_state["records"])
    agg = df.groupby("hasil_prediksi").agg(rata_confidence=("confidence", "mean")).reset_index()

    # If no numeric values (possible), show info
    if agg.empty:
        st.info("Belum ada data untuk divisualisasikan.")
    else:
        fig_tree = px.treemap(
            agg,
            path=["hasil_prediksi"],
            values="rata_confidence",
            color="rata_confidence",
            color_continuous_scale="YlGnBu",
            title="Treemap: Rata-rata Confidence per Kelas"
        )
        fig_tree.update_layout(margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)
else:
    st.info("Belum ada gambar yang diunggah — unggah gambar untuk menyimpan dan melihat treemap.")

# Footer
st.markdown('<div class="footer-note">✨ Crafted with subtle personal touch — SeeBil</div>', unsafe_allow_html=True)
