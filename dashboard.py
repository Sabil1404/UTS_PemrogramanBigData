import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
import datetime

# --- contoh fungsi dummy predict (ganti pakai modelmu) ---
def dummy_predict(img: Image.Image):
    # return (class_name, confidence, bbox)
    # untuk demo: acak kelas
    classes = ["Anjing", "Kucing", "Burung"]
    idx = np.random.randint(0, len(classes))
    conf = float(np.round(np.random.uniform(0.5, 0.98), 3))
    bbox = (10, 10, img.width-10, img.height-10)
    return classes[idx], conf, bbox

# Inisialisasi storage sementara
if "records" not in st.session_state:
    st.session_state["records"] = []  # list of dicts

st.title("Upload gambar — simpan metadata & lihat Treemap kelas")

uploaded_file = st.file_uploader("Upload gambar (satu-satu)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Panggil model di sini (gunakan lazy loading pada app utama)
    # Contoh: pred_class, conf, bbox = yolo_model_predict(img)
    pred_class, conf, bbox = dummy_predict(img)

    st.markdown(f"**Prediksi:** `{pred_class}`  — Confidence: **{conf*100:.1f}%**")

    # Simpan record
    record = {
        "filename": getattr(uploaded_file, "name", f"img_{len(st.session_state['records'])+1}.jpg"),
        "uploaded_at": datetime.datetime.now().isoformat(timespec='seconds'),
        "pred_class": pred_class,
        "confidence": conf,
        "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]
    }
    st.session_state["records"].append(record)
    st.success("Metadata gambar disimpan (session only).")

# Jika ada data, buat DataFrame agregat untuk treemap
if len(st.session_state["records"]) > 0:
    df = pd.DataFrame(st.session_state["records"])
    st.subheader("Ringkasan data (preview)")
    st.dataframe(df.tail(10))

    # Buat agregasi: count per kelas & rata2 confidence
    agg = df.groupby("pred_class").agg(
        count=("filename","count"),
        avg_confidence=("confidence","mean")
    ).reset_index()

    # Treemap: path by pred_class, size=count, color=avg_confidence
    fig = px.treemap(
        agg,
        path=["pred_class"],
        values="count",
        color="avg_confidence",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=agg["avg_confidence"].mean(),
        hover_data={"count":True, "avg_confidence":True}
    )
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
    st.subheader("Treemap: Distribusi kelas (size=count, color=avg confidence)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Belum ada gambar diupload — upload satu gambar untuk mulai mengumpulkan metadata.")
