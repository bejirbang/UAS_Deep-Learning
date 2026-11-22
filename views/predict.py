"""Demo utama: prediksi satu foto (upload / kamera / contoh) + keputusan human-in-the-loop + Grad-CAM."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.business import CLASS_INFO, decide
from src.gradcam import gradcam, model_view, overlay
from src.model import LOW_RES_SHORT_SIDE, predict, prepare
from src.ui import device_switcher, footer, get_bundle, is_mobile, settings_panel

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "samples"
IMG_EXT = ["jpg", "jpeg", "png", "webp", "bmp"]

bundle = get_bundle()
mobile = is_mobile()

device_switcher()
st.title("🔍 Cek Foto Paket")
st.caption("Unggah foto paket yang diterima pembeli. Model akan menilai kondisinya dan merekomendasikan tindak lanjut klaim.")
threshold, show_cam = settings_panel(bundle, with_cam=True)

# ---------------------------------------------------------------- input
samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower().lstrip(".") in IMG_EXT) if SAMPLE_DIR.exists() else []
tab_names = (["📷 Kamera", "📁 Galeri"] if mobile else ["📁 Unggah foto", "📷 Kamera"]) + (["🖼️ Contoh"] if samples else [])
tabs = st.tabs(tab_names)
t_upload, t_cam = (tabs[1], tabs[0]) if mobile else (tabs[0], tabs[1])

source, source_name = None, None
with t_upload:
    up = st.file_uploader("Pilih foto paket", type=IMG_EXT, key="single_upload",
                          help="Di HP, tombol ini juga bisa membuka kamera atau galeri.")
    if up is not None:
        source, source_name = up.getvalue(), up.name
with t_cam:
    st.caption("Izinkan akses kamera di browser. Di HP, gunakan kamera belakang untuk hasil terbaik.")
    if st.toggle("Aktifkan kamera", key="cam_on"):
        shot = st.camera_input("Ambil foto paket", label_visibility="collapsed")
        if shot is not None:
            source, source_name = shot.getvalue(), "foto_kamera.jpg"
if samples:
    with tabs[2]:
        pick = st.selectbox("Pilih foto contoh", ["—"] + [p.name for p in samples])
        if pick != "—":
            p = SAMPLE_DIR / pick
            source, source_name = p.read_bytes(), p.name

if source is None:
    st.info("Belum ada foto. Unggah foto atau ambil gambar dengan kamera untuk memulai.", icon="⬆️")
    footer()
    st.stop()

# ---------------------------------------------------------------- inference
try:
    prep = prepare(bundle, source)
except Exception as exc:  # file rusak / bukan gambar
    st.error(f"Gambar tidak bisa dibuka: {exc}")
    st.stop()

with st.spinner("Menganalisis foto…"):
    pred = predict(bundle, prep)
info = CLASS_INFO[pred.label]
dec = decide(pred.label, pred.confidence, threshold)
classes = bundle.classes


def result_block():
    st.markdown(
        f'<div class="result-card" style="background:{info["color"]}">'
        f'<div class="lbl">Hasil prediksi</div>'
        f'<div class="cls">{info["emoji"]} {pred.label}</div>'
        f'<div class="conf">Confidence {pred.confidence:.1%} · {info["arti"]}</div></div>'
        f'<div class="decision" style="border-color:{dec.color}">'
        f'<div class="t">{dec.icon} {dec.title}</div><div class="d">{dec.detail}</div></div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure(go.Bar(
        x=[pred.probs[c] for c in classes][::-1], y=classes[::-1], orientation="h",
        marker_color=[CLASS_INFO[c]["color"] for c in classes][::-1],
        text=[f"{pred.probs[c]:.1%}" for c in classes][::-1], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_dash="dot", line_color="#8d8d8d")
    fig.update_layout(
        height=190, margin=dict(l=0, r=45, t=30, b=0), showlegend=False,
        xaxis=dict(range=[0, 1], tickvals=[0, .25, .5, .75, 1], tickformat=".0%", fixedrange=True),
        yaxis=dict(fixedrange=True),
        title=dict(text=f"Probabilitas per kelas (garis = threshold {threshold:.2f})", font=dict(size=13)),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    if prep.low_res:
        st.warning(f"Resolusi foto rendah (sisi pendek {prep.short_side}px < {LOW_RES_SHORT_SIDE}px). "
                   "Detail kerusakan bisa tidak terlihat.", icon="⚠️")
    if prep.heavy_noise:
        st.warning(f"Foto mengandung noise berat (σ≈{prep.noise:.3f}). Pada foto seperti ini model lebih sering "
                   "salah menebak **Aman**, jadi pertimbangkan review manual.", icon="⚠️")
    st.caption(f"Waktu inferensi: {pred.latency_ms:.0f} ms (CPU server)")


if mobile:
    thumb = prep.original.copy()
    thumb.thumbnail((640, 360))
    st.image(thumb, caption=f"{source_name} · {prep.width}×{prep.height}px", width="stretch")
    result_block()
else:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(prep.original, caption=f"{source_name} · {prep.width}×{prep.height}px", width="stretch")
    with right:
        result_block()

# ---------------------------------------------------------------- Grad-CAM
if show_cam:
    st.subheader("🔥 Apa yang dilihat model?")
    with st.spinner("Membuat Grad-CAM…"):
        heat = gradcam(bundle, prep.tensor, classes.index(pred.label))
    view, cam = model_view(bundle, prep.tensor), overlay(bundle, prep.tensor, heat)
    # gabungkan jadi satu gambar agar tetap berdampingan di layar HP
    gap = 8
    combo = Image.new("RGB", (view.width * 2 + gap, view.height), "white")
    combo.paste(view, (0, 0)); combo.paste(cam, (view.width + gap, 0))
    if mobile:
        st.image(combo, width="stretch")
    else:
        c, _ = st.columns([2, 1])
        c.image(combo, width="stretch")
    st.caption(f"Kiri: potongan 224×224 yang dilihat model. Kanan: Grad-CAM untuk prediksi “{pred.label}”. "
               "Merah = area paling berpengaruh. Model sebaiknya fokus pada badan kardus, bukan latar atau watermark.")

with st.expander("Detail teknis"):
    st.write({
        "arsitektur": bundle.meta.get("arch"),
        "ukuran input": f"{bundle.img_size}×{bundle.img_size}",
        "probabilitas": {k: round(v, 4) for k, v in pred.probs.items()},
        "estimasi noise σ": round(prep.noise, 4),
        "ukuran asli": f"{prep.width}×{prep.height}",
    })

footer()
