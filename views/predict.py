from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.business import CLASS_INFO, decide
from src.gradcam import gradcam, model_view, overlay
from src.model import LOW_RES_SHORT_SIDE, predict, prepare
from src.ui import chart, footer, get_bundle, is_mobile, settings_panel

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "samples"
IMG_EXT = ["jpg", "jpeg", "png", "webp", "bmp"]
ROUTE_ALERT = {"auto": st.success, "cs": st.info, "reupload": st.warning}
ROUTE_ICON = {"auto": ":material/check_circle:", "cs": ":material/support_agent:", "reupload": ":material/replay:"}

bundle = get_bundle()
mobile = is_mobile()

st.title("Cek foto paket", icon=":material/search:")
st.caption("Unggah foto paket yang diterima pembeli. Model akan menilai kondisinya dan merekomendasikan tindak lanjut klaim.")
threshold, show_cam = settings_panel(bundle, with_cam=True)

samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower().lstrip(".") in IMG_EXT) if SAMPLE_DIR.exists() else []
T_UP, T_CAM, T_EX = ":material/upload: Unggah foto", ":material/photo_camera: Kamera", ":material/image: Contoh"
tab_names = ([T_CAM, T_UP] if mobile else [T_UP, T_CAM]) + ([T_EX] if samples else [])
tabs = dict(zip(tab_names, st.tabs(tab_names)))

source, source_name = None, None
with tabs[T_UP]:
    up = st.file_uploader("Pilih foto paket", type=IMG_EXT, key="single_upload",
                          help="Di HP, tombol ini juga bisa membuka kamera atau galeri.")
    if up is not None:
        source, source_name = up.getvalue(), up.name
with tabs[T_CAM]:
    st.caption("Izinkan akses kamera di browser. Di HP, gunakan kamera belakang untuk hasil terbaik.")
    if st.toggle("Aktifkan kamera", key="cam_on"):
        shot = st.camera_input("Ambil foto paket", label_visibility="collapsed")
        if shot is not None:
            source, source_name = shot.getvalue(), "foto_kamera.jpg"
if samples:
    with tabs[T_EX]:
        pick = st.selectbox("Pilih foto contoh", [p.name for p in samples], index=None, placeholder="Pilih foto…")
        if pick:
            p = SAMPLE_DIR / pick
            source, source_name = p.read_bytes(), p.name

if source is None:
    st.info("Unggah foto atau ambil gambar dengan kamera untuk memulai.", icon=":material/upload:")
    footer()
    st.stop()

try:
    prep = prepare(bundle, source)
except Exception as exc:
    st.error(f"Gambar tidak bisa dibuka: {exc}", icon=":material/broken_image:")
    st.stop()

with st.spinner("Menganalisis foto…"):
    pred = predict(bundle, prep)
info = CLASS_INFO[pred.label]
dec = decide(pred.label, pred.confidence, threshold)
classes = bundle.classes


def result_block():
    st.html(
        f'<div class="result-card" style="background:{info["color"]}">'
        f'<div class="lbl">Hasil prediksi</div>'
        f'<div class="cls">{info["emoji"]} {pred.label}</div>'
        f'<div class="conf">Confidence {pred.confidence:.1%} · {info["arti"]}</div></div>'
    )
    ROUTE_ALERT[dec.route](dec.detail, title=dec.title, icon=ROUTE_ICON[dec.route])

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
    chart(fig, alt=f"Probabilitas: " + ", ".join(f"{c} {pred.probs[c]:.0%}" for c in classes))

    if prep.low_res:
        st.warning(f"Sisi pendek {prep.short_side}px (< {LOW_RES_SHORT_SIDE}px), detail kerusakan bisa tidak terlihat.",
                   title="Resolusi foto rendah", icon=":material/warning:")
    if prep.heavy_noise:
        st.warning(f"Estimasi σ≈{prep.noise:.3f}. Pada foto seperti ini model lebih sering salah menebak Aman, "
                   "jadi pertimbangkan review manual.", title="Foto mengandung noise berat", icon=":material/warning:")
    st.caption(f"Waktu inferensi: {pred.latency_ms:.0f} ms (CPU server)")


photo_alt = f"Foto yang diunggah: {source_name}"
if mobile:
    thumb = prep.original.copy()
    thumb.thumbnail((640, 360))
    st.image(thumb, caption=f"{source_name} · {prep.width}×{prep.height}px", width="stretch", alt=photo_alt)
    result_block()
else:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(prep.original, caption=f"{source_name} · {prep.width}×{prep.height}px", width="stretch", alt=photo_alt)
    with right:
        result_block()

if show_cam:
    st.subheader(":material/local_fire_department: Apa yang dilihat model?")
    with st.spinner("Membuat Grad-CAM…"):
        heat = gradcam(bundle, prep.tensor, classes.index(pred.label))
    view, cam = model_view(bundle, prep.tensor), overlay(bundle, prep.tensor, heat)
    gap = 8
    combo = Image.new("RGBA", (view.width * 2 + gap, view.height), (0, 0, 0, 0))
    combo.paste(view, (0, 0)); combo.paste(cam, (view.width + gap, 0))
    cam_alt = f"Kiri: potongan foto 224×224. Kanan: peta panas Grad-CAM untuk prediksi {pred.label}"
    if mobile:
        st.image(combo, width="stretch", alt=cam_alt)
    else:
        c, _ = st.columns([2, 1])
        c.image(combo, width="stretch", alt=cam_alt)
    st.caption(f"Kiri: potongan 224×224 yang dilihat model. Kanan: Grad-CAM untuk prediksi “{pred.label}”. "
               "Merah = area paling berpengaruh. Model sebaiknya fokus pada badan kardus, bukan latar atau watermark.")

with st.expander("Detail teknis", icon=":material/code:"):
    st.json({
        "arsitektur": bundle.meta.get("arch"),
        "ukuran input": f"{bundle.img_size}×{bundle.img_size}",
        "probabilitas": {k: round(v, 4) for k, v in pred.probs.items()},
        "estimasi noise σ": round(prep.noise, 4),
        "ukuran asli": f"{prep.width}×{prep.height}",
    })

footer()
