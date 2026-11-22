from __future__ import annotations

import streamlit as st

from src.model import load_bundle

DEVICES = ["Laptop", "HP"]

CSS = """
<style>

.st-key-topbar { margin-bottom: .2rem; }

.hero { background: linear-gradient(135deg, #1f5fae 0%, #2a78d6 55%, #4aa3f0 100%); color: #fff;
  border-radius: 20px; padding: 2.2rem 2.2rem; margin-bottom: .6rem; }
.hero h1 { color: #fff !important; font-size: 2.2rem; line-height: 1.15; margin: 0 0 .6rem 0; padding: 0; }
.hero p  { color: #eaf2fd !important; font-size: 1.05rem; line-height: 1.55; margin: 0; max-width: 720px; }
.hero b  { color: #fff !important; }
@media (max-width: 640px) {
  .hero { padding: 1.4rem 1.2rem; border-radius: 16px; }
  .hero h1 { font-size: 1.6rem; }
  .hero p { font-size: .97rem; }
}

.result-card { border-radius: 16px; padding: 1.1rem 1.3rem; color: #fff; margin-bottom: .4rem; }
.result-card div { color: #fff !important; }
.result-card .lbl  { font-size: .8rem; opacity: .9; text-transform: uppercase; letter-spacing: .06em; }
.result-card .cls  { font-size: 2rem; font-weight: 800; line-height: 1.1; margin: .15rem 0; }
.result-card .conf { font-size: .98rem; opacity: .95; }

.num { display:inline-flex; width:28px; height:28px; border-radius:50%; background:#2a78d6; color:#fff !important;
  align-items:center; justify-content:center; font-weight:700; font-size:.9rem; }
</style>
"""

def ua_looks_mobile() -> bool:
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    return any(k in ua for k in ("Mobi", "Android", "iPhone", "iPad"))

def device_bar() -> str:
    if "mode" not in st.session_state and "mode" not in st.query_params:
        st.session_state["mode"] = "HP" if ua_looks_mobile() else "Laptop"
    with st.container(key="topbar", horizontal=True, horizontal_alignment="right"):
        mode = st.segmented_control(
            "Tampilan", DEVICES, key="mode", bind="query-params",
            required=True, label_visibility="collapsed",
            help="Pilih tampilan sesuai perangkat. Tema terang/gelap ada di menu ⋮ di kanan atas.",
        )
    return mode or "Laptop"

def is_mobile() -> bool:
    return st.session_state.get("mode") == "HP"

def choose_device(d: str) -> None:
    st.session_state["mode"] = d
    st.session_state["device_confirmed"] = True

@st.dialog("Selamat datang", icon=":material/waving_hand:", dismissible=False)
def device_dialog():
    st.write("Anda mengakses CekPaket lewat perangkat apa? Tampilan akan disesuaikan.")
    guess = "HP" if ua_looks_mobile() else "Laptop"
    st.caption(f"Terdeteksi: {'HP' if guess == 'HP' else 'laptop / komputer'}")
    with st.container(horizontal=True):
        if st.button("HP", icon=":material/smartphone:", width="stretch", on_click=choose_device, args=("HP",),
                     type="primary" if guess == "HP" else "secondary"):
            st.rerun()
        if st.button("Laptop", icon=":material/laptop:", width="stretch", on_click=choose_device, args=("Laptop",),
                     type="primary" if guess == "Laptop" else "secondary"):
            st.rerun()
    st.caption("Bisa diubah kapan saja lewat tombol di kanan atas. Tema terang/gelap ada di menu ⋮ di kanan atas.")

def inject_css() -> None:
    st.html(CSS)

def stat_grid(items: list[tuple[str, str, str]]) -> None:
    with st.grid(4, min_column_width=150, gap="small"):
        for label, value, note in items:
            st.metric(label, value, note, delta_color="off", delta_arrow="off", border=True)

def chart(fig, alt: str, height: int | None = None, container=None) -> None:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      **({"height": height} if height else {}))
    (container or st).plotly_chart(fig, width="stretch", alt=alt,
                                   config={"displayModeBar": False, "staticPlot": is_mobile()})

def settings_panel(bundle, with_cam: bool = False):
    def body():
        th = st.slider("Threshold confidence otomatis", 0.50, 0.99, bundle.threshold, 0.01, key="threshold",
                       help="Prediksi dengan confidence ≥ threshold diputus otomatis, sisanya ke Customer Service. "
                            "Default 0,90 dari notebook (akurasi klaim otomatis ≥ 95%).")
        cam = st.toggle("Tampilkan Grad-CAM", value=True, key="show_cam",
                        help="Peta panas area foto yang paling memengaruhi keputusan model.") if with_cam else False
        return th, cam

    if is_mobile():
        with st.expander("Pengaturan", icon=":material/tune:"):
            return body()
    with st.sidebar:
        st.subheader(":material/tune: Pengaturan")
        return body()

@st.cache_resource(show_spinner="Memuat model MobileNetV3…")
def get_bundle():
    return load_bundle()

def footer() -> None:
    st.divider()
    st.caption("Tugas Besar Deep Learning · Verifikasi otomatis foto klaim kerusakan paket · "
               "MobileNetV3-Large (transfer learning)", text_alignment="center")
