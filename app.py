"""Entry point Streamlit: `streamlit run app.py`."""
import streamlit as st

from src.ui import device_chosen, device_dialog, get_device, inject_css

mobile = get_device() == "hp"
st.set_page_config(
    page_title="CekPaket · Verifikasi Klaim Paket Rusak",
    page_icon="📦",
    layout="centered" if mobile else "wide",
    initial_sidebar_state="collapsed" if mobile else "expanded",
)
inject_css()

PAGES = {
    "home": st.Page("views/home.py", title="Beranda", icon="🏠", default=True),
    "predict": st.Page("views/predict.py", title="Cek Foto Paket", icon="🔍", url_path="cek"),
    "batch": st.Page("views/batch.py", title="Cek Banyak Foto", icon="🗂️", url_path="batch"),
    "performance": st.Page("views/performance.py", title="Performa Model", icon="📊", url_path="performa"),
    "about": st.Page("views/about.py", title="Tentang", icon="ℹ️", url_path="tentang"),
}
st.session_state["PAGES"] = PAGES

nav = st.navigation(list(PAGES.values()), position="top" if mobile else "sidebar")

# kunjungan pertama: tanyakan perangkat (kecuali sudah ada ?mode= di URL)
if not device_chosen() and st.query_params.get("mode") not in ("hp", "laptop"):
    device_dialog()

nav.run()
