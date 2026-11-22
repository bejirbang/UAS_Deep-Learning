import streamlit as st

from pathlib import Path

from src.ui import device_bar, device_dialog, inject_css

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="CekPaket · Verifikasi klaim paket rusak", page_icon=str(ASSETS / "logo_icon.png"))
st.logo(str(ASSETS / "logo.png"), icon_image=str(ASSETS / "logo_icon.png"), size="large")

PAGES = {
    "home": st.Page("views/home.py", title="Beranda", icon=":material/home:", default=True),
    "predict": st.Page("views/predict.py", title="Cek foto paket", icon=":material/search:", url_path="cek"),
    "batch": st.Page("views/batch.py", title="Cek banyak foto", icon=":material/photo_library:", url_path="batch"),
    "performance": st.Page("views/performance.py", title="Performa model", icon=":material/monitoring:",
                           url_path="performa"),
    "about": st.Page("views/about.py", title="Tentang", icon=":material/info:", url_path="tentang"),
}
st.session_state["PAGES"] = PAGES

nav = st.navigation(list(PAGES.values()))
inject_css()

if "mode" in st.query_params:
    st.session_state["device_confirmed"] = True

mobile = device_bar() == "HP"
st.set_page_config(layout="centered" if mobile else "wide",
                   initial_sidebar_state="collapsed" if mobile else "expanded")

if not st.session_state.get("device_confirmed"):
    device_dialog()

nav.run()
