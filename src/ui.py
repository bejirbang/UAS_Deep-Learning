"""Komponen tampilan bersama: mode perangkat (HP / Laptop), CSS, loader model ter-cache, kartu statistik, footer."""
from __future__ import annotations

import streamlit as st

from src.model import load_bundle

DEVICES = {"hp": "📱 HP", "laptop": "💻 Laptop"}

# ---------------------------------------------------------------- CSS dasar (berlaku di semua mode)
BASE_CSS = """
<style>
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1150px; }
h1, h2, h3 { letter-spacing: -0.01em; }

/* hero */
.hero { background: linear-gradient(135deg, #1f5fae 0%, #2a78d6 55%, #4aa3f0 100%); color: #fff;
  border-radius: 20px; padding: 2.4rem 2.4rem; margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(42,120,214,.22); }
.hero h1 { color: #fff; font-size: 2.3rem; line-height: 1.15; margin: 0 0 .6rem 0; padding: 0; }
.hero p  { color: #eaf2fd; font-size: 1.05rem; line-height: 1.55; margin: 0; max-width: 720px; }
.hero .badge { display:inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  padding: .25rem .7rem; border-radius: 999px; font-size: .8rem; margin-bottom: .9rem; }

/* kartu menu: tinggi sama, tombol selalu di bawah */
.menu-card { display:flex; flex-direction:column; gap:.25rem; padding-bottom: 1.1rem; }
.menu-icon  { font-size: 2rem; line-height: 1; margin-bottom: .3rem; }
.menu-title { font-weight: 700; font-size: 1.08rem; }
.menu-desc  { color: #5b6476; font-size: .92rem; line-height: 1.45; min-height: 4.4em; margin-bottom: .7rem; }
[data-testid="stPageLink"] a { background: #2a78d6; border-radius: 10px; justify-content: center; padding: .45rem .8rem; }
[data-testid="stPageLink"] a p, [data-testid="stPageLink"] a span { color: #fff !important; font-weight: 600; }
[data-testid="stPageLink"] a:hover { background: #1f5fae; }

/* grid statistik (pengganti st.metric agar bisa 2x2 di HP & tanpa panah delta) */
.stats { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.75rem; margin:.4rem 0 1rem 0; }
.stat { background:#f3f6fb; border-radius:14px; padding:.85rem 1rem; }
.stat .l { color:#5b6476; font-size:.82rem; }
.stat .v { font-size:1.55rem; font-weight:700; color:#1d2433; line-height:1.25; }
.stat .s { color:#5b6476; font-size:.78rem; }

/* langkah alur */
.steps { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.75rem; }
.step { background:#f3f6fb; border-radius: 14px; padding: 1rem; }
.step .n { display:inline-flex; width:28px; height:28px; border-radius:50%; background:#2a78d6; color:#fff;
  align-items:center; justify-content:center; font-weight:700; font-size:.9rem; margin-bottom:.4rem; }
.step .t { font-weight: 700; margin-bottom: .15rem; }
.step .d { color:#5b6476; font-size:.9rem; }

/* hasil prediksi */
.result-card { border-radius: 16px; padding: 1.1rem 1.3rem; color: #fff; margin-bottom: .8rem; }
.result-card .lbl  { font-size: .8rem; opacity: .9; text-transform: uppercase; letter-spacing: .06em; }
.result-card .cls  { font-size: 2rem; font-weight: 800; line-height: 1.1; margin: .15rem 0; }
.result-card .conf { font-size: .98rem; opacity: .95; }
.decision { border-left: 6px solid; border-radius: 10px; padding: .85rem 1rem; background: #f6f8fc; margin-bottom: .8rem; }
.decision .t { font-weight: 700; font-size: 1.03rem; margin-bottom: .2rem; }
.decision .d { color: #4a5263; font-size: .92rem; }

.team { display:grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap:.75rem; }
.member { background:#f3f6fb; border-radius:14px; padding:.9rem 1rem; display:flex; align-items:center; gap:.75rem; }
.member .av { width:42px; height:42px; border-radius:50%; background:#2a78d6; color:#fff; display:flex;
  align-items:center; justify-content:center; font-weight:700; flex-shrink:0; }
.member .nm { font-weight:600; line-height:1.3; }

.footer { color:#8a92a3; font-size:.82rem; text-align:center; margin-top: 2.5rem; }

/* layar sempit (fallback walau mode Laptop dipilih di HP) */
@media (max-width: 640px) {
  .block-container { padding: 3.6rem .9rem 2.5rem .9rem; }
  .stats, .steps { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .hero { padding: 1.5rem 1.2rem; }
  .hero h1 { font-size: 1.6rem; }
  .menu-desc { height: auto; }
}
</style>
"""

# ---------------------------------------------------------------- CSS tambahan khusus mode HP
MOBILE_CSS = """
<style>
.block-container { max-width: 480px !important; padding-left: .9rem !important; padding-right: .9rem !important; }
h1 { font-size: 1.65rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.12rem !important; }
.hero { padding: 1.5rem 1.2rem; border-radius: 16px; }
.hero h1 { font-size: 1.6rem !important; }
.hero p { font-size: .97rem; }
.stats, .steps { grid-template-columns: repeat(2, minmax(0,1fr)); gap:.6rem; }
.stat { padding:.7rem .8rem; }
.stat .v { font-size:1.3rem; }
.menu-desc { height: auto; }
.result-card .cls { font-size: 1.7rem; }
/* target sentuh besar */
.stButton button, [data-testid="stPageLink"] a, .stDownloadButton button { min-height: 48px; width: 100%; }
[data-testid="stTabs"] button { padding: .55rem .6rem; }
[data-testid="stTabs"] [role="tablist"] { overflow-x: auto; scrollbar-width: none; }
</style>
"""


# ---------------------------------------------------------------- mode perangkat
def ua_looks_mobile() -> bool:
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    return any(k in ua for k in ("Mobi", "Android", "iPhone", "iPad"))


def device_chosen() -> bool:
    return st.session_state.get("device") in DEVICES


def get_device() -> str:
    """Mode tampilan aktif. Urutan: pilihan user di sesi ini → ?mode= di URL → tebakan dari browser."""
    d = st.session_state.get("device")
    if d in DEVICES:
        return d
    q = st.query_params.get("mode")
    if q in DEVICES:
        st.session_state["device"] = q
        return q
    return "hp" if ua_looks_mobile() else "laptop"


def set_device(d: str) -> None:
    st.session_state["device"] = d
    st.query_params["mode"] = d


def is_mobile() -> bool:
    return get_device() == "hp"


def inject_css() -> None:
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if is_mobile():
        st.markdown(MOBILE_CSS, unsafe_allow_html=True)


@st.dialog("Selamat datang 👋", dismissible=False)
def device_dialog():
    st.write("Anda mengakses CekPaket lewat perangkat apa? Tampilan akan disesuaikan.")
    guess = "hp" if ua_looks_mobile() else "laptop"
    st.caption(f"Terdeteksi: {'HP' if guess == 'hp' else 'laptop / komputer'}")
    c1, c2 = st.columns(2)
    if c1.button("📱 HP", width="stretch", type="primary" if guess == "hp" else "secondary"):
        set_device("hp"); st.rerun()
    if c2.button("💻 Laptop", width="stretch", type="primary" if guess == "laptop" else "secondary"):
        set_device("laptop"); st.rerun()
    st.caption("Pilihan ini bisa diubah kapan saja lewat tombol di bagian atas halaman.")


def device_switcher() -> None:
    """Tombol kecil di atas setiap halaman untuk mengganti mode tampilan."""
    cur = get_device()
    _, right = st.columns([3, 1.3]) if not is_mobile() else (None, st.container())
    with right:
        pick = st.segmented_control("Tampilan", list(DEVICES), default=cur, format_func=DEVICES.get,
                                    key="device_switch", label_visibility="collapsed", width="stretch")
    if pick and pick != cur:
        set_device(pick); st.rerun()


# ---------------------------------------------------------------- komponen
def stat_grid(items: list[tuple[str, str, str]]) -> None:
    """items: (label, nilai, keterangan). 4 kolom di laptop, 2x2 di HP."""
    html = "".join(f'<div class="stat"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
                   for l, v, s in items)
    st.markdown(f'<div class="stats">{html}</div>', unsafe_allow_html=True)


def settings_panel(bundle, with_cam: bool = False):
    """Pengaturan threshold (& Grad-CAM). Laptop: di sidebar. HP: di panel yang bisa dibuka di halaman."""
    def body():
        th = st.slider("Threshold confidence otomatis", 0.50, 0.99, bundle.threshold, 0.01, key="threshold",
                       help="Prediksi dengan confidence ≥ threshold diputus otomatis, sisanya ke Customer Service. "
                            "Default 0,90 dari notebook (akurasi klaim otomatis ≥ 95%).")
        cam = st.toggle("Tampilkan Grad-CAM", value=True, key="show_cam",
                        help="Peta panas area foto yang paling memengaruhi keputusan model.") if with_cam else False
        return th, cam

    if is_mobile():
        with st.expander("⚙️ Pengaturan", expanded=False):
            return body()
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        return body()


@st.cache_resource(show_spinner="Memuat model MobileNetV3…")
def get_bundle():
    return load_bundle()


def footer() -> None:
    st.markdown(
        '<div class="footer">Tugas Besar Deep Learning · Verifikasi Otomatis Foto Klaim Kerusakan Paket · '
        "MobileNetV3-Large (transfer learning)</div>",
        unsafe_allow_html=True,
    )
