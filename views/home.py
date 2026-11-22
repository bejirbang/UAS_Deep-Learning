"""Landing page: sambutan, menu utama, cara kerja sistem."""
import streamlit as st

from src.ui import device_switcher, footer, is_mobile, stat_grid

PAGES = st.session_state["PAGES"]
mobile = is_mobile()

device_switcher()

st.markdown(
    """
<div class="hero">
  <span class="badge">📦 Deep Learning · Klasifikasi Gambar</span>
  <h1>Selamat datang di CekPaket</h1>
  <p>Sistem verifikasi otomatis foto klaim paket rusak untuk e-commerce. Unggah foto paket yang diterima,
  dan model <b>MobileNetV3</b> akan menilai apakah paket <b>Aman</b>, <b>Rusak</b>, atau fotonya
  <b>tidak relevan</b>, lengkap dengan rekomendasi tindak lanjut klaim.</p>
</div>
""",
    unsafe_allow_html=True,
)

if mobile:
    st.page_link(PAGES["predict"], label="Mulai cek paket sekarang", icon="📷", width="stretch")
else:
    c1, _ = st.columns([1, 2.6])
    c1.page_link(PAGES["predict"], label="Mulai cek paket sekarang", icon="🔍", width="stretch")

st.subheader("Menu")
MENU = [
    ("predict", "🔍", "Cek Foto Paket", "Unggah atau foto langsung dengan kamera, lalu lihat hasil dan alasan model."),
    ("batch", "🗂️", "Cek Banyak Foto", "Proses banyak foto klaim sekaligus dan unduh hasil triasenya."),
    ("performance", "📊", "Performa Model", "Hasil evaluasi: perbandingan model, confusion matrix, dampak bisnis."),
    ("about", "ℹ️", "Tentang", "Latar belakang, dataset, metode, keterbatasan, dan tim pengembang."),
]

if mobile:
    # HP: daftar ringkas, satu baris per menu
    for key, icon, title, desc in MENU:
        with st.container(border=True):
            st.markdown(f'<div class="menu-title">{icon} {title}</div><div class="menu-desc">{desc}</div>',
                        unsafe_allow_html=True)
            st.page_link(PAGES[key], label=f"Buka {title}", width="stretch")
else:
    cols = st.columns(4)
    for col, (key, icon, title, desc) in zip(cols, MENU):
        with col.container(border=True):
            st.markdown(f'<div class="menu-card"><div class="menu-icon">{icon}</div><div class="menu-title">{title}</div>'
                        f'<div class="menu-desc">{desc}</div></div>', unsafe_allow_html=True)
            st.page_link(PAGES[key], label="Buka", width="stretch")

st.subheader("Cara kerja")
steps = [
    ("Unggah foto", "Pembeli mengunggah foto paket sebagai bukti klaim."),
    ("Analisis model", "MobileNetV3 menghitung probabilitas 3 kelas."),
    ("Keputusan", "Confidence tinggi diputus otomatis, rendah ke CS."),
    ("Tindak lanjut", "Refund / kirim ulang, review CS, atau foto ulang."),
]
st.markdown('<div class="steps">' + "".join(
    f'<div class="step"><div class="n">{i + 1}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
    for i, (t, d) in enumerate(steps)) + "</div>", unsafe_allow_html=True)

st.write("")
st.subheader("Sekilas performa")
stat_grid([
    ("Macro F1 (test)", "0,801", "± 0,012 · rata-rata 3 seed"),
    ("Akurasi (test)", "85,7%", "± 0,6%"),
    ("Klaim otomatis", "37%", "dengan akurasi ≥ 95%"),
    ("Waktu inferensi", "±7 ms", "per foto (GPU)"),
])

st.info("Model ini dirancang sebagai **alat bantu triase** untuk Customer Service, bukan pengganti penuh. "
        "Klaim dengan confidence rendah tetap diperiksa petugas.", icon="💡")

footer()
