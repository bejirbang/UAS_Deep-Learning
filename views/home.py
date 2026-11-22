import streamlit as st

from src.ui import footer, is_mobile, stat_grid

PAGES = st.session_state["PAGES"]
mobile = is_mobile()

st.html(
    """
<div class="hero">
  <h1>Selamat datang di CekPaket</h1>
  <p>Sistem verifikasi otomatis foto klaim paket rusak untuk e-commerce. Unggah foto paket yang diterima,
  dan model <b>MobileNetV3</b> akan menilai apakah paket <b>Aman</b>, <b>Rusak</b>, atau fotonya
  <b>tidak relevan</b>, lengkap dengan rekomendasi tindak lanjut klaim.</p>
</div>
"""
)

if st.button("Mulai cek paket sekarang", icon=":material/photo_camera:" if mobile else ":material/search:",
             type="primary", width="stretch" if mobile else "content"):
    st.switch_page(PAGES["predict"])

st.subheader("Menu")
MENU = [
    ("predict", ":material/search:", "Cek foto paket",
     "Unggah atau foto langsung dengan kamera, lalu lihat hasil dan alasan model."),
    ("batch", ":material/photo_library:", "Cek banyak foto",
     "Proses banyak foto klaim sekaligus dan unduh hasil triasenya."),
    ("performance", ":material/monitoring:", "Performa model",
     "Hasil evaluasi: perbandingan model, confusion matrix, dampak bisnis."),
    ("about", ":material/info:", "Tentang",
     "Latar belakang, dataset, metode, keterbatasan, dan tim pengembang."),
]
with st.grid(4, min_column_width=200 if not mobile else 260, gap="small"):
    for key, icon, title, desc in MENU:
        with st.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            if st.button("Buka", key=f"open_{key}", icon=":material/arrow_forward:", width="stretch"):
                st.switch_page(PAGES[key])

st.subheader("Cara kerja")
STEPS = [
    ("Unggah foto", "Pembeli mengunggah foto paket sebagai bukti klaim."),
    ("Analisis model", "MobileNetV3 menghitung probabilitas 3 kelas."),
    ("Keputusan", "Confidence tinggi diputus otomatis, rendah ke CS."),
    ("Tindak lanjut", "Refund / kirim ulang, review CS, atau foto ulang."),
]
with st.grid(4, min_column_width=150, gap="small"):
    for i, (t, d) in enumerate(STEPS, start=1):
        with st.container(border=True, height="stretch"):
            st.html(f'<span class="num">{i}</span>')
            st.markdown(f"**{t}**")
            st.caption(d)

st.subheader("Sekilas performa")
stat_grid([
    ("Macro F1 (test)", "0,801", "± 0,012 (3 seed)"),
    ("Akurasi (test)", "85,7%", "± 0,6%"),
    ("Klaim otomatis", "37%", "akurasi ≥ 95%"),
    ("Waktu inferensi", "±7 ms", "per foto (GPU)"),
])

st.info("Klaim dengan confidence rendah tetap diperiksa petugas.", title="Alat bantu triase, bukan pengganti CS",
        icon=":material/lightbulb:")

footer()
