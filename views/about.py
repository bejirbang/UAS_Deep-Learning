from pathlib import Path

import streamlit as st

from src.business import CLASS_INFO
from src.ui import footer, is_mobile, stat_grid

FIG = Path(__file__).resolve().parent.parent / "assets" / "figures"
OK = ":green[:material/check_circle:]"

TEAM = [
    "Qalam Raydathul Sabani",
    "Christian Toga Raja Pasaribu",
    "Muhammad Rayfan Pashya",
    "Athallah Pandhu Nitisara",
    "Grace Magaretha Sirait",
]

st.title("Tentang", icon=":material/info:")

tabs = st.tabs([":material/target: Masalah", ":material/database: Dataset", ":material/psychology: Metode",
                ":material/warning: Keterbatasan", ":material/groups: Tim"])

with tabs[0]:
    st.subheader("Latar belakang")
    st.markdown(
        "Pembeli yang menerima paket rusak mengajukan klaim dengan mengunggah foto sebagai bukti. Saat ini setiap foto "
        "diperiksa **manual** oleh Customer Service, sehingga prosesnya:\n"
        "- **lambat**: klaim mengantre review,\n"
        "- **mahal**: ±Rp 5.000 per klaim × ±2.000 klaim per hari,\n"
        "- **tidak konsisten**: penilaian berbeda antar petugas.\n\n"
        "Proyek ini membangun model deep learning yang mengklasifikasikan foto klaim ke dalam tiga kelas, lalu "
        "memakainya sebagai **alat triase** (human-in-the-loop): kasus yang jelas diputus otomatis, kasus ragu "
        "diteruskan ke petugas."
    )
    with st.grid(3, min_column_width=200):
        for c, inf in CLASS_INFO.items():
            with st.container(border=True, height="stretch"):
                st.markdown(f"#### :{inf['md']}[{inf['icon']} {c}]")
                st.write(inf["arti"])
                st.caption(f"Tindakan: {inf['aksi']}")
    st.subheader("Target (KPI)")
    st.markdown(
        "| KPI | Target | Hasil |\n|---|---|---|\n"
        f"| Macro F1 test | ≥ 0,80 | **0,801 ± 0,012** {OK} (rata-rata 3 seed) |\n"
        f"| Latensi per foto | < 100 ms | **±7 ms** (GPU), ±15 ms (CPU) {OK} |\n"
        "| Klaim otomatis dengan akurasi ≥ 95% | sebesar mungkin | **37%** |\n"
        f"| Biaya per klaim | < manual (Rp 5.000) | **Rp 4.605** pada threshold 0,94 {OK} |"
    )

with tabs[1]:
    st.subheader("Dataset DATASCAPE 2026")
    stat_grid([
        ("Gambar berlabel", "12.171", "Aman · Rusak · Lainnya"),
        ("Setelah dibersihkan", "11.824", "347 konflik label dibuang"),
        ("Foto lapangan", "3.057", "tanpa label"),
        ("Rasio imbalance", "21 : 1", "Aman : Lainnya"),
    ])
    st.image(str(FIG / "class_distribution.jpg"), caption="Distribusi kelas: sangat tidak seimbang (21:1)",
             width="stretch" if is_mobile() else 520,
             alt="Distribusi kelas: Aman 8.773, Rusak 2.975, Lainnya 423 gambar")
    st.markdown(
        "**Temuan audit data**\n"
        "- **Ketidakseimbangan kelas**: Lainnya hanya 3,5%, sehingga dipakai *class-weighted loss* dan metrik macro F1.\n"
        "- **Near-duplicate**: 3.481 gambar punya kembaran. Split dibuat *group-aware* agar tidak bocor ke test.\n"
        "- **Label bertentangan**: 347 gambar (gambar sama, label beda) dibuang.\n"
        "- **Jalan pintas**: resolusi rendah berkorelasi dengan Lainnya, noise buatan berkorelasi dengan Aman. "
        "Diatasi dengan augmentasi downscale & noise pada semua kelas.\n"
        "- **Label noise**: banyak foto stok hasil scraping yang labelnya keliru."
    )
    with st.expander("Contoh gambar per kelas", icon=":material/image:"):
        for name in ["aman", "rusak", "lainnya"]:
            st.image(str(FIG / f"samples_{name}.jpg"), width="stretch", alt=f"Contoh 12 gambar kelas {name.title()}")
    st.markdown("**Split**: train / val / test = 70 / 15 / 15 (8.481 / 1.636 / 1.707), stratified & group-aware.")

with tabs[2]:
    st.subheader("Tiga pendekatan yang dibandingkan")
    st.markdown(
        "| Kode | Model | Peran |\n|---|---|---|\n"
        "| M0 | Histogram HSV + HOG → Logistic Regression | Baseline non-DL |\n"
        "| M1 | CNN sederhana 4 blok (422 ribu parameter), dilatih dari nol | Arsitektur NN sederhana |\n"
        "| **M2** | **MobileNetV3-Large pretrained ImageNet, transfer learning 2 tahap** | **Model yang di-deploy** |"
    )
    st.subheader("Pipeline model yang di-deploy")
    st.code(
        "Foto → koreksi orientasi EXIF → RGB (transparan → latar putih)\n"
        "     → resize sisi pendek 256 px → resize 255 → center crop 224×224\n"
        "     → normalisasi ImageNet → MobileNetV3-Large → softmax 3 kelas\n"
        "     → keputusan: confidence ≥ threshold → otomatis | < threshold → review CS | Lainnya → minta foto ulang",
        language=None,
    )
    st.markdown(
        "**Training**: tahap 1 hanya melatih head (3 epoch, LR 1e-3), tahap 2 fine-tuning seluruh jaringan "
        "(hingga 15 epoch, LR 3e-4, warmup + cosine). Loss cross-entropy berbobot kelas (akar invers frekuensi) "
        "dengan label smoothing 0,05, optimizer AdamW, early stopping pada macro F1 validasi.\n\n"
        "**Augmentasi**: random resized crop, flip, rotasi ±10°, color jitter, kompresi JPEG acak, ditambah "
        "downscale & noise acak untuk mitigasi jalan pintas."
    )
    with st.expander("Contoh augmentasi", icon=":material/auto_awesome:"):
        st.image(str(FIG / "augmentation.jpg"), width="stretch", alt="Satu foto dengan 14 variasi augmentasi")

with tabs[3]:
    st.subheader("Keterbatasan")
    st.markdown(
        "- **Label noise** cukup besar: sekitar separuh kesalahan model yang paling yakin ternyata label datanya yang salah.\n"
        "- **Domain shift**: banyak data latih berupa foto stok, bukan foto klaim asli dari aplikasi.\n"
        "- **Kelas Lainnya sempit**: hampir semuanya kemasan produk ritel. Foto tidak relevan jenis lain (selfie, "
        "dokumen, screenshot chat) bisa salah diklasifikasikan.\n"
        "- **Rusak → Aman** masih menjadi kesalahan terbanyak (31% foto Rusak di test).\n"
        "- **Matriks biaya** bersifat ilustratif, dan threshold dipilih di test set.\n"
        "- Model belum diuji pada foto klaim asli berlabel dari aplikasi."
    )
    st.subheader("Rekomendasi")
    st.markdown(
        "1. Pakai model sebagai **alat triase** CS, bukan pengganti.\n"
        "2. **Bersihkan label**, investasi paling murah untuk menaikkan performa.\n"
        "3. Catat keputusan CS sebagai label baru lalu lakukan **retraining berkala**.\n"
        "4. **Pantau drift**: distribusi confidence dan proporsi kelas pada foto yang masuk."
    )

with tabs[4]:
    st.subheader("Tim pengembang")
    with st.grid("auto", min_column_width=220):
        for nama in TEAM:
            with st.container(border=True):
                st.markdown(f":material/person: **{nama}**")
    st.caption("Mata kuliah Deep Learning · Tugas Besar")

footer()
