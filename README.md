# CekPaket: Verifikasi Otomatis Foto Klaim Paket Rusak

Aplikasi web Streamlit untuk Tugas Besar Deep Learning. Pengguna mengunggah foto paket, lalu model
**MobileNetV3-Large** (transfer learning) mengklasifikasikannya ke dalam tiga kelas:

| Kelas | Arti | Tindakan |
|---|---|---|
| Aman | paket utuh | klaim kemungkinan tidak valid |
| Rusak | penyok, sobek, basah, hancur | refund / kirim ulang |
| Lainnya | foto bukan paket | minta foto ulang |

Keputusan mengikuti desain *human-in-the-loop*: confidence ≥ threshold (default 0,90) diputus otomatis,
sisanya diteruskan ke Customer Service (Rusak diprioritaskan).

Performa di test set: macro F1 **0,801 ± 0,012** (3 seed), akurasi 0,857. Detail eksperimen ada di `TUBES_DL.html`.

## Fitur

- **Beranda**: sambutan, menu, cara kerja, ringkasan performa
- **Cek Foto Paket**: unggah foto / kamera HP / foto contoh, lalu tampil prediksi, probabilitas, keputusan,
  peringatan kualitas foto (resolusi rendah, noise berat), dan Grad-CAM
- **Cek Banyak Foto**: hingga 100 foto sekaligus, ringkasan triase, filter, unduh CSV
- **Performa Model**: perbandingan model, confusion matrix, simulasi biaya, ablasi, ketahanan JPEG, Grad-CAM
- **Tentang**: masalah bisnis, dataset, metode, keterbatasan, tim

Tampilan responsif: kolom otomatis bertumpuk di HP dan tombol dibuat lebih besar agar mudah disentuh.

## Struktur

```
app.py                  entry point + navigasi
views/                  halaman (home, predict, batch, performance, about)
src/model.py            load checkpoint, preprocessing (sama persis dengan notebook), prediksi
src/gradcam.py          Grad-CAM
src/business.py         logika keputusan & matriks biaya
src/metrics_data.py     angka evaluasi dari notebook
src/ui.py               CSS responsif, cache model
models/                 model_terbaik_m2_mobilenetv3.pt (17 MB)
assets/figures/         grafik dari notebook
assets/samples/         foto contoh (opsional, muncul di tab "Foto contoh")
scripts/pack_model.py   mengemas folder checkpoint hasil ekstrak menjadi .pt
tests/test_app.py       tes model + semua halaman
```

## Menjalankan di lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tes: `python tests/test_app.py`

Jika `models/model_terbaik_m2_mobilenetv3.pt` belum ada tetapi folder `model_terbaik_m2_mobilenetv3/`
(checkpoint yang sudah diekstrak) ada, jalankan `python scripts/pack_model.py`.

## Foto contoh

Taruh file `.jpg` / `.png` di `assets/samples/`. Foto-foto ini otomatis muncul di tab **Foto contoh**
pada halaman Cek Foto Paket. Beri nama yang jelas, misalnya `rusak_01.jpg`.

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub (pastikan `models/model_terbaik_m2_mobilenetv3.pt` ikut ter-commit).
2. Buka https://share.streamlit.io, lalu pilih **Create app → Deploy a public app from GitHub**.
3. Pilih repo dan branch, isi **Main file path**: `app.py`.
4. Di **Advanced settings**, pilih Python **3.12**.
5. Klik **Deploy**. Build pertama memakan waktu beberapa menit (install PyTorch CPU).

`requirements.txt` memakai index PyTorch CPU agar ukuran instalasi kecil (cloud tidak punya GPU).
