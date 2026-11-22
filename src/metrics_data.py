import pandas as pd

TABLE4 = pd.DataFrame([
    dict(Model="M0 Baseline non-DL (HOG + LogReg)", **{"Macro F1": "0,491", "Akurasi": "0,647", "Precision": "0,474",
         "Recall": "0,523", "F1 Rusak": "0,418", "Param (juta)": "0,005", "Inferensi (ms)": "–"}),
    dict(Model="M1 CNN sederhana (dari nol)", **{"Macro F1": "0,569 ± 0,032", "Akurasi": "0,684 ± 0,023",
         "Precision": "0,556 ± 0,030", "Recall": "0,610 ± 0,055", "F1 Rusak": "0,496 ± 0,023",
         "Param (juta)": "0,42", "Inferensi (ms)": "0,8"}),
    dict(Model="M2 MobileNetV3 (transfer learning) ✅", **{"Macro F1": "0,801 ± 0,012", "Akurasi": "0,857 ± 0,006",
         "Precision": "0,799 ± 0,010", "Recall": "0,808 ± 0,013", "F1 Rusak": "0,710 ± 0,010",
         "Param (juta)": "4,21", "Inferensi (ms)": "7,0"}),
])

MACRO_F1_BARS = pd.DataFrame(dict(
    model=["M0 Baseline", "M1 CNN sederhana", "M2 MobileNetV3"],
    mean=[0.491, 0.569, 0.801], std=[0.0, 0.032, 0.012],
))

CLASS_REPORT = pd.DataFrame({
    "Kelas": ["Aman", "Rusak", "Lainnya"],
    "Precision": [0.8952, 0.7694, 0.7647],
    "Recall": [0.9249, 0.6773, 0.8667],
    "F1": [0.9098, 0.7204, 0.8125],
    "Jumlah (test)": [1238, 409, 60],
})

CONFUSION = [
    [1145, 82, 11],
    [127, 277, 5],
    [7, 1, 52],
]

ABLATION = pd.DataFrame([
    ("M2 lengkap", 0.8142, 0.0),
    ("Tanpa augmentasi", 0.8052, -0.0091),
    ("Tanpa mitigasi jalan pintas", 0.8270, +0.0128),
    ("Tanpa fine-tuning", 0.6808, -0.1335),
    ("Tanpa pretrained (dari nol)", 0.5655, -0.2487),
], columns=["Eksperimen", "Macro F1", "Δ vs M2"])

JPEG = pd.DataFrame(dict(kualitas=["asli", "90", "70", "50", "30", "15"],
                         m2=[0.8142, 0.8143, 0.8041, 0.8040, 0.8132, 0.7904],
                         m1=[0.5478, 0.5453, 0.5463, 0.5460, 0.5478, 0.5385]))

OPERATING_POINTS = pd.DataFrame([
    dict(skenario="Manual 100% (kondisi sekarang)", threshold="–", coverage=0.0, akurasi_otomatis=None, biaya=5_000),
    dict(skenario="Otomasi penuh (semua diputus model)", threshold="0,00", coverage=1.0, akurasi_otomatis=0.8635, biaya=16_186),
    dict(skenario="Threshold 0,90 (akurasi otomatis ≥ 95%)", threshold="0,90", coverage=0.372, akurasi_otomatis=0.953, biaya=5_269),
    dict(skenario="Threshold 0,94 (biaya minimum)", threshold="0,94", coverage=0.132, akurasi_otomatis=0.9822, biaya=4_605),
])
