"""Logika keputusan human-in-the-loop (desain notebook 2.3 dan rekomendasi 8.3)."""
from __future__ import annotations

from dataclasses import dataclass

CLASS_INFO = {
    "Aman": dict(emoji="📦", color="#1baf7a", arti="Paket terlihat utuh",
                 aksi="Klaim kemungkinan tidak valid"),
    "Rusak": dict(emoji="💥", color="#e34948", arti="Paket penyok, sobek, basah, atau hancur",
                  aksi="Klaim valid: refund / kirim ulang"),
    "Lainnya": dict(emoji="🖼️", color="#eda100", arti="Foto bukan paket kiriman (produk, dokumen, gambar lain)",
                    aksi="Minta pembeli mengunggah ulang foto"),
}

# Matriks biaya ilustratif dari notebook 1.4 / 7.3 (baris = sebenarnya, kolom = prediksi), urutan Aman, Rusak, Lainnya
COST_MATRIX = [
    [0, 100_000, 5_000],
    [150_000, 0, 10_000],
    [25_000, 100_000, 0],
]
MANUAL_REVIEW_COST = 5_000
CLAIMS_PER_DAY = 2_000


@dataclass
class Decision:
    route: str        # "auto" | "cs" | "reupload"
    title: str
    detail: str
    priority: str     # "tinggi" | "normal" | "-"
    color: str
    icon: str


def decide(label: str, confidence: float, threshold: float) -> Decision:
    if label == "Lainnya":
        return Decision("reupload", "Minta foto ulang",
                        "Foto tidak menunjukkan paket kiriman. Sistem otomatis meminta pembeli mengunggah foto paket yang jelas.",
                        "-", "#eda100", "🔁")
    if confidence >= threshold:
        if label == "Rusak":
            return Decision("auto", "Diputus otomatis: klaim disetujui",
                            f"Model yakin paket rusak (confidence ≥ {threshold:.2f}). Klaim diteruskan ke proses refund / kirim ulang.",
                            "-", "#1baf7a", "✅")
        return Decision("auto", "Diputus otomatis: paket terlihat utuh",
                        f"Model yakin paket utuh (confidence ≥ {threshold:.2f}). Klaim ditandai tidak terbukti rusak.",
                        "-", "#2a78d6", "✅")
    prio = "tinggi" if label == "Rusak" else "normal"
    return Decision("cs", f"Review Customer Service (prioritas {prio})",
                    f"Confidence di bawah threshold {threshold:.2f}, jadi keputusan diserahkan ke petugas CS."
                    + (" Didahulukan karena kemungkinan klaim sah." if prio == "tinggi" else ""),
                    prio, "#6250d6", "👤")
