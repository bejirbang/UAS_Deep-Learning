from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import metrics_data as md
from src.business import CLAIMS_PER_DAY, COST_MATRIX, MANUAL_REVIEW_COST
from src.ui import chart, footer, is_mobile, stat_grid

FIG = Path(__file__).resolve().parent.parent / "assets" / "figures"
CLASSES = ["Aman", "Rusak", "Lainnya"]
mobile = is_mobile()


def rp(x: float) -> str:
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def fig_img(name: str, caption: str):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch", alt=caption)


def table(df: pd.DataFrame, alt: str, hide_index: bool = True):
    st.table(df, hide_index=hide_index, alt=alt)


st.title("Performa model", icon=":material/monitoring:")
st.caption("Semua angka berasal dari evaluasi di **test set (1.707 gambar)** yang tidak pernah dipakai saat training "
           "maupun tuning. Model yang di-deploy: MobileNetV3-Large seed 42 (macro F1 test 0,814).")

stat_grid([
    ("Macro F1", "0,801", "± 0,012 · target ≥ 0,80"),
    ("Akurasi", "85,7%", "± 0,6%"),
    ("Recall Rusak", "67,7%", "kelas terpenting"),
    ("Latensi", "±7 ms", "target < 100 ms"),
])

TAB_ICONS = [":material/trophy:", ":material/target:", ":material/payments:", ":material/science:",
             ":material/local_fire_department:"]
TAB_NAMES = (["Model", "Kelas", "Bisnis", "Ablasi", "Interpretasi"] if mobile else
             ["Perbandingan", "Per kelas", "Dampak bisnis", "Eksperimen", "Interpretasi"])
tabs = st.tabs([f"{i} {n}" for i, n in zip(TAB_ICONS, TAB_NAMES)])

with tabs[0]:
    st.subheader("Baseline vs CNN sederhana vs transfer learning")
    t4 = md.TABLE4[["Model", "Macro F1", "Akurasi"]].assign(Model=["M0 Baseline", "M1 CNN", "M2 MobileNetV3 ✅"]) if mobile else md.TABLE4
    table(t4, alt="Perbandingan M0, M1, dan M2 di test set; M2 unggul di semua metrik")
    b = md.MACRO_F1_BARS
    fig = go.Figure(go.Bar(x=b.model, y=b["mean"], error_y=dict(type="data", array=b["std"]),
                           marker_color=["#8d8d8d", "#eb6834", "#2a78d6"],
                           text=[f"{v:.3f}" for v in b["mean"]], textposition="inside", insidetextanchor="start",
                           textfont=dict(color="white", size=14)))
    fig.add_hline(y=0.80, line_dash="dot", line_color="#1baf7a", annotation_text="target KPI 0,80")
    fig.update_layout(height=280 if mobile else 340, yaxis=dict(range=[0, 1], title="Macro F1 (test)"),
                      margin=dict(l=0, r=0, t=20, b=0))
    chart(fig, alt="Macro F1 test: M0 0,491, M1 0,569, M2 0,801; hanya M2 melewati target 0,80")
    st.markdown(
        "- **Transfer learning adalah kunci**: M2 unggul +0,23 macro F1 dari CNN sederhana, jauh di atas variasi antar-seed.\n"
        "- Seed terburuk M2 (0,791) masih jauh di atas seed terbaik M1 (0,606).\n"
        "- Harga akurasi: 10× parameter dan ±8× waktu inferensi, tetapi ±7 ms per foto masih jauh di bawah target 100 ms."
    )
    with st.expander("Kurva training M2 (rerata ± std 3 seed)"):
        fig_img("seed_curves_m2.jpg", "Garis putus-putus = akhir tahap 1 (hanya head) → mulai fine-tuning penuh")

with tabs[1]:
    c1, c2 = (st.container(), st.container()) if mobile else st.columns([1, 1.1], gap="large")
    with c1:
        st.subheader("Classification report")
        table(md.CLASS_REPORT.assign(**{c: md.CLASS_REPORT[c].map("{:.3f}".format)
                                        for c in ["Precision", "Recall", "F1"]}),
              alt="Precision, recall, dan F1 per kelas untuk model M2")
        st.markdown(
            "- **Aman** paling mudah (kelas mayoritas).\n"
            "- **Lainnya** jarang tertukar (recall 0,87) meski datanya hanya 3,5%.\n"
            "- **Rusak → Aman** adalah kesalahan dominan (127 dari 409 foto Rusak) sekaligus yang paling mahal, "
            "karena klaim sah ikut ditolak."
        )
    with c2:
        st.subheader("Confusion matrix")
        normalize = st.toggle("Tampilkan persentase per baris", value=True)
        cm = md.CONFUSION
        z = [[v / sum(r) for v in r] for r in cm] if normalize else cm
        text = [[f"{z[i][j]:.0%}<br>({cm[i][j]})" if normalize else str(cm[i][j]) for j in range(3)] for i in range(3)]
        fig = go.Figure(go.Heatmap(z=z, x=CLASSES, y=CLASSES, text=text, texttemplate="%{text}",
                                   colorscale="Blues", showscale=False, hoverinfo="skip"))
        fig.update_layout(height=300 if mobile else 360, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(title="Prediksi", side="bottom"),
                          yaxis=dict(title="Label sebenarnya", autorange="reversed"))
        chart(fig, alt="Confusion matrix M2: kesalahan terbanyak Rusak diprediksi Aman (127 foto)")

with tabs[2]:
    st.subheader("Human-in-the-loop: berapa klaim yang bisa diotomasi?")
    fig_img("automation_curve.jpg",
            "Makin tinggi threshold: makin sedikit klaim otomatis, tetapi makin akurat. Garis merah = biaya minimum.")

    st.subheader(":material/calculate: Simulasi biaya")
    claims = st.number_input("Jumlah klaim per hari", 100, 100_000, CLAIMS_PER_DAY, 100)
    op = md.OPERATING_POINTS.copy()
    op["Klaim otomatis"] = op.coverage.map(lambda v: f"{v:.1%}")
    op["Akurasi otomatis"] = op.akurasi_otomatis.map(lambda v: "–" if v != v or v is None else f"{v:.1%}")
    op["Beban CS / hari"] = (claims * (1 - op.coverage)).round().astype(int).map(lambda v: f"{v:,}".replace(",", "."))
    op["Biaya / klaim"] = op.biaya.map(rp)
    op["Biaya / bulan"] = (op.biaya * claims * 30).map(rp)
    op_cols = (["skenario", "Klaim otomatis", "Biaya / bulan"] if mobile else
               ["skenario", "threshold", "Klaim otomatis", "Akurasi otomatis", "Beban CS / hari", "Biaya / klaim", "Biaya / bulan"])
    if mobile:
        op["skenario"] = ["Manual 100%", "Otomasi penuh", "Threshold 0,90", "Threshold 0,94"]
    table(op[op_cols].rename(columns={"skenario": "Skenario", "threshold": "Threshold"}),
          alt="Simulasi biaya per skenario otomasi")
    st.markdown(
        "- Otomasi penuh justru **lebih mahal** dari manual, karena satu kesalahan Rusak → Aman (Rp 150.000) "
        "setara 30× biaya review CS.\n"
        "- **Threshold 0,90** (dipakai di demo): 37% klaim diputus otomatis dengan akurasi 95%, beban CS turun "
        "dari 2.000 menjadi ±1.256 klaim/hari.\n"
        "- **Threshold 0,94**: biaya per klaim terendah (hemat ±8% dari manual)."
    )
    with st.expander("Asumsi matriks biaya (ilustratif)"):
        cmx = pd.DataFrame(COST_MATRIX, index=[f"Sebenarnya {c}" for c in CLASSES],
                           columns=[f"Prediksi {c}" for c in CLASSES]).map(rp)
        table(cmx, alt="Matriks biaya kesalahan ilustratif", hide_index=False)
        st.caption(f"Review manual oleh CS: {rp(MANUAL_REVIEW_COST)} per klaim.")

with tabs[3]:
    st.subheader("Studi ablasi: apa yang membuat M2 unggul?")
    a = md.ABLATION
    fig = px.bar(a.iloc[::-1], x="Macro F1", y="Eksperimen", orientation="h", text="Macro F1",
                 color=["#e34948"] * 4 + ["#2a78d6"], color_discrete_map="identity")
    fig.update_traces(texttemplate="%{x:.3f}")
    fig.update_layout(height=280 if mobile else 320, xaxis=dict(range=[0, 1]), yaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
    chart(fig, alt="Ablasi: tanpa pretrained turun ke 0,566, tanpa fine-tuning 0,681")
    st.markdown(
        "- **Bobot pretrained** paling penting (−0,25). MobileNetV3 yang dilatih dari nol hanya setara CNN sederhana.\n"
        "- **Fine-tuning** backbone menyumbang +0,13.\n"
        "- Tanpa mitigasi jalan pintas skor naik tipis (+0,01), tetapi foto non-Aman ber-noise jauh lebih sering "
        "ditebak Aman (70% vs 47%). Karena itu model dengan mitigasi tetap dipilih."
    )
    st.subheader("Studi arsitektur (CNN sederhana)")
    fig_img("arch_activation.jpg", "Fungsi aktivasi: LeakyReLU dan SiLU unggul ±0,05 dari ReLU")
    with st.expander("Papan skor seluruh eksperimen"):
        fig_img("scoreboard.jpg", "Semua eksperimen di test set, diurutkan berdasarkan macro F1")

with tabs[4]:
    st.subheader("Ketahanan terhadap kompresi JPEG")
    j = md.JPEG
    fig = go.Figure()
    fig.add_scatter(x=j.kualitas, y=j.m2, mode="lines+markers", name="M2 MobileNetV3", line_color="#2a78d6")
    fig.add_scatter(x=j.kualitas, y=j.m1, mode="lines+markers", name="M1 CNN sederhana", line_color="#eb6834")
    fig.update_layout(height=300, yaxis=dict(range=[0, 1], title="Macro F1"), xaxis_title="Kualitas JPEG",
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12))
    chart(fig, alt="Macro F1 M2 stabil di sekitar 0,80 hingga kualitas JPEG 30, turun ke 0,79 di kualitas 15")
    st.caption("Stabil sampai kualitas 30. Di kualitas 15 hanya turun 2,4 poin, jadi aman untuk foto yang dikompres aplikasi chat.")

    st.subheader("Grad-CAM")
    fig_img("gradcam_correct.jpg", "Prediksi benar: model fokus pada badan kardus")
    fig_img("gradcam_errors.jpg", "Kesalahan dengan confidence tertinggi: sebagian besar akibat label dataset yang keliru")

footer()
