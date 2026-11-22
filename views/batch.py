import base64
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.business import CLASS_INFO, CLAIMS_PER_DAY, decide
from src.model import predict_tensors, prepare
from src.ui import chart, footer, get_bundle, is_mobile, settings_panel, stat_grid

MAX_FILES = 100
IMG_EXT = ["jpg", "jpeg", "png", "webp", "bmp"]
ROUTE_LABEL = {"auto": "Otomatis", "cs": "Review CS", "reupload": "Minta foto ulang"}
ROUTE_COLOR = {"Otomatis": "#1baf7a", "Review CS": "#6250d6", "Minta foto ulang": "#eda100"}

bundle = get_bundle()
mobile = is_mobile()

st.title("Cek banyak foto", icon=":material/photo_library:")
st.caption(f"Simulasi antrean klaim: unggah hingga {MAX_FILES} foto sekaligus, lalu lihat berapa yang bisa diputus "
           "otomatis dan berapa yang perlu diperiksa CS.")
threshold, _ = settings_panel(bundle)

files = st.file_uploader("Pilih beberapa foto paket", type=IMG_EXT, accept_multiple_files=True, key="batch_upload")
if not files:
    st.info("Unggah beberapa foto untuk memulai.", icon=":material/upload:")
    footer()
    st.stop()
if len(files) > MAX_FILES:
    st.warning(f"Hanya {MAX_FILES} foto pertama yang diproses.", icon=":material/warning:")
    files = files[:MAX_FILES]


def thumb_uri(img, size=160):
    im = img.copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False, max_entries=500)
def analyze(name: str, data: bytes) -> dict:
    p = prepare(bundle, data)
    pr = predict_tensors(bundle, [p.tensor])[0]
    return dict(foto=thumb_uri(p.original), nama_file=name, lebar=p.width, tinggi=p.height,
                resolusi_rendah=p.low_res, noise_berat=p.heavy_noise,
                prediksi=pr.label, confidence=pr.confidence, **{f"p_{c}": v for c, v in pr.probs.items()})


rows, failed = [], []
bar = st.progress(0.0, text="Memproses foto…")
for i, f in enumerate(files):
    try:
        rows.append(analyze(f.name, f.getvalue()))
    except Exception as exc:
        failed.append(f"{f.name}: {exc}")
    bar.progress((i + 1) / len(files), text=f"Memproses foto {i + 1}/{len(files)}")
bar.empty()
df = pd.DataFrame(rows)
for msg in failed:
    st.error(msg, title="Gagal dibuka", icon=":material/broken_image:")
if df.empty:
    st.stop()

decs = [decide(l, c, threshold) for l, c in zip(df.prediksi, df.confidence)]
df["keputusan"] = [ROUTE_LABEL[d.route] for d in decs]
df["prioritas_cs"] = [d.priority for d in decs]


def ribuan(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


n = len(df)
n_auto = int((df.keputusan == "Otomatis").sum())
n_cs = int((df.keputusan == "Review CS").sum())
n_re = int((df.keputusan == "Minta foto ulang").sum())
stat_grid([
    ("Total foto", str(n), "diproses model"),
    ("Diputus otomatis", str(n_auto), f"{n_auto / n:.0%} dari total"),
    ("Ke Customer Service", str(n_cs), f"{int((df.prioritas_cs == 'tinggi').sum())} prioritas tinggi"),
    ("Minta foto ulang", str(n_re), f"{n_re / n:.0%} dari total"),
])

ch = 240 if mobile else 300
cnt = df.prediksi.value_counts().reindex(bundle.classes, fill_value=0).reset_index()
cnt.columns = ["kelas", "jumlah"]
fig_bar = px.bar(cnt, x="kelas", y="jumlah", color="kelas", text="jumlah", title="Distribusi prediksi",
                 color_discrete_map={c: CLASS_INFO[c]["color"] for c in bundle.classes})
fig_bar.update_layout(showlegend=False, height=ch, margin=dict(l=0, r=0, t=40, b=0), xaxis_title=None,
                      yaxis=dict(title=None, dtick=max(1, int(cnt.jumlah.max() // 5) or 1), rangemode="tozero"))
route = df.keputusan.value_counts().reset_index()
route.columns = ["keputusan", "jumlah"]
fig_pie = px.pie(route, names="keputusan", values="jumlah", hole=.55, title="Rute keputusan",
                 color="keputusan", color_discrete_map=ROUTE_COLOR)
fig_pie.update_traces(textinfo="value+percent", textfont=dict(color="white", size=13),
                      insidetextorientation="horizontal")
fig_pie.update_layout(height=ch, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.05))
alt_bar = "Distribusi prediksi: " + ", ".join(f"{k} {v}" for k, v in zip(cnt.kelas, cnt.jumlah))
alt_pie = "Rute keputusan: " + ", ".join(f"{k} {v}" for k, v in zip(route.keputusan, route.jumlah))
if mobile:
    chart(fig_pie, alt=alt_pie)
    chart(fig_bar, alt=alt_bar)
else:
    c1, c2 = st.columns(2)
    chart(fig_bar, alt=alt_bar, container=c1)
    chart(fig_pie, alt=alt_pie, container=c2)

st.caption(f"Jika proporsi ini berlaku untuk {ribuan(CLAIMS_PER_DAY)} klaim/hari, beban CS menjadi sekitar "
           f"**{ribuan(CLAIMS_PER_DAY * n_cs / n)} klaim/hari**.")

st.subheader("Hasil per foto")
with st.container(horizontal=True, vertical_alignment="bottom"):
    filt_cls = st.pills("Prediksi", bundle.classes, selection_mode="multi", default=bundle.classes, key="filt_cls")
    filt_route = st.pills("Keputusan", list(ROUTE_LABEL.values()), selection_mode="multi",
                          default=list(ROUTE_LABEL.values()), key="filt_route")
view = df[df.prediksi.isin(filt_cls) & df.keputusan.isin(filt_route)].sort_values("confidence")

if mobile:
    view = view.assign(hasil=view.prediksi + " · " + view.keputusan)
cols = (["foto", "hasil", "confidence"] if mobile else
        ["foto", "nama_file", "prediksi", "confidence", "keputusan", "prioritas_cs", "resolusi_rendah", "noise_berat"])
if view.empty:
    st.info("Tidak ada foto yang cocok dengan filter.", icon=":material/filter_alt_off:")
else:
    st.dataframe(
        view[cols], hide_index=True, row_height=64 if mobile else 72,
        alt=f"Hasil prediksi {len(view)} foto: prediksi, confidence, dan keputusan",
        column_config={
            "foto": st.column_config.ImageColumn("Foto", width="small"),
            "nama_file": "Nama file",
            "hasil": st.column_config.TextColumn("Hasil", width="medium"),
            "prediksi": "Prediksi",
            "confidence": (st.column_config.NumberColumn("Conf.", format="percent", width="small") if mobile else
                           st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent")),
            "keputusan": "Keputusan",
            "prioritas_cs": "Prioritas CS",
            "resolusi_rendah": st.column_config.CheckboxColumn("Res. rendah"),
            "noise_berat": st.column_config.CheckboxColumn("Noise berat"),
        },
    )

csv = df.drop(columns=["foto"]).to_csv(index=False).encode("utf-8")
st.download_button("Unduh hasil (CSV)", csv, file_name="hasil_cek_paket.csv", mime="text/csv",
                   icon=":material/download:", width="stretch")

footer()
