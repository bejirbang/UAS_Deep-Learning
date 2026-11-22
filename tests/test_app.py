import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from src.business import decide
from src.gradcam import gradcam
from src.model import load_bundle, predict, prepare


def sample_tiles():
    for name, label in [("aman", "Aman"), ("rusak", "Rusak"), ("lainnya", "Lainnya")]:
        g = Image.open(ROOT / "assets" / "figures" / f"samples_{name}.jpg")
        w, h = g.size
        for j in (0, 2):
            buf = io.BytesIO()
            g.crop((j * w / 6 + 10, 50, (j + 1) * w / 6 - 10, h / 2 - 10)).save(buf, "JPEG")
            yield label, buf.getvalue()


def test_model():
    b = load_bundle()
    assert b.classes == ["Aman", "Rusak", "Lainnya"] and b.img_size == 224
    correct = 0
    for label, data in sample_tiles():
        p = prepare(b, data)
        r = predict(b, p)
        assert abs(sum(r.probs.values()) - 1) < 1e-4
        correct += r.label == label
    assert correct >= 5, f"hanya {correct}/6 contoh benar"
    heat = gradcam(b, p.tensor)
    assert heat.shape == (224, 224) and 0 <= heat.min() and heat.max() <= 1
    buf = io.BytesIO(); Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(buf, "PNG")
    assert prepare(b, buf.getvalue()).low_res
    print(f"model OK ({correct}/6 contoh benar)")


def test_decision():
    assert decide("Lainnya", 0.99, 0.9).route == "reupload"
    assert decide("Rusak", 0.95, 0.9).route == "auto"
    d = decide("Rusak", 0.6, 0.9)
    assert d.route == "cs" and d.priority == "tinggi"
    assert decide("Aman", 0.6, 0.9).priority == "normal"
    print("keputusan OK")


def test_pages():
    from streamlit.testing.v1 import AppTest

    def app(mode):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        if mode:
            at.query_params["mode"] = mode
        return at

    at = app(None); at.run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    assert "HP" in labels and "Laptop" in labels, f"dialog tidak muncul: {labels}"
    at.button[labels.index("HP")].click().run()
    assert at.session_state["mode"] == "HP" and at.session_state["device_confirmed"] and not at.exception
    at.run()
    assert "HP" not in [b.label for b in at.button], "dialog masih muncul setelah memilih"
    print("dialog pilihan perangkat OK")

    tiles = list(sample_tiles())
    files = [(f"{lab}_{i}.jpg", d, "image/jpeg") for i, (lab, d) in enumerate(tiles)] + [("bukan_gambar.jpg", b"xx", "image/jpeg")]
    for mode in ["Laptop", "HP"]:
        for page in ["home", "predict", "batch", "performance", "about"]:
            at = app(mode); at.run()
            if page != "home":
                at.switch_page(f"views/{page}.py").run()
            assert not at.exception, f"{mode}/{page}: {at.exception}"
            assert at.session_state["mode"] == mode, f"?mode={mode} di URL diabaikan ({at.session_state['mode']})"

        at = app(mode); at.run(); at.switch_page("views/predict.py").run()
        at.file_uploader(key="single_upload").set_value(("rusak.jpg", tiles[2][1], "image/jpeg")).run()
        assert not at.exception, at.exception
        assert at.success and "klaim disetujui" in at.success[0].proto.title, "keputusan otomatis tidak muncul"
        assert len(at.image) >= 2, f"gambar yang tampil: {len(at.image)}"
        at.slider(key="threshold").set_value(0.99).run()
        assert at.info and "Review Customer Service" in at.info[0].proto.title, "threshold tidak berpengaruh"

        at = app(mode); at.run(); at.switch_page("views/batch.py").run()
        at.file_uploader(key="batch_upload").set_value(files).run()
        assert not at.exception, at.exception
        assert at.metric[0].label == "Total foto" and at.metric[0].value == "6", f"total foto = {at.metric[0].value}"
        assert any("bukan_gambar" in e.value for e in at.error), "file rusak tidak dilaporkan"
        at.slider(key="threshold").set_value(0.5).run()
        assert not at.exception, at.exception
        print(f"mode {mode}: semua halaman + upload OK")

if __name__ == "__main__":
    test_model()
    test_decision()
    test_pages()
