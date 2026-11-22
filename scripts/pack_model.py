"""Kemas ulang folder checkpoint PyTorch yang sudah diekstrak menjadi file .pt (zip).

Pemakaian:  python scripts/pack_model.py
Bobot tidak diubah sama sekali, hanya dibungkus kembali ke format zip yang dibaca torch.load.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "model_terbaik_m2_mobilenetv3"
DST = ROOT / "models" / "model_terbaik_m2_mobilenetv3.pt"


def main():
    if not (SRC / "data.pkl").exists():
        sys.exit(f"folder checkpoint tidak ditemukan: {SRC}")
    DST.parent.mkdir(parents=True, exist_ok=True)
    prefix = DST.stem  # nama folder di dalam zip (bebas, torch hanya butuh 1 folder induk)
    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    # data.pkl harus ditulis lebih dulu, mengikuti urutan torch.save
    files.sort(key=lambda p: (p.name != "data.pkl", str(p)))
    with zipfile.ZipFile(DST, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in files:
            zf.write(p, f"{prefix}/{p.relative_to(SRC).as_posix()}")
    print(f"OK: {DST} ({DST.stat().st_size / 1e6:.1f} MB, {len(files)} file)")


if __name__ == "__main__":
    main()
