from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2 as T

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(max(1, torch.get_num_threads()))

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model_terbaik_m2_mobilenetv3.pt"

LOW_RES_SHORT_SIDE = 200
NOISE_THRESHOLD = 0.04


@dataclass
class Bundle:
    model: torch.nn.Module
    classes: list[str]
    img_size: int
    resize_short: int
    mean: tuple
    std: tuple
    threshold: float
    meta: dict = field(default_factory=dict)
    transform: T.Compose | None = None


def load_bundle(path: Path = MODEL_PATH) -> Bundle:
    if not path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}. Jalankan `python scripts/pack_model.py` terlebih dahulu."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(ckpt["classes"])
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    b = Bundle(
        model=model,
        classes=classes,
        img_size=int(ckpt["img_size"]),
        resize_short=int(ckpt["resize_short"]),
        mean=tuple(ckpt["mean"]),
        std=tuple(ckpt["std"]),
        threshold=float(ckpt.get("hitl_threshold", 0.9)),
        meta={k: ckpt[k] for k in ("arch", "val_macro_f1", "test_macro_f1", "train_cfg") if k in ckpt},
    )
    b.transform = T.Compose([
        T.PILToTensor(),
        T.Resize(int(b.img_size * 1.14), antialias=True),
        T.CenterCrop(b.img_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(b.mean, b.std),
    ])
    return b


def load_rgb(src) -> Image.Image:
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    img = ImageOps.exif_transpose(Image.open(src))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    if min(w, h) <= short:
        return img
    s = short / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def noise_sigma(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    c = (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:]
         - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
         + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
    return float(np.sqrt(np.pi / 2) * np.abs(c).sum() / (6 * (w - 2) * (h - 2)))


@dataclass
class Prepared:
    original: Image.Image
    small: Image.Image
    tensor: torch.Tensor
    width: int
    height: int
    noise: float

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def low_res(self) -> bool:
        return self.short_side < LOW_RES_SHORT_SIDE

    @property
    def heavy_noise(self) -> bool:
        return self.noise > NOISE_THRESHOLD


def prepare(bundle: Bundle, src) -> Prepared:
    img = load_rgb(src)
    small = resize_short_side(img, bundle.resize_short)
    return Prepared(original=img, small=small, tensor=bundle.transform(small),
                    width=img.width, height=img.height, noise=noise_sigma(small))


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict[str, float]
    latency_ms: float


@torch.no_grad()
def predict_tensors(bundle: Bundle, tensors: list[torch.Tensor], batch_size: int = 16) -> list[Prediction]:
    out: list[Prediction] = []
    for i in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[i:i + batch_size])
        t0 = time.perf_counter()
        p = torch.softmax(bundle.model(x), dim=1).numpy()
        per_img = (time.perf_counter() - t0) * 1000 / len(x)
        for row in p:
            k = int(row.argmax())
            out.append(Prediction(label=bundle.classes[k], confidence=float(row[k]),
                                  probs={c: float(v) for c, v in zip(bundle.classes, row)}, latency_ms=per_img))
    return out


def predict(bundle: Bundle, prepared: Prepared) -> Prediction:
    return predict_tensors(bundle, [prepared.tensor])[0]
