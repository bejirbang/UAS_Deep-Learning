"""Grad-CAM untuk MobileNetV3 (layer target = blok terakhir, sama dengan notebook 7.7)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def jet(x: np.ndarray) -> np.ndarray:
    """Colormap 'jet' (biru → hijau → kuning → merah) tanpa matplotlib. x: 0..1 -> H x W x 3."""
    return np.stack([np.clip(1.5 - np.abs(4 * x - k), 0, 1) for k in (3, 2, 1)], axis=-1)


def gradcam(bundle, tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
    """Kembalikan heatmap 0..1 berukuran sama dengan input (H x W)."""
    model = bundle.model
    layer = model.blocks[-1]
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("acts", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grads", go[0]))
    try:
        x = tensor.unsqueeze(0).clone().requires_grad_(True)
        with torch.enable_grad():
            logits = model(x)
            k = int(logits.argmax(1)) if class_idx is None else class_idx
            model.zero_grad(set_to_none=True)
            logits[0, k].backward()
        acts, grads = store["acts"].detach(), store["grads"].detach()
        cam = F.relu((grads.mean((2, 3), keepdim=True) * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.numpy()
    finally:
        h1.remove(); h2.remove()
        model.zero_grad(set_to_none=True)


def denormalize(bundle, tensor: torch.Tensor) -> np.ndarray:
    m = torch.tensor(bundle.mean).view(3, 1, 1)
    s = torch.tensor(bundle.std).view(3, 1, 1)
    return (tensor * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(bundle, tensor: torch.Tensor, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = denormalize(bundle, tensor)
    color = jet(heat)
    mixed = (1 - alpha) * base + alpha * color
    return Image.fromarray((mixed.clip(0, 1) * 255).astype(np.uint8))


def model_view(bundle, tensor: torch.Tensor) -> Image.Image:
    """Potongan 224x224 yang benar-benar dilihat model (setelah resize + center crop)."""
    return Image.fromarray((denormalize(bundle, tensor) * 255).astype(np.uint8))
