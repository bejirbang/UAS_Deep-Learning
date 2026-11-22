import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
S = 4
H = 128
BLUE_DARK, BLUE, BLUE_LIGHT = (31, 95, 174), (42, 120, 214), (74, 163, 240)
GREEN = (27, 175, 122)


def icon(size: int) -> Image.Image:
    n = size * S
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    grad = Image.new("RGBA", (n, n))
    px = grad.load()
    for y in range(n):
        for x in range(n):
            t = (x + y) / (2 * n - 2)
            px[x, y] = tuple(int(BLUE_DARK[i] + (BLUE_LIGHT[i] - BLUE_DARK[i]) * t) for i in range(3)) + (255,)
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * .24), fill=255)
    img.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(img)
    w = max(2, int(n * .055))
    cx, top, mid, bot = n * .46, n * .20, n * .37, n * .80
    left, right = n * .17, n * .75
    top_face = [(cx, top), (right, mid - n * .02), (cx, mid + n * .14), (left, mid - n * .02)]
    face = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(face).polygon(top_face, fill=(255, 255, 255, 70))
    img.alpha_composite(face)
    d = ImageDraw.Draw(img)
    d.line(top_face + [top_face[0]], fill="white", width=w, joint="curve")
    d.line([(left, mid - n * .02), (left, bot - n * .15), (cx, bot), (right, bot - n * .15), (right, mid - n * .02)],
           fill="white", width=w, joint="curve")
    d.line([(cx, mid + n * .14), (cx, bot)], fill="white", width=w)
    d.line([((cx + left) / 2, (top + mid) / 2 - n * .01), ((cx + right) / 2 + n * .02, mid + n * .06)],
           fill="white", width=w)
    r = n * .17
    bx, by = n * .76, n * .76
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=GREEN, outline="white", width=int(w * .8))
    d.line([(bx - r * .45, by + r * .02), (bx - r * .08, by + r * .38), (bx + r * .5, by - r * .35)],
           fill="white", width=int(w * .85), joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def wordmark(font_path: str) -> Image.Image:
    ic = icon(H)
    font = ImageFont.truetype(font_path, int(H * .62 * S))
    parts = [("Cek", BLUE), ("Paket", BLUE_LIGHT)]
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    widths = [tmp.textbbox((0, 0), t, font=font)[2] for t, _ in parts]
    gap = int(H * .22)
    text_w = sum(widths) // S
    out = Image.new("RGBA", (H + gap + text_w + 4, H), (0, 0, 0, 0))
    out.paste(ic, (0, 0), ic)
    layer = Image.new("RGBA", ((text_w + 4) * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = 0
    asc, desc = font.getmetrics()
    y = (H * S - (asc + desc)) // 2 - int(H * S * .02)
    for (t, c), wdt in zip(parts, widths):
        d.text((x, y), t, font=font, fill=c)
        x += wdt
    layer = layer.resize((text_w + 4, H), Image.LANCZOS)
    out.paste(layer, (H + gap, 0), layer)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default="C:/Windows/Fonts/segoeuib.ttf")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    icon(H).save(OUT / "logo_icon.png")
    wordmark(a.font).save(OUT / "logo.png")
    print("OK:", OUT / "logo.png", OUT / "logo_icon.png")


if __name__ == "__main__":
    main()
