"""Generate the Geospatial Atlas app icon set.

Design brief:
  * Minimal, modern, YC-adjacent palette (deep navy + amber accent + off-white).
  * Three filled circles forming an asymmetric cluster — reads as both a
    map-point scatter and an embedding projection.
  * No literal pins, globes, or typography. Survives 16×16.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFilter

HERE = pathlib.Path(__file__).resolve().parent

# Palette
BG_TOP = (11, 16, 31, 255)  # #0B101F — near-black navy
BG_BOT = (17, 24, 39, 255)  # #111827 — slate-900
ACCENT = (249, 115, 22, 255)  # #F97316 — amber-500 (the warm anchor)
FOREGROUND = (248, 250, 252, 255)  # #F8FAFC — off-white

# Circle positions & sizes (fractions of canvas width/height and radius).
# The layout is the whole identity — tuned to look balanced at every scale.
CIRCLES = [
    # (cx_frac, cy_frac, r_frac, fill)
    (0.58, 0.46, 0.155, ACCENT),  # anchor — largest, amber
    (0.30, 0.30, 0.075, FOREGROUND),  # companion — upper-left
    (0.76, 0.74, 0.055, FOREGROUND),  # outlier — lower-right
]


def _gradient(size: int) -> Image.Image:
    """Vertical linear gradient from BG_TOP to BG_BOT."""
    grad = Image.new("RGBA", (1, size))
    for y in range(size):
        t = y / max(size - 1, 1)
        pixel = tuple(int(BG_TOP[i] * (1 - t) + BG_BOT[i] * t) for i in range(3)) + (
            255,
        )
        grad.putpixel((0, y), pixel)
    return grad.resize((size, size))


def _mask_rounded_square(size: int, radius_frac: float = 0.225) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    radius = int(size * radius_frac)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def _draw_dot(
    img: Image.Image, cx: int, cy: int, r: int, fill: tuple[int, int, int, int]
) -> None:
    """A flat, anti-aliased filled circle. No highlights, no gradients."""
    # Super-sample 4× for clean edges at small sizes.
    scale = 4
    big = Image.new(
        "RGBA", (r * 2 * scale + scale, r * 2 * scale + scale), (0, 0, 0, 0)
    )
    ImageDraw.Draw(big).ellipse(
        [0, 0, r * 2 * scale, r * 2 * scale],
        fill=fill,
    )
    small = big.resize((r * 2, r * 2), Image.LANCZOS)
    img.alpha_composite(small, (cx - r, cy - r))


def make_base(size: int) -> Image.Image:
    # 1. Rounded-square gradient background (very subtle).
    bg = _gradient(size)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(bg, (0, 0), _mask_rounded_square(size))

    # 2. Three flat dots — the entire identity.
    for cx_f, cy_f, r_f, fill in CIRCLES:
        cx = int(size * cx_f)
        cy = int(size * cy_f)
        r = max(2, int(size * r_f))
        _draw_dot(canvas, cx, cy, r, fill)

    # 3. Re-apply the rounded mask so nothing bleeds past the corners.
    masked = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    masked.paste(canvas, (0, 0), _mask_rounded_square(size))
    return masked


def main() -> None:
    sizes = {
        "32x32.png": 32,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "icon.png": 512,
    }
    pngs: dict[str, pathlib.Path] = {}
    for name, size in sizes.items():
        out = HERE / name
        img = make_base(size)
        img.save(out)
        pngs[name] = out
        print(f"wrote {out}")

    # Windows .ico — multi-resolution container readable on all OSes.
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    ico_path = HERE / "icon.ico"
    make_base(256).save(ico_path, format="ICO", sizes=[(s, s) for s in ico_sizes])
    print(f"wrote {ico_path}")

    # macOS .icns — only build when iconutil is available (macOS host).
    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory() as td:
            iconset = pathlib.Path(td) / "icon.iconset"
            iconset.mkdir()
            icns_sizes = {
                "icon_16x16.png": 16,
                "icon_16x16@2x.png": 32,
                "icon_32x32.png": 32,
                "icon_32x32@2x.png": 64,
                "icon_128x128.png": 128,
                "icon_128x128@2x.png": 256,
                "icon_256x256.png": 256,
                "icon_256x256@2x.png": 512,
                "icon_512x512.png": 512,
                "icon_512x512@2x.png": 1024,
            }
            for name, size in icns_sizes.items():
                make_base(size).save(iconset / name)
            out = HERE / "icon.icns"
            subprocess.run(
                ["iconutil", "-c", "icns", "-o", str(out), str(iconset)],
                check=True,
            )
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
