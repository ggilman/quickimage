"""Generate the Quick Image app icon: assets/quickimage.ico (multi-size) and assets/quickimage.png.

Motif: a picture reduced to a few flat colors, stacked like 3D-print color layers,
with a lightning bolt for "quick". Each size is rendered separately (supersampled),
so small sizes stay crisp instead of being downscaled from 256 px.

Run:  .venv\\Scripts\\python tools\\make_icon.py [--sheet preview.png]
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SS = 4  # supersampling factor

# Limited "print filament" palette
BG = (24, 33, 56)
SKY = (46, 64, 104)
SUN = (255, 112, 87)
FAR = (64, 140, 214)
NEAR = (38, 190, 160)
BOLT = (255, 211, 63)
BOLT_EDGE = (24, 33, 56)


def render(size):
    """Draw the icon on a 256-unit design grid, scaled to `size` px."""
    s = size * SS
    u = s / 256.0  # one design unit in supersampled px

    def P(*pts):
        return [(x * u, y * u) for x, y in pts]

    small = size <= 24
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    # Tile
    m = 6 if small else 10
    r = 40 if small else 52
    d.rounded_rectangle([m * u, m * u, (256 - m) * u, (256 - m) * u], radius=r * u, fill=BG)

    # Picture area (clip by drawing into a mask)
    pic = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pic)
    pm = 30 if small else 36
    pd.rectangle([0, 0, s, s], fill=SKY)
    pd.ellipse(P((62, 58), (122, 118)), fill=SUN)
    # Far range: stepped silhouette = visible "layers"
    pd.polygon(P((0, 196), (0, 150), (50, 120), (96, 146), (150, 92), (210, 140), (256, 118), (256, 196)), fill=FAR)
    # Near range
    pd.polygon(P((0, 256), (0, 190), (70, 128), (118, 168), (160, 140), (256, 210), (256, 256)), fill=NEAR)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [pm * u, pm * u, (256 - pm) * u, (256 - pm) * u], radius=(22 if small else 26) * u, fill=255
    )
    im.paste(pic, (0, 0), mask)

    # Lightning bolt badge, overlapping the top-right corner of the picture
    # (kept inside the tile's rounded corner, outline included)
    if small:  # chunkier bolt so it still reads at 16-24 px
        bolt = ((160, 22), (112, 128), (152, 128), (128, 222), (224, 96), (182, 96), (210, 22))
    else:
        bolt = ((170, 26), (128, 118), (160, 118), (136, 200), (214, 92), (180, 92), (204, 26))
    edge = 10 if small else 8
    d.line(P(*bolt, bolt[0]), fill=BOLT_EDGE, width=int(round(edge * 2 * u)), joint="curve")
    d.polygon(P(*bolt), fill=BOLT_EDGE)
    d.polygon(P(*bolt), fill=BOLT)

    return im.resize((size, size), Image.LANCZOS)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    frames = {sz: render(sz) for sz in SIZES}
    big = frames[max(SIZES)]
    big.save(OUT_DIR / "quickimage.png")
    # Pillow's ICO writer takes per-size frames from append_images when the sizes match exactly
    big.save(
        OUT_DIR / "quickimage.ico",
        format="ICO",
        sizes=[(sz, sz) for sz in SIZES],
        append_images=[frames[sz] for sz in SIZES if sz != max(SIZES)],
    )
    print("wrote", OUT_DIR / "quickimage.ico", OUT_DIR / "quickimage.png")
    if "--sheet" in sys.argv:
        # Contact sheet of every size side by side, for reviewing the design
        sheet_path = Path(sys.argv[sys.argv.index("--sheet") + 1])
        sheet = Image.new("RGBA", (sum(SIZES) + 10 * (len(SIZES) + 1), max(SIZES) + 20), (255, 255, 255, 255))
        x = 10
        for sz in SIZES:
            sheet.alpha_composite(frames[sz], (x, 10 + (max(SIZES) - sz)))
            x += sz + 10
        sheet.save(sheet_path)
        print("wrote", sheet_path)


if __name__ == "__main__":
    main()
