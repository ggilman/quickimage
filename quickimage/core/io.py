"""Loading images, output paths and .ico writing."""

from pathlib import Path

from PIL import Image, ImageOps

from .config import ICON_SIZES, IMG_EXT


def load_rgba(path):
    with Image.open(path) as src:
        try:
            src.seek(0)
        except Exception:
            pass
        im = ImageOps.exif_transpose(src)
        return im.convert("RGBA")

def next_save_path(path, ext=".png"):
    out = path.with_name(path.stem + f"_icon{ext}")
    n = 2
    while out.exists():
        out = path.with_name(path.stem + f"_icon_{n}{ext}")
        n += 1
    return out


def make_icon_frame(im, size):
    rgba = im.convert("RGBA")
    w, h = rgba.size
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = rgba.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def save_ico(im, out, sizes=ICON_SIZES):
    frames = [make_icon_frame(im, s) for s in sorted(sizes, reverse=True)]
    frames[0].save(
        out,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[1:],
    )

def is_image_file(p):
    return Path(p).suffix.lower() in IMG_EXT
