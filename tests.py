import importlib.util
import io
import sys
import tempfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location("iconizer", "iconizer.pyw")
iconizer = importlib.util.module_from_spec(_spec)
sys.modules["iconizer"] = iconizer
_spec.loader.exec_module(iconizer)

import numpy as np
from PIL import Image, ImageDraw
from iconizer import make_transparent, content_bbox, process, ICON_SIZES


def make_test_jpg():
    im = Image.new("RGB", (240, 240), (250, 248, 250))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 50, 180, 190], fill=(20, 30, 200))
    d.rectangle([105, 100, 135, 160], fill=(250, 248, 250))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf)


def make_gradient_bg_image():
    h, w = 160, 160
    yy = np.linspace(245, 175, h)[:, None]
    xx = np.linspace(250, 190, w)[None, :]
    r = np.clip(0.5 * (yy + xx), 0, 255).astype(np.uint8)
    g = np.clip(0.5 * (yy + xx) - 5, 0, 255).astype(np.uint8)
    b = np.clip(0.5 * (yy + xx) + 5, 0, 255).astype(np.uint8)
    arr = np.dstack([r, g, b])
    im = Image.fromarray(arr, "RGB")
    ImageDraw.Draw(im).ellipse([45, 45, 115, 115], fill=(210, 40, 30))
    return im


def main():
    im = make_test_jpg()

    out, note = make_transparent(im)
    a = np.asarray(out)
    assert a[0, 0, 3] == 0 and a[-1, -1, 3] == 0, "corners must be transparent"
    assert a[120, 70, 3] == 255, "ellipse interior must stay opaque"
    assert a[120, 120, 3] == 255, "hole inside ellipse must stay opaque by default"
    assert (a[..., 3] == 0).sum() > 240 * 240 * 0.5, "must remove a background-sized region"

    # Interior hole removal when remove_holes=True
    out_holes, _ = make_transparent(im, remove_holes=True)
    a_holes = np.asarray(out_holes)
    assert a_holes[120, 120, 3] == 0, "hole inside ellipse must become transparent when remove_holes=True"
    assert a_holes[120, 70, 3] == 255, "ellipse ring must remain opaque when remove_holes=True"

    # Seeded click selection on an enclosed or distinct region
    out_seed, _ = make_transparent(im, seeds=[(120, 120)])
    a_seed = np.asarray(out_seed)
    assert a_seed[120, 120, 3] == 0, "clicked interior region must become transparent"
    assert a_seed[120, 70, 3] == 255, "ellipse body must remain opaque after seed click"

    # Smooth gradient background removal
    grad_im = make_gradient_bg_image()
    grad_out, grad_note = make_transparent(grad_im)
    ga = np.asarray(grad_out)
    assert grad_note is None, f"unexpected note on gradient image: {grad_note}"
    assert ga[0, 0, 3] == 0 and ga[-1, -1, 3] == 0 and ga[0, -1, 3] == 0 and ga[-1, 0, 3] == 0, (
        "all 4 corners of gradient background must be transparent"
    )
    assert ga[80, 80, 3] == 255, "foreground subject on gradient background must stay opaque"
    grad_box = content_bbox(grad_im)
    assert grad_box is not None and 42 <= grad_box[0] <= 48 and 112 <= grad_box[2] <= 118, (
        f"gradient content_bbox {grad_box} should hug foreground"
    )

    # Fringe feathering & color decontamination
    fringe_im = Image.new("RGB", (60, 60), (250, 250, 250))
    ImageDraw.Draw(fringe_im).rectangle([20, 20, 40, 40], fill=(180, 180, 180))
    ImageDraw.Draw(fringe_im).rectangle([22, 22, 38, 38], fill=(20, 20, 20))
    fringe_out, _ = make_transparent(fringe_im)
    fo = np.asarray(fringe_out)
    fa = fo[..., 3]
    assert ((fa > 0) & (fa < 255)).any(), "fringe pixels must get partial alpha"
    # Decontaminated fringe RGB should be darker than raw (180,180,180) because white (250) was unmixed
    assert fo[20, 30, 0] < 180, f"expected decontaminated fringe pixel < 180, got {fo[20, 30, 0]}"

    # Tolerance sensitivity test
    two_tone = Image.new("RGB", (80, 80), (240, 240, 240))
    ImageDraw.Draw(two_tone).rectangle([10, 10, 70, 70], fill=(200, 200, 200))
    ImageDraw.Draw(two_tone).rectangle([25, 25, 55, 55], fill=(20, 120, 30))
    low_tol_out, _ = make_transparent(two_tone, tol=25)
    high_tol_out, _ = make_transparent(two_tone, tol=140)
    assert np.asarray(low_tol_out)[15, 15, 3] == 255, "inner gray frame should stay opaque at low tolerance"
    assert np.asarray(high_tol_out)[15, 15, 3] == 0, "inner gray frame should become transparent at high tolerance"
    assert np.asarray(high_tol_out)[40, 40, 3] == 255, "center subject must remain opaque at high tolerance"

    pal = Image.new("P", (40, 40), 0)
    pal.putpalette([255, 0, 0, 0, 255, 0] + [0, 0, 0] * 254)
    ImageDraw.Draw(pal).rectangle([10, 10, 30, 30], fill=1)
    pal.info["transparency"] = 0
    out_p, note_p = make_transparent(pal)
    assert note_p == "already transparent", f"expected 'already transparent' for P-mode with tRNS, got {note_p}"
    assert content_bbox(pal) == (10, 10, 31, 31), f"P-mode transparency bbox: {content_bbox(pal)}"

    box = content_bbox(im)
    assert box is not None
    assert 55 <= box[0] <= 62 and 45 <= box[1] <= 52 and 178 <= box[2] <= 185 and 188 <= box[3] <= 195, f"bbox {box} must hug the ellipse"

    b2 = content_bbox(out)
    assert b2 == box, f"crop after transparency {b2} must match {box}"

    alpha = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(alpha).rectangle([30, 25, 70, 80], fill=(9, 9, 9, 255))
    b = content_bbox(alpha)
    assert b == (30, 25, 71, 81), f"alpha-based crop {b}"

    opaque_rgba = Image.new("RGBA", (100, 100), (250, 248, 250, 255))
    ImageDraw.Draw(opaque_rgba).rectangle([30, 25, 70, 80], fill=(20, 30, 200, 255))
    b_opaque = content_bbox(opaque_rgba)
    assert b_opaque == (30, 25, 71, 81), f"opaque RGBA crop must fall back to color distance, got {b_opaque}"

    all_bg = Image.new("RGB", (60, 60), (200, 200, 200))
    out_bg, note_bg = make_transparent(all_bg)
    assert "skipped" in (note_bg or ""), f"expected skipped note, got {note_bg}"
    assert content_bbox(out_bg) is None, "all-background image must have no content bbox"

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "sample.png"
        alpha.save(src, "PNG")
        ico_out, note = process(src, True, True, True)
        assert ico_out.suffix == ".ico" and ico_out.exists(), "must write .ico file"
        with Image.open(ico_out) as ico_im:
            assert ico_im.format == "ICO"
            expected = {(s, s) for s in ICON_SIZES}
            assert ico_im.info.get("sizes") == expected, f"got sizes {ico_im.info.get('sizes')}"

    print("all tests passed")


if __name__ == "__main__":
    main()