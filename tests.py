import importlib.util
import io
import sys
import tempfile
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("quickimage", Path(__file__).resolve().parent / "quickimage.pyw")
quickimage = importlib.util.module_from_spec(_spec)
sys.modules["quickimage"] = quickimage
_spec.loader.exec_module(quickimage)

import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from quickimage import (
    ICON_SIZES,
    content_bbox,
    image_to_svg,
    make_transparent,
    process,
    quantize_foreground_colors,
    save_svg,
)


def svg_path_points(path_el):
    """Set of (x, y) vertices in an SVG path 'd' made of M/L/Z commands."""
    toks = [t for t in path_el.attrib["d"].split() if t not in ("M", "L", "Z")]
    return {(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks), 2)}


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

    # 4-color foreground quantization & despeckling
    multi_im = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    d_m = ImageDraw.Draw(multi_im)
    d_m.rectangle([10, 10, 55, 55], fill=(220, 30, 30, 255))
    d_m.rectangle([65, 10, 110, 55], fill=(30, 190, 40, 255))
    d_m.rectangle([10, 65, 55, 110], fill=(35, 60, 225, 255))
    d_m.rectangle([65, 65, 110, 110], fill=(240, 200, 25, 255))
    # Add a 2x2 speckle of a 5th color inside the first quadrant
    d_m.rectangle([25, 25, 26, 26], fill=(180, 0, 180, 255))

    binned_im, palette = quantize_foreground_colors(multi_im, n_colors=4)
    assert len(palette) == 4, f"expected 4 binned colors, got {len(palette)}: {palette}"
    ba = np.asarray(binned_im)
    assert ba[0, 0, 3] == 0, "transparent background must stay transparent after color binning"
    assert set(np.unique(ba[..., 3]).tolist()) <= {0, 255}, "binned alpha must be crisp binary {0, 255}"
    fg_pixels = ba[ba[..., 3] == 255][:, :3]
    unique_fg = {tuple(int(v) for v in row) for row in np.unique(fg_pixels, axis=0)}
    assert unique_fg == set(palette), f"foreground colors {unique_fg} must match palette {palette}"
    # Speckle at (25, 25) should be absorbed into surrounding red quadrant
    assert tuple(int(v) for v in ba[25, 25, :3]) == tuple(int(v) for v in ba[20, 20, :3]), (
        "tiny 2x2 speckle should be despeckled into surrounding color"
    )

    # Custom N-color binning (2 colors)
    _, pal2 = quantize_foreground_colors(multi_im, n_colors=2)
    assert len(pal2) == 2, f"expected 2 colors when n_colors=2, got {len(pal2)}"

    # Vector SVG generation & XML validity
    svg_str = image_to_svg(multi_im, n_colors=4)
    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg"), f"expected <svg> root, got {root.tag}"
    assert root.attrib.get("width") == "120" and root.attrib.get("height") == "120"
    paths = [el for el in root if el.tag.endswith("path")]
    assert len(paths) == 4, f"expected 4 <path> layers in SVG, got {len(paths)}"
    for p_el in paths:
        assert p_el.attrib.get("fill", "").startswith("#")
        assert p_el.attrib.get("fill-rule") == "evenodd"
        assert "M " in p_el.attrib.get("d", "") and "Z" in p_el.attrib.get("d", "")

    # Shared color boundaries must be emitted identically by both layers (no seams)
    nested = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    ImageDraw.Draw(nested).ellipse([5, 5, 54, 54], fill=(30, 60, 220, 255))
    ImageDraw.Draw(nested).rectangle([20, 20, 39, 39], fill=(220, 30, 30, 255))
    n_root = ET.fromstring(image_to_svg(nested, n_colors=2))
    n_paths = {el.attrib["fill"]: el for el in n_root if el.tag.endswith("path")}
    red_pts = svg_path_points(n_paths["#dc1e1e"])
    blue_pts = svg_path_points(n_paths["#1e3cdc"])
    assert {(20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)} <= red_pts, f"red square corners: {red_pts}"
    assert red_pts <= blue_pts, "inner square outline must match the surrounding layer's hole exactly"

    # Binning for PNG/ICO keeps feathered alpha; SVG binning stays crisp
    soft = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    ImageDraw.Draw(soft).rectangle([10, 10, 29, 29], fill=(200, 40, 40, 255))
    sa = np.array(soft)
    sa[9, 10:30] = (190, 50, 50, 90)  # semi-transparent fringe row
    soft = Image.fromarray(sa, "RGBA")
    kept, _ = quantize_foreground_colors(soft, n_colors=2, keep_alpha=True)
    ka = np.asarray(kept)
    assert ka[9, 15, 3] == 90, f"keep_alpha must preserve fringe alpha, got {ka[9, 15, 3]}"
    assert tuple(ka[9, 15, :3]) == tuple(ka[15, 15, :3]), "fringe pixel must be recolored to a palette color"
    crisp, _ = quantize_foreground_colors(soft, n_colors=2)
    assert np.asarray(crisp)[9, 15, 3] == 0, "default binning must drop alpha < 128"

    # Picked background color must not leak to same-colored regions far from the click
    leak = Image.new("RGB", (100, 60), (250, 250, 250))
    ImageDraw.Draw(leak).rectangle([5, 5, 45, 55], fill=(20, 20, 160))
    ImageDraw.Draw(leak).rectangle([15, 15, 35, 45], fill=(200, 200, 40))  # picked region
    ImageDraw.Draw(leak).rectangle([55, 5, 95, 55], fill=(20, 20, 160))
    ImageDraw.Draw(leak).rectangle([65, 15, 85, 45], fill=(200, 200, 40))  # same color, not picked
    leak_out, _ = make_transparent(leak, seeds=[(25, 30)], remove_holes=True)
    la = np.asarray(leak_out)
    assert la[30, 25, 3] == 0, "picked region must become transparent"
    assert la[30, 75, 3] == 255, "unpicked region of the same color must stay opaque"

    # Large noisy image: binning must finish quickly (despeckle used to be O(specks * pixels))
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8), "RGB").convert("RGBA")
    t0 = time.perf_counter()
    quantize_foreground_colors(noisy, n_colors=16)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20, f"binning a noisy 600x600 image took {elapsed:.1f}s"

    with tempfile.TemporaryDirectory() as td:
        # Transparency + crop + ICO on a single dark subject over transparency
        alpha_src = Path(td) / "alpha.png"
        alpha.save(alpha_src, "PNG")
        a_ico, _ = process(alpha_src, True, True, True)
        with Image.open(a_ico) as ico_im:
            assert ico_im.info.get("sizes") == {(s, s) for s in ICON_SIZES}
            ico_im.size = (256, 256)
            big = np.asarray(ico_im.convert("RGBA"))
        # 41x56 subject scaled to fit 256 tall -> about 187 wide, centered
        cols = np.where((big[..., 3] > 0).any(axis=0))[0]
        assert big[128, 128, 3] == 255 and 180 <= cols[-1] - cols[0] + 1 <= 194, (
            f"cropped subject should fill the icon height, width {cols[-1] - cols[0] + 1}"
        )

        src = Path(td) / "sample.png"
        multi_im.save(src, "PNG")
        ico_out, note = process(src, True, True, True)
        assert ico_out.suffix == ".ico" and ico_out.exists(), "must write .ico file"
        with Image.open(ico_out) as ico_im:
            assert ico_im.format == "ICO"
            expected = {(s, s) for s in ICON_SIZES}
            assert ico_im.info.get("sizes") == expected, f"got sizes {ico_im.info.get('sizes')}"

        svg_out, svg_note = process(src, False, True, do_svg=True, n_colors=4)
        assert svg_out.suffix == ".svg" and svg_out.exists(), "must write .svg file"
        assert "4 color(s)" in svg_note and "svg" in svg_note, f"unexpected svg note: {svg_note}"
        svg_root = ET.fromstring(svg_out.read_text(encoding="utf-8"))
        assert len([el for el in svg_root if el.tag.endswith("path")]) == 4

        bin_out, bin_note = process(src, False, False, bin_colors=True, n_colors=3)
        assert bin_out.suffix == ".png" and bin_out.exists(), "must write binned .png file"
        assert "3 color(s)" in bin_note, f"unexpected bin note: {bin_note}"

        # Binning combined with icon export
        bin_ico, bin_ico_note = process(src, False, True, True, bin_colors=True, n_colors=2)
        assert bin_ico.suffix == ".ico" and "2 color(s)" in bin_ico_note, f"bin+ico note: {bin_ico_note}"
        # The image fed to the icon resizer (before LANCZOS blends edges) must hold exactly 2 colors
        binned_src, _, _, _ = quickimage.render_preview(
            quickimage.load_rgba(src), False, True, do_icon=True, bin_colors=True, n_colors=2
        )
        bs = np.asarray(binned_src)
        assert len(np.unique(bs[bs[..., 3] > 0][:, :3], axis=0)) == 2, "binned icon source must have 2 colors"

    print("all tests passed")


if __name__ == "__main__":
    main()