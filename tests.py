import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from quickimage import bootstrap
from quickimage.core.config import ICON_SIZES
from quickimage.core.io import load_rgba
from quickimage.core.ops import (
    apply_crop_options,
    apply_palette_edits,
    content_bbox,
    make_transparent,
    refine_alpha_mask,
    transform_image,
)
from quickimage.core.pipeline import batch_process, process, render_preview
from quickimage.core.quantize import quantize_foreground_colors
from quickimage.core.svg import image_to_svg
from quickimage.ui.app import App


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
        binned_src, _, _, _ = render_preview(
            load_rgba(src), False, True, do_icon=True, bin_colors=True, n_colors=2
        )
        bs = np.asarray(binned_src)
        assert len(np.unique(bs[bs[..., 3] > 0][:, :3], axis=0)) == 2, "binned icon source must have 2 colors"

    # Click-to-delete color mode (use_border=False): only clicked colors are deleted, not unpicked borders
    click_im = Image.new("RGB", (100, 100), (240, 240, 240))  # light border background
    ImageDraw.Draw(click_im).rectangle([15, 15, 45, 85], fill=(220, 30, 30))  # red bar
    ImageDraw.Draw(click_im).rectangle([55, 15, 85, 85], fill=(30, 60, 220))  # blue bar
    # Clicking only the red bar deletes red while keeping the unpicked border and blue bar opaque
    click_red, _ = make_transparent(click_im, seeds=[(30, 50)], use_border=False)
    cra = np.asarray(click_red)
    assert cra[50, 30, 3] == 0, "clicked red bar must become transparent"
    assert cra[2, 2, 3] == 255, "unclicked border must stay opaque when use_border=False"
    assert cra[50, 70, 3] == 255, "unclicked blue bar must stay opaque"
    # Clicking both red and border deletes both while keeping blue opaque
    click_both, _ = make_transparent(click_im, seeds=[(30, 50), (2, 2)], use_border=False)
    cba = np.asarray(click_both)
    assert cba[50, 30, 3] == 0 and cba[2, 2, 3] == 0, "both clicked colors must become transparent"
    assert cba[50, 70, 3] == 255, "blue bar must remain opaque"

    test_lasso_extraction()
    test_mode_and_undo()
    test_transforms_and_zoom()
    test_edge_and_mask_refinement()
    test_palette_edits_and_svg()
    test_batch_processing()
    test_dependencies()
    print("all tests passed")


def test_dependencies():
    # The in-app installer and requirements.txt must list the same packages
    req = Path(__file__).resolve().parent / "requirements.txt"
    req_names = {
        line.split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for line in req.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert req_names == {d[1].lower() for d in bootstrap.DEPENDENCIES}, (req_names, bootstrap.DEPENDENCIES)

    # Missing packages are detected
    fake = ("no_such_module_xyz", "no-such-package", "testing", False)
    assert bootstrap.missing_dependencies((fake,) + bootstrap.DEPENDENCIES[:1]) == [fake]

    with tempfile.TemporaryDirectory() as td:
        # No venv yet: create it, then install every package into it
        venv = Path(td) / ".venv"
        cmds = bootstrap.install_commands(venv=venv)
        assert len(cmds) == 2 and cmds[0][1:3] == ["-m", "venv"] and cmds[0][-1] == str(venv), cmds
        assert cmds[1][:4] == [str(bootstrap.venv_python(venv)), "-m", "pip", "install"], cmds[1]
        assert set(cmds[1]) >= {d[1] for d in bootstrap.DEPENDENCIES}
        assert not cmds[0][0].lower().endswith("pythonw.exe"), "venv must be created with a console python"
        # Existing venv: only install
        bootstrap.venv_python(venv).parent.mkdir(parents=True)
        bootstrap.venv_python(venv).write_bytes(b"")
        assert len(bootstrap.install_commands(venv=venv)) == 1

    # Command runner streams output and reports failures
    lines, done = [], []
    bootstrap.run_commands(
        [[sys.executable, "-c", "print('one'); print('two')"]], lines.append, lambda ok, err: done.append((ok, err))
    )
    assert "one" in lines and "two" in lines and done == [(True, None)], (lines, done)
    done.clear()
    bootstrap.run_commands(
        [[sys.executable, "-c", "import sys; sys.exit(3)"], [sys.executable, "-c", "print('never')"]],
        lines.append,
        lambda ok, err: done.append((ok, err)),
    )
    assert done and done[0][0] is False and "code 3" in done[0][1], done
    assert "never" not in lines, "commands after a failure must not run"


def make_busy_scene():
    """Striped, noisy background where one stripe has the object's color; object = orange ellipse."""
    rng = np.random.default_rng(3)
    cols = [(40, 120, 200), (230, 140, 30), (60, 170, 80), (200, 60, 120), (240, 230, 220)]
    bg = np.zeros((200, 200, 3), np.uint8)
    for i in range(0, 200, 16):
        bg[:, i : i + 16] = cols[(i // 16) % len(cols)]
    bg = np.clip(bg.astype(int) + rng.integers(-20, 21, bg.shape), 0, 255).astype(np.uint8)
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    d.ellipse([60, 50, 140, 150], fill=(235, 135, 25))
    d.ellipse([85, 80, 115, 110], fill=(30, 30, 30))  # dark interior detail (not a hole)
    t = Image.new("L", (200, 200))
    ImageDraw.Draw(t).ellipse([60, 50, 140, 150], fill=1)
    truth = np.asarray(t).astype(bool)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    lasso = [(100 + 58 * np.cos(a) + rng.uniform(-4, 4), 100 + 64 * np.sin(a) + rng.uniform(-4, 4)) for a in ang]
    return im, truth, lasso


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def test_lasso_extraction():
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("SKIPPED outline (GrabCut) tests: opencv-python-headless is not installed")
        return

    im, truth, lasso = make_busy_scene()

    # The border-color method cannot isolate the object on a busy background...
    old, _ = make_transparent(im)
    assert iou(np.asarray(old)[..., 3] > 127, truth) < 0.9, "scene must be too busy for border-color matching"

    # ...but a loose outline pulls it out
    out, note = make_transparent(im, lasso=lasso)
    oa = np.asarray(out)[..., 3]
    assert note is None, f"unexpected note: {note}"
    assert iou(oa > 127, truth) >= 0.9, f"outline cutout IoU {iou(oa > 127, truth):.3f}"
    assert oa[100, 100] == 255 and oa[70, 100] == 255, "object interior (incl. dark detail) must be opaque"
    assert oa[5, 5] == 0 and oa[100, 45] == 0, "background outside and inside the outline must be transparent"

    # A one-color object cut from a multi-color scene bins to one print color (edges are
    # un-mixed against their local background, and near-duplicate palette shades merge)
    solid_scene = Image.fromarray(np.asarray(im).copy())
    ImageDraw.Draw(solid_scene).ellipse([60, 50, 140, 150], fill=(235, 135, 25))
    solid_out, _ = make_transparent(solid_scene, lasso=lasso)
    _, solid_pal = quantize_foreground_colors(solid_out, n_colors=4)
    assert len(solid_pal) == 1, f"single-color object should bin to 1 color, got {solid_pal}"

    # Interior detail counts as a hole only when "remove interior holes" is on
    holes, _ = make_transparent(im, lasso=lasso, remove_holes=True)
    assert np.asarray(holes)[100, 100, 3] == 255, "dark detail differs from the background: stays opaque"

    # A background hint inside the outline forces that spot to transparent
    gap = Image.new("RGB", (120, 120), (240, 240, 240))
    ImageDraw.Draw(gap).rectangle([30, 30, 90, 90], fill=(200, 40, 40))
    ImageDraw.Draw(gap).rectangle([50, 50, 70, 70], fill=(240, 240, 240))  # enclosed bg-colored gap
    box = [(20, 20), (100, 20), (100, 100), (20, 100)]
    no_hint, _ = make_transparent(gap, lasso=box)
    hinted, _ = make_transparent(gap, lasso=box, seeds=[(60, 60)], remove_holes=True)
    assert np.asarray(no_hint)[60, 60, 3] == 255, "enclosed gap is filled when holes are kept"
    assert np.asarray(hinted)[60, 60, 3] == 0, "hinted gap must become transparent"
    assert np.asarray(hinted)[40, 40, 3] == 255, "object around the gap must stay opaque"

    # Outline around plain background: no crash, falls back to the outline itself
    flat = Image.new("RGB", (80, 80), (120, 120, 120))
    fout, fnote = make_transparent(flat, lasso=[(10, 10), (70, 10), (70, 70), (10, 70)])
    assert fnote == "outline used as-is", f"expected fallback note, got {fnote}"
    assert np.asarray(fout)[40, 40, 3] == 255 and np.asarray(fout)[2, 2, 3] == 0

    # Crop + transparency through the full pipeline hugs the object
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "busy.png"
        im.save(src)
        png, pnote = process(src, True, True, lasso=lasso)
        assert "outline mode" in pnote, pnote
        with Image.open(png) as res:
            w, h = res.size
        assert abs(w - 81) <= 3 and abs(h - 101) <= 3, f"cropped size {w}x{h} should hug the 81x101 ellipse"

        # Crop to the outlined object without transparency keeps the surroundings opaque
        crop_only, _ = process(src, False, True, lasso=lasso)
        with Image.open(crop_only) as res:
            assert abs(res.width - 81) <= 3 and abs(res.height - 101) <= 3, res.size
            assert np.asarray(res.convert("RGBA"))[..., 3].min() == 255


def test_mode_and_undo():
    app = App()
    app.build_ui()
    app.bind_events()
    try:
        assert app.opt_mode.get() == "color", "default mode must be 'color'"
        im = Image.new("RGBA", (100, 100), (240, 240, 240, 255))
        ImageDraw.Draw(im).rectangle([20, 20, 45, 80], fill=(220, 30, 30, 255))
        ImageDraw.Draw(im).rectangle([55, 20, 80, 80], fill=(30, 60, 220, 255))
        app.bw, app.bh = 200, 200
        app.load_picture(Path("dummy.png"), im)

        # Before any click, no transparency is active
        assert app._active_transp_state() == (False, None, None, False)

        # Simulate clicking the red bar in 'color' mode -> deletes red, accumulates clicks, Ctrl+Z undoes
        cx_red, cy_red = app._source_to_canvas(30, 50)
        evt = type("E", (), {"x": int(round(cx_red)), "y": int(round(cy_red)), "state": 0, "widget": app.canvas})()
        app.on_press(evt)
        assert len(app.seeds) == 1
        assert np.asarray(app.last_preview)[50, 30, 3] == 0, "clicked red color must become transparent"
        assert np.asarray(app.last_preview)[2, 2, 3] == 255, "unclicked background border must stay opaque"

        # Second click on border adds to seeds without needing Shift
        cx_bg, cy_bg = app._source_to_canvas(5, 5)
        evt2 = type("E", (), {"x": int(round(cx_bg)), "y": int(round(cy_bg)), "state": 0, "widget": app.canvas})()
        app.on_press(evt2)
        assert len(app.seeds) == 2
        assert np.asarray(app.last_preview)[50, 30, 3] == 0 and np.asarray(app.last_preview)[2, 2, 3] == 0
        assert np.asarray(app.last_preview)[50, 70, 3] == 255, "blue bar must remain opaque"

        # Ctrl+Z undoes second click, keeping the first
        app.undo()
        assert len(app.seeds) == 1
        assert np.asarray(app.last_preview)[50, 30, 3] == 0 and np.asarray(app.last_preview)[2, 2, 3] == 255

        # Ctrl+Z again returns to original image (0 seeds)
        app.undo()
        assert len(app.seeds) == 0
        assert app.last_preview is app.source_rgba

        # Ctrl+Y / Redo restores first click, then second click; a new action clears redo stack
        app.redo()
        assert len(app.seeds) == 1
        assert np.asarray(app.last_preview)[50, 30, 3] == 0 and np.asarray(app.last_preview)[2, 2, 3] == 255
        app.redo()
        assert len(app.seeds) == 2
        assert np.asarray(app.last_preview)[50, 30, 3] == 0 and np.asarray(app.last_preview)[2, 2, 3] == 0
        app.undo()
        app.undo()
        assert len(app.seeds) == 0

        # Switching to outline mode enables drag-to-outline
        app.opt_mode.set("outline")
        app._on_mode_change()
        assert "disabled" in app.tol_scale.state(), "tolerance slider should be disabled in outline mode"

        # Dragging in outline mode draws a closing line back to the initial point and closes the loop on release
        p0x, p0y = [int(round(c)) for c in app._source_to_canvas(15, 15)]
        p1x, p1y = [int(round(c)) for c in app._source_to_canvas(50, 15)]
        p2x, p2y = [int(round(c)) for c in app._source_to_canvas(50, 85)]
        p3x, p3y = [int(round(c)) for c in app._source_to_canvas(15, 85)]
        mk_evt = lambda x, y: type("E", (), {"x": x, "y": y, "state": 0, "widget": app.canvas})()
        app.on_press(mk_evt(p0x, p0y))
        app.on_motion(mk_evt(p1x, p1y))
        app.on_motion(mk_evt(p2x, p2y))
        live_items = app.canvas.find_withtag("lasso_live")
        assert len(live_items) == 1
        live_coords = [int(round(c)) for c in app.canvas.coords("lasso_live")]
        assert live_coords[:2] == [p0x, p0y] and live_coords[-2:] == [p0x, p0y], (
            f"live outline must close back to start point, got {live_coords}"
        )
        app.on_release(mk_evt(p3x, p3y))
        assert len(app.lasso) >= 4
        assert app.lasso[-1] == app.lasso[0], "released outline must close back to initial point"
        outline_preview = app.last_preview
        assert outline_preview is not app.source_rgba, "outline must produce a cutout preview"

        # Switching back to 'Delete clicked color' preserves the outline cutout instead of reverting
        app.opt_mode.set("color")
        app._on_mode_change()
        assert app._active_transp_state() == (True, None, list(app.lasso), False)
        assert np.array_equal(np.asarray(app.last_preview), np.asarray(outline_preview)), (
            "switching to 'Delete clicked color' must keep the outlined image"
        )
        assert len(app.canvas.find_withtag("lasso")) == 0, "outline overlay must stay hidden in color mode"

        # Clicking a color inside the kept object deletes that color on top of the outline cutout
        cx_r2, cy_r2 = app._source_to_canvas(30, 50)
        app.on_press(mk_evt(int(round(cx_r2)), int(round(cy_r2))))
        assert np.asarray(app.last_preview)[50, 30, 3] == 0, "clicked color inside outlined object must be deleted"

        # Crop padding & 1:1 square canvas options
        app.reset_all()
        rect_im = Image.new("RGBA", (80, 40), (20, 120, 220, 255))
        padded = apply_crop_options(rect_im, pad_px=5, square=True)
        assert padded.size == (90, 90), f"expected 90x90 square padded canvas, got {padded.size}"
        assert np.asarray(padded)[0, 0, 3] == 0 and np.asarray(padded)[45, 45, 3] == 255

        app.opt_pad.set(6)
        app.opt_square.set(True)
        app.update_live_preview()
        assert app.last_preview.size == (112, 112), f"expected 112x112 preview, got {app.last_preview.size}"

        # WebP export & custom out_path ("Save As")
        with tempfile.TemporaryDirectory() as td:
            src_p = Path(td) / "sample.png"
            rect_im.save(src_p)
            custom_webp = Path(td) / "custom_out.webp"
            out_w, note_w = process(src_p, False, False, do_webp=True, out_path=custom_webp, crop_pad=4, square_canvas=True)
            assert out_w == custom_webp and custom_webp.is_file()
            assert "webp" in note_w and "pad 4px" in note_w and "1:1" in note_w
            with Image.open(custom_webp) as wim:
                assert wim.size == (88, 88)
    finally:
        app.root.destroy()


def test_transforms_and_zoom():
    from PIL import Image
    import numpy as np

    arr = np.zeros((20, 40, 4), dtype=np.uint8)
    arr[0:5, 0:10] = (255, 0, 0, 255)
    arr[15:20, 30:40] = (0, 255, 0, 255)
    im = Image.fromarray(arr, "RGBA")

    r90 = transform_image(im, rotate_deg=90)
    assert r90.size == (20, 40)
    r180 = transform_image(im, rotate_deg=180)
    assert r180.size == (40, 20)
    assert tuple(np.asarray(r180)[19, 39]) == (255, 0, 0, 255)
    r270 = transform_image(im, rotate_deg=270)
    assert r270.size == (20, 40)

    fh = transform_image(im, flip_h=True)
    assert tuple(np.asarray(fh)[0, 39]) == (255, 0, 0, 255)
    fv = transform_image(im, flip_v=True)
    assert tuple(np.asarray(fv)[19, 0]) == (255, 0, 0, 255)

    app = App()
    app.build_ui()
    try:
        app.load_picture(Path("test.png"), im)
        assert app.source_rgba.size == (40, 20)
        app.rotate_cw()
        assert app.source_rgba.size == (20, 40)
        app.flip_horizontal()
        assert app.source_rgba.size == (20, 40)

        class DummyEvent:
            def __init__(self, delta, x=100, y=100):
                self.delta = delta
                self.x = x
                self.y = y

        app.on_zoom(DummyEvent(120))
        assert app.zoom > 1.0
        app.reset_zoom()
        assert app.zoom == 1.0 and app.pan_x == 0.0 and app.pan_y == 0.0

        for bd, expected in (("light", (244, 244, 246)), ("dark", (28, 28, 32)), ("checker", None)):
            app.opt_backdrop.set(bd)
            app._on_backdrop_change()
            tile = app.checker_board(16, 16)
            assert tile.shape == (16, 16, 3)
            if expected is not None:
                assert tuple(tile[0, 0]) == expected
    finally:
        app.root.destroy()


def test_edge_and_mask_refinement():
    from PIL import Image
    import numpy as np

    arr = np.zeros((40, 40, 4), dtype=np.uint8)
    arr[:, :] = (240, 240, 240, 255)
    arr[10:30, 10:30] = (20, 120, 220, 255)
    orig = Image.fromarray(arr, "RGBA")
    cut, _ = make_transparent(orig, tol=30, use_border=True)
    base_fg = int((np.asarray(cut)[..., 3] > 127).sum())

    shrunk = refine_alpha_mask(cut, source_im=orig, edge_choke=-2)
    grown = refine_alpha_mask(cut, source_im=orig, edge_choke=2)
    assert int((np.asarray(shrunk)[..., 3] > 127).sum()) < base_fg
    assert int((np.asarray(grown)[..., 3] > 127).sum()) > base_fg

    feathered = refine_alpha_mask(cut, source_im=orig, edge_feather=2.0)
    alpha_f = np.asarray(feathered)[..., 3]
    assert ((alpha_f > 0) & (alpha_f < 255)).any()

    erased = refine_alpha_mask(cut, source_im=orig, brush_strokes=[("erase", 4, [(20, 20)])])
    assert np.asarray(erased)[20, 20, 3] == 0
    restored = refine_alpha_mask(erased, source_im=orig, brush_strokes=[("restore", 5, [(20, 20), (5, 5)])])
    assert np.asarray(restored)[20, 20, 3] == 255
    assert np.asarray(restored)[5, 5, 3] == 255

    app = App()
    app.build_ui()
    try:
        app.load_picture(Path("brush_test.png"), orig)
        assert app._active_transp_state() == (False, None, None, False)
        app.opt_mode.set("brush")
        app._on_mode_change()
        app.opt_brush_mode.set("erase")
        app.opt_brush_size.set(5)
        cx, cy = app._source_to_canvas(20, 20)

        class Ev:
            def __init__(self, x, y):
                self.x = int(round(x))
                self.y = int(round(y))
                self.widget = app.canvas

        app.on_press(Ev(cx, cy))
        app.on_motion(Ev(cx + 2, cy + 2))
        app.on_release(Ev(cx + 2, cy + 2))
        assert len(app.brush_strokes) == 1
        app.undo()
        assert len(app.brush_strokes) == 0
        app.redo()
        assert len(app.brush_strokes) == 1
    finally:
        app.root.destroy()


def test_palette_edits_and_svg():
    from PIL import Image
    import numpy as np

    arr = np.zeros((30, 30, 4), dtype=np.uint8)
    arr[2:14, 2:28] = (220, 30, 30, 255)
    arr[16:28, 2:28] = (30, 60, 220, 255)
    im = Image.fromarray(arr, "RGBA")
    binned, pal = quantize_foreground_colors(im, n_colors=2)
    assert pal and len(pal) == 2

    edited, edited_pal = apply_palette_edits(binned, pal, hidden_indices={0}, color_overrides={1: (10, 200, 50)})
    earr = np.asarray(edited)
    assert (earr[..., 3] == 0).any()
    fg = earr[earr[..., 3] > 0]
    assert len(fg) > 0 and tuple(fg[0, :3]) == (10, 200, 50)
    assert edited_pal == [(10, 200, 50)]

    svg = image_to_svg(im, n_colors=2, eps=1.2, hidden_indices={0}, color_overrides={1: (10, 200, 50)})
    root = ET.fromstring(svg)
    paths = [el for el in root if el.tag.endswith("path")]
    assert len(paths) == 1
    assert "#0ac832" in svg


def test_batch_processing():
    from PIL import Image
    import numpy as np
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        paths = []
        for i, col in enumerate(((220, 40, 40, 255), (40, 180, 60, 255))):
            arr = np.full((24, 24, 4), (250, 250, 250, 255), dtype=np.uint8)
            arr[6:18, 6:18] = col
            p = td_path / f"item_{i}.png"
            Image.fromarray(arr, "RGBA").save(p)
            paths.append(p)
        outputs, errors = batch_process(paths, do_transp=True, do_crop=True, use_border=True)
        assert not errors
        assert len(outputs) == 2
        for _, out_p, _ in outputs:
            assert out_p.is_file()
            with Image.open(out_p) as out_im:
                assert out_im.size == (12, 12)


if __name__ == "__main__":
    main()