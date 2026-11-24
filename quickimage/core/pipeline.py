"""The full processing pipeline: in-memory previews, single-file export and batches."""

from pathlib import Path

import numpy as np

from .config import BG_TOL, DEFAULT_SVG_COLORS, ICON_SIZES
from .io import DEFAULT_SUFFIX, load_rgba, make_icon_frame, next_save_path, save_ico
from .filaments import filament_overrides
from .ops import (
    apply_crop_options,
    apply_palette_edits,
    content_bbox,
    make_transparent,
    refine_alpha_mask,
    resize_to_longest,
    transform_image,
)
from .outline import _mask_bbox, get_lasso_mask
from .quantize import quantize_foreground_colors
from .svg import binned_to_svg


def palette_overrides(raw_palette, filaments=None, color_overrides=None):
    """Swatch recolors on top of the filament mapping: a color you recolored by hand wins."""
    out = filament_overrides(raw_palette, filaments) if filaments else {}
    out.update(color_overrides or {})
    return out or None


def render_preview(
    im,
    do_transp,
    do_crop,
    do_icon=False,
    tol=BG_TOL,
    remove_holes=False,
    seeds=None,
    bin_colors=False,
    n_colors=DEFAULT_SVG_COLORS,
    lasso=None,
    use_border=True,
    crop_pad=0,
    square_canvas=False,
    edge_choke=0,
    edge_feather=0.0,
    brush_strokes=None,
    hidden_indices=None,
    color_overrides=None,
    out_size=0,
    filaments=None,
):
    """Run transparency/mask-refinement/crop/binning/icon steps in-memory without writing files.

    `lasso` (an outline around one object, in image pixels) switches transparency to
    "keep only that object"; with transparency off it still crops to the object.
    `out_size` scales the result so its longer side is that many px (0 keeps the size); it runs
    before color binning, so binned colors stay flat. `filaments` (list of rgb) maps every binned
    color onto one of those filament colors.
    """
    notes = []
    orig_im = im.convert("RGBA")
    crop_box = (0, 0, im.width, im.height)
    lasso = list(lasso) if lasso else None
    # Background picks are part of transparency; ignore them when it is switched off.
    seeds = seeds if do_transp else None
    if lasso:
        notes.append("outline mode")
    if do_transp:
        im, note = make_transparent(
            im, tol=tol, remove_holes=remove_holes, seeds=seeds, lasso=lasso, use_border=use_border
        )
        if note:
            notes.append(note)
    if brush_strokes or edge_choke != 0 or edge_feather > 0:
        im = refine_alpha_mask(
            im,
            source_im=orig_im,
            edge_choke=edge_choke,
            edge_feather=edge_feather,
            brush_strokes=brush_strokes,
        )
        if brush_strokes:
            notes.append(f"{len(brush_strokes)} stroke(s)")
        if edge_choke != 0:
            notes.append(f"edge {int(round(edge_choke)):+d}px")
        if edge_feather > 0:
            notes.append(f"feather {edge_feather:g}px")
    if do_crop:
        if lasso and not do_transp and not brush_strokes:
            # Crop to the outlined object but keep its surroundings opaque
            alpha_f, _, _ = get_lasso_mask(np.asarray(im.convert("RGB")), lasso, remove_holes=remove_holes)
            box = _mask_bbox(alpha_f > 16 / 255.0)
        else:
            box = content_bbox(im, tol=tol, remove_holes=remove_holes, seeds=seeds, use_border=use_border)
        if box and box != (0, 0, im.width, im.height):
            crop_box = box
            im = im.crop(box)
            notes.append(f"cropped to {box[2]-box[0]}x{box[3]-box[1]}")
        elif not (crop_pad > 0 or square_canvas):
            notes.append("nothing to crop")
    if crop_pad > 0 or square_canvas:
        im = apply_crop_options(im, pad_px=crop_pad, square=square_canvas)
        if crop_pad > 0:
            notes.append(f"pad {int(round(crop_pad))}px")
        if square_canvas:
            notes.append(f"1:1 ({im.width}x{im.height})")
    if out_size and max(im.size) != int(out_size):
        im = resize_to_longest(im, out_size)
        notes.append(f"resized to {im.width}x{im.height}")
    raw_palette = None
    if bin_colors:
        im, raw_palette = quantize_foreground_colors(im, n_colors=n_colors, keep_alpha=True)
        im, active_palette = apply_palette_edits(
            im, raw_palette, hidden_indices=hidden_indices,
            color_overrides=palette_overrides(raw_palette, filaments, color_overrides),
        )
        im.info["raw_palette"] = list(raw_palette)
        im.info["active_palette"] = list(active_palette)
        notes.append(f"{len(active_palette)} {'filament' if filaments else 'color'}(s)")
    preview = make_icon_frame(im, max(ICON_SIZES)) if do_icon else im
    if raw_palette is not None and preview is not im:
        preview.info["raw_palette"] = list(raw_palette)
        preview.info["active_palette"] = list(im.info.get("active_palette", raw_palette))
    return im, preview, notes, crop_box


def _process_with_preview(
    path,
    do_transp,
    do_crop,
    do_icon=False,
    tol=BG_TOL,
    remove_holes=False,
    seeds=None,
    do_svg=False,
    bin_colors=False,
    n_colors=DEFAULT_SVG_COLORS,
    lasso=None,
    use_border=True,
    do_webp=False,
    out_path=None,
    crop_pad=0,
    square_canvas=False,
    source_im=None,
    edge_choke=0,
    edge_feather=0.0,
    brush_strokes=None,
    hidden_indices=None,
    color_overrides=None,
    svg_eps=0.85,
    out_size=0,
    filaments=None,
    out_dir=None,
    suffix=DEFAULT_SUFFIX,
):
    path = Path(path)
    im = source_im.copy() if source_im is not None else load_rgba(path)
    im, preview, notes, crop_box = render_preview(
        im,
        do_transp,
        do_crop,
        do_icon=do_icon and not do_svg and not do_webp,
        tol=tol,
        remove_holes=remove_holes,
        seeds=seeds,
        bin_colors=bool(bin_colors and not do_svg),  # SVG bins below, exactly once
        n_colors=n_colors,
        lasso=lasso,
        use_border=use_border,
        crop_pad=crop_pad,
        square_canvas=square_canvas,
        edge_choke=edge_choke,
        edge_feather=edge_feather,
        brush_strokes=brush_strokes,
        hidden_indices=hidden_indices,
        color_overrides=color_overrides,
        out_size=0 if do_icon else out_size,  # icons have their own fixed sizes
        filaments=filaments,
    )

    def target(ext):
        return Path(out_path) if out_path else next_save_path(path, ext, out_dir=out_dir, suffix=suffix)

    if do_svg:
        binned, raw_palette = quantize_foreground_colors(im, n_colors=n_colors)
        binned, palette = apply_palette_edits(
            binned, raw_palette, hidden_indices=hidden_indices,
            color_overrides=palette_overrides(raw_palette, filaments, color_overrides),
        )
        binned.info["raw_palette"] = list(raw_palette)
        binned.info["active_palette"] = list(palette)
        notes.append(f"{len(palette)} {'filament' if filaments else 'color'}(s)")
        out = target(".svg")
        out.write_text(binned_to_svg(binned, palette, eps=svg_eps), encoding="utf-8")
        notes.append("svg")
        preview = binned
    elif do_icon:
        out = target(".ico")
        save_ico(im, out)
        notes.append(f"ico ({','.join(str(s) for s in ICON_SIZES)}px)")
    elif do_webp:
        out = target(".webp")
        im.save(out, "WEBP", lossless=True, quality=100)
        notes.append("webp")
    else:
        out = target(".png")
        im.save(out, "PNG")
    return out, ("; ".join(notes) if notes else "saved"), preview, crop_box


def process(
    path,
    do_transp,
    do_crop,
    do_icon=False,
    tol=BG_TOL,
    remove_holes=False,
    seeds=None,
    do_svg=False,
    bin_colors=False,
    n_colors=DEFAULT_SVG_COLORS,
    lasso=None,
    use_border=True,
    do_webp=False,
    out_path=None,
    crop_pad=0,
    square_canvas=False,
    edge_choke=0,
    edge_feather=0.0,
    brush_strokes=None,
    hidden_indices=None,
    color_overrides=None,
    svg_eps=0.85,
    out_size=0,
    filaments=None,
):
    out, note, _, _ = _process_with_preview(
        path,
        do_transp,
        do_crop,
        do_icon=do_icon,
        tol=tol,
        remove_holes=remove_holes,
        seeds=seeds,
        do_svg=do_svg,
        bin_colors=bin_colors,
        n_colors=n_colors,
        lasso=lasso,
        use_border=use_border,
        do_webp=do_webp,
        out_path=out_path,
        crop_pad=crop_pad,
        square_canvas=square_canvas,
        edge_choke=edge_choke,
        edge_feather=edge_feather,
        brush_strokes=brush_strokes,
        hidden_indices=hidden_indices,
        color_overrides=color_overrides,
        svg_eps=svg_eps,
        out_size=out_size,
        filaments=filaments,
    )
    return out, note


def batch_process(
    paths,
    do_transp=True,
    do_crop=False,
    do_icon=False,
    tol=BG_TOL,
    remove_holes=False,
    do_svg=False,
    bin_colors=False,
    n_colors=DEFAULT_SVG_COLORS,
    use_border=True,
    do_webp=False,
    crop_pad=0,
    square_canvas=False,
    edge_choke=0,
    edge_feather=0.0,
    svg_eps=0.85,
    rotate_deg=0,
    flip_h=False,
    flip_v=False,
    out_size=0,
    filaments=None,
    out_dir=None,
    suffix=DEFAULT_SUFFIX,
):
    """Process a sequence of image paths with shared settings. Returns `(outputs, errors)`.

    One output format per batch: `do_svg`, `do_icon`, `do_webp`, or PNG when none is set.
    Outputs go next to each original, or into `out_dir`, named `<stem><suffix>.<ext>`.
    """
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    outputs = []
    errors = []
    for p in paths:
        p = Path(p)
        try:
            src_im = load_rgba(p)
            if rotate_deg or flip_h or flip_v:
                src_im = transform_image(src_im, rotate_deg=rotate_deg, flip_h=flip_h, flip_v=flip_v)
            out, note, _, _ = _process_with_preview(
                p,
                do_transp,
                do_crop,
                do_icon=do_icon,
                tol=tol,
                remove_holes=remove_holes,
                seeds=None,
                do_svg=do_svg,
                bin_colors=bin_colors,
                n_colors=n_colors,
                lasso=None,
                use_border=use_border,
                do_webp=do_webp,
                crop_pad=crop_pad,
                square_canvas=square_canvas,
                source_im=src_im,
                edge_choke=edge_choke,
                edge_feather=edge_feather,
                svg_eps=svg_eps,
                out_size=out_size,
                filaments=filaments,
                out_dir=out_dir,
                suffix=suffix,
            )
            outputs.append((p, out, note))
        except Exception as exc:
            errors.append((p, str(exc)))
    return outputs, errors
