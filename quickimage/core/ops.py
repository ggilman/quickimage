"""Image operations: transparency, crop, transforms, mask refinement, palette edits."""


import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, uniform_filter

from .background import compute_bg_selection
from .config import BG_TOL, MAX_BG_FRAC, SOFT_TOL
from .outline import _decontaminate, _decontaminate_local, _mask_bbox, get_lasso_mask


def make_transparent(im, tol=BG_TOL, remove_holes=False, seeds=None, lasso=None, use_border=True):
    """Make the background transparent.

    Without `lasso`: background = colors matching the image border (when `use_border` is True)
    plus picked `seeds`.
    With `lasso`: only the object inside the outline is kept (GrabCut); when `use_border` is True
    `seeds` are GrabCut background hints, otherwise `seeds` also delete clicked colors.
    """
    im = im.convert("RGBA")
    a_u8 = np.asarray(im)
    notes = []
    if lasso:
        hints = seeds if use_border else None
        alpha_f, bg_col, note = get_lasso_mask(a_u8[..., :3], lasso, bg_hints=hints, remove_holes=remove_holes)
        if note:
            notes.append(note)
        alpha = np.minimum(a_u8[..., 3], np.round(alpha_f * 255.0)).astype(np.uint8)
        rgb_out = a_u8[..., :3].astype(np.float32)
        _decontaminate_local(rgb_out, alpha, fallback_bg=bg_col)
        a_u8 = np.dstack([rgb_out.astype(np.uint8), alpha])
        im = Image.fromarray(a_u8)
        if use_border or not seeds:
            return im, ("; ".join(notes) if notes else None)
    if not use_border and not seeds:
        return im, None
    if (a_u8[..., 3] < 255).any() and not seeds:
        return im, "already transparent"
    a = a_u8.astype(np.int16)
    vis, dist, bg_field, eff_tol = compute_bg_selection(
        a, tol=tol, remove_holes=remove_holes, seeds=seeds, use_border=use_border
    )
    if vis.mean() > MAX_BG_FRAC:
        return im, "looks all background - removal skipped"
    if not vis.any():
        return im, "no background matched"

    soft_span = max(80.0, float(SOFT_TOL - BG_TOL) * (eff_tol / max(1.0, float(BG_TOL))))
    soft_tol = eff_tol + soft_span

    # Multi-pixel spatial feathering (up to 2px from background boundary)
    fg_dist = distance_transform_edt(~vis)
    fringe_zone = (~vis) & (fg_dist <= 2.0) & (dist <= soft_tol)

    alpha = a_u8[:, :, 3].copy()
    alpha[vis] = 0

    if fringe_zone.any():
        color_ramp = np.clip((dist[fringe_zone] - eff_tol) / (soft_tol - eff_tol), 0.0, 1.0)
        # Pixels 2px deep get mostly opaque unless very close in color to background
        spatial_boost = np.clip((fg_dist[fringe_zone] - 1.0) * 0.45, 0.0, 0.45)
        a_norm = np.clip(color_ramp + spatial_boost * color_ramp, 0.0, 1.0)
        new_alpha = np.round(255.0 * a_norm).astype(np.uint8)
        alpha[fringe_zone] = np.minimum(alpha[fringe_zone], new_alpha)

    # Decontaminate background color spill from semi-transparent edge pixels
    rgb_out = _decontaminate(a_u8[:, :, :3].astype(np.float32), alpha, bg_field, fringe_zone)

    out = np.dstack([rgb_out.astype(np.uint8), alpha])
    return Image.fromarray(out), ("; ".join(notes) if notes else None)


def content_bbox(im, tol=BG_TOL, remove_holes=False, seeds=None, use_border=True):
    a_u8 = np.asarray(im.convert("RGBA"))
    if (a_u8[..., 3] < 255).any():
        fg = a_u8[..., 3] > 16
    else:
        a = a_u8.astype(np.int16)
        vis, dist, _, eff_tol = compute_bg_selection(
            a, tol=tol, remove_holes=remove_holes, seeds=seeds, use_border=use_border
        )
        if vis.any():
            fg = ~vis & (dist > eff_tol)
        else:
            fg = dist > eff_tol
    return _mask_bbox(fg)


def apply_crop_options(im, pad_px=0, square=False):
    """Add transparent margin (`pad_px` on each side) and/or center on a 1:1 square canvas."""
    pad = max(0, int(round(pad_px)))
    w, h = im.size
    nw, nh = w + 2 * pad, h + 2 * pad
    if square:
        side = max(nw, nh)
        nw, nh = side, side
    if (nw, nh) == (w, h):
        return im
    rgba = im.convert("RGBA")
    canvas = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    canvas.paste(rgba, ((nw - w) // 2, (nh - h) // 2))
    return canvas


def transform_image(im, rotate_deg=0, flip_h=False, flip_v=False):
    """Rotate by a multiple of 90 degrees clockwise and/or flip horizontally/vertically."""
    out = im.convert("RGBA")
    deg = int(rotate_deg) % 360
    if deg == 90:
        out = out.transpose(Image.Transpose.ROTATE_270)
    elif deg == 180:
        out = out.transpose(Image.Transpose.ROTATE_180)
    elif deg == 270:
        out = out.transpose(Image.Transpose.ROTATE_90)
    if flip_h:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return out


def refine_alpha_mask(im, source_im=None, edge_choke=0, edge_feather=0.0, brush_strokes=None):
    """Apply touch-up brush strokes (erase/restore), edge shrink/grow (choke), and feathering.

    `brush_strokes`: list of `(mode, radius, points)` where `mode` is `'erase'` or `'restore'`,
    `radius` is in pixels, and `points` is a list of `(x, y)` coordinates in `im` pixel space.
    `edge_choke`: negative erodes (shrinks) foreground, positive dilates (grows) foreground.
    `edge_feather`: Gaussian/box softness radius in pixels applied to the alpha edge.
    """
    rgba = np.asarray(im.convert("RGBA")).copy()
    src_rgba = np.asarray((source_im if source_im is not None else im).convert("RGBA"))
    h, w, _ = rgba.shape
    alpha = rgba[..., 3].astype(np.float32)

    if brush_strokes:
        mask_im = Image.fromarray(np.clip(np.round(alpha), 0, 255).astype(np.uint8), "L")
        draw = ImageDraw.Draw(mask_im)
        restored_any = False
        for mode, radius, pts in brush_strokes:
            if not pts:
                continue
            r = max(1, int(round(radius)))
            val = 0 if mode == "erase" else 255
            if mode == "restore":
                restored_any = True
            if len(pts) == 1:
                x, y = pts[0]
                draw.ellipse([x - r, y - r, x + r, y + r], fill=val)
            else:
                flat = [int(round(c)) for pt in pts for c in pt]
                draw.line(flat, fill=val, width=2 * r, joint="curve")
                for x, y in pts:
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=val)
        new_alpha = np.asarray(mask_im).astype(np.float32)
        if restored_any and src_rgba.shape[:2] == (h, w):
            restored = (new_alpha > alpha) & (rgba[..., 3] == 0)
            rgba[restored, :3] = src_rgba[restored, :3]
        alpha = new_alpha

    choke = int(round(edge_choke))
    if choke != 0 and (alpha > 0).any() and (alpha < 255).any() or (choke < 0 and (alpha > 0).any()):
        fg = alpha >= 128.0
        if fg.any():
            r = abs(choke)
            yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
            disk = (xx * xx + yy * yy) <= r * r
            if choke < 0:
                eroded = binary_erosion(fg, structure=disk, border_value=0)
                removed = fg & ~eroded
                alpha[removed] = 0.0
                # Also pull down semi-transparent fringe adjacent to the eroded boundary
                dist_in = distance_transform_edt(eroded)
                fringe = eroded & (dist_in <= 1.5)
                alpha[fringe] = np.minimum(alpha[fringe], dist_in[fringe] * 180.0)
            else:
                dilated = binary_dilation(fg, structure=disk)
                added = dilated & ~fg
                if added.any():
                    if src_rgba.shape[:2] == (h, w):
                        rgba[added, :3] = src_rgba[added, :3]
                    alpha[added] = 255.0

    feather = float(edge_feather)
    if feather > 0.0 and (alpha < 255).any() and (alpha > 0).any():
        ksize = max(3, int(round(feather * 2)) | 1)
        smoothed = uniform_filter(alpha, size=ksize, mode="nearest")
        # Blend edge zone toward smoothed alpha while preserving deep interior opacity
        fg = alpha >= 128.0
        dist_in = distance_transform_edt(fg)
        dist_out = distance_transform_edt(~fg)
        band = (dist_in <= feather * 2.0 + 1.0) & (dist_out <= feather * 2.0 + 1.0)
        alpha = np.where(band, smoothed, alpha)

    rgba[..., 3] = np.clip(np.round(alpha), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def apply_palette_edits(binned_im, palette, hidden_indices=None, color_overrides=None):
    """Apply interactive swatch edits (hidden/deleted color bins and custom recolors) to a binned image.

    `hidden_indices`: set/iterable of 0-based palette indices to make transparent.
    `color_overrides`: dict mapping 0-based palette index -> `(r, g, b)` replacement tuple.
    Returns `(edited_image, active_palette)`.
    """
    hidden = set(hidden_indices or ())
    overrides = dict(color_overrides or {})
    if not hidden and not overrides:
        return binned_im, list(palette)
    rgba = np.asarray(binned_im.convert("RGBA")).copy()
    opaque = rgba[..., 3] > 0
    active_palette = []
    for idx, rgb in enumerate(palette):
        mask = opaque & (rgba[..., 0] == rgb[0]) & (rgba[..., 1] == rgb[1]) & (rgba[..., 2] == rgb[2])
        if idx in hidden:
            rgba[mask] = 0
        else:
            new_rgb = tuple(int(v) for v in overrides.get(idx, rgb))
            if new_rgb != rgb:
                rgba[mask, 0] = new_rgb[0]
                rgba[mask, 1] = new_rgb[1]
                rgba[mask, 2] = new_rgb[2]
            if new_rgb not in active_palette:
                active_palette.append(new_rgb)
    return Image.fromarray(rgba, "RGBA"), active_palette
