"""Object outline cutout: GrabCut inside a lasso, edge refinement, color decontamination."""

from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_opening, distance_transform_edt, label, uniform_filter

from .background import _ANALYSIS_LOCK, _image_key
from .config import BOX3, CROSS


GRABCUT_MAX_SIDE = 640  # GrabCut runs on the outline's window downscaled to at most this
GRABCUT_ITERS = 5
OPENCV_HINT = "Object outline needs OpenCV (opencv-python-headless) — click Dependencies to install it."

_LASSO_CACHE = OrderedDict()  # (image key, outline, hints, holes) -> (alpha, bg color, note)
_LASSO_CACHE_SIZE = 4


def _mask_bbox(fg):
    """(left, top, right, bottom) of the True pixels in a 2D mask, or None if there are none."""
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _polygon_mask(width, height, pts):
    m = Image.new("L", (width, height), 0)
    ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in pts], fill=1)
    return np.asarray(m).astype(bool)


def lasso_object_mask(rgb, lasso, bg_hints=None, remove_holes=False):
    """Cut out the object inside a loose user outline with GrabCut.

    rgb: (h, w, 3) uint8. lasso: [(x, y), ...] polygon in pixel coordinates around the object.
    bg_hints: optional (x, y) points known to be background (e.g. a gap inside the outline).
    Returns (alpha float32 0..1 of shape (h, w), background RGB color near the outline, note or None).
    Everything outside the outline is background; inside, GrabCut decides along color edges.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(OPENCV_HINT) from exc

    h, w = rgb.shape[:2]
    pts = np.asarray(lasso, dtype=np.float64)
    # Work window: the outline's bounding box plus a margin, clipped to the image
    margin = 0.1 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) + 4
    x0 = int(max(0, np.floor(pts[:, 0].min() - margin)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - margin)))
    x1 = int(min(w, np.ceil(pts[:, 0].max() + margin) + 1))
    y1 = int(min(h, np.ceil(pts[:, 1].max() + margin) + 1))
    win_w, win_h = x1 - x0, y1 - y0
    alpha = np.zeros((h, w), dtype=np.float32)
    if win_w < 3 or win_h < 3:
        return alpha, np.zeros(3, np.float32), "outline too small"

    s = min(1.0, GRABCUT_MAX_SIDE / max(win_w, win_h))
    sw, sh = max(3, round(win_w * s)), max(3, round(win_h * s))
    sx, sy = sw / win_w, sh / win_h
    win_im = Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
    small = np.asarray(win_im.resize((sw, sh), Image.BOX) if s < 1.0 else win_im)
    inside = _polygon_mask(sw, sh, [((x - x0) * sx, (y - y0) * sy) for x, y in pts])

    # Background color right around the outline: used to un-mix soft edge pixels
    ring = binary_dilation(inside, structure=BOX3, iterations=2) & ~inside
    bg_col = np.median(small[ring], axis=0).astype(np.float32) if ring.any() else np.zeros(3, np.float32)

    def polygon_fallback(note):
        poly = Image.fromarray((inside * 255).astype(np.uint8)).resize((win_w, win_h), Image.BILINEAR)
        alpha[y0:y1, x0:x1] = np.asarray(poly, dtype=np.float32) / 255.0
        return alpha, bg_col, note

    gc = np.where(inside, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    for hx, hy in bg_hints or ():
        cx, cy = int(round((hx - x0) * sx)), int(round((hy - y0) * sy))
        if 0 <= cx < sw and 0 <= cy < sh:
            gc[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2] = cv2.GC_BGD
    n_bg = int((gc == cv2.GC_BGD).sum())
    n_fg = int((gc == cv2.GC_PR_FGD).sum())
    if n_bg < 16 or n_fg < 16:  # GrabCut needs samples of both
        return polygon_fallback("outline used as-is")

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(np.ascontiguousarray(small[..., ::-1]), gc, None, bgd_model, fgd_model, GRABCUT_ITERS,
                cv2.GC_INIT_WITH_MASK)
    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    # Drop 1-px slivers (downscaling blends background edges into colors GrabCut may call object)
    opened = binary_opening(fg, structure=CROSS)
    if opened.any():
        fg = opened

    # Keep only the pieces in the heart of the outline; drop stray blobs near its edge
    lab, n = label(fg, structure=CROSS)
    if n > 1:
        depth = distance_transform_edt(inside)
        core = depth >= 0.25 * depth.max()
        keep = np.unique(lab[core & fg])
        keep = keep[keep > 0]
        if keep.size == 0:
            keep = np.array([np.argmax(np.bincount(lab.ravel())[1:]) + 1])
        fg = np.isin(lab, keep)
    if not remove_holes:
        fg = binary_fill_holes(fg)
    if not fg.any():
        return polygon_fallback("outline used as-is")

    # Back to full resolution; bilinear upscaling gives a soft ~1 source-px edge ramp
    up = Image.fromarray((fg * 255).astype(np.uint8))
    if s < 1.0:
        up = up.resize((win_w, win_h), Image.BILINEAR)
    a_win = np.asarray(up, dtype=np.float32) / 255.0
    if s < 1.0:
        a_win = _refine_edge_band(rgb[y0:y1, x0:x1], a_win, radius=int(np.ceil(1.0 / s)) + 1)
    alpha[y0:y1, x0:x1] = a_win
    return alpha, bg_col, None


def _refine_edge_band(rgb, a_up, radius):
    """Sharpen an upscaled (blurry, misplaced by up to ~1 work px) mask at full resolution.

    Only pixels within `radius` of the coarse edge change. Each one is unmixed between the
    local object color F and local background color B (box-filtered means of the confident
    pixels nearby): alpha = projection of the pixel color onto the B->F segment.
    """
    edge = (a_up > 0.02) & (a_up < 0.98)
    if not edge.any():
        return a_up
    band = binary_dilation(edge, structure=BOX3, iterations=radius)
    ys, xs = np.nonzero(band)
    by0, by1 = max(0, ys.min() - 2 * radius), min(a_up.shape[0], ys.max() + 2 * radius + 1)
    bx0, bx1 = max(0, xs.min() - 2 * radius), min(a_up.shape[1], xs.max() + 2 * radius + 1)
    sl = (slice(by0, by1), slice(bx0, bx1))
    c = rgb[sl].astype(np.float32)
    a = a_up[sl]
    fgc = (a >= 0.98) & ~band[sl]
    bgc = (a <= 0.02) & ~band[sl]
    size = 4 * radius + 1

    def local_mean(conf):
        n = uniform_filter(conf.astype(np.float32), size=size, mode="nearest")
        sums = uniform_filter(c * conf[..., None], size=(size, size, 1), mode="nearest")
        return sums / np.maximum(n, 1e-6)[..., None], n > 1e-3

    F, has_f = local_mean(fgc)
    B, has_b = local_mean(bgc)
    fb = F - B
    fb2 = (fb * fb).sum(axis=-1)
    usable = band[sl] & has_f & has_b & (fb2 > 30.0 ** 2)  # need contrast between the two sides
    proj = ((c - B) * fb).sum(axis=-1) / np.maximum(fb2, 1e-6)
    out = a_up.copy()
    region = out[sl]
    region[usable] = np.clip(proj[usable], 0.0, 1.0)
    return out


def get_lasso_mask(rgb, lasso, bg_hints=None, remove_holes=False):
    """Cached lasso_object_mask: toggling crop/binning or exporting reuses the GrabCut result."""
    key = (
        _image_key(rgb),
        tuple((round(float(x), 2), round(float(y), 2)) for x, y in lasso),
        tuple((round(float(x), 2), round(float(y), 2)) for x, y in bg_hints or ()),
        bool(remove_holes),
        GRABCUT_MAX_SIDE,
    )
    with _ANALYSIS_LOCK:
        hit = _LASSO_CACHE.get(key)
        if hit is not None:
            _LASSO_CACHE.move_to_end(key)
            return hit
    result = lasso_object_mask(rgb, lasso, bg_hints=bg_hints, remove_holes=remove_holes)
    result[0].flags.writeable = False
    with _ANALYSIS_LOCK:
        _LASSO_CACHE[key] = result
        while len(_LASSO_CACHE) > _LASSO_CACHE_SIZE:
            _LASSO_CACHE.popitem(last=False)
    return result


def _decontaminate(rgb, alpha, bg, zone):
    """Un-mix background color spill from semi-transparent pixels in `zone` (in place on float rgb).

    bg is either a per-pixel (h, w, 3) background field or a single (3,) color.
    """
    zone = zone & (alpha > 0) & (alpha < 250)
    if zone.any():
        al = (alpha[zone].astype(np.float32) / 255.0)[:, None]
        bg_z = bg[zone] if np.ndim(bg) == 3 else np.asarray(bg, np.float32)[None, :]
        fg_est = (rgb[zone] - (1.0 - al) * bg_z) / np.maximum(al, 0.2)
        rgb[zone] = np.clip(fg_est, 0.0, 255.0)
    return rgb


def _decontaminate_local(rgb, alpha, fallback_bg, reach=6):
    """Un-mix soft edge pixels against the color of their nearest fully transparent neighbor.

    Busy backgrounds have different colors on different sides of an object, so a single
    background color would tint the fringe; the nearest background pixel is what actually
    bled into each edge pixel. Works on the edge's bounding box only (in place on float rgb).
    """
    soft = (alpha > 0) & (alpha < 255)
    box = _mask_bbox(soft)
    if box is None:
        return rgb
    h, w = alpha.shape
    x0, y0 = max(0, box[0] - reach), max(0, box[1] - reach)
    x1, y1 = min(w, box[2] + reach), min(h, box[3] + reach)
    sub_a = alpha[y0:y1, x0:x1]
    sub_rgb = rgb[y0:y1, x0:x1]  # view: writes go to rgb
    clear = sub_a == 0
    if clear.any():
        iy, ix = distance_transform_edt(~clear, return_distances=False, return_indices=True)
        bg_field = sub_rgb[iy, ix]
    else:
        bg_field = np.broadcast_to(np.asarray(fallback_bg, np.float32), sub_rgb.shape)
    _decontaminate(sub_rgb, sub_a, bg_field, soft[y0:y1, x0:x1])
    return rgb
