"""Background detection: estimated background field, border and seed masks."""

import hashlib
import threading
from collections import OrderedDict

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, label, median_filter

from .config import BG_TOL, BOX3, DIST_CHUNK, SEED_REACH


def color_dist(a, bg):
    """Perceptual redmean-weighted distance scaled to match Manhattan [0..765] range."""
    rgb = a[..., :3].astype(np.float32)
    bg_f = np.asarray(bg, dtype=np.float32)
    diff = rgb - bg_f
    r_mean = 0.5 * (rgb[..., 0] + bg_f[..., 0])
    w_r = 2.0 + r_mean / 256.0
    w_g = 4.0
    w_b = 2.0 + (255.0 - r_mean) / 256.0
    # Max possible value of sqrt(w_r*dR^2 + w_g*dG^2 + w_b*dB^2) is sqrt(9 * 255^2) = 3 * 255 = 765.
    euclid = np.sqrt(w_r * diff[..., 0] ** 2 + w_g * diff[..., 1] ** 2 + w_b * diff[..., 2] ** 2)
    l1 = np.abs(diff).sum(axis=-1)
    return 0.55 * euclid + 0.45 * l1


def nearest_center(rgb, centers, chunk=DIST_CHUNK):
    """Index of the closest center (by color_dist) for each row of an (N, 3) array, in bounded memory."""
    centers = np.asarray(centers, dtype=np.float32)
    out = np.empty(len(rgb), dtype=np.int32)
    for s in range(0, len(rgb), chunk):
        part = rgb[s : s + chunk]
        dists = np.stack([color_dist(part, c) for c in centers], axis=1)
        out[s : s + chunk] = np.argmin(dists, axis=1)
    return out


def _clean_edge_strip(strip, global_med, outlier_thresh):
    dev = np.abs(strip - global_med).sum(axis=1)
    bad = dev > outlier_thresh
    out = strip.astype(np.float32).copy()
    if bad.any() and (~bad).any():
        good_idx = np.where(~bad)[0]
        bad_idx = np.where(bad)[0]
        for c in range(3):
            out[bad_idx, c] = np.interp(bad_idx, good_idx, out[good_idx, c])
    elif bad.all():
        out[:] = global_med
    k = max(3, min(15, (len(strip) // 8) | 1))
    return median_filter(out, size=(k, 1), mode="nearest")


def estimate_bg_field(a):
    """Estimate a smooth 2D background RGB surface from the image borders (Coons patch)."""
    h, w = a.shape[:2]
    rgb = a[:, :, :3].astype(np.float32)
    border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[1:-1, 0], rgb[1:-1, -1]], axis=0)
    global_med = np.median(border, axis=0)
    l1_dev = np.abs(border - global_med).sum(axis=1)
    mad = float(np.median(l1_dev))
    if h < 8 or w < 8 or mad < 2.0:
        return np.broadcast_to(global_med.reshape(1, 1, 3), (h, w, 3)).copy(), mad

    # Allow smooth gradients across the border while rejecting foreground objects touching an edge
    p85 = float(np.percentile(l1_dev, 85))
    outlier_thresh = max(90.0, p85 * 1.6, mad * 4.5)

    top = _clean_edge_strip(rgb[0, :], global_med, outlier_thresh)
    bot = _clean_edge_strip(rgb[-1, :], global_med, outlier_thresh)
    left = _clean_edge_strip(rgb[:, 0], global_med, outlier_thresh)
    right = _clean_edge_strip(rgb[:, -1], global_med, outlier_thresh)

    v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]

    tb = (1.0 - v) * top[None, :, :] + v * bot[None, :, :]
    lr = (1.0 - u) * left[:, None, :] + u * right[:, None, :]
    c00 = 0.5 * (top[0] + left[0])
    c10 = 0.5 * (top[-1] + right[0])
    c01 = 0.5 * (bot[0] + left[-1])
    c11 = 0.5 * (bot[-1] + right[-1])
    corners = (
        (1.0 - u) * (1.0 - v) * c00
        + u * (1.0 - v) * c10
        + (1.0 - u) * v * c01
        + u * v * c11
    )
    field = np.clip(tb + lr - corners, 0.0, 255.0)

    # Compute residual noise relative to the smooth border model
    border_model = np.concatenate([field[0, :], field[-1, :], field[1:-1, 0], field[1:-1, -1]], axis=0)
    res_mad = float(np.median(np.abs(border - border_model).sum(axis=1)))
    return field, res_mad


def border_bg_mask(dist, tol):
    near = dist <= tol
    lab, n = label(near)
    if n == 0:
        return np.zeros(dist.shape, dtype=bool)
    bl = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bl = bl[bl > 0]
    if bl.size == 0:
        return np.zeros(dist.shape, dtype=bool)
    return np.isin(lab, bl)


class _Analysis:
    """Tolerance-independent per-image results, so slider drags and option toggles only re-threshold."""

    __slots__ = ("bg_field", "noise_mad", "dist", "seed_dist")

    def __init__(self, a):
        self.bg_field, self.noise_mad = estimate_bg_field(a)
        self.dist = color_dist(a, self.bg_field)
        self.seed_dist = {}  # (ix, iy) -> (seed color, distance map)
        for arr in (self.bg_field, self.dist):
            arr.flags.writeable = False  # shared between calls: must never be modified in place

    def for_seed(self, a, ix, iy):
        hit = self.seed_dist.get((ix, iy))
        if hit is None:
            h, w = a.shape[:2]
            patch = a[max(0, iy - 1) : min(h, iy + 2), max(0, ix - 1) : min(w, ix + 2), :3].reshape(-1, 3)
            scol = np.median(patch, axis=0)
            d = color_dist(a, scol)
            d.flags.writeable = False
            hit = self.seed_dist[(ix, iy)] = (scol, d)
        return hit


_ANALYSIS_CACHE = OrderedDict()  # image digest -> _Analysis; small LRU (live preview + an export)
_ANALYSIS_CACHE_SIZE = 2
_ANALYSIS_LOCK = threading.Lock()


def _image_key(a):
    """Exact content key for an image array (used by the analysis and outline caches)."""
    return (a.shape, a.dtype.str, hashlib.blake2b(np.ascontiguousarray(a).data, digest_size=16).digest())


def get_analysis(a):
    """Cached _Analysis for image array `a`, keyed by an exact content digest."""
    key = _image_key(a)
    with _ANALYSIS_LOCK:
        hit = _ANALYSIS_CACHE.get(key)
        if hit is not None:
            _ANALYSIS_CACHE.move_to_end(key)
            return hit
    result = _Analysis(a)  # computed outside the lock so an export never blocks the preview
    with _ANALYSIS_LOCK:
        _ANALYSIS_CACHE[key] = result
        _ANALYSIS_CACHE.move_to_end(key)
        while len(_ANALYSIS_CACHE) > _ANALYSIS_CACHE_SIZE:
            _ANALYSIS_CACHE.popitem(last=False)
    return result


def seed_tolerance(seed, default):
    """A picked point is `(x, y)` or `(x, y, tolerance)`; the tolerance sticks with that click."""
    return float(seed[2]) if len(seed) > 2 and seed[2] is not None else float(default)


def seed_bg_mask(a, seeds, tol, analysis=None, noise_pad=0.0):
    """Flood-fill background regions from user-picked seed points.

    Each seed uses its own tolerance when it has one (else `tol`), plus `noise_pad`.
    Returns (mask, distance to the owning seed's color, that color, that seed's tolerance).
    """
    h, w = a.shape[:2]
    analysis = analysis or get_analysis(a)
    mask = np.zeros((h, w), dtype=bool)
    dist_map = np.full((h, w), np.inf, dtype=np.float32)
    bg_ref = np.zeros((h, w, 3), dtype=np.float32)
    tol_map = np.full((h, w), float(tol) + noise_pad, dtype=np.float32)

    for seed in seeds:
        sx, sy = seed[0], seed[1]
        seed_tol = seed_tolerance(seed, tol) + noise_pad
        ix = int(np.clip(round(sx), 0, w - 1))
        iy = int(np.clip(round(sy), 0, h - 1))
        scol, d = analysis.for_seed(a, ix, iy)
        near = d <= seed_tol
        near[iy, ix] = True
        lab, _ = label(near, structure=BOX3)
        target = lab[iy, ix]
        if target == 0:
            continue
        comp = lab == target
        mask |= comp
        # The picked color only counts as background in and just around the picked region,
        # so same-colored foreground elsewhere is not feathered or removed as a "hole".
        reach = binary_dilation(comp, structure=BOX3, iterations=SEED_REACH)
        better = reach & (d < dist_map)
        dist_map = np.where(better, d, dist_map)
        bg_ref[better] = scol
        tol_map[better] = seed_tol
    return mask, dist_map, bg_ref, tol_map


def compute_bg_selection(a, tol=BG_TOL, remove_holes=False, seeds=None, use_border=True):
    """Return (bg_mask, dist_field, bg_color_field, eff_tol) for image array `a`.

    `eff_tol` is a number, or a per-pixel array when seeds carry their own tolerances (each pixel
    then uses the tolerance of the seed whose color it is compared with).
    The returned dist/bg_color arrays may be shared cache entries: treat them as read-only.
    """
    an = get_analysis(a)
    bg_field, dist = an.bg_field, an.dist
    noise_pad = min(18.0, an.noise_mad * 0.75)
    eff_tol = float(tol) + noise_pad
    if use_border:
        vis = border_bg_mask(dist, eff_tol)
    else:
        h, w = a.shape[:2]
        vis = np.zeros((h, w), dtype=bool)
        dist = np.full((h, w), np.inf, dtype=np.float32)
        bg_field = np.zeros((h, w, 3), dtype=np.float32)

    if seeds:
        s_mask, s_dist, s_bg, s_tol = seed_bg_mask(a, seeds, tol, analysis=an, noise_pad=noise_pad)
        vis = vis | s_mask
        use_seed = s_dist < dist
        dist = np.where(use_seed, s_dist, dist)
        bg_field = np.where(use_seed[:, :, None], s_bg, bg_field)
        if any(len(s) > 2 for s in seeds):
            eff_tol = np.where(use_seed, s_tol, np.float32(eff_tol))

    if remove_holes:
        # Enclosed background holes are regions matching background color that sit inside the foreground silhouette
        fg_Rough = ~vis
        fg_filled = binary_fill_holes(fg_Rough)
        interior = fg_filled & ~vis
        hole_candidates = interior & (dist <= eff_tol)
        if hole_candidates.any():
            vis = vis | hole_candidates

    return vis, dist, bg_field, eff_tol
