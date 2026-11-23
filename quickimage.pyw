#!python3
# Runs on any Python 3.9+. Dependencies: see requirements.txt (installed into .venv next to this file).
import hashlib
import os
import queue
import site
import subprocess
import sys
import threading
import tkinter
from collections import OrderedDict
from pathlib import Path

_here = Path(__file__).resolve().parent
_venv = _here / ".venv"
_venv_sp = _venv / "Lib" / "site-packages"
_RELAUNCH_ENV = "QUICKIMAGE_RELAUNCHED"

APP_NAME = "Quick Image"
APP_ICON_ICO = _here / "assets" / "quickimage.ico"
APP_ICON_PNG = _here / "assets" / "quickimage.png"
APP_USER_MODEL_ID = "QuickImage.App"  # Windows taskbar identity, so it shows our icon, not Python's


def _venv_python_version(venv):
    """Return (major, minor) from the venv's pyvenv.cfg, or None if unknown."""
    try:
        for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, val = line.partition("=")
            if key.strip() in ("version", "version_info"):
                parts = val.strip().split(".")
                return int(parts[0]), int(parts[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _running_in_venv(venv):
    try:
        return Path(sys.prefix).resolve() == venv.resolve()
    except OSError:
        return False


if _venv_sp.is_dir() and not _running_in_venv(_venv):
    # Re-launch under the venv interpreter only when run as a script, and only once.
    if __name__ == "__main__" and not os.environ.get(_RELAUNCH_ENV):
        _exe = "pythonw.exe" if sys.executable.lower().endswith("w.exe") else "python.exe"
        _venv_py = _venv / "Scripts" / _exe
        if _venv_py.is_file():
            _env = dict(os.environ, **{_RELAUNCH_ENV: "1"})
            raise SystemExit(subprocess.call([str(_venv_py), str(Path(__file__).resolve()), *sys.argv[1:]], env=_env))
    # Borrowing the venv's packages is only safe when the interpreter versions match.
    if _venv_python_version(_venv) == sys.version_info[:2]:
        site.addsitedir(str(_venv_sp))

try:
    import numpy as np
    from PIL import Image, ImageOps, ImageTk
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        distance_transform_edt,
        label,
        median_filter,
    )

    from tkinterdnd2 import TkinterDnD, DND_FILES, COPY
except ImportError as _exc:
    if __name__ != "__main__":
        raise
    # pythonw has no console, so a missing package would otherwise fail silently
    _msg = (
        f"{APP_NAME} could not start: {_exc}\n\n"
        f"Python {sys.version.split()[0]} at {sys.executable}\n\n"
        "Install the dependencies into a venv next to quickimage.pyw:\n"
        "    py -m venv .venv\n"
        "    .venv\\Scripts\\python -m pip install -r requirements.txt"
    )
    try:
        from tkinter import messagebox

        _r = tkinter.Tk()
        _r.withdraw()
        messagebox.showerror(APP_NAME, _msg)
        _r.destroy()
    except Exception:
        print(_msg, file=sys.stderr)
    raise SystemExit(1)

BG_TOL = 90
SOFT_TOL = 330
MAX_BG_FRAC = 0.97
DEFAULT_SVG_COLORS = 4
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ico"}
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
PREVIEW_MAX_SIDE = 1600  # live previews run on a proxy no larger than this
DIST_CHUNK = 1 << 18  # pixels per chunk when assigning pixels to palette colors
SEED_REACH = 3  # px around a picked region where the picked color acts as background

BOX3 = np.ones((3, 3), dtype=bool)
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


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


def get_analysis(a):
    """Cached _Analysis for image array `a`, keyed by an exact content digest."""
    key = (a.shape, a.dtype.str, hashlib.blake2b(np.ascontiguousarray(a).data, digest_size=16).digest())
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


def seed_bg_mask(a, seeds, tol, analysis=None):
    """Flood-fill background regions from user-picked (x, y) seed points."""
    h, w = a.shape[:2]
    analysis = analysis or get_analysis(a)
    mask = np.zeros((h, w), dtype=bool)
    dist_map = np.full((h, w), np.inf, dtype=np.float32)
    bg_ref = np.zeros((h, w, 3), dtype=np.float32)

    for sx, sy in seeds:
        ix = int(np.clip(round(sx), 0, w - 1))
        iy = int(np.clip(round(sy), 0, h - 1))
        scol, d = analysis.for_seed(a, ix, iy)
        near = d <= tol
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
    return mask, dist_map, bg_ref


def compute_bg_selection(a, tol=BG_TOL, remove_holes=False, seeds=None):
    """Return (bg_mask, dist_field, bg_color_field, eff_tol) for image array `a`.

    The returned dist/bg_color arrays may be shared cache entries: treat them as read-only.
    """
    an = get_analysis(a)
    bg_field, dist = an.bg_field, an.dist
    eff_tol = float(tol) + min(18.0, an.noise_mad * 0.75)
    vis = border_bg_mask(dist, eff_tol)

    if seeds:
        s_mask, s_dist, s_bg = seed_bg_mask(a, seeds, eff_tol, analysis=an)
        vis = vis | s_mask
        use_seed = s_dist < dist
        dist = np.where(use_seed, s_dist, dist)
        bg_field = np.where(use_seed[:, :, None], s_bg, bg_field)

    if remove_holes:
        # Enclosed background holes are regions matching background color that sit inside the foreground silhouette
        fg_Rough = ~vis
        fg_filled = binary_fill_holes(fg_Rough)
        interior = fg_filled & ~vis
        hole_candidates = interior & (dist <= eff_tol)
        if hole_candidates.any():
            vis = vis | hole_candidates

    return vis, dist, bg_field, eff_tol


def load_rgba(path):
    with Image.open(path) as src:
        try:
            src.seek(0)
        except Exception:
            pass
        im = ImageOps.exif_transpose(src)
        return im.convert("RGBA")


def make_transparent(im, tol=BG_TOL, remove_holes=False, seeds=None):
    im = im.convert("RGBA")
    a_u8 = np.asarray(im)
    if (a_u8[..., 3] < 255).any() and not seeds:
        return im, "already transparent"
    a = a_u8.astype(np.int16)
    vis, dist, bg_field, eff_tol = compute_bg_selection(a, tol=tol, remove_holes=remove_holes, seeds=seeds)
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
    rgb_out = a_u8[:, :, :3].astype(np.float32).copy()
    decontam = fringe_zone & (alpha > 0) & (alpha < 250)
    if decontam.any():
        al = (alpha[decontam].astype(np.float32) / 255.0)[:, None]
        denom = np.maximum(al, 0.2)
        fg_est = (rgb_out[decontam] - (1.0 - al) * bg_field[decontam]) / denom
        rgb_out[decontam] = np.clip(fg_est, 0.0, 255.0)

    out = np.dstack([rgb_out.astype(np.uint8), alpha])
    return Image.fromarray(out), None


def content_bbox(im, tol=BG_TOL, remove_holes=False, seeds=None):
    a_u8 = np.asarray(im.convert("RGBA"))
    if (a_u8[..., 3] < 255).any():
        fg = a_u8[..., 3] > 16
    else:
        a = a_u8.astype(np.int16)
        vis, dist, _, eff_tol = compute_bg_selection(a, tol=tol, remove_holes=remove_holes, seeds=seeds)
        if vis.any():
            fg = ~vis & (dist > eff_tol)
        else:
            fg = dist > eff_tol
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


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


def _pack_rgb(rgb):
    rgb = rgb.astype(np.uint32)
    return (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


def _few_unique_colors(rgb, limit):
    """(colors, counts) if an (N, 3) uint8 array has at most `limit` distinct colors, else None.

    A subsample is checked first, so photos (many colors) never pay for a full-size unique.
    """
    probe = rgb[:: max(1, len(rgb) // 65536)]
    if len(np.unique(_pack_rgb(probe))) > limit:
        return None
    keys, counts = np.unique(_pack_rgb(rgb), return_counts=True)
    if len(keys) > limit:
        return None
    colors = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], axis=1).astype(np.uint8)
    return colors, counts


def _label_color_components(idx_map, n_bins):
    """Globally unique 4-connected component ids per color bin (0 = transparent)."""
    comp = np.zeros(idx_map.shape, dtype=np.int64)
    total = 0
    for k in range(n_bins):
        lab, n = label(idx_map == k, structure=CROSS)
        if n:
            comp += np.where(lab > 0, lab + total, 0)  # bins are disjoint, so adding is a merge
            total += n
    return comp, total


def _neighbor_pairs(a):
    """(this, neighbor) slice pairs for all four 4-neighborhood directions."""
    return (
        (a[:-1, :], a[1:, :]),
        (a[1:, :], a[:-1, :]),
        (a[:, :-1], a[:, 1:]),
        (a[:, 1:], a[:, :-1]),
    )


def _despeckle(idx_map, fg_mask, n_bins, min_area, max_passes=8):
    """Merge 4-connected color components smaller than `min_area` into their dominant neighbor color.

    Vectorized: in each pass every small component votes over the colors of its edge-adjacent
    pixels, but only toward strictly larger neighbor components (size, then id). Merges
    therefore always flow toward bigger regions and cannot cycle; passes repeat until stable.
    """
    big_vote = 1 << 40
    for _ in range(max_passes):
        comp, total = _label_color_components(idx_map, n_bins)
        if total == 0:
            return
        sizes = np.bincount(comp.ravel(), minlength=total + 1)
        small = sizes < min_area
        small[0] = False
        if not small.any():
            return
        keys, weights = [], []
        for (c_a, c_b), (i_a, i_b) in zip(_neighbor_pairs(comp), _neighbor_pairs(idx_map)):
            sel = small[c_a] & (i_b >= 0) & (i_b != i_a)
            ca, cb, ib = c_a[sel], c_b[sel], i_b[sel]
            sa, sb = sizes[ca], sizes[cb]
            larger = (sb > sa) | ((sb == sa) & (cb > ca))
            ca, cb, ib, sb = ca[larger], cb[larger], ib[larger], sb[larger]
            keys.append(ca * n_bins + ib)
            weights.append(np.where(small[cb], sb, big_vote))
        keys = np.concatenate(keys)
        if keys.size == 0:
            return
        weights = np.concatenate(weights)
        # Sum vote weight per (component, color), then keep the heaviest color per component
        order = np.argsort(keys, kind="stable")
        keys, weights = keys[order], weights[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        uk = keys[starts]
        tw = np.add.reduceat(weights, starts)
        cid, col = uk // n_bins, uk % n_bins
        best = np.lexsort((-tw, cid))
        first = np.r_[True, cid[best][1:] != cid[best][:-1]]
        new_color = np.full(total + 1, -1, dtype=np.int64)
        new_color[cid[best][first]] = col[best][first]
        target = new_color[comp]
        change = target >= 0
        if not change.any():
            return
        idx_map[change] = target[change]


def quantize_foreground_colors(im, n_colors=DEFAULT_SVG_COLORS, despeckle_area=9, keep_alpha=False):
    """Quantize opaque foreground pixels into the N best representative colors.

    With keep_alpha=False the result has crisp {0, 255} alpha (needed for SVG tracing).
    With keep_alpha=True the original alpha is kept and semi-transparent edge pixels are
    recolored to their nearest palette color too, so feathered edges survive.
    """
    n_colors = max(1, min(16, int(n_colors)))
    rgba = np.asarray(im.convert("RGBA")).copy()
    h, w, _ = rgba.shape
    fg_mask = rgba[..., 3] >= 128

    if not fg_mask.any():
        rgba[:] = 0
        return Image.fromarray(rgba, "RGBA"), []

    core_mask = binary_erosion(fg_mask, structure=np.ones((3, 3), dtype=bool))
    sample_mask = core_mask if int(core_mask.sum()) >= max(16, n_colors * 4) else fg_mask
    sample_rgb = rgba[..., :3][sample_mask]

    few = _few_unique_colors(sample_rgb, n_colors)
    if few is not None:
        uniq_colors, uniq_counts = few
        order = np.argsort(-uniq_counts)
        centers = uniq_colors[order].astype(np.float32)
    else:
        n_samp = len(sample_rgb)
        side = max(1, int(np.ceil(np.sqrt(n_samp))))
        # Pad to a square with evenly spread (deterministic) picks, not the first rows repeated
        extra = side * side - n_samp
        fill_idx = np.linspace(0, n_samp - 1, extra).round().astype(np.intp) if extra else np.empty(0, np.intp)
        pad_rgb = np.concatenate([sample_rgb, sample_rgb[fill_idx]]).reshape(side, side, 3)
        q_im = Image.fromarray(pad_rgb, "RGB").quantize(
            colors=n_colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
        )
        raw_pal = q_im.getpalette() or []
        used_idx = sorted(set(np.asarray(q_im).ravel().tolist()))
        init_list = [raw_pal[idx * 3 : idx * 3 + 3] for idx in used_idx if idx * 3 + 2 < len(raw_pal)]
        if not init_list:
            init_list = sample_rgb[:: max(1, n_samp // n_colors)][:n_colors].tolist()
        centers = np.array(init_list[:n_colors], dtype=np.float32)

        step = max(1, n_samp // 16000) if n_samp > 16000 else 1
        train_rgb = sample_rgb[::step].astype(np.float32)

        for _ in range(8):
            labels_k = nearest_center(train_rgb, centers)
            new_centers = centers.copy()
            for k in range(len(centers)):
                pts = train_rgb[labels_k == k]
                if len(pts) > 0:
                    new_centers[k] = np.median(pts, axis=0)
            if np.allclose(new_centers, centers, atol=0.5):
                centers = new_centers
                break
            centers = new_centers

    palette = np.unique(np.clip(np.round(centers), 0, 255).astype(np.uint8), axis=0)

    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[fg_mask] = nearest_center(rgba[..., :3][fg_mask].astype(np.float32), palette)

    if despeckle_area > 1 and len(palette) > 1:
        # Work only inside the foreground's bounding box (views write through to idx_map)
        rows = np.flatnonzero(fg_mask.any(axis=1))
        cols = np.flatnonzero(fg_mask.any(axis=0))
        win = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        _despeckle(idx_map[win], fg_mask[win], len(palette), despeckle_area)

    bin_counts = np.bincount(idx_map[fg_mask], minlength=len(palette))
    used_bins = np.flatnonzero(bin_counts)
    order = np.argsort(-bin_counts[used_bins], kind="stable")
    ordered_palette = []
    remap = np.zeros(len(palette), dtype=np.int32)
    for new_i, old_i in enumerate(used_bins[order]):
        remap[old_i] = new_i
        ordered_palette.append(tuple(int(v) for v in palette[old_i]))

    idx_map[fg_mask] = remap[idx_map[fg_mask]]
    pal_arr = np.array(ordered_palette, dtype=np.uint8)

    out_rgba = np.zeros_like(rgba)
    out_rgba[fg_mask, :3] = pal_arr[idx_map[fg_mask]]
    if keep_alpha:
        soft = (rgba[..., 3] > 0) & ~fg_mask
        if soft.any():
            out_rgba[soft, :3] = pal_arr[nearest_center(rgba[..., :3][soft].astype(np.float32), pal_arr)]
        out_rgba[..., 3] = rgba[..., 3]
    else:
        out_rgba[fg_mask, 3] = 255
    return Image.fromarray(out_rgba, "RGBA"), ordered_palette


def _trace_mask_loops(mask):
    """Extract closed boundary loops along the pixel grid for a binary mask."""
    m = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    top_y, top_x = np.where(m[1:-1, 1:-1] & ~m[:-2, 1:-1])
    right_y, right_x = np.where(m[1:-1, 1:-1] & ~m[1:-1, 2:])
    bot_y, bot_x = np.where(m[1:-1, 1:-1] & ~m[2:, 1:-1])
    left_y, left_x = np.where(m[1:-1, 1:-1] & ~m[1:-1, :-2])

    adj = {}

    def _add_edges(xs0, ys0, xs1, ys1):
        for x0, y0, x1, y1 in zip(xs0.tolist(), ys0.tolist(), xs1.tolist(), ys1.tolist()):
            adj.setdefault((x0, y0), []).append((x1, y1))

    _add_edges(top_x, top_y, top_x + 1, top_y)
    _add_edges(right_x + 1, right_y, right_x + 1, right_y + 1)
    _add_edges(bot_x + 1, bot_y + 1, bot_x, bot_y + 1)
    _add_edges(left_x, left_y + 1, left_x, left_y)

    loops = []
    while adj:
        start = next(iter(adj))
        loop = [start]
        curr = start
        prev_dx, prev_dy = 0, 0
        while True:
            nxts = adj.get(curr)
            if not nxts:
                break
            if len(nxts) == 1:
                nxt = nxts.pop()
            else:
                best_idx = 0
                for idx, cand in enumerate(nxts):
                    if (cand[0] - curr[0], cand[1] - curr[1]) == (-prev_dy, prev_dx):
                        best_idx = idx
                        break
                nxt = nxts.pop(best_idx)
            if not nxts:
                del adj[curr]
            prev_dx, prev_dy = nxt[0] - curr[0], nxt[1] - curr[1]
            if nxt == start:
                break
            loop.append(nxt)
            curr = nxt
        if len(loop) >= 3:
            simplified = []
            n = len(loop)
            for i in range(n):
                p_prev, p_curr, p_next = loop[(i - 1) % n], loop[i], loop[(i + 1) % n]
                if (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) != (p_curr[1] - p_prev[1]) * (
                    p_next[0] - p_curr[0]
                ):
                    simplified.append(p_curr)
            if len(simplified) >= 3:
                loops.append(np.array(simplified, dtype=np.float64))
    return loops


def _rdp_open(pts, eps):
    if len(pts) <= 2:
        return pts
    p0, p1 = pts[0], pts[-1]
    vec = p1 - p0
    norm = float(np.hypot(vec[0], vec[1]))
    if norm < 1e-9:
        dists = np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
    else:
        dists = np.abs(vec[0] * (p0[1] - pts[:, 1]) - (p0[0] - pts[:, 0]) * vec[1]) / norm
    idx = int(np.argmax(dists))
    if dists[idx] > eps:
        left = _rdp_open(pts[: idx + 1], eps)
        right = _rdp_open(pts[idx:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([p0, p1])


def _shared_vertex_mask(idx_map):
    """(h+1, w+1) bool: grid vertices touching two or more different opaque color bins.

    Boundaries through these vertices are shared by two color layers, so they must be
    emitted identically by both layers (no smoothing / independent simplification).
    """
    p = np.pad(idx_map, 1, mode="constant", constant_values=-1)
    quads = np.stack([p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]])
    hi = quads.max(axis=0)
    lo = np.where(quads >= 0, quads, np.iinfo(quads.dtype).max).min(axis=0)
    return (hi >= 0) & (lo < hi)


def _smooth_and_simplify_loop(pts, eps=0.85, locked=None):
    if len(pts) <= 4:
        return pts
    if locked is not None and locked.any():
        return _smooth_and_simplify_pinned(pts, locked, eps)
    seg_lens = np.hypot(np.roll(pts[:, 0], -1) - pts[:, 0], np.roll(pts[:, 1], -1) - pts[:, 1])
    is_sharp_corner = (seg_lens >= 3.0) & (np.roll(seg_lens, 1) >= 3.0)
    smoothed = _smooth_loop(pts, is_sharp_corner)
    dists = np.hypot(smoothed[:, 0] - smoothed[0, 0], smoothed[:, 1] - smoothed[0, 1])
    i1 = int(np.argmax(dists))
    if i1 == 0 or len(smoothed) < 5:
        return smoothed
    half1 = _rdp_open(smoothed[: i1 + 1], eps)
    half2 = _rdp_open(np.vstack([smoothed[i1:], smoothed[:1]]), eps)
    res = np.vstack([half1[:-1], half2[:-1]])
    return res if len(res) >= 3 else smoothed


def _smooth_loop(pts, keep):
    smoothed = pts.copy()
    for _ in range(2):
        mid = 0.25 * np.roll(smoothed, 1, axis=0) + 0.5 * smoothed + 0.25 * np.roll(smoothed, -1, axis=0)
        smoothed = np.where(keep[:, None], smoothed, mid)
    return smoothed


def _smooth_and_simplify_pinned(pts, locked, eps):
    """Smooth/simplify only the runs between pinned (shared-boundary) vertices."""
    seg_lens = np.hypot(np.roll(pts[:, 0], -1) - pts[:, 0], np.roll(pts[:, 1], -1) - pts[:, 1])
    is_sharp_corner = (seg_lens >= 3.0) & (np.roll(seg_lens, 1) >= 3.0)
    smoothed = _smooth_loop(pts, is_sharp_corner | locked)
    pins = np.flatnonzero(locked)
    n = len(pts)
    out = []
    for i, a in enumerate(pins):
        b = pins[(i + 1) % len(pins)]
        span = (b - a) % n or n  # one pin: the run wraps the whole loop
        run = smoothed[(a + np.arange(span + 1)) % n]
        simp = _rdp_open(run, eps) if span > 1 else run
        out.append(simp[:-1])
    res = np.vstack(out)
    return res if len(res) >= 3 else smoothed


def image_to_svg(im, n_colors=DEFAULT_SVG_COLORS):
    """Bin foreground to N colors and trace each color bin into a vector SVG string."""
    binned_im, palette = quantize_foreground_colors(im, n_colors=n_colors)
    return binned_to_svg(binned_im, palette)


def binned_to_svg(binned_im, palette):
    """Trace an already color-binned RGBA image (one path per palette color) into an SVG string."""
    rgba = np.asarray(binned_im.convert("RGBA"))
    h, w, _ = rgba.shape
    opaque = rgba[..., 3] >= 128
    idx_map = np.full((h, w), -1, dtype=np.int32)
    masks = []
    for idx, rgb in enumerate(palette):
        mask = opaque & (rgba[..., 0] == rgb[0]) & (rgba[..., 1] == rgb[1]) & (rgba[..., 2] == rgb[2])
        idx_map[mask] = idx
        masks.append(mask)
    shared = _shared_vertex_mask(idx_map)

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for idx, (rgb, mask) in enumerate(zip(palette, masks)):
        if not mask.any():
            continue
        subpaths = []
        for lp in _trace_mask_loops(mask):
            vi = lp.astype(np.intp)
            simp = _smooth_and_simplify_loop(lp, locked=shared[vi[:, 1], vi[:, 0]])
            if len(simp) >= 3:
                coords = [f"{round(float(p[0]), 2):g} {round(float(p[1]), 2):g}" for p in simp]
                subpaths.append("M " + " L ".join(coords) + " Z")
        if subpaths:
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            lines.append(
                f'  <path id="color_{idx + 1}" fill="{hex_color}" fill-rule="evenodd" d="{" ".join(subpaths)}"/>'
            )
    lines.append("</svg>\n")
    return "\n".join(lines)


def save_svg(im, out, n_colors=DEFAULT_SVG_COLORS):
    svg_text = image_to_svg(im, n_colors=n_colors)
    Path(out).write_text(svg_text, encoding="utf-8")
    return svg_text


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
):
    """Run transparency/crop/binning/icon steps in-memory without writing files."""
    notes = []
    crop_box = (0, 0, im.width, im.height)
    # Background picks are part of transparency; ignore them when it is switched off.
    seeds = seeds if do_transp else None
    if do_transp:
        im, note = make_transparent(im, tol=tol, remove_holes=remove_holes, seeds=seeds)
        if note:
            notes.append(note)
    if do_crop:
        box = content_bbox(im, tol=tol, remove_holes=remove_holes, seeds=seeds)
        if box and box != (0, 0, im.width, im.height):
            crop_box = box
            im = im.crop(box)
            notes.append(f"cropped to {box[2]-box[0]}x{box[3]-box[1]}")
        else:
            notes.append("nothing to crop")
    if bin_colors:
        im, palette = quantize_foreground_colors(im, n_colors=n_colors, keep_alpha=True)
        notes.append(f"{len(palette)} color(s)")
    preview = make_icon_frame(im, max(ICON_SIZES)) if do_icon else im
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
):
    path = Path(path)
    im = load_rgba(path)
    im, preview, notes, crop_box = render_preview(
        im,
        do_transp,
        do_crop,
        do_icon=do_icon and not do_svg,
        tol=tol,
        remove_holes=remove_holes,
        seeds=seeds,
        bin_colors=bool(bin_colors and not do_svg),  # SVG bins below, exactly once
        n_colors=n_colors,
    )
    if do_svg:
        binned, palette = quantize_foreground_colors(im, n_colors=n_colors)
        notes.append(f"{len(palette)} color(s)")
        out = next_save_path(path, ".svg")
        out.write_text(binned_to_svg(binned, palette), encoding="utf-8")
        notes.append("svg")
        preview = binned
    elif do_icon:
        out = next_save_path(path, ".ico")
        save_ico(im, out)
        notes.append(f"ico ({','.join(str(s) for s in ICON_SIZES)}px)")
    else:
        out = next_save_path(path, ".png")
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
    )
    return out, note


def is_image_file(p):
    return Path(p).suffix.lower() in IMG_EXT


def set_app_user_model_id():
    """Give the process its own Windows taskbar identity.

    Without this, Windows groups the window under pythonw.exe and shows Python's icon
    in the taskbar instead of ours.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def set_window_icon(root):
    """Title bar / taskbar / Alt-Tab icon. Missing icon files are not fatal."""
    try:
        if sys.platform == "win32" and APP_ICON_ICO.is_file():
            root.iconbitmap(default=str(APP_ICON_ICO))  # .ico carries crisp small sizes
        elif APP_ICON_PNG.is_file():
            photo = tkinter.PhotoImage(file=str(APP_ICON_PNG))
            root.iconphoto(True, photo)
            root._app_icon = photo  # keep a reference so Tk does not drop the image
    except tkinter.TclError:
        pass


class App:
    CAP_W = 880
    CAP_H = 640

    def __init__(self):
        set_app_user_model_id()  # must happen before the first window is created
        self.root = TkinterDnD.Tk()
        self.root.title(APP_NAME)
        set_window_icon(self.root)
        self.root.geometry("920x580")
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.canvas = None
        self.bw = 0
        self.bh = 0
        self.current_path = None
        self.source_rgba = None
        self.preview_src = None  # source_rgba, downscaled for fast live previews of big images
        self.preview_scale = 1.0  # preview_src size / source_rgba size
        self.hover_path = None
        self.tkimg = None
        self.last_preview = None
        self.last_label = None
        self.seeds = []
        self._preview_job = None
        self._disp_scale = 1.0
        self._disp_size = (1, 1)
        self._crop_box = None
        self._is_icon_preview = False
        self._busy = False  # an export is running on the worker thread

    def _get_color_count(self):
        """Return (count, valid). Invalid spinbox text falls back to the default count."""
        try:
            return max(1, min(16, int(self.opt_colors.get()))), True
        except (tkinter.TclError, ValueError):
            return DEFAULT_SVG_COLORS, False

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.set_box(440, 380)

        self.opt_transp = tkinter.BooleanVar(value=False)
        self.opt_holes = tkinter.BooleanVar(value=False)
        self.opt_crop = tkinter.BooleanVar(value=False)
        self.opt_bin = tkinter.BooleanVar(value=False)
        self.opt_colors = tkinter.IntVar(value=DEFAULT_SVG_COLORS)
        self.opt_tol = tkinter.IntVar(value=BG_TOL)

        panel = tkinter.Frame(self.root)
        panel.grid(row=0, column=1, padx=24, pady=24, sticky="n")
        self.chk_t = tkinter.Checkbutton(
            panel, variable=self.opt_transp, text="make background transparent", command=self.schedule_live_preview
        )
        self.chk_h = tkinter.Checkbutton(
            panel, variable=self.opt_holes, text="remove interior holes", command=self.schedule_live_preview
        )
        self.chk_c = tkinter.Checkbutton(
            panel, variable=self.opt_crop, text="crop to image", command=self.schedule_live_preview
        )

        bin_row = tkinter.Frame(panel)
        self.chk_b = tkinter.Checkbutton(
            bin_row, variable=self.opt_bin, text="bin colors:", command=self.schedule_live_preview
        )
        self.spin_colors = tkinter.Spinbox(
            bin_row,
            from_=1,
            to=16,
            width=4,
            textvariable=self.opt_colors,
        )
        # Fires on arrow clicks and on every keystroke, so the preview always matches the export
        self.opt_colors.trace_add("write", lambda *_a: self.schedule_live_preview())
        self.chk_b.pack(side="left")
        self.spin_colors.pack(side="left", padx=(4, 0))

        self.tol_label = tkinter.Label(panel, text="tolerance:")
        self.tol_scale = tkinter.Scale(
            panel,
            from_=10,
            to=240,
            orient="horizontal",
            variable=self.opt_tol,
            length=180,
            command=lambda _v: self.schedule_live_preview(),
        )
        self.seed_info = tkinter.Label(
            panel, text="tip: click image to pick bg\n(Shift+click add, right-click reset)", justify="left", fg="#555555"
        )
        self.btn_reset_seeds = tkinter.Button(panel, text="Reset picked bg", command=self.clear_seeds)

        btn_row = tkinter.Frame(panel)
        self.btn = tkinter.Button(btn_row, text="Apply (PNG)", command=self.apply)
        self.btn_ico = tkinter.Button(btn_row, text="Export Icon (.ico)", command=self.export_icon)
        self.btn_svg = tkinter.Button(btn_row, text="Export SVG (.svg)", command=self.export_svg)
        self.btn.pack(side="left", padx=(0, 6))
        self.btn_ico.pack(side="left", padx=(0, 6))
        self.btn_svg.pack(side="left")

        self.chk_t.grid(row=0, sticky="w")
        self.chk_h.grid(row=1, sticky="w", padx=(18, 0))
        self.chk_c.grid(row=2, sticky="w")
        bin_row.grid(row=3, sticky="w", pady=(2, 0))
        self.tol_label.grid(row=4, sticky="w", pady=(10, 0))
        self.tol_scale.grid(row=5, sticky="w")
        self.seed_info.grid(row=6, sticky="w", pady=(8, 4))
        self.btn_reset_seeds.grid(row=7, sticky="w", pady=(2, 4))
        btn_row.grid(row=8, pady=(10, 0), sticky="w")

        self.status = tkinter.Label(
            self.root,
            text="Drop a picture on the box. Use Apply (PNG), Export Icon (.ico), or Export SVG (.svg).",
            justify="left",
        )
        self.status.grid(row=1, column=0, columnspan=2, pady=(16, 8), sticky="n")

    def set_box(self, bw, bh):
        self.bw, self.bh = bw, bh
        if self.canvas is None:
            self.canvas = tkinter.Canvas(self.root, width=bw, height=bh, cursor="crosshair")
            self.canvas.grid(row=0, column=0, padx=24, pady=(56, 24), sticky="n")
            self.canvas.create_rectangle(0, 0, bw, bh, fill="#F4F4F4", outline="#777777", width=2, tags="box")
            self.canvas.create_text(bw // 2, bh - 18, text="drag a picture onto this box", fill="#999999", tags="hint")
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.canvas.bind("<Button-3>", lambda _e: self.clear_seeds())
        else:
            self.canvas.configure(width=bw, height=bh)
            self.canvas.coords("box", 0, 0, bw, bh)
            self.canvas.coords("hint", bw // 2, bh - 18)

    def reset_canvas(self):
        """Back to the empty drop box (no picture loaded)."""
        self.canvas.delete("preview")
        self.tkimg = None
        if not self.canvas.find_withtag("hint"):
            self.canvas.create_text(
                self.bw // 2, self.bh - 18, text="drag a picture onto this box", fill="#999999", tags="hint"
            )
        self.canvas.itemconfigure("box", fill="#F4F4F4")

    def fit_display(self, size):
        w, h = size
        s = min(self.CAP_W / w, self.CAP_H / h, 2.0)
        bw = max(200, int(round(w * s)))
        bh = max(180, int(round(h * s)))
        if (bw, bh) != (self.bw, self.bh):
            self.set_box(bw, bh)
            self.root.geometry(f"{bw + 440}x{max(bh + 170, 560)}")

    def set_msg(self, text):
        self.status.configure(text=text)

    def show_image(self, pil_img, label):
        bw, bh = self.bw, self.bh
        scale = min((bw - 8) / pil_img.width, (bh - 8) / pil_img.height, 3.0)
        w = max(1, int(round(pil_img.width * scale)))
        h = max(1, int(round(pil_img.height * scale)))
        self._disp_scale = scale
        self._disp_size = (w, h)
        small = pil_img.resize((w, h), Image.LANCZOS)
        if small.mode == "RGBA" and (np.asarray(small)[..., 3] < 255).any():
            yy = np.arange(h)[:, None] // 6
            xx = np.arange(w)[None, :] // 6
            tile = np.where(((yy + xx) % 2).astype(bool), 204, 255).astype(np.uint8)
            back = Image.fromarray(np.repeat(tile[:, :, None], 3, axis=2))
            small = Image.alpha_composite(back.convert("RGBA"), small).convert("RGB")
        photo = ImageTk.PhotoImage(small)
        self.canvas.delete("preview")
        self.canvas.delete("hint")
        self.canvas.create_image(bw // 2, bh // 2, image=photo, anchor="center", tags="preview")
        self.tkimg = photo
        if label:
            self.status.configure(text=label)

    def on_canvas_click(self, event):
        if self.source_rgba is None:
            return
        src_w, src_h = self.source_rgba.size
        disp_w, disp_h = self._disp_size
        x0 = (self.bw - disp_w) / 2.0
        y0 = (self.bh - disp_h) / 2.0
        rel_x = (event.x - x0) / max(1.0, float(disp_w))
        rel_y = (event.y - y0) / max(1.0, float(disp_h))
        if not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
            return

        cx0, cy0, cx1, cy1 = self._crop_box or (0, 0, src_w, src_h)
        cw = max(1, cx1 - cx0)
        ch = max(1, cy1 - cy0)

        if self._is_icon_preview:
            icon_s = float(max(ICON_SIZES))
            sc = min(icon_s / cw, icon_s / ch)
            nw = max(1.0, round(cw * sc))
            nh = max(1.0, round(ch * sc))
            pad_x = (icon_s - nw) / 2.0
            pad_y = (icon_s - nh) / 2.0
            px = rel_x * icon_s
            py = rel_y * icon_s
            if not (pad_x <= px <= pad_x + nw and pad_y <= py <= pad_y + nh):
                return
            rel_x = (px - pad_x) / nw
            rel_y = (py - pad_y) / nh

        ix = int(np.clip(round(cx0 + rel_x * (cw - 1)), 0, src_w - 1))
        iy = int(np.clip(round(cy0 + rel_y * (ch - 1)), 0, src_h - 1))

        shift_held = bool(event.state & 0x0001)
        if shift_held:
            self.seeds.append((ix, iy))
        else:
            self.seeds = [(ix, iy)]
        if not self.opt_transp.get():
            self.opt_transp.set(True)
        self.update_live_preview()

    def clear_seeds(self):
        if not self.seeds:
            return
        self.seeds = []
        self.update_live_preview()

    def schedule_live_preview(self):
        if self.source_rgba is None:
            return
        if self._preview_job is not None:
            try:
                self.root.after_cancel(self._preview_job)
            except Exception:
                pass
        self._preview_job = self.root.after(50, self.update_live_preview)

    def update_live_preview(self):
        self._preview_job = None
        if self.source_rgba is None:
            return
        do_t = self.opt_transp.get()
        do_h = self.opt_holes.get()
        do_c = self.opt_crop.get()
        do_b = self.opt_bin.get()
        n_colors, colors_ok = self._get_color_count()
        tol = self.opt_tol.get()
        if not (do_t or do_c or do_b):
            self._crop_box = (0, 0, self.source_rgba.width, self.source_rgba.height)
            self._is_icon_preview = False
            self.last_preview = self.source_rgba
            self.last_label = f"preview: {self.current_path.name} (original)"
            self.show_image(self.source_rgba, self.last_label)
            return
        s = self.preview_scale
        seeds = [(x * s, y * s) for x, y in self.seeds] or None
        try:
            _, preview, notes, crop_box = render_preview(
                self.preview_src,
                do_t,
                do_c,
                do_icon=False,
                tol=tol,
                remove_holes=do_h,
                seeds=seeds,
                bin_colors=do_b,
                n_colors=n_colors,
            )
        except Exception as exc:  # pythonw has no console: surface the error in the UI
            self.set_msg(f"preview error: {exc}")
            return
        # Click mapping works in full-resolution source coordinates
        src_w, src_h = self.source_rgba.size
        x0, y0, x1, y1 = crop_box
        self._crop_box = (
            int(np.floor(x0 / s)),
            int(np.floor(y0 / s)),
            min(src_w, int(np.ceil(x1 / s))),
            min(src_h, int(np.ceil(y1 / s))),
        )
        self._is_icon_preview = False
        seed_txt = f", {len(self.seeds)} bg pick(s)" if self.seeds else ""
        if do_b and not colors_ok:
            notes.append(f"invalid color count, using {n_colors}")
        note_str = "; ".join(notes) if notes else "ready"
        self.last_preview = preview
        self.last_label = f"preview: {self.current_path.name} ({note_str}, tol={tol}{seed_txt}) - click Apply/Export to save"
        self.show_image(preview, self.last_label)

    def dropped_paths(self, event):
        raw = getattr(event, "data", None)
        if not raw:
            return []
        try:
            return list(self.root.tk.splitlist(raw))
        except Exception:
            return [raw]

    def on_hover(self, event):
        self.canvas.itemconfigure("box", fill="#FFF8D8")
        paths = self.dropped_paths(event)
        p = paths[0] if paths else None
        if p and is_image_file(p) and p != self.hover_path:
            self.hover_path = p
            try:
                rgba = load_rgba(p)
                self.fit_display(rgba.size)
                self.canvas.itemconfigure("box", fill="#FFF8D8")
                self.show_image(rgba, f"hovering: {Path(p).name}")
            except Exception:
                pass
        return COPY

    def on_leave(self, event):
        had_hover = self.hover_path is not None
        self.hover_path = None
        if self.current_path is None:
            self.reset_canvas()
            if had_hover:
                self.set_msg("Drop a picture on the box. Use Apply (PNG), Export Icon (.ico), or Export SVG (.svg).")
        else:
            if had_hover and self.last_preview is not None:
                self.fit_display(self.last_preview.size)
                self.show_image(self.last_preview, self.last_label)
            self.canvas.itemconfigure("box", fill="#EAF4E4")
        return COPY

    def _restore_after_failed_drop(self, msg):
        if self.current_path is None:
            self.reset_canvas()
        elif self.last_preview is not None:
            self.fit_display(self.last_preview.size)
            self.show_image(self.last_preview, None)
            self.canvas.itemconfigure("box", fill="#EAF4E4")
        self.set_msg(msg)

    def on_drop(self, event):
        self.hover_path = None
        paths = self.dropped_paths(event)
        images = [p for p in paths if is_image_file(p)]
        if not images:
            self._restore_after_failed_drop("not an image file - drop a .png, .jpg, .bmp, .gif, .tif, .webp or .ico")
            return COPY
        # One picture at a time: use the first one that opens
        path = rgba = None
        failed = []
        for p in images:
            try:
                rgba = load_rgba(p)
                path = Path(p)
                break
            except Exception:
                failed.append(Path(p).name)
        if path is None:
            self._restore_after_failed_drop(f"could not open: {', '.join(failed)}")
            return COPY

        self.current_path = path
        self.source_rgba = rgba
        s = min(1.0, PREVIEW_MAX_SIDE / max(rgba.size))
        if s < 1.0:
            pw = max(1, round(rgba.width * s))
            ph = max(1, round(rgba.height * s))
            self.preview_src = rgba.resize((pw, ph), Image.LANCZOS)
            self.preview_scale = pw / rgba.width
        else:
            self.preview_src = rgba
            self.preview_scale = 1.0
        self.seeds = []
        self._crop_box = (0, 0, rgba.width, rgba.height)
        self._is_icon_preview = False
        self.fit_display(rgba.size)
        do_t, do_c, do_b = self.opt_transp.get(), self.opt_crop.get(), self.opt_bin.get()
        self.canvas.itemconfigure("box", fill="#EAF4E4")
        self.last_preview = rgba
        self.last_label = f"dropped: {path.name}"
        self.show_image(rgba, self.last_label)
        skipped = len(paths) - 1 - len(failed)
        extra = f" ({skipped} other file(s) ignored - one picture at a time)" if skipped > 0 else ""
        if do_t or do_c or do_b:
            self.update_live_preview()
            if extra:
                self.last_label += extra
                self.set_msg(self.last_label)
        else:
            self.last_label = f"dropped: {path.name} - tick options, click bg, or click Apply / Export{extra}"
            self.set_msg(self.last_label)
        return COPY

    def _set_busy(self, busy):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.btn, self.btn_ico, self.btn_svg):
            b.configure(state=state)
        self.root.configure(cursor="watch" if busy else "")

    def _run_process(self, do_t, do_c, do_icon=False, do_svg=False, bin_colors=False):
        """Export on a worker thread so the window stays responsive; results come back via a queue.

        Tk is not thread-safe, so every option is read here on the UI thread and the worker
        never touches a widget; _poll_export applies the result on the UI thread.
        """
        if self._busy:
            return
        n_colors, _ = self._get_color_count()
        path = self.current_path
        kwargs = dict(
            do_icon=do_icon,
            tol=self.opt_tol.get(),
            remove_holes=self.opt_holes.get(),
            seeds=list(self.seeds) or None,
            do_svg=do_svg,
            bin_colors=bin_colors,
            n_colors=n_colors,
        )
        results = queue.Queue()

        def work():
            try:
                results.put(("ok", _process_with_preview(path, do_t, do_c, **kwargs)))
            except Exception as exc:
                results.put(("error", exc))

        kind = "SVG" if do_svg else "icon" if do_icon else "PNG"
        self._set_busy(True)
        self.set_msg(f"exporting {kind} for {path.name}...")
        threading.Thread(target=work, name="quickimage-export", daemon=True).start()
        self.root.after(50, self._poll_export, results, path, bool(do_icon))

    def _poll_export(self, results, path, do_icon):
        try:
            kind, value = results.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_export, results, path, do_icon)
            return
        self._set_busy(False)
        if kind == "error":
            self.set_msg(f"error: {value}")
            return
        out, note, preview, crop_box = value
        label = f"{path.name} -> {out.name}: {note}"
        if path != self.current_path:  # another picture was dropped meanwhile: don't replace its preview
            self.set_msg(f"saved {label}")
            return
        self._crop_box = crop_box
        self._is_icon_preview = do_icon
        self.last_preview = preview
        self.last_label = label
        self.show_image(preview, self.last_label)

    def apply(self):
        if self.current_path is None:
            self.set_msg("no picture dropped yet")
            return
        do_t, do_c, do_b = self.opt_transp.get(), self.opt_crop.get(), self.opt_bin.get()
        if not (do_t or do_c or do_b):
            self.set_msg("no options ticked")
            return
        self._run_process(do_t, do_c, do_icon=False, do_svg=False, bin_colors=do_b)

    def export_icon(self):
        if self.current_path is None:
            self.set_msg("no picture dropped yet")
            return
        do_t, do_c, do_b = self.opt_transp.get(), self.opt_crop.get(), self.opt_bin.get()
        self._run_process(do_t, do_c, do_icon=True, do_svg=False, bin_colors=do_b)

    def export_svg(self):
        if self.current_path is None:
            self.set_msg("no picture dropped yet")
            return
        do_t, do_c = self.opt_transp.get(), self.opt_crop.get()
        self._run_process(do_t, do_c, do_icon=False, do_svg=True, bin_colors=True)

    def bind_events(self):
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.on_drop)
        self.root.dnd_bind("<<DropPosition>>", self.on_hover)
        self.root.dnd_bind("<<DropLeave>>", self.on_leave)


def main():
    check = "--check" in sys.argv
    app = App()
    app.build_ui()
    app.bind_events()
    if check:
        app.root.destroy()
        return
    app.root.mainloop()


if __name__ == "__main__":
    main()