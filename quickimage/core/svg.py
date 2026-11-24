"""Tracing flat-color images into SVG paths."""

from pathlib import Path

import numpy as np

from .config import DEFAULT_SVG_COLORS
from .ops import apply_palette_edits
from .quantize import quantize_foreground_colors


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


def image_to_svg(im, n_colors=DEFAULT_SVG_COLORS, eps=0.85, hidden_indices=None, color_overrides=None):
    """Bin foreground to N colors and trace each color bin into a vector SVG string."""
    binned_im, palette = quantize_foreground_colors(im, n_colors=n_colors)
    if hidden_indices or color_overrides:
        binned_im, palette = apply_palette_edits(
            binned_im, palette, hidden_indices=hidden_indices, color_overrides=color_overrides
        )
    return binned_to_svg(binned_im, palette, eps=eps)


def binned_to_svg(binned_im, palette, eps=0.85):
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
            simp = _smooth_and_simplify_loop(lp, eps=eps, locked=shared[vi[:, 1], vi[:, 0]])
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


def save_svg(im, out, n_colors=DEFAULT_SVG_COLORS, eps=0.85):
    svg_text = image_to_svg(im, n_colors=n_colors, eps=eps)
    Path(out).write_text(svg_text, encoding="utf-8")
    return svg_text
