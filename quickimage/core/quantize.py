"""Reducing the foreground to a few flat colors, with despeckling."""


import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, label

from .background import color_dist, nearest_center
from .config import CROSS, DEFAULT_SVG_COLORS


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


PALETTE_MERGE_DIST = 30.0  # color_dist below which two palette colors are the same "filament"


def _merge_close_centers(centers, labels):
    """Drop palette colors nearly identical to a more common one.

    Asking for more colors than the picture has makes k-means split one real color into
    near-identical shades, which would mean pointless extra print colors.
    """
    counts = np.bincount(labels, minlength=len(centers))
    kept = []
    for k in np.argsort(-counts, kind="stable"):
        if counts[k] == 0:
            continue
        if all(color_dist(centers[k][None], centers[j])[0] >= PALETTE_MERGE_DIST for j in kept):
            kept.append(k)
    return centers[sorted(kept)]


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
        centers = _merge_close_centers(centers, nearest_center(train_rgb, centers))

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
