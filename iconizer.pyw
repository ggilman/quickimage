#!python3.12
import site
import subprocess
import sys
import tkinter
from pathlib import Path

_here = Path(__file__).resolve().parent
_venv_sp = _here / ".venv" / "Lib" / "site-packages"
if _venv_sp.is_dir():
    if sys.version_info[:2] != (3, 12):
        _exe = "pythonw.exe" if sys.executable.lower().endswith("w.exe") else "python.exe"
        _venv_py = _here / ".venv" / "Scripts" / _exe
        if _venv_py.is_file():
            raise SystemExit(subprocess.call([str(_venv_py), str(Path(__file__).resolve()), *sys.argv[1:]]))
    site.addsitedir(str(_venv_sp))

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

BG_TOL = 90
SOFT_TOL = 330
MAX_BG_FRAC = 0.97
DEFAULT_SVG_COLORS = 4
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ico"}
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
BOX3 = np.ones((3, 3), dtype=bool)


def manhattan_dist(a, col):
    return np.abs(a[:, :, :3] - col).sum(axis=2)


def color_dist(a, bg):
    """Perceptual redmean-weighted distance scaled to match Manhattan [0..765] range."""
    rgb = a[:, :, :3].astype(np.float32)
    bg_f = np.asarray(bg, dtype=np.float32)
    diff = rgb - bg_f
    r_mean = 0.5 * (rgb[:, :, 0] + (bg_f[:, :, 0] if bg_f.ndim == 3 else bg_f[0]))
    w_r = 2.0 + r_mean / 256.0
    w_g = 4.0
    w_b = 2.0 + (255.0 - r_mean) / 256.0
    # Max possible value of sqrt(w_r*dR^2 + w_g*dG^2 + w_b*dB^2) is sqrt(9 * 255^2) = 3 * 255 = 765.
    euclid = np.sqrt(w_r * diff[:, :, 0] ** 2 + w_g * diff[:, :, 1] ** 2 + w_b * diff[:, :, 2] ** 2)
    l1 = np.abs(diff).sum(axis=2)
    return 0.55 * euclid + 0.45 * l1


def sample_bg_color(a):
    border = np.concatenate([a[0, :, :3], a[-1, :, :3], a[1:-1, 0, :3], a[1:-1, -1, :3]])
    return np.median(border, axis=0)


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


def seed_bg_mask(a, seeds, tol):
    """Flood-fill background regions from user-picked (x, y) seed points."""
    h, w = a.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    dist_map = np.full((h, w), np.inf, dtype=np.float32)
    bg_ref = np.zeros((h, w, 3), dtype=np.float32)

    for sx, sy in seeds:
        ix = int(np.clip(round(sx), 0, w - 1))
        iy = int(np.clip(round(sy), 0, h - 1))
        y0, y1 = max(0, iy - 1), min(h, iy + 2)
        x0, x1 = max(0, ix - 1), min(w, ix + 2)
        patch = a[y0:y1, x0:x1, :3].reshape(-1, 3)
        scol = np.median(patch, axis=0)
        d = color_dist(a, scol)
        near = d <= tol
        near[iy, ix] = True
        lab, _ = label(near, structure=BOX3)
        target = lab[iy, ix]
        if target > 0:
            comp = lab == target
            mask |= comp
        better = d < dist_map
        dist_map = np.where(better, d, dist_map)
        bg_ref[better] = scol
    return mask, dist_map, bg_ref


def compute_bg_selection(a, tol=BG_TOL, remove_holes=False, seeds=None):
    """Return (bg_mask, dist_field, bg_color_field, eff_tol) for image array `a`."""
    bg_field, noise_mad = estimate_bg_field(a)
    eff_tol = float(tol) + min(18.0, noise_mad * 0.75)
    dist = color_dist(a, bg_field)
    vis = border_bg_mask(dist, eff_tol)

    if seeds:
        s_mask, s_dist, s_bg = seed_bg_mask(a, seeds, eff_tol)
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


def render_preview(im, do_transp, do_crop, do_icon=False, tol=BG_TOL, remove_holes=False, seeds=None):
    """Run transparency/crop/icon steps in-memory without writing files."""
    notes = []
    crop_box = (0, 0, im.width, im.height)
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
    preview = make_icon_frame(im, max(ICON_SIZES)) if do_icon else im
    return im, preview, notes, crop_box


def _process_with_preview(path, do_transp, do_crop, do_icon=False, tol=BG_TOL, remove_holes=False, seeds=None):
    path = Path(path)
    im = load_rgba(path)
    im, preview, notes, crop_box = render_preview(
        im, do_transp, do_crop, do_icon, tol=tol, remove_holes=remove_holes, seeds=seeds
    )
    if do_icon:
        out = next_save_path(path, ".ico")
        save_ico(im, out)
        notes.append(f"ico ({','.join(str(s) for s in ICON_SIZES)}px)")
    else:
        out = next_save_path(path, ".png")
        im.save(out, "PNG")
    return out, ("; ".join(notes) if notes else "saved"), preview, crop_box


def process(path, do_transp, do_crop, do_icon=False, tol=BG_TOL, remove_holes=False, seeds=None):
    out, note, _, _ = _process_with_preview(
        path, do_transp, do_crop, do_icon, tol=tol, remove_holes=remove_holes, seeds=seeds
    )
    return out, note


def is_image_file(p):
    return Path(p).suffix.lower() in IMG_EXT


class App:
    CAP_W = 880
    CAP_H = 640

    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Iconizer")
        self.root.geometry("920x580")
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.canvas = None
        self.bw = 0
        self.bh = 0
        self.current_path = None
        self.source_rgba = None
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

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.set_box(440, 380)

        self.opt_transp = tkinter.BooleanVar(value=False)
        self.opt_holes = tkinter.BooleanVar(value=False)
        self.opt_crop = tkinter.BooleanVar(value=False)
        self.opt_icon = tkinter.BooleanVar(value=False)
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
        self.chk_i = tkinter.Checkbutton(
            panel, variable=self.opt_icon, text="create icon", command=self.schedule_live_preview
        )

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
        self.btn = tkinter.Button(panel, text="Apply (Save)", command=self.apply)

        self.chk_t.grid(row=0, sticky="w")
        self.chk_h.grid(row=1, sticky="w", padx=(18, 0))
        self.chk_c.grid(row=2, sticky="w")
        self.chk_i.grid(row=3, sticky="w")
        self.tol_label.grid(row=4, sticky="w", pady=(10, 0))
        self.tol_scale.grid(row=5, sticky="w")
        self.seed_info.grid(row=6, sticky="w", pady=(8, 4))
        self.btn_reset_seeds.grid(row=7, sticky="w", pady=(2, 4))
        self.btn.grid(row=8, pady=(10, 0), sticky="w")

        self.status = tkinter.Label(
            self.root,
            text="Drop a picture on the box. Options apply on drop; Apply saves a new file.",
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
        do_i = self.opt_icon.get()
        tol = self.opt_tol.get()
        if not (do_t or do_c or do_i):
            self._crop_box = (0, 0, self.source_rgba.width, self.source_rgba.height)
            self._is_icon_preview = False
            self.last_preview = self.source_rgba
            self.last_label = f"preview: {self.current_path.name} (original)"
            self.show_image(self.source_rgba, self.last_label)
            return
        _, preview, notes, crop_box = render_preview(
            self.source_rgba,
            do_t,
            do_c,
            do_i,
            tol=tol,
            remove_holes=do_h,
            seeds=self.seeds if self.seeds else None,
        )
        self._crop_box = crop_box
        self._is_icon_preview = bool(do_i)
        seed_txt = f", {len(self.seeds)} bg pick(s)" if self.seeds else ""
        note_str = "; ".join(notes) if notes else "ready"
        self.last_preview = preview
        self.last_label = f"preview: {self.current_path.name} ({note_str}, tol={tol}{seed_txt}) - click Apply to save"
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
            self.canvas.itemconfigure("box", fill="#F4F4F4")
        else:
            if had_hover and self.last_preview is not None:
                self.fit_display(self.last_preview.size)
                self.show_image(self.last_preview, self.last_label)
            self.canvas.itemconfigure("box", fill="#EAF4E4")
        return COPY

    def on_drop(self, event):
        self.hover_path = None
        for p in self.dropped_paths(event):
            if not is_image_file(p):
                continue
            path = Path(p)
            try:
                rgba = load_rgba(path)
            except Exception:
                self.set_msg(f"could not open: {path.name}")
                continue
            self.current_path = path
            self.source_rgba = rgba
            self.seeds = []
            self._crop_box = (0, 0, rgba.width, rgba.height)
            self._is_icon_preview = False
            self.fit_display(rgba.size)
            do_t, do_c, do_i = self.opt_transp.get(), self.opt_crop.get(), self.opt_icon.get()
            self.canvas.itemconfigure("box", fill="#EAF4E4")
            self.last_preview = rgba
            self.last_label = f"dropped: {self.current_path.name}"
            self.show_image(rgba, self.last_label)
            if do_t or do_c or do_i:
                self._run_process(do_t, do_c, do_i)
            else:
                self.last_label = f"dropped: {self.current_path.name} - tick options, click bg, or click Apply"
                self.set_msg(self.last_label)
        return COPY

    def _run_process(self, do_t, do_c, do_i):
        try:
            out, note, preview, crop_box = _process_with_preview(
                self.current_path,
                do_t,
                do_c,
                do_i,
                tol=self.opt_tol.get(),
                remove_holes=self.opt_holes.get(),
                seeds=self.seeds if self.seeds else None,
            )
        except Exception as exc:
            self.set_msg(f"error: {exc}")
            return
        self._crop_box = crop_box
        self._is_icon_preview = bool(do_i)
        self.last_preview = preview
        self.last_label = f"{self.current_path.name} -> {out.name}: {note}"
        self.show_image(preview, self.last_label)

    def apply(self):
        if self.current_path is None:
            self.set_msg("no picture dropped yet")
            return
        do_t, do_c, do_i = self.opt_transp.get(), self.opt_crop.get(), self.opt_icon.get()
        if not (do_t or do_c or do_i):
            self.set_msg("no options ticked")
            return
        self._run_process(do_t, do_c, do_i)

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