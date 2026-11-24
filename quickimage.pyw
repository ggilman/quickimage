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
import tkinter.font
from tkinter import filedialog, ttk
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


def set_app_user_model_id():
    """Give the process its own Windows taskbar identity and make it DPI-aware.

    Without the identity, Windows groups the window under pythonw.exe and shows Python's
    icon in the taskbar. Without DPI awareness, Windows bitmap-stretches the whole window
    on high-DPI screens, which makes all text blurry. Both must happen before Tk starts.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
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


# --- dependencies (standard library only: this must work when packages are missing) -----------

DEPENDENCIES = (
    # (import name, pip package, what it is used for, required to start the app)
    ("numpy", "numpy", "Image math", True),
    ("PIL", "pillow", "Opening and saving images", True),
    ("scipy", "scipy", "Background removal and color cleanup", True),
    ("tkinterdnd2", "tkinterdnd2", "Drag and drop", True),
    ("cv2", "opencv-python-headless", "Object outline (cut out one object)", False),
    ("sv_ttk", "sv-ttk", "Modern Windows 11 look", False),
    ("darkdetect", "darkdetect", "Follow the Windows light / dark setting", False),
)
STATUS_OK = "#2E9E5B"
STATUS_MISSING = "#D64545"


def missing_dependencies(deps=None):
    """The DEPENDENCIES entries that the running interpreter cannot import."""
    import importlib
    import importlib.util

    importlib.invalidate_caches()  # see packages installed while the app is running
    missing = []
    for dep in DEPENDENCIES if deps is None else deps:
        try:
            found = importlib.util.find_spec(dep[0]) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(dep)
    return missing


def venv_python(venv=_venv, windowed=False):
    if sys.platform == "win32":
        return venv / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
    return venv / "bin" / "python"


def _console_python(exe):
    """python.exe next to pythonw.exe: pip and venv need a console interpreter."""
    p = Path(exe)
    if p.name.lower() == "pythonw.exe" and p.with_name("python.exe").is_file():
        return str(p.with_name("python.exe"))
    return str(p)


def install_commands(venv=_venv, deps=None):
    """Commands that make the project's .venv complete: create it if needed, then pip install.

    Everything goes into .venv (where the launcher runs the app from). All packages are
    listed; pip skips the ones already installed, and a brand-new venv gets everything.
    """
    deps = DEPENDENCIES if deps is None else deps
    cmds = []
    if not venv_python(venv).is_file():
        cmds.append([_console_python(sys.executable), "-m", "venv", str(venv)])
    cmds.append(
        [str(venv_python(venv)), "-m", "pip", "install", "--disable-pip-version-check", *[d[1] for d in deps]]
    )
    return cmds


def run_commands(cmds, on_line, on_done):
    """Run commands one after another (call from a worker thread), streaming their output."""
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flash
    try:
        for cmd in cmds:
            on_line("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            for line in proc.stdout:
                on_line(line.rstrip())
            code = proc.wait()
            if code != 0:
                on_done(False, f"{Path(cmd[0]).name} exited with code {code}")
                return
    except OSError as exc:
        on_done(False, str(exc))
        return
    on_done(True, None)


def restart_app():
    """Start a fresh copy of the app (under .venv when it exists); the caller then exits."""
    exe = venv_python(windowed=True)
    if not exe.is_file():
        exe = Path(sys.executable)
    env = {k: v for k, v in os.environ.items() if k != _RELAUNCH_ENV}
    subprocess.Popen([str(exe), str(Path(__file__).resolve())], env=env)


class DependencyDialog:
    """Lists the packages Quick Image uses, shows which are missing, and installs them.

    Uses only tkinter/ttk, so it also works at startup when required packages are missing.
    """

    def __init__(self, parent, startup_error=None, on_restart=None):
        self.parent = parent
        self.on_restart = on_restart
        self.events = queue.Queue()
        self.top = top = tkinter.Toplevel(parent)
        top.title(f"{APP_NAME} — Dependencies")
        top.resizable(True, True)
        try:
            if parent.winfo_viewable():
                top.transient(parent)
        except tkinter.TclError:
            pass
        self.frame = frame = ttk.Frame(top, padding=18)
        frame.grid(sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.missing = missing_dependencies()
        if startup_error:
            intro = (
                f"{APP_NAME} could not start because required packages are missing "
                f"({startup_error}). It can install them now into the .venv folder next to quickimage.pyw."
            )
        elif self.missing:
            intro = "Some packages are missing. Quick Image can install them into its .venv folder."
        else:
            intro = "Everything Quick Image uses is installed."
        ttk.Label(frame, text=intro, wraplength=520, justify="left").grid(row=0, column=0, sticky="w")

        # Fonts derived from the theme's label font, so they match it at any DPI
        body = tkinter.font.Font(top, font=ttk.Style(top).lookup("TLabel", "font") or "TkDefaultFont")
        self._head_font = body.copy()
        self._head_font.configure(weight="bold")
        self._log_font = tkinter.font.Font(top, family="Consolas" if sys.platform == "win32" else "TkFixedFont",
                                           size=body.actual("size"))

        table = ttk.Frame(frame)
        table.grid(row=1, column=0, sticky="ew", pady=(14, 10))
        table.columnconfigure(1, weight=1)
        for c, head in enumerate(("Package", "Used for", "Status")):
            ttk.Label(table, text=head, font=self._head_font).grid(
                row=0, column=c, sticky="w", padx=(0, 18), pady=(0, 4)
            )
        self.status_labels = {}
        missing_names = {d[0] for d in self.missing}
        for r, (mod, pip_name, purpose, required) in enumerate(DEPENDENCIES, start=1):
            ttk.Label(table, text=pip_name).grid(row=r, column=0, sticky="w", padx=(0, 18), pady=1)
            ttk.Label(table, text=purpose + ("" if required else " (optional)")).grid(
                row=r, column=1, sticky="w", padx=(0, 18), pady=1
            )
            is_missing = mod in missing_names
            lbl = ttk.Label(
                table,
                text=("Missing" if is_missing else "Installed"),
                foreground=STATUS_MISSING if is_missing else STATUS_OK,
            )
            lbl.grid(row=r, column=2, sticky="w", pady=1)
            self.status_labels[mod] = lbl

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.log = tkinter.Text(frame, height=10, width=80, wrap="none", font=self._log_font, state="disabled")
        self.result = ttk.Label(frame, text="", wraplength=520, justify="left")

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, sticky="e", pady=(12, 0))
        self.btn_install = ttk.Button(
            buttons,
            text=f"Install missing ({len(self.missing)})" if self.missing else "Reinstall all",
            command=self.install,
        )
        try:
            self.btn_install.configure(style="Accent.TButton")
        except tkinter.TclError:
            pass
        self.btn_restart = ttk.Button(buttons, text=f"Restart {APP_NAME}", command=self.restart)
        self.btn_close = ttk.Button(buttons, text="Close", command=self.close)
        self.btn_install.grid(row=0, column=0, padx=(0, 8))
        self.btn_close.grid(row=0, column=2)
        top.protocol("WM_DELETE_WINDOW", self.close)
        top.bind("<Escape>", lambda _e: self.close())
        self._installing = False

    def install(self):
        if self._installing:
            return
        self._installing = True
        self.btn_install.state(["disabled"])
        self.btn_close.state(["disabled"])
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.progress.start(12)
        self.log.grid(row=3, column=0, sticky="nsew")
        self.frame.rowconfigure(3, weight=1)
        self.result.configure(text="")
        cmds = install_commands()
        threading.Thread(
            target=run_commands,
            args=(cmds, lambda line: self.events.put(("line", line)), lambda ok, err: self.events.put(("done", (ok, err)))),
            name="quickimage-install",
            daemon=True,
        ).start()
        self.top.after(80, self._poll)

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    self.log.configure(state="normal")
                    self.log.insert("end", value + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                else:
                    self._finished(*value)
                    return
        except queue.Empty:
            pass
        self.top.after(80, self._poll)

    def _finished(self, ok, err):
        self._installing = False
        self.progress.stop()
        self.progress.grid_remove()
        self.btn_close.state(["!disabled"])
        still = missing_dependencies()
        missing_names = {d[0] for d in still}
        for mod, lbl in self.status_labels.items():
            gone = mod not in missing_names
            lbl.configure(
                text="Installed" if gone else "Missing", foreground=STATUS_OK if gone else STATUS_MISSING
            )
        self.result.grid(row=4, column=0, sticky="w", pady=(8, 0))
        if ok:
            self.result.configure(
                text=f"Done. Restart {APP_NAME} to use the new packages.", foreground=STATUS_OK
            )
            self.btn_restart.grid(row=0, column=1, padx=(0, 8))
            try:
                self.btn_restart.configure(style="Accent.TButton")
                self.btn_install.configure(style="TButton")
            except tkinter.TclError:
                pass
        else:
            self.result.configure(text=f"Installation failed: {err}. See the log above.", foreground=STATUS_MISSING)
            self.btn_install.state(["!disabled"])

    def restart(self):
        if self.on_restart is not None:
            self.on_restart()
        else:
            restart_app()
            self.close()

    def close(self):
        if self._installing:
            return
        self.top.destroy()


def _startup_missing_packages(exc):
    """Required packages are missing: offer to install them instead of failing silently."""
    try:
        set_app_user_model_id()  # DPI awareness + taskbar identity, before the window exists
        root = tkinter.Tk()
        root.withdraw()
        root.title(APP_NAME)
        set_window_icon(root)
        try:
            import sv_ttk

            sv_ttk.set_theme("light")
        except Exception:
            pass
        root.update()  # let the theme finish applying (fonts, colors) before building the dialog
        dlg = DependencyDialog(root, startup_error=exc)
        root.wait_window(dlg.top)
        root.destroy()
    except Exception:
        print(f"{APP_NAME} could not start: {exc}\nInstall the packages in requirements.txt.", file=sys.stderr)


import io
import tempfile

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageGrab, ImageOps, ImageTk
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        binary_opening,
        distance_transform_edt,
        label,
        median_filter,
        uniform_filter,
    )

    from tkinterdnd2 import TkinterDnD, DND_FILES, COPY
except ImportError as _exc:
    if __name__ != "__main__":
        raise
    # pythonw has no console, so a missing package would otherwise fail silently
    _startup_missing_packages(_exc)
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


def compute_bg_selection(a, tol=BG_TOL, remove_holes=False, seeds=None, use_border=True):
    """Return (bg_mask, dist_field, bg_color_field, eff_tol) for image array `a`.

    The returned dist/bg_color arrays may be shared cache entries: treat them as read-only.
    """
    an = get_analysis(a)
    bg_field, dist = an.bg_field, an.dist
    eff_tol = float(tol) + min(18.0, an.noise_mad * 0.75)
    if use_border:
        vis = border_bg_mask(dist, eff_tol)
    else:
        h, w = a.shape[:2]
        vis = np.zeros((h, w), dtype=bool)
        dist = np.full((h, w), np.inf, dtype=np.float32)
        bg_field = np.zeros((h, w, 3), dtype=np.float32)

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


GRABCUT_MAX_SIDE = 640  # GrabCut runs on the outline's window downscaled to at most this
GRABCUT_ITERS = 5
OPENCV_HINT = "Object outline needs OpenCV (opencv-python-headless) — click Dependencies to install it."

_LASSO_CACHE = OrderedDict()  # (image key, outline, hints, holes) -> (alpha, bg color, note)
_LASSO_CACHE_SIZE = 4


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


def load_rgba(path):
    with Image.open(path) as src:
        try:
            src.seek(0)
        except Exception:
            pass
        im = ImageOps.exif_transpose(src)
        return im.convert("RGBA")


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


def _mask_bbox(fg):
    """(left, top, right, bottom) of the True pixels in a 2D mask, or None if there are none."""
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


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


def copy_image_to_clipboard(im):
    """Copy a PIL Image to the system clipboard (CF_DIB + PNG on Windows). Returns True on success."""
    rgba = im.convert("RGBA")
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        # Composite onto white for apps that only read CF_DIB (24-bit BMP), and also register "PNG"
        # format for apps that support full RGBA transparency (Word, PowerPoint, browsers, etc.).
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(white, rgba).convert("RGB")
        bmp_buf = io.BytesIO()
        flat.save(bmp_buf, "BMP")
        dib_bytes = bmp_buf.getvalue()[14:]  # strip 14-byte BITMAPFILEHEADER

        png_buf = io.BytesIO()
        rgba.save(png_buf, "PNG")
        png_bytes = png_buf.getvalue()

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        GMEM_MOVEABLE = 0x0002
        CF_DIB = 8

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        user32.RegisterClipboardFormatW.restype = wintypes.UINT

        def _put_Clip(fmt_id, raw):
            hmem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(raw))
            if not hmem:
                return False
            ptr = kernel32.GlobalLock(hmem)
            if not ptr:
                kernel32.GlobalFree(hmem)
                return False
            ctypes.memmove(ptr, raw, len(raw))
            kernel32.GlobalUnlock(hmem)
            if not user32.SetClipboardData(fmt_id, hmem):
                kernel32.GlobalFree(hmem)
                return False
            return True

        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            ok = _put_Clip(CF_DIB, dib_bytes)
            cf_png = user32.RegisterClipboardFormatW("PNG")
            if cf_png:
                _put_Clip(cf_png, png_bytes)
            return bool(ok)
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def grab_clipboard_image():
    """Return `(path_or_none, rgba_image)` from the system clipboard, or `(None, None)` if empty."""
    try:
        clip = ImageGrab.grabclipboard()
    except Exception:
        return None, None
    if isinstance(clip, Image.Image):
        return None, ImageOps.exif_transpose(clip).convert("RGBA")
    if isinstance(clip, (list, tuple)):
        for item in clip:
            if isinstance(item, (str, Path)) and is_image_file(item):
                try:
                    return Path(item), load_rgba(item)
                except Exception:
                    continue
    return None, None


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
    lasso=None,
    use_border=True,
    crop_pad=0,
    square_canvas=False,
):
    """Run transparency/crop/binning/icon steps in-memory without writing files.

    `lasso` (an outline around one object, in image pixels) switches transparency to
    "keep only that object"; with transparency off it still crops to the object.
    """
    notes = []
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
    if do_crop:
        if lasso and not do_transp:
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
    lasso=None,
    use_border=True,
    do_webp=False,
    out_path=None,
    crop_pad=0,
    square_canvas=False,
    source_im=None,
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
    )
    if do_svg:
        binned, palette = quantize_foreground_colors(im, n_colors=n_colors)
        notes.append(f"{len(palette)} color(s)")
        out = Path(out_path) if out_path else next_save_path(path, ".svg")
        out.write_text(binned_to_svg(binned, palette), encoding="utf-8")
        notes.append("svg")
        preview = binned
    elif do_icon:
        out = Path(out_path) if out_path else next_save_path(path, ".ico")
        save_ico(im, out)
        notes.append(f"ico ({','.join(str(s) for s in ICON_SIZES)}px)")
    elif do_webp:
        out = Path(out_path) if out_path else next_save_path(path, ".webp")
        im.save(out, "WEBP", lossless=True, quality=100)
        notes.append("webp")
    else:
        out = Path(out_path) if out_path else next_save_path(path, ".png")
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
    )
    return out, note


def is_image_file(p):
    return Path(p).suffix.lower() in IMG_EXT


# Colors for the parts the app draws itself (canvas, checkerboard, outline, links)
PALETTES = {
    "light": {
        "canvas": "#E8E8E8",
        "check": (255, 255, 255, 228, 228, 228),
        "text": "#1F1F1F",
        "muted": "#6B6B6B",
        "accent": "#005FB8",
        "outline": "#E5484D",
    },
    "dark": {
        "canvas": "#161616",
        "check": (66, 66, 66, 52, 52, 52),
        "text": "#F3F3F3",
        "muted": "#9E9E9E",
        "accent": "#60CDFF",
        "outline": "#FF6B6B",
    },
}


def apply_theme(root):
    """Sun Valley (Windows 11 style) ttk theme, light or dark following the OS. Returns (palette, mode).

    Falls back to a built-in ttk theme if sv-ttk / darkdetect are not installed.
    """
    mode = "light"
    try:
        import darkdetect

        mode = "dark" if (darkdetect.theme() or "").lower() == "dark" else "light"
    except Exception:
        pass
    try:
        import sv_ttk

        sv_ttk.set_theme(mode)
    except Exception:
        mode = "light"
        style = ttk.Style(root)
        for name in ("vista", "clam"):
            if name in style.theme_names():
                style.theme_use(name)
                break
    return PALETTES[mode], mode


def set_dark_title_bar(root, dark):
    """Match the Windows title bar to the theme (Windows 10 20H1+ / 11); no-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        root.update_idletasks()  # the native window must exist
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
        value = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (newer, then older builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except Exception:
        pass


def reveal_in_folder(path):
    """Open the file manager with `path` selected (or its folder, where selecting is not supported)."""
    path = Path(path)
    if sys.platform == "win32":
        subprocess.Popen(f'explorer /select,"{path}"')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


class App:
    DRAG_MIN_PX = 4  # total mouse movement below this is a click, not an outline
    VIEW_MARGIN = 12  # px between the preview and the canvas edge

    def __init__(self):
        set_app_user_model_id()  # must happen before the first window is created
        self.root = TkinterDnD.Tk()
        self.root.title(APP_NAME)
        set_window_icon(self.root)
        self.palette, self.theme_mode = apply_theme(self.root)
        set_dark_title_bar(self.root, self.theme_mode == "dark")
        self.ui_scale = self.root.winfo_fpixels("1i") / 96.0  # 1.0 at 100 %, 1.5 at 150 %
        self.root.geometry(f"{self.px(1180)}x{self.px(760)}")
        self.root.minsize(self.px(900), self.px(600))
        self.root.bind("<Escape>", lambda _e: self.cancel_drag())
        self.root.bind("<Control-o>", lambda _e: self.open_dialog())
        self.root.bind("<Control-v>", self.paste_clipboard)
        self.root.bind("<Control-V>", self.paste_clipboard)
        self.root.bind("<Control-c>", self.copy_clipboard)
        self.root.bind("<Control-C>", self.copy_clipboard)
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-Z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-Y>", self.redo)
        self.root.bind("<Control-Shift-Z>", self.redo)
        self.root.bind("<Control-Shift-z>", self.redo)
        self.root.bind("<KeyPress-space>", self.compare_start)
        self.root.bind("<KeyRelease-space>", self.compare_end)

        self.canvas = None
        self.bw = 1
        self.bh = 1
        self.current_path = None
        self.source_rgba = None
        self.preview_src = None  # source_rgba, downscaled for fast live previews of big images
        self.preview_scale = 1.0  # preview_src size / source_rgba size
        self.hover_path = None
        self.tkimg = None
        self.last_preview = None
        self.last_label = None
        self.last_output = None  # most recent exported file (for "Show in folder")
        self.seeds = []
        self.lasso = []  # outline around one object, in full-resolution source pixels
        self._history = []  # stack of (seeds, lasso) snapshots for Ctrl+Z undo
        self._redo_stack = []  # stack of (seeds, lasso) snapshots for Ctrl+Y / Ctrl+Shift+Z redo
        self._drag = None  # canvas points of the outline being drawn
        self._drag_far = False  # the press moved enough to be a drag, not a click
        self._preview_job = None
        self._resize_job = None
        self._disp_scale = 1.0
        self._disp_size = (1, 1)
        self._crop_box = None
        self._is_icon_preview = False
        self._busy = False  # an export is running on the worker thread
        self._comparing = False  # "hold to compare" is showing the original
        self._empty_icon = None

    def px(self, v):
        """Scale a design size in px (at 100 %) to the screen's DPI."""
        return int(round(v * self.ui_scale))

    def _get_color_count(self):
        """Return (count, valid). Invalid spinbox text falls back to the default count."""
        try:
            return max(1, min(16, int(self.opt_colors.get()))), True
        except (tkinter.TclError, ValueError):
            return DEFAULT_SVG_COLORS, False

    def _get_crop_pad(self):
        """Return crop padding in pixels (0..256). Invalid text falls back to 0."""
        try:
            return max(0, min(256, int(self.opt_pad.get())))
        except (tkinter.TclError, ValueError):
            return 0

    # --- layout -----------------------------------------------------------------------------

    def _init_styles(self):
        pal = self.palette
        style = ttk.Style(self.root)
        base = tkinter.font.nametofont("TkDefaultFont")
        family = base.actual("family")
        size = base.actual("size")
        semibold = "Segoe UI Semibold" if sys.platform == "win32" else family
        style.configure("Section.TLabel", font=(semibold, max(8, size - 1)), foreground=pal["muted"])
        style.configure("Hint.TLabel", foreground=pal["muted"])
        style.configure("Title.TLabel", font=(semibold, size + 2))
        style.configure("Muted.TLabel", foreground=pal["muted"])
        # The link must match the status text next to it, so derive it from the theme's label font
        self._link_font = tkinter.font.Font(self.root, font=style.lookup("TLabel", "font") or "TkDefaultFont")
        self._link_font.configure(underline=True)
        style.configure("Link.TLabel", foreground=pal["accent"], font=self._link_font)
        style.configure("Status.TLabel", foreground=pal["text"])

    def build_ui(self):
        self._init_styles()
        P = self.px
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.opt_mode = tkinter.StringVar(value="color")
        self.opt_holes = tkinter.BooleanVar(value=False)
        self.opt_crop = tkinter.BooleanVar(value=False)
        self.opt_pad = tkinter.IntVar(value=0)
        self.opt_square = tkinter.BooleanVar(value=False)
        self.opt_bin = tkinter.BooleanVar(value=False)
        self.opt_colors = tkinter.IntVar(value=DEFAULT_SVG_COLORS)
        self.opt_tol = tkinter.IntVar(value=BG_TOL)

        # ---- main area: header + preview ----
        main = ttk.Frame(root, padding=(P(16), P(12), P(8), P(8)))
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, P(10)))
        header.columnconfigure(4, weight=1)
        self.btn_open = ttk.Button(header, text="Open…", command=self.open_dialog)
        self.btn_open.grid(row=0, column=0, padx=(0, P(6)))
        self.btn_paste = ttk.Button(header, text="Paste", command=self.paste_clipboard)
        self.btn_paste.grid(row=0, column=1, padx=(0, P(6)))
        self.btn_copy = ttk.Button(header, text="Copy", command=self.copy_clipboard)
        self.btn_copy.grid(row=0, column=2, padx=(0, P(12)))
        self.file_label = ttk.Label(header, text="No image", style="Title.TLabel")
        self.file_label.grid(row=0, column=3, sticky="w")
        self.info_label = ttk.Label(header, text="", style="Muted.TLabel")
        self.info_label.grid(row=0, column=4, sticky="w", padx=(P(10), 0))
        self.btn_compare = ttk.Button(header, text="Hold to compare")
        self.btn_compare.grid(row=0, column=5, sticky="e")
        self.btn_compare.bind("<ButtonPress-1>", self.compare_start)
        self.btn_compare.bind("<ButtonRelease-1>", self.compare_end)

        self.canvas = tkinter.Canvas(
            main, highlightthickness=0, borderwidth=0, background=self.palette["canvas"], cursor="crosshair"
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        # sv-ttk recolors every classic Tk widget (tk_setPalette) when the theme event arrives,
        # which happens after this point; put the canvas color back afterwards.
        root.bind("<<ThemeChanged>>", lambda _e: root.after_idle(self._apply_canvas_color), add="+")
        root.after_idle(self._apply_canvas_color)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", lambda _e: self.clear_picks())

        # ---- sidebar ----
        ttk.Separator(root, orient="vertical").grid(row=0, column=1, sticky="ns")
        side = ttk.Frame(root, padding=(P(18), P(14), P(18), P(12)))
        side.grid(row=0, column=2, sticky="nsew")
        side.columnconfigure(0, weight=1)
        wrap = P(250)
        row = 0

        def section(title, first=False):
            nonlocal row
            if not first:
                ttk.Separator(side).grid(row=row, column=0, sticky="ew", pady=(P(14), P(12)))
                row += 1
            ttk.Label(side, text=title.upper(), style="Section.TLabel").grid(
                row=row, column=0, sticky="w", pady=(0, P(6))
            )
            row += 1

        def place(widget, **kw):
            nonlocal row
            widget.grid(row=row, column=0, sticky=kw.pop("sticky", "w"), **kw)
            row += 1

        section("Mode", first=True)
        self.rad_color = ttk.Radiobutton(
            side,
            variable=self.opt_mode,
            value="color",
            text="Delete clicked color",
            command=self._on_mode_change,
        )
        place(self.rad_color, pady=(0, P(2)))
        self.rad_outline = ttk.Radiobutton(
            side,
            variable=self.opt_mode,
            value="outline",
            text="Outline object to keep",
            command=self._on_mode_change,
        )
        place(self.rad_outline, pady=(0, P(6)))
        self.chk_h = ttk.Checkbutton(
            side, variable=self.opt_holes, text="Remove interior holes", command=self.schedule_live_preview
        )
        place(self.chk_h, pady=(0, P(8)))
        tol_row = ttk.Frame(side)
        tol_row.columnconfigure(0, weight=1)
        self.tol_label = ttk.Label(tol_row, text="Tolerance")
        self.tol_label.grid(row=0, column=0, sticky="w")
        self.tol_value = ttk.Label(tol_row, text=str(BG_TOL), style="Muted.TLabel")
        self.tol_value.grid(row=0, column=1, sticky="e")
        place(tol_row, sticky="ew")
        self.tol_scale = ttk.Scale(side, from_=10, to=240, orient="horizontal", command=self._on_tol_drag)
        self.tol_scale.set(BG_TOL)
        place(self.tol_scale, sticky="ew", pady=(P(2), P(6)))
        self.mode_hint = ttk.Label(
            side,
            text="",
            style="Hint.TLabel",
            wraplength=wrap,
            justify="left",
        )
        place(self.mode_hint, pady=(0, P(8)))
        self.btn_reset_seeds = ttk.Button(side, text="Clear picks & outline", command=self.clear_picks)
        place(self.btn_reset_seeds)

        section("Crop & colors")
        self.chk_c = ttk.Checkbutton(
            side, variable=self.opt_crop, text="Crop to content", command=self.schedule_live_preview
        )
        place(self.chk_c, pady=(0, P(4)))
        crop_opts = ttk.Frame(side)
        self.chk_square = ttk.Checkbutton(
            crop_opts, variable=self.opt_square, text="Square (1:1)", command=self.schedule_live_preview
        )
        self.chk_square.grid(row=0, column=0, sticky="w")
        ttk.Label(crop_opts, text="Pad (px)").grid(row=0, column=1, padx=(P(10), P(4)))
        self.spin_pad = ttk.Spinbox(crop_opts, from_=0, to=256, width=4, textvariable=self.opt_pad)
        self.spin_pad.grid(row=0, column=2)
        place(crop_opts, pady=(0, P(6)))
        self.opt_pad.trace_add("write", lambda *_a: self.schedule_live_preview())

        bin_row = ttk.Frame(side)
        self.chk_b = ttk.Checkbutton(bin_row, variable=self.opt_bin, text="Limit colors to", command=self.schedule_live_preview)
        self.chk_b.grid(row=0, column=0, sticky="w")
        self.spin_colors = ttk.Spinbox(bin_row, from_=1, to=16, width=4, textvariable=self.opt_colors)
        self.spin_colors.grid(row=0, column=1, padx=(P(8), 0))
        place(bin_row)
        place(
            ttk.Label(
                side,
                text="Flat colors, e.g. one per filament for 3D printing. SVG export always uses this count.",
                style="Hint.TLabel",
                wraplength=wrap,
                justify="left",
            ),
            pady=(P(6), 0),
        )
        # Fires on arrow clicks and on every keystroke, so the preview always matches the export
        self.opt_colors.trace_add("write", lambda *_a: self.schedule_live_preview())

        section("Export")
        self.btn = ttk.Button(side, text="Save PNG", style="Accent.TButton", command=self.apply)
        self.btn.bind("<Shift-ButtonRelease-1>", lambda _e: self._shift_export(self.apply))
        place(self.btn, sticky="ew", pady=(0, P(6)))
        self.btn_webp = ttk.Button(side, text="Save WebP (.webp)", command=self.export_webp)
        self.btn_webp.bind("<Shift-ButtonRelease-1>", lambda _e: self._shift_export(self.export_webp))
        place(self.btn_webp, sticky="ew", pady=(0, P(6)))
        self.btn_ico = ttk.Button(side, text="Export icon (.ico)", command=self.export_icon)
        self.btn_ico.bind("<Shift-ButtonRelease-1>", lambda _e: self._shift_export(self.export_icon))
        place(self.btn_ico, sticky="ew", pady=(0, P(6)))
        self.btn_svg = ttk.Button(side, text="Export SVG (.svg)", command=self.export_svg)
        self.btn_svg.bind("<Shift-ButtonRelease-1>", lambda _e: self._shift_export(self.export_svg))
        place(self.btn_svg, sticky="ew")
        place(
            ttk.Label(
                side,
                text="Files are saved next to the original (Shift-click to Save As…).",
                style="Hint.TLabel",
                wraplength=wrap,
            ),
            pady=(P(6), 0),
        )

        side.rowconfigure(row, weight=1)  # push the bottom row down
        row += 1
        bottom = ttk.Frame(side)
        bottom.columnconfigure(0, weight=1)
        self.btn_deps = ttk.Button(bottom, text="Dependencies…", command=self.show_dependencies)
        self.btn_deps.grid(row=0, column=0, sticky="w")
        self.btn_reset_all = ttk.Button(bottom, text="Reset all", command=self.reset_all)
        self.btn_reset_all.grid(row=0, column=1, sticky="e")
        place(bottom, sticky="ew")

        # ---- status bar ----
        bar = ttk.Frame(root)
        bar.grid(row=1, column=0, columnspan=3, sticky="ew")
        bar.columnconfigure(0, weight=1)
        ttk.Separator(bar).grid(row=0, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(bar, text="Open or drop an image to get started.", style="Status.TLabel", anchor="w")
        self.status.grid(row=1, column=0, sticky="ew", padx=(P(16), P(8)), pady=P(6))
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=P(140))
        self.progress.grid(row=1, column=1, padx=P(8))
        self.progress.grid_remove()
        self.link = ttk.Label(bar, text="Show in folder", style="Link.TLabel", cursor="hand2")
        self.link.grid(row=1, column=2, padx=(P(8), P(16)))
        self.link.bind("<Button-1>", lambda _e: self.show_in_folder())
        self.link.grid_remove()

        self._update_mode_widgets()
        self._update_compare_state()
        self._update_deps_button(announce=True)

    # --- dependencies -------------------------------------------------------------------------

    def _update_deps_button(self, announce=False):
        """Show how many optional packages are missing on the button; mention it once at startup."""
        missing = missing_dependencies()
        self.btn_deps.configure(text=f"Dependencies ({len(missing)} missing)" if missing else "Dependencies…")
        if missing and announce:
            names = ", ".join(d[1] for d in missing)
            self.set_msg(f"Some features are unavailable ({names} not installed) — click Dependencies to install.")

    def show_dependencies(self):
        dlg = DependencyDialog(self.root, on_restart=self.restart)
        self.root.wait_window(dlg.top)
        try:
            self._update_deps_button()
        except tkinter.TclError:
            pass  # the app restarted from the dialog and this window is gone

    def restart(self):
        """Restart after installing packages (the theme and imports are set up at startup)."""
        if self._busy:
            self.set_msg("Wait for the export to finish, then restart.")
            return
        restart_app()
        self.root.destroy()

    # --- canvas drawing ---------------------------------------------------------------------

    def _apply_canvas_color(self):
        try:
            self.canvas.configure(background=self.palette["canvas"])
        except tkinter.TclError:
            pass  # window already closed

    def on_canvas_resize(self, event):
        self.bw, self.bh = max(1, event.width), max(1, event.height)
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_job = None
        if self._comparing and self.source_rgba is not None:
            self.show_image(self.source_rgba, None, overlay=False)
        elif self.last_preview is not None:
            self.show_image(self.last_preview, None)
        else:
            self.reset_canvas()
        if self.canvas.find_withtag("drophint"):
            self._draw_drop_hint()

    def reset_canvas(self):
        """The empty state: app icon, 'Drop an image here', and the supported formats."""
        c, pal, P = self.canvas, self.palette, self.px
        c.delete("all")
        self.tkimg = None
        cx, cy = self.bw / 2, self.bh / 2
        if self._empty_icon is None and APP_ICON_PNG.is_file():
            try:
                with Image.open(APP_ICON_PNG) as ic:
                    self._empty_icon = ImageTk.PhotoImage(ic.convert("RGBA").resize((P(88), P(88)), Image.LANCZOS))
            except Exception:
                self._empty_icon = False
        if self._empty_icon:
            c.create_image(cx, cy - P(62), image=self._empty_icon, tags="empty")
        base = tkinter.font.nametofont("TkDefaultFont").actual()
        c.create_text(cx, cy + P(8), text="Drop an image here", fill=pal["text"],
                      font=(base["family"], base["size"] + 4), tags="empty")
        c.create_text(cx, cy + P(36), text="or click to open (Ctrl+O)", fill=pal["muted"],
                      font=(base["family"], base["size"]), tags="empty")
        fmts = "  ".join(e.lstrip(".").upper() for e in sorted(IMG_EXT) if e not in (".jpeg", ".tiff"))
        c.create_text(cx, cy + P(62), text=fmts, fill=pal["muted"],
                      font=(base["family"], max(7, base["size"] - 1)), tags="empty")

    def _draw_drop_hint(self):
        self.canvas.delete("drophint")
        m = self.px(6)
        self.canvas.create_rectangle(
            m, m, self.bw - m, self.bh - m, outline=self.palette["accent"], width=self.px(2),
            dash=(8, 6), tags="drophint",
        )

    def set_msg(self, text):
        self.status.configure(text=text)

    def show_image(self, pil_img, label, overlay=True):
        """Show a picture in the box. overlay=False skips the outline (e.g. for a hovered file)."""
        bw, bh = self.bw, self.bh
        m = 2 * self.VIEW_MARGIN
        scale = min(max(1, bw - m) / pil_img.width, max(1, bh - m) / pil_img.height, 3.0)
        w = max(1, int(round(pil_img.width * scale)))
        h = max(1, int(round(pil_img.height * scale)))
        self._disp_scale = scale
        self._disp_size = (w, h)
        small = pil_img.resize((w, h), Image.LANCZOS)
        if small.mode == "RGBA" and (np.asarray(small)[..., 3] < 255).any():
            r1, g1, b1, r2, g2, b2 = self.palette["check"]
            cell = self.px(8)
            yy = np.arange(h)[:, None] // cell
            xx = np.arange(w)[None, :] // cell
            odd = ((yy + xx) % 2).astype(bool)
            tile = np.empty((h, w, 3), np.uint8)
            tile[~odd] = (r1, g1, b1)
            tile[odd] = (r2, g2, b2)
            small = Image.alpha_composite(Image.fromarray(tile).convert("RGBA"), small).convert("RGB")
        photo = ImageTk.PhotoImage(small)
        self.canvas.delete("preview", "empty", "lasso")
        self.canvas.create_image(bw // 2, bh // 2, image=photo, anchor="center", tags="preview")
        self.canvas.tag_lower("preview")
        self.tkimg = photo
        if overlay:
            self._draw_lasso()
        if label:
            self.set_msg(label)

    # --- mapping between canvas pixels and full-resolution source pixels ---------------------

    def _view_geometry(self):
        """What the box shows: displayed size/offset, crop box, and icon padding (if an icon preview)."""
        src_w, src_h = self.source_rgba.size
        disp_w, disp_h = self._disp_size
        ox = (self.bw - disp_w) / 2.0
        oy = (self.bh - disp_h) / 2.0
        cx0, cy0, cx1, cy1 = self._crop_box or (0, 0, src_w, src_h)
        cw, ch = max(1, cx1 - cx0), max(1, cy1 - cy0)
        icon = None
        if self._is_icon_preview:
            icon_s = float(max(ICON_SIZES))
            sc = min(icon_s / cw, icon_s / ch)
            nw, nh = max(1.0, round(cw * sc)), max(1.0, round(ch * sc))
            icon = (icon_s, (icon_s - nw) / 2.0, (icon_s - nh) / 2.0, nw, nh)
        return ox, oy, disp_w, disp_h, cx0, cy0, cw, ch, icon

    def _canvas_to_source(self, x, y, clamp=False):
        """Canvas point -> float source pixel, or None if outside the picture (unless clamp)."""
        src_w, src_h = self.source_rgba.size
        ox, oy, disp_w, disp_h, cx0, cy0, cw, ch, icon = self._view_geometry()
        rel_x = (x - ox) / max(1.0, float(disp_w))
        rel_y = (y - oy) / max(1.0, float(disp_h))
        if clamp:
            rel_x, rel_y = min(max(rel_x, 0.0), 1.0), min(max(rel_y, 0.0), 1.0)
        elif not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
            return None
        if icon:
            icon_s, pad_x, pad_y, nw, nh = icon
            px, py = rel_x * icon_s, rel_y * icon_s
            if clamp:
                px, py = min(max(px, pad_x), pad_x + nw), min(max(py, pad_y), pad_y + nh)
            elif not (pad_x <= px <= pad_x + nw and pad_y <= py <= pad_y + nh):
                return None
            rel_x, rel_y = (px - pad_x) / nw, (py - pad_y) / nh
        sx = min(max(cx0 + rel_x * (cw - 1), 0.0), src_w - 1.0)
        sy = min(max(cy0 + rel_y * (ch - 1), 0.0), src_h - 1.0)
        return sx, sy

    def _source_to_canvas(self, sx, sy):
        """Inverse of _canvas_to_source (points may land outside the shown picture)."""
        ox, oy, disp_w, disp_h, cx0, cy0, cw, ch, icon = self._view_geometry()
        rel_x = (sx - cx0) / max(1.0, cw - 1.0)
        rel_y = (sy - cy0) / max(1.0, ch - 1.0)
        if icon:
            icon_s, pad_x, pad_y, nw, nh = icon
            rel_x, rel_y = (pad_x + rel_x * nw) / icon_s, (pad_y + rel_y * nh) / icon_s
        return ox + rel_x * disp_w, oy + rel_y * disp_h

    def _draw_lasso(self):
        if not self.lasso or self.source_rgba is None or self.opt_mode.get() != "outline":
            return
        flat = [c for p in self.lasso for c in self._source_to_canvas(*p)]
        self.canvas.create_polygon(
            *flat, outline=self.palette["outline"], fill="", width=self.px(2), dash=(6, 4), tags="lasso"
        )

    # --- mouse: mode-dependent click to remove color or drag to outline object -----------------

    def _push_history(self):
        self._history.append((list(self.seeds), list(self.lasso)))
        self._redo_stack.clear()

    def undo(self, event=None):
        """Ctrl+Z: undo the last color click or outline action one step at a time."""
        if event is not None and isinstance(event.widget, (ttk.Spinbox, tkinter.Entry)):
            return
        if self._busy or self._drag:
            return
        if not self._history:
            self.set_msg("Nothing to undo.")
            return
        self._redo_stack.append((list(self.seeds), list(self.lasso)))
        self.seeds, self.lasso = self._history.pop()
        self.canvas.delete("lasso")
        self._update_mode_widgets()
        self.update_live_preview()

    def redo(self, event=None):
        """Ctrl+Y / Ctrl+Shift+Z: redo the most recently undone color click or outline action."""
        if event is not None and isinstance(event.widget, (ttk.Spinbox, tkinter.Entry)):
            return
        if self._busy or self._drag:
            return
        if not self._redo_stack:
            self.set_msg("Nothing to redo.")
            return
        self._history.append((list(self.seeds), list(self.lasso)))
        self.seeds, self.lasso = self._redo_stack.pop()
        self.canvas.delete("lasso")
        self._update_mode_widgets()
        self.update_live_preview()

    def on_press(self, event):
        self.canvas.focus_set()  # so Space (compare) is not swallowed by a focused button
        if self.source_rgba is None:
            self.open_dialog()
            return
        if self._comparing or self._busy:
            return
        if self.opt_mode.get() == "color":
            self.pick_background(event)
            return
        self._drag = [(event.x, event.y)]
        self._drag_far = False

    def on_motion(self, event):
        if not self._drag:
            return
        lx, ly = self._drag[-1]
        if np.hypot(event.x - lx, event.y - ly) >= 3:
            self._drag.append((event.x, event.y))
        sx, sy = self._drag[0]
        if not self._drag_far and np.hypot(event.x - sx, event.y - sy) >= self.DRAG_MIN_PX:
            self._drag_far = True
        if len(self._drag) < 2 and (event.x, event.y) == (sx, sy):
            return
        flat = [c for p in self._drag for c in p]
        if self._drag[-1] != (event.x, event.y):
            flat.extend((event.x, event.y))
        flat.extend((sx, sy))
        if self.canvas.find_withtag("lasso_live"):
            self.canvas.coords("lasso_live", *flat)
        else:
            self.canvas.create_line(*flat, fill=self.palette["outline"], width=self.px(2), tags="lasso_live")

    def cancel_drag(self):
        """Escape: abandon an outline that is being drawn."""
        if self._drag:
            self._drag, self._drag_far = None, False
            self.canvas.delete("lasso_live")
            self.set_msg("Outline cancelled.")

    def on_release(self, event):
        pts, far = self._drag, self._drag_far
        self._drag, self._drag_far = None, False
        self.canvas.delete("lasso_live")
        if not pts or self.source_rgba is None:
            return
        if not far:
            self.set_msg("Drag a loop around the object to cut it out.")
            return
        pts.append((event.x, event.y))
        src = []
        for x, y in pts:
            p = self._canvas_to_source(x, y, clamp=True)
            if not src or np.hypot(p[0] - src[-1][0], p[1] - src[-1][1]) >= 2.0:
                src.append(p)
        if len(src) < 3:
            self.set_msg("Outline too small — drag a loop around the object.")
            self._draw_lasso()
            return
        if src[-1] != src[0]:
            src.append(src[0])
        self._push_history()
        self.lasso = src
        self.opt_crop.set(True)
        self._update_mode_widgets()
        self.update_live_preview()

    def pick_background(self, event):
        p = self._canvas_to_source(event.x, event.y)
        if p is None:
            return
        src_w, src_h = self.source_rgba.size
        ix = int(np.clip(round(p[0]), 0, src_w - 1))
        iy = int(np.clip(round(p[1]), 0, src_h - 1))
        self._push_history()
        self.seeds.append((ix, iy))
        self._update_mode_widgets()
        self.update_live_preview()

    # --- options ------------------------------------------------------------------------------

    def _on_mode_change(self):
        self.cancel_drag()
        self.canvas.delete("lasso")
        self._update_mode_widgets()
        self.update_live_preview()

    def _active_transp_state(self):
        """Return (do_transp, seeds, lasso, use_border) for the currently selected mode."""
        seeds = list(self.seeds) or None
        lasso = list(self.lasso) or None
        return bool(seeds or lasso), seeds, lasso, False

    def _on_tol_drag(self, value):
        v = int(round(float(value)))
        self.tol_value.configure(text=str(v))
        if v != self.opt_tol.get():
            self.opt_tol.set(v)
            self.schedule_live_preview()

    def _set_tolerance(self, v):
        self.opt_tol.set(int(v))
        self.tol_scale.set(int(v))
        self.tol_value.configure(text=str(int(v)))

    def _update_mode_widgets(self):
        """Update hints and grey out controls that do not apply to the active mode."""
        is_outline = self.opt_mode.get() == "outline"
        self.tol_scale.state(["disabled"] if is_outline else ["!disabled"])
        self.tol_label.configure(style="Muted.TLabel" if is_outline else "TLabel")
        has_transp = bool(self.lasso or self.seeds)
        self.chk_h.state(["!disabled"] if has_transp else ["disabled"])
        if is_outline:
            self.mode_hint.configure(
                text="Drag a loop around an object to cut it out. "
                "Ctrl+Z undoes, right-click clears."
            )
        else:
            self.mode_hint.configure(
                text="Click any color in the preview to delete it (click more colors to remove them too). "
                "Ctrl+Z undoes, right-click clears."
            )

    def _update_compare_state(self):
        can = self.source_rgba is not None and self.last_preview is not self.source_rgba
        self.btn_compare.state(["!disabled"] if can else ["disabled"])
        has_img = self.source_rgba is not None
        if hasattr(self, "btn_copy"):
            self.btn_copy.state(["!disabled"] if has_img else ["disabled"])

    def clear_picks(self):
        if not self.seeds and not self.lasso:
            return
        self._push_history()
        self.seeds = []
        self.lasso = []
        self.canvas.delete("lasso")
        self._update_mode_widgets()
        self.update_live_preview()

    def reset_all(self):
        """Back to the original picture: no outline or picks, all options off, defaults restored."""
        self.seeds = []
        self.lasso = []
        self._history = []
        self._redo_stack = []
        self.canvas.delete("lasso")
        self.opt_mode.set("color")
        for var in (self.opt_holes, self.opt_crop, self.opt_square, self.opt_bin):
            var.set(False)
        self.opt_pad.set(0)
        self._set_tolerance(BG_TOL)
        self.opt_colors.set(DEFAULT_SVG_COLORS)  # its trace schedules a preview; cancelled below
        self._update_mode_widgets()
        self._hide_link()
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None
        if self.source_rgba is None:
            return
        self._crop_box = (0, 0, self.source_rgba.width, self.source_rgba.height)
        self._is_icon_preview = False
        self.last_preview = self.source_rgba
        self.last_label = "Reset to the original image."
        self.show_image(self.source_rgba, self.last_label)
        self._update_compare_state()

    # --- hold to compare ------------------------------------------------------------------------

    def compare_start(self, event=None):
        if event is not None and isinstance(event.widget, (ttk.Spinbox, tkinter.Entry)):
            return  # typing in the color count box
        if self._comparing or self.btn_compare.instate(["disabled"]) or self._drag:
            return
        self._comparing = True
        self.show_image(self.source_rgba, "Showing the original — release to return.", overlay=False)

    def compare_end(self, event=None):
        if not self._comparing:
            return
        self._comparing = False
        self.show_image(self.last_preview, self.last_label)

    # --- live preview ---------------------------------------------------------------------------

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
        self._comparing = False
        do_t, raw_seeds, raw_lasso, use_border = self._active_transp_state()
        do_h = self.opt_holes.get()
        do_c = self.opt_crop.get()
        crop_pad = self._get_crop_pad()
        square_canvas = self.opt_square.get()
        do_b = self.opt_bin.get()
        n_colors, colors_ok = self._get_color_count()
        tol = self.opt_tol.get()
        if not (do_t or do_c or crop_pad > 0 or square_canvas or do_b):
            self._crop_box = (0, 0, self.source_rgba.width, self.source_rgba.height)
            self._is_icon_preview = False
            self.last_preview = self.source_rgba
            if self.opt_mode.get() == "outline":
                self.last_label = "Original image. Drag a loop around an object to cut it out."
            else:
                self.last_label = "Original image. Click a color in the image to delete it."
            self.show_image(self.source_rgba, self.last_label)
            self._update_compare_state()
            return
        s = self.preview_scale
        seeds = [(x * s, y * s) for x, y in raw_seeds] if raw_seeds else None
        lasso = [(x * s, y * s) for x, y in raw_lasso] if raw_lasso else None
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
                lasso=lasso,
                use_border=use_border,
                crop_pad=max(0, round(crop_pad * s)),
                square_canvas=square_canvas,
            )
        except Exception as exc:  # pythonw has no console: surface the error in the UI
            self.set_msg(f"Preview error: {exc}")
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
        if s != 1.0:  # the preview ran on a downscaled copy: report the crop in real image pixels
            cx0, cy0, cx1, cy1 = self._crop_box
            notes = [f"cropped to {cx1 - cx0}x{cy1 - cy0}" if n.startswith("cropped to") else n for n in notes]
        if raw_seeds:
            notes.append(f"{len(raw_seeds)} color pick(s)")
        if do_b and not colors_ok:
            notes.append(f"invalid color count, using {n_colors}")
        if raw_seeds or (do_t and not raw_lasso):
            notes.append(f"tolerance {tol}")
        self.last_preview = preview
        self.last_label = "Preview: " + (" · ".join(notes) if notes else "ready")
        self.show_image(preview, self.last_label)
        self._update_compare_state()

    # --- opening files --------------------------------------------------------------------------

    def open_dialog(self):
        if self._busy:
            return
        pattern = " ".join(f"*{e}" for e in sorted(IMG_EXT))
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open image",
            filetypes=[("Images", pattern), ("All files", "*.*")],
            initialdir=str(self.current_path.parent) if self.current_path else None,
        )
        if not path:
            return
        try:
            rgba = load_rgba(path)
        except Exception as exc:
            self.set_msg(f"Could not open {Path(path).name}: {exc}")
            return
        self.load_picture(Path(path), rgba)

    def paste_clipboard(self, event=None):
        """Ctrl+V / Paste button: load an image or image file path from the system clipboard."""
        if event is not None and isinstance(event.widget, (ttk.Spinbox, tkinter.Entry)):
            return
        if self._busy:
            return
        path, rgba = grab_clipboard_image()
        if rgba is None:
            self.set_msg("No image found on the clipboard.")
            return
        if path is None:
            base_dir = self.current_path.parent if self.current_path else Path(tempfile.gettempdir())
            path = base_dir / "clipboard.png"
        self.load_picture(path, rgba, extra_msg="(Pasted from clipboard.)")

    def copy_clipboard(self, event=None):
        """Ctrl+C / Copy button: copy the current processed image (at full resolution) to the clipboard."""
        if event is not None and isinstance(event.widget, (ttk.Spinbox, tkinter.Entry)):
            return
        if self._busy or self.source_rgba is None:
            if self.source_rgba is None:
                self.set_msg("Open, drop, or paste an image first.")
            return
        do_t, raw_seeds, raw_lasso, use_border = self._active_transp_state()
        do_h = self.opt_holes.get()
        do_c = self.opt_crop.get()
        crop_pad = self._get_crop_pad()
        square_canvas = self.opt_square.get()
        do_b = self.opt_bin.get()
        n_colors, _ = self._get_color_count()
        tol = self.opt_tol.get()
        try:
            if do_t or do_c or crop_pad > 0 or square_canvas or do_b:
                im, _, _, _ = render_preview(
                    self.source_rgba,
                    do_t,
                    do_c,
                    do_icon=False,
                    tol=tol,
                    remove_holes=do_h,
                    seeds=raw_seeds,
                    bin_colors=do_b,
                    n_colors=n_colors,
                    lasso=raw_lasso,
                    use_border=use_border,
                    crop_pad=crop_pad,
                    square_canvas=square_canvas,
                )
            else:
                im = self.source_rgba
            if copy_image_to_clipboard(im):
                self.set_msg(f"Copied {im.width} × {im.height} px image to clipboard.")
            else:
                self.set_msg("Could not copy image to the clipboard.")
        except Exception as exc:
            self.set_msg(f"Copy failed: {exc}")

    def load_picture(self, path, rgba, extra_msg=""):
        """Make `rgba` (loaded from `path`) the current picture and show it, keeping the options."""
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
        self.lasso = []
        self._history = []
        self._redo_stack = []
        self._comparing = False
        self._update_mode_widgets()
        self._hide_link()
        self._crop_box = (0, 0, rgba.width, rgba.height)
        self._is_icon_preview = False
        self.root.title(f"{APP_NAME} — {path.name}")
        self.file_label.configure(text=path.name)
        self.info_label.configure(text=f"{rgba.width} × {rgba.height} px")
        self.canvas.delete("drophint")
        self.last_preview = rgba
        if self.opt_mode.get() == "outline":
            self.last_label = "Original image. Drag a loop around an object to cut it out."
        else:
            self.last_label = "Original image. Click a color in the image to delete it."
        self.show_image(rgba, self.last_label)
        if self.opt_crop.get() or self._get_crop_pad() > 0 or self.opt_square.get() or self.opt_bin.get():
            self.update_live_preview()
        self._update_compare_state()
        if extra_msg:
            self.set_msg(f"{self.status.cget('text')} {extra_msg}")

    def dropped_paths(self, event):
        raw = getattr(event, "data", None)
        if not raw:
            return []
        try:
            return list(self.root.tk.splitlist(raw))
        except Exception:
            return [raw]

    def on_hover(self, event):
        if not self.canvas.find_withtag("drophint"):
            self._draw_drop_hint()
        paths = self.dropped_paths(event)
        p = paths[0] if paths else None
        if p and is_image_file(p) and p != self.hover_path:
            self.hover_path = p
            try:
                rgba = load_rgba(p)
                self.show_image(rgba, f"Drop to open {Path(p).name}", overlay=False)
                self.canvas.tag_raise("drophint")
            except Exception:
                pass
        return COPY

    def _restore_view(self):
        """After a drag-and-drop hover ends: back to what was shown before."""
        self.canvas.delete("drophint")
        if self.current_path is None:
            self.reset_canvas()
        elif self.last_preview is not None:
            self.show_image(self.last_preview, None)

    def on_leave(self, event):
        had_hover = self.hover_path is not None
        self.hover_path = None
        self._restore_view()
        if had_hover:
            self.set_msg(self.last_label or "Open or drop an image to get started.")
        return COPY

    def on_drop(self, event):
        self.hover_path = None
        paths = self.dropped_paths(event)
        images = [p for p in paths if is_image_file(p)]
        if not images:
            self._restore_view()
            self.set_msg("Not an image file. Supported: PNG, JPG, BMP, GIF, TIF, WEBP, ICO.")
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
            self._restore_view()
            self.set_msg(f"Could not open {', '.join(failed)}.")
            return COPY
        skipped = len(paths) - 1 - len(failed)
        extra = f"({skipped} other file(s) ignored — one image at a time.)" if skipped > 0 else ""
        self.load_picture(path, rgba, extra_msg=extra)
        return COPY

    # --- exporting ------------------------------------------------------------------------------

    def _set_busy(self, busy):
        self._busy = busy
        for b in (self.btn, self.btn_webp, self.btn_ico, self.btn_svg, self.btn_open, self.btn_paste):
            b.state(["disabled"] if busy else ["!disabled"])
        if busy:
            self._hide_link()
            self.progress.grid()
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.grid_remove()
        self.root.configure(cursor="watch" if busy else "")

    def _hide_link(self):
        self.link.grid_remove()

    def show_in_folder(self):
        if self.last_output is None:
            return
        try:
            reveal_in_folder(self.last_output)
        except Exception as exc:
            self.set_msg(f"Could not open the folder: {exc}")

    def _shift_export(self, fn):
        """Handle Shift-click on an export button by opening a Save As dialog."""
        self._save_as_next = True
        try:
            fn()
        finally:
            self._save_as_next = False
        return "break"

    def _ask_save_as_path(self, ext, label):
        default_out = next_save_path(self.current_path, ext)
        chosen = filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Save {label} as",
            initialdir=str(default_out.parent),
            initialfile=default_out.name,
            defaultextension=ext,
            filetypes=[(label, f"*{ext}"), ("All files", "*.*")],
        )
        return Path(chosen) if chosen else None

    def _run_process(self, do_t, do_c, do_icon=False, do_svg=False, bin_colors=False, do_webp=False):
        """Export on a worker thread so the window stays responsive; results come back via a queue.

        Tk is not thread-safe, so every option is read here on the UI thread and the worker
        never touches a widget; _poll_export applies the result on the UI thread.
        """
        if self._busy:
            return
        ext = ".svg" if do_svg else ".ico" if do_icon else ".webp" if do_webp else ".png"
        kind = "SVG" if do_svg else "icon" if do_icon else "WebP" if do_webp else "PNG"
        out_path = None
        if getattr(self, "_save_as_next", False):
            self._save_as_next = False
            out_path = self._ask_save_as_path(ext, kind)
            if out_path is None:
                return
        n_colors, _ = self._get_color_count()
        _, raw_seeds, raw_lasso, use_border = self._active_transp_state()
        path = self.current_path
        kwargs = dict(
            do_icon=do_icon,
            tol=self.opt_tol.get(),
            remove_holes=self.opt_holes.get(),
            seeds=raw_seeds,
            do_svg=do_svg,
            bin_colors=bin_colors,
            n_colors=n_colors,
            lasso=raw_lasso,
            use_border=use_border,
            do_webp=do_webp,
            out_path=out_path,
            crop_pad=self._get_crop_pad(),
            square_canvas=self.opt_square.get(),
            source_im=self.source_rgba,
        )
        results = queue.Queue()

        def work():
            try:
                results.put(("ok", _process_with_preview(path, do_t, do_c, **kwargs)))
            except Exception as exc:
                results.put(("error", exc))

        self._comparing = False
        self._set_busy(True)
        self.set_msg(f"Exporting {kind}…")
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
            self.set_msg(f"Export failed: {value}")
            return
        out, note, preview, crop_box = value
        self.last_output = out
        self.link.grid()
        label = f"Saved {out.name} · {note.replace('; ', ' · ')}"
        if path != self.current_path:  # another picture was opened meanwhile: don't replace its preview
            self.set_msg(label)
            return
        self._crop_box = crop_box
        self._is_icon_preview = do_icon
        self.last_preview = preview
        self.last_label = label
        self.show_image(preview, self.last_label)
        self._update_compare_state()

    def _need_picture(self):
        if self.current_path is None:
            self.set_msg("Open or drop an image first.")
            return True
        return False

    def apply(self):
        if self._need_picture():
            return
        do_t = self._active_transp_state()[0]
        do_c, do_b = self.opt_crop.get(), self.opt_bin.get()
        has_crop_opts = self._get_crop_pad() > 0 or self.opt_square.get()
        if not (do_t or do_c or has_crop_opts or do_b or getattr(self, "_save_as_next", False)):
            if self.opt_mode.get() == "outline":
                self.set_msg("Nothing to save yet — drag an outline around an object or choose an option on the right.")
            else:
                self.set_msg("Nothing to save yet — click a color to delete or choose an option on the right.")
            return
        self._run_process(do_t, do_c, do_icon=False, do_svg=False, bin_colors=do_b, do_webp=False)

    def export_webp(self):
        if self._need_picture():
            return
        do_t = self._active_transp_state()[0]
        do_c, do_b = self.opt_crop.get(), self.opt_bin.get()
        self._run_process(do_t, do_c, do_icon=False, do_svg=False, bin_colors=do_b, do_webp=True)

    def export_icon(self):
        if self._need_picture():
            return
        do_t = self._active_transp_state()[0]
        do_c, do_b = self.opt_crop.get(), self.opt_bin.get()
        self._run_process(do_t, do_c, do_icon=True, do_svg=False, bin_colors=do_b, do_webp=False)

    def export_svg(self):
        if self._need_picture():
            return
        do_t = self._active_transp_state()[0]
        do_c = self.opt_crop.get()
        self._run_process(do_t, do_c, do_icon=False, do_svg=True, bin_colors=True, do_webp=False)

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