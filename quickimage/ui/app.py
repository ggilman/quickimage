"""The Quick Image main window."""

import queue
import sys
import tempfile
import threading
import tkinter
import tkinter.font
from pathlib import Path
from tkinter import colorchooser, filedialog, ttk

import numpy as np
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES, COPY

from ..bootstrap import APP_ICON_PNG, APP_NAME, DependencyDialog, missing_dependencies, restart_app, set_app_user_model_id, set_window_icon
from ..core.config import BG_TOL, DEFAULT_SVG_COLORS, ICON_SIZES, IMG_EXT, PREVIEW_MAX_SIDE
from ..core.io import is_image_file, load_rgba, next_save_path
from ..core.ops import transform_image
from ..core.filaments import DEFAULT_MATERIAL, filament_overrides, load_catalog, material_filaments, materials
from ..core.pipeline import _process_with_preview, batch_process, render_preview
from .batch_options import BatchOptions
from .desktop import copy_image_to_clipboard, grab_clipboard_image, reveal_in_folder, set_dark_title_bar
from .filament_picker import FilamentPicker, filament_label
from .theme import apply_theme


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
        self.root.bind("<bracketleft>", lambda e: self._step_brush_size(e, -1))
        self.root.bind("<bracketright>", lambda e: self._step_brush_size(e, +1))

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
        self._history = []  # stack of (seeds, lasso, brush_strokes) snapshots for Ctrl+Z undo
        self._redo_stack = []  # stack of (seeds, lasso, brush_strokes) snapshots for Ctrl+Y / Ctrl+Shift+Z redo
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

        # 1. Canvas, Zoom & Transforms
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None
        self.opt_backdrop = tkinter.StringVar(value="checker")

        # 2. Edge & Mask Refinement
        self.opt_brush_mode = tkinter.StringVar(value="erase")
        self.opt_brush_size = tkinter.IntVar(value=12)
        self.opt_brush_soft = tkinter.IntVar(value=0)  # % of the radius that fades out
        self._pointer = None  # last mouse position over the canvas (for the brush cursor)
        self.brush_strokes = []
        self._Active_stroke = None
        self._active_stroke = None
        self.opt_choke = tkinter.IntVar(value=0)
        self.opt_feather = tkinter.DoubleVar(value=0.0)

        # 3. Interactive Color Palette & 3D Print
        self.hidden_colors = set()
        self.color_overrides = {}
        self._last_raw_palette = []
        self.opt_svg_eps = tkinter.DoubleVar(value=0.85)
        self.opt_filaments = tkinter.BooleanVar(value=False)
        self.opt_fil_mode = tkinter.StringVar(value="auto")  # "auto": best match from a material; "manual": picked
        self.opt_fil_material = tkinter.StringVar(value=DEFAULT_MATERIAL)  # Auto's material, and the picker's start
        self.filaments = []  # manually picked Bambu filaments (dicts with material, name, hex, rgb)

        # 4. Export
        self.opt_out_size = tkinter.IntVar(value=0)  # longest side in px; 0 keeps the size
        self._batch_opts = None  # last batch dialog choices

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

    def _get_brush_size(self):
        """Return brush radius in pixels (1..100). Invalid text falls back to 12."""
        try:
            return max(1, min(100, int(self.opt_brush_size.get())))
        except (tkinter.TclError, ValueError):
            return 12

    def _get_brush_soft(self):
        """Return brush softness as a fraction 0.0..1.0 (spinbox shows 0..100 %)."""
        try:
            return max(0, min(100, int(self.opt_brush_soft.get()))) / 100.0
        except (tkinter.TclError, ValueError):
            return 0.0

    def _get_out_size(self):
        """Return the export size (longest side, px; 0 = unchanged). Invalid text falls back to 0."""
        try:
            return max(0, min(16384, int(self.opt_out_size.get())))
        except (tkinter.TclError, ValueError):
            return 0

    def _filament_candidates(self):
        """The filaments binned colors may become: all colors of the Auto material, or the manual picks."""
        if not self.opt_filaments.get():
            return []
        if self.opt_fil_mode.get() == "auto":
            return material_filaments(self.opt_fil_material.get())
        return list(self.filaments)

    def _active_filaments(self):
        """Filament colors to map binned colors onto, or None when that option is off."""
        cands = self._filament_candidates()
        return [tuple(f["rgb"]) for f in cands] or None

    def _get_edge_choke(self):
        """Return edge choke in pixels (-10..+10). Invalid text falls back to 0."""
        try:
            return max(-10, min(10, int(round(float(self.opt_choke.get())))))
        except (tkinter.TclError, ValueError):
            return 0

    def _get_edge_feather(self):
        """Return edge feather in pixels (0.0..10.0). Invalid text falls back to 0.0."""
        try:
            return max(0.0, min(10.0, float(self.opt_feather.get())))
        except (tkinter.TclError, ValueError):
            return 0.0

    def _get_svg_eps(self):
        """Return SVG simplification tolerance (0.1..5.0). Invalid text falls back to 0.85."""
        try:
            return max(0.1, min(5.0, float(self.opt_svg_eps.get())))
        except (tkinter.TclError, ValueError):
            return 0.85

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
        self.btn_open.grid(row=0, column=0, padx=(0, P(4)))
        self.btn_paste = ttk.Button(header, text="Paste", command=self.paste_clipboard)
        self.btn_paste.grid(row=0, column=1, padx=(0, P(4)))
        self.btn_copy = ttk.Button(header, text="Copy", command=self.copy_clipboard)
        self.btn_copy.grid(row=0, column=2, padx=(0, P(8)))
        self.file_label = ttk.Label(header, text="No image", style="Title.TLabel")
        self.file_label.grid(row=0, column=3, sticky="w")
        self.info_label = ttk.Label(header, text="", style="Muted.TLabel")
        self.info_label.grid(row=0, column=4, sticky="w", padx=(P(8), 0))

        tools = ttk.Frame(header)
        tools.grid(row=0, column=5, sticky="e", padx=(P(6), P(6)))
        self.btn_rot_ccw = ttk.Button(tools, text="↶", width=2, command=self.rotate_ccw)
        self.btn_rot_ccw.grid(row=0, column=0, padx=(0, P(2)))
        self.btn_rot_cw = ttk.Button(tools, text="↷", width=2, command=self.rotate_cw)
        self.btn_rot_cw.grid(row=0, column=1, padx=(0, P(2)))
        self.btn_flip_h = ttk.Button(tools, text="⇆", width=2, command=self.flip_horizontal)
        self.btn_flip_h.grid(row=0, column=2, padx=(0, P(2)))
        self.btn_flip_v = ttk.Button(tools, text="⇅", width=2, command=self.flip_vertical)
        self.btn_flip_v.grid(row=0, column=3, padx=(0, P(4)))
        self.btn_zoom_reset = ttk.Button(tools, text="1:1", width=3, command=self.reset_zoom)
        self.btn_zoom_reset.grid(row=0, column=4, padx=(0, P(6)))
        for col_i, (bd_val, bd_lbl) in enumerate((("checker", "Checker"), ("light", "Light"), ("dark", "Dark")), start=5):
            rb = ttk.Radiobutton(
                tools,
                text=bd_lbl,
                variable=self.opt_backdrop,
                value=bd_val,
                command=self._on_backdrop_change,
            )
            rb.grid(row=0, column=col_i, padx=(0, P(2)))

        self.btn_compare = ttk.Button(header, text="Hold to compare")
        self.btn_compare.grid(row=0, column=6, sticky="e")
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
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", self.on_canvas_leave)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", lambda _e: self.clear_picks())
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Double-Button-2>", lambda _e: self.reset_zoom())

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
                ttk.Separator(side).grid(row=row, column=0, sticky="ew", pady=(P(10), P(8)))
                row += 1
            ttk.Label(side, text=title.upper(), style="Section.TLabel").grid(
                row=row, column=0, sticky="w", pady=(0, P(4))
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
        place(self.rad_outline, pady=(0, P(2)))
        self.rad_brush = ttk.Radiobutton(
            side,
            variable=self.opt_mode,
            value="brush",
            text="Touch-up brush (erase / restore)",
            command=self._on_mode_change,
        )
        place(self.rad_brush, pady=(0, P(4)))

        self.brush_frame = ttk.Frame(side)
        self.rad_brush_erase = ttk.Radiobutton(
            self.brush_frame, variable=self.opt_brush_mode, value="erase", text="Erase"
        )
        self.rad_brush_erase.grid(row=0, column=0, sticky="w")
        self.rad_brush_restore = ttk.Radiobutton(
            self.brush_frame, variable=self.opt_brush_mode, value="restore", text="Restore"
        )
        self.rad_brush_restore.grid(row=0, column=1, sticky="w", padx=(P(6), P(8)))
        ttk.Label(self.brush_frame, text="Size").grid(row=0, column=2, padx=(0, P(4)))
        self.spin_brush_size = ttk.Spinbox(
            self.brush_frame, from_=1, to=100, width=3, textvariable=self.opt_brush_size
        )
        self.spin_brush_size.grid(row=0, column=3)
        ttk.Label(self.brush_frame, text="Soft %").grid(row=1, column=2, padx=(0, P(4)), pady=(P(4), 0))
        self.spin_brush_soft = ttk.Spinbox(
            self.brush_frame, from_=0, to=100, increment=10, width=3, textvariable=self.opt_brush_soft
        )
        self.spin_brush_soft.grid(row=1, column=3, pady=(P(4), 0))
        place(self.brush_frame, pady=(0, P(4)))
        for var in (self.opt_brush_size, self.opt_brush_soft, self.opt_brush_mode):
            var.trace_add("write", lambda *_a: self._redraw_brush_cursor())

        self.chk_h = ttk.Checkbutton(
            side, variable=self.opt_holes, text="Remove interior holes", command=self.schedule_live_preview
        )
        place(self.chk_h, pady=(0, P(4)))
        tol_row = ttk.Frame(side)
        tol_row.columnconfigure(0, weight=1)
        self.tol_label = ttk.Label(tol_row, text="Tolerance")
        self.tol_label.grid(row=0, column=0, sticky="w")
        self.tol_value = ttk.Label(tol_row, text=str(BG_TOL), style="Muted.TLabel")
        self.tol_value.grid(row=0, column=1, sticky="e")
        place(tol_row, sticky="ew")
        self.tol_scale = ttk.Scale(side, from_=10, to=240, orient="horizontal", command=self._on_tol_drag)
        self.tol_scale.set(BG_TOL)
        place(self.tol_scale, sticky="ew", pady=(P(2), P(4)))

        edge_row = ttk.Frame(side)
        ttk.Label(edge_row, text="Edge (px)").grid(row=0, column=0, sticky="w")
        self.spin_choke = ttk.Spinbox(
            edge_row, from_=-10, to=10, width=3, textvariable=self.opt_choke, command=self.schedule_live_preview
        )
        self.spin_choke.grid(row=0, column=1, padx=(P(4), P(10)))
        ttk.Label(edge_row, text="Feather").grid(row=0, column=2, sticky="w")
        self.spin_feather = ttk.Spinbox(
            edge_row, from_=0.0, to=10.0, increment=0.5, width=4, textvariable=self.opt_feather, command=self.schedule_live_preview
        )
        self.spin_feather.grid(row=0, column=3, padx=(P(4), 0))
        place(edge_row, pady=(0, P(4)))
        self.opt_choke.trace_add("write", lambda *_a: self.schedule_live_preview())
        self.opt_feather.trace_add("write", lambda *_a: self.schedule_live_preview())

        self.mode_hint = ttk.Label(
            side,
            text="",
            style="Hint.TLabel",
            wraplength=wrap,
            justify="left",
        )
        place(self.mode_hint, pady=(0, P(6)))
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

        fil_row = ttk.Frame(side)
        self.chk_fil = ttk.Checkbutton(
            fil_row, variable=self.opt_filaments, text="Bambu filaments", command=self._on_filaments_toggle
        )
        self.chk_fil.grid(row=0, column=0, sticky="w")
        self.rad_fil_auto = ttk.Radiobutton(
            fil_row, variable=self.opt_fil_mode, value="auto", text="Auto", command=self._on_fil_mode_change
        )
        self.rad_fil_auto.grid(row=0, column=1, padx=(P(10), P(4)))
        self.rad_fil_manual = ttk.Radiobutton(
            fil_row, variable=self.opt_fil_mode, value="manual", text="Manual", command=self._on_fil_mode_change
        )
        self.rad_fil_manual.grid(row=0, column=2)
        place(fil_row, pady=(P(4), 0))

        # Second line: the material for Auto, or the Pick button for Manual (same place, one at a time)
        self.fil_opts = ttk.Frame(side)
        self.fil_auto_frame = ttk.Frame(self.fil_opts)
        ttk.Label(self.fil_auto_frame, text="Material").grid(row=0, column=0, sticky="w")
        self.cmb_fil_material = ttk.Combobox(
            self.fil_auto_frame,
            textvariable=self.opt_fil_material,
            values=materials(load_catalog()),
            state="readonly",
            width=20,
        )
        self.cmb_fil_material.grid(row=0, column=1, padx=(P(8), 0))
        self.cmb_fil_material.bind("<<ComboboxSelected>>", lambda _e: self._on_fil_material_change())
        self.fil_manual_frame = ttk.Frame(self.fil_opts)
        self.btn_fil = ttk.Button(self.fil_manual_frame, text="Pick…", width=6, command=self.pick_filaments)
        self.btn_fil.grid(row=0, column=0, sticky="w")
        self.lbl_fil_picked = ttk.Label(self.fil_manual_frame, text="", style="Muted.TLabel")
        self.lbl_fil_picked.grid(row=0, column=1, sticky="w", padx=(P(8), 0))
        for f in (self.fil_auto_frame, self.fil_manual_frame):
            f.grid(row=0, column=0, sticky="w")
        place(self.fil_opts, pady=(P(4), 0))
        self._update_filament_widgets()

        self.swatch_frame = ttk.Frame(side)
        place(self.swatch_frame, sticky="ew", pady=(P(4), 0))

        svg_row = ttk.Frame(side)
        ttk.Label(svg_row, text="SVG smoothing").grid(row=0, column=0, sticky="w")
        self.spin_svg_eps = ttk.Spinbox(
            svg_row, from_=0.1, to=5.0, increment=0.15, width=4, textvariable=self.opt_svg_eps
        )
        self.spin_svg_eps.grid(row=0, column=1, padx=(P(8), 0))
        place(svg_row, pady=(P(4), 0))

        place(
            ttk.Label(
                side,
                text="Click a swatch to hide it, right-click to recolor.",
                style="Hint.TLabel",
                wraplength=wrap,
                justify="left",
            ),
            pady=(P(4), 0),
        )
        # Fires on arrow clicks and on every keystroke, so the preview always matches the export
        self.opt_colors.trace_add("write", self._on_color_count_change)

        section("Export")
        size_row = ttk.Frame(side)
        ttk.Label(size_row, text="Longest side (px)").grid(row=0, column=0, sticky="w")
        self.spin_out_size = ttk.Spinbox(
            size_row, from_=0, to=16384, increment=64, width=6, textvariable=self.opt_out_size
        )
        self.spin_out_size.grid(row=0, column=1, padx=(P(8), P(6)))
        ttk.Label(size_row, text="0 = original", style="Hint.TLabel").grid(row=0, column=2, sticky="w")
        place(size_row, pady=(0, P(6)))
        self.btn = ttk.Button(side, text="Save PNG", style="Accent.TButton", command=self.apply)
        self.btn.bind("<Shift-ButtonRelease-1>", lambda _e: self._shift_export(self.apply))
        place(self.btn, sticky="ew", pady=(0, P(4)))
        formats = ttk.Frame(side)
        self.btn_webp = ttk.Button(formats, text="WebP", command=self.export_webp)
        self.btn_ico = ttk.Button(formats, text="Icon (.ico)", command=self.export_icon)
        self.btn_svg = ttk.Button(formats, text="SVG", command=self.export_svg)
        for col, (b, fn) in enumerate(
            ((self.btn_webp, self.export_webp), (self.btn_ico, self.export_icon), (self.btn_svg, self.export_svg))
        ):
            b.bind("<Shift-ButtonRelease-1>", lambda _e, fn=fn: self._shift_export(fn))
            b.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else P(4), 0))
            formats.columnconfigure(col, weight=1, uniform="fmt")
        place(formats, sticky="ew", pady=(0, P(4)))
        self.btn_batch = ttk.Button(side, text="Batch…", command=self.batch_dialog)
        place(self.btn_batch, sticky="ew")
        place(
            ttk.Label(
                side,
                text="Saved next to the original. Shift-click to choose where.",
                style="Hint.TLabel",
                wraplength=wrap,
            ),
            pady=(P(4), 0),
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

    def checker_board(self, w, h, backdrop=None):
        """Build a (h, w, 3) uint8 backdrop tile for `"checker"`, `"light"`, or `"dark"`."""
        mode = backdrop or self.opt_backdrop.get()
        if mode == "light":
            return np.full((h, w, 3), (244, 244, 246), dtype=np.uint8)
        if mode == "dark":
            return np.full((h, w, 3), (28, 28, 32), dtype=np.uint8)
        r1, g1, b1, r2, g2, b2 = self.palette["check"]
        cell = max(1, self.px(8))
        yy = np.arange(h)[:, None] // cell
        xx = np.arange(w)[None, :] // cell
        odd = ((yy + xx) % 2).astype(bool)
        tile = np.empty((h, w, 3), np.uint8)
        tile[~odd] = (r1, g1, b1)
        tile[odd] = (r2, g2, b2)
        return tile

    def _on_backdrop_change(self):
        if self._comparing and self.source_rgba is not None:
            self.show_image(self.source_rgba, None, overlay=False)
        elif self.last_preview is not None:
            self.show_image(self.last_preview, None)

    def show_image(self, pil_img, label, overlay=True):
        """Show a picture in the box. overlay=False skips the outline (e.g. for a hovered file)."""
        bw, bh = self.bw, self.bh
        m = 2 * self.VIEW_MARGIN
        base_scale = min(max(1, bw - m) / pil_img.width, max(1, bh - m) / pil_img.height, 3.0)
        scale = base_scale * self.zoom
        w = max(1, int(round(pil_img.width * scale)))
        h = max(1, int(round(pil_img.height * scale)))
        self._disp_scale = scale
        self._disp_size = (w, h)
        small = pil_img.resize((w, h), Image.LANCZOS)
        if small.mode == "RGBA" and (np.asarray(small)[..., 3] < 255).any():
            tile = self.checker_board(w, h, backdrop=self.opt_backdrop.get())
            small = Image.alpha_composite(Image.fromarray(tile).convert("RGBA"), small).convert("RGB")
        photo = ImageTk.PhotoImage(small)
        self.canvas.delete("preview", "empty", "lasso", "brush_live")
        cx = int(round(bw / 2.0 + self.pan_x)) if (self.pan_x or self.pan_y) else bw // 2
        cy = int(round(bh / 2.0 + self.pan_y)) if (self.pan_x or self.pan_y) else bh // 2
        self.canvas.create_image(cx, cy, image=photo, anchor="center", tags="preview")
        self.canvas.tag_lower("preview")
        self.tkimg = photo
        if overlay:
            self._draw_lasso()
        self._redraw_brush_cursor()
        if label:
            self.set_msg(label)

    # --- zoom, pan, and image transforms ----------------------------------------------------

    def on_zoom(self, event):
        if self.source_rgba is None:
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            num = getattr(event, "num", 0)
            delta = 120 if num == 4 else -120 if num == 5 else 120
        factor = 1.2 if delta > 0 else (1.0 / 1.2)
        old_zoom = self.zoom
        new_zoom = float(np.clip(old_zoom * factor, 0.25, 8.0))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        cx = getattr(event, "x", self.bw / 2.0)
        cy = getattr(event, "y", self.bh / 2.0)
        ratio = new_zoom / old_zoom
        self.pan_x = (cx - self.bw / 2.0) * (1.0 - ratio) + self.pan_x * ratio
        self.pan_y = (cy - self.bh / 2.0) * (1.0 - ratio) + self.pan_y * ratio
        self.zoom = new_zoom
        if self._comparing:
            self.show_image(self.source_rgba, None, overlay=False)
        elif self.last_preview is not None:
            self.show_image(self.last_preview, None)

    def reset_zoom(self, event=None):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None
        if self._comparing and self.source_rgba is not None:
            self.show_image(self.source_rgba, None, overlay=False)
        elif self.last_preview is not None:
            self.show_image(self.last_preview, None)

    def on_pan_start(self, event):
        if self.source_rgba is None:
            return
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)

    def on_pan_motion(self, event):
        if not self._pan_start or self.source_rgba is None:
            return
        sx, sy, px0, py0 = self._pan_start
        self.pan_x = px0 + (event.x - sx)
        self.pan_y = py0 + (event.y - sy)
        if self._comparing:
            self.show_image(self.source_rgba, None, overlay=False)
        elif self.last_preview is not None:
            self.show_image(self.last_preview, None)

    def on_pan_end(self, event):
        self._pan_start = None

    def _apply_transform(self, rotate_deg=0, flip_h=False, flip_v=False):
        if self._busy or self.source_rgba is None:
            return
        rgba = transform_image(self.source_rgba, rotate_deg=rotate_deg, flip_h=flip_h, flip_v=flip_v)
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
        self.brush_strokes = []
        self._history = []
        self._redo_stack = []
        self._crop_box = (0, 0, rgba.width, rgba.height)
        self.info_label.configure(text=f"{rgba.width} × {rgba.height} px")
        self.canvas.delete("lasso", "brush_live")
        self._update_mode_widgets()
        self.update_live_preview()

    def rotate_ccw(self):
        self._apply_transform(rotate_deg=270)

    def rotate_cw(self):
        self._apply_transform(rotate_deg=90)

    def flip_horizontal(self):
        self._apply_transform(flip_h=True)

    def flip_vertical(self):
        self._apply_transform(flip_v=True)

    # --- mapping between canvas pixels and full-resolution source pixels ---------------------

    def _view_geometry(self, img_w=None, img_h=None):
        """What the box shows: displayed size/offset, crop box, and icon padding (if an icon preview)."""
        src_w, src_h = (img_w, img_h) if (img_w is not None and img_h is not None) else self.source_rgba.size
        disp_w, disp_h = self._disp_size
        ox = (self.bw - disp_w) / 2.0 + self.pan_x
        oy = (self.bh - disp_h) / 2.0 + self.pan_y
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

    # --- mouse: mode-dependent click to remove color, drag to outline, or brush touch-up -----

    def _snapshot(self):
        return (list(self.seeds), list(self.lasso), list(self.brush_strokes))

    def _restore_snapshot(self, snap):
        if len(snap) == 3:
            seeds, lasso, strokes = snap
            self.seeds, self.lasso, self.brush_strokes = list(seeds), list(lasso), list(strokes)
        else:
            seeds, lasso = snap
            self.seeds, self.lasso = list(seeds), list(lasso)
            self.brush_strokes = []

    def _push_history(self):
        self._history.append(self._snapshot())
        self._redo_stack.clear()

    def undo(self, event=None):
        """Ctrl+Z: undo the last color click, outline, or brush stroke one step at a time."""
        if event is not None and isinstance(event.widget, (ttk.Spinbox, tkinter.Entry)):
            return
        if self._busy or self._drag or self._active_stroke:
            return
        if not self._history:
            self.set_msg("Nothing to undo.")
            return
        self._redo_stack.append(self._snapshot())
        self._restore_snapshot(self._history.pop())
        self.canvas.delete("lasso", "brush_live")
        self._update_mode_widgets()
        self.update_live_preview()

    def redo(self, event=None):
        """Ctrl+Y / Ctrl+Shift+Z: redo the most recently undone action."""
        if event is not None and isinstance(event.widget, (ttk.Spinbox, tkinter.Entry)):
            return
        if self._busy or self._drag or self._active_stroke:
            return
        if not self._redo_stack:
            self.set_msg("Nothing to redo.")
            return
        self._history.append(self._snapshot())
        self._restore_snapshot(self._redo_stack.pop())
        self.canvas.delete("lasso", "brush_live")
        self._update_mode_widgets()
        self.update_live_preview()

    def _brush_radius_on_canvas(self):
        """The brush radius (in source-image pixels) as canvas pixels.

        Strokes are applied in source pixels, but the shown picture may be a downscaled proxy,
        a crop, or an icon preview, so map through the same geometry as the mouse coordinates.
        """
        r_src = float(self._get_brush_size())
        x0, _ = self._source_to_canvas(0.0, 0.0)
        x1, _ = self._source_to_canvas(r_src, 0.0)
        return abs(x1 - x0)

    def _brush_color(self):
        return "#ff5c5c" if self.opt_brush_mode.get() == "erase" else "#4cd964"

    def _draw_brush_cursor(self, cx, cy):
        """The brush outline under the mouse while hovering, so its size is visible before painting."""
        self.canvas.delete("brush_cursor")
        if (
            self.opt_mode.get() != "brush"
            or self.source_rgba is None
            or self._comparing
            or self._active_stroke
            or self.hover_path is not None
        ):
            return
        r = max(2.0, self._brush_radius_on_canvas())
        color = self._brush_color()
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=self.px(2), tags="brush_cursor")
        soft = self._get_brush_soft()
        if soft > 0:  # the inner, full-strength part of a soft brush
            ri = max(1.0, r * (1.0 - soft))
            self.canvas.create_oval(
                cx - ri, cy - ri, cx + ri, cy + ri, outline=color, width=1, dash=(3, 3), tags="brush_cursor"
            )

    def _redraw_brush_cursor(self):
        if self.canvas is not None and self._pointer is not None:
            self._draw_brush_cursor(*self._pointer)

    def on_canvas_motion(self, event):
        self._pointer = (event.x, event.y)
        self._draw_brush_cursor(event.x, event.y)

    def on_canvas_leave(self, _event=None):
        self._pointer = None
        self.canvas.delete("brush_cursor")

    def _step_brush_size(self, event, direction):
        """[ and ]: smaller / larger brush (steps grow with the size)."""
        if isinstance(event.widget, (ttk.Entry, ttk.Spinbox, tkinter.Entry)):
            return None  # typing a value
        if self.opt_mode.get() != "brush":
            return None
        cur = self._get_brush_size()
        step = 1 if cur < 10 else 2 if cur < 30 else 5
        new = max(1, min(100, cur + direction * step))
        self.opt_brush_size.set(new)
        self.set_msg(f"Brush size {new}")
        return "break"

    def _draw_brush_live(self, cx, cy, canvas_pts):
        self.canvas.delete("brush_live")
        r_disp = max(2.0, self._brush_radius_on_canvas())
        color = self._brush_color()
        if len(canvas_pts) >= 2:
            flat = [c for p in canvas_pts for c in p]
            self.canvas.create_line(
                *flat, fill=color, width=max(1, int(round(2 * r_disp))), capstyle="round", joinstyle="round", tags="brush_live"
            )
        self.canvas.create_oval(
            cx - r_disp, cy - r_disp, cx + r_disp, cy + r_disp, outline=color, width=self.px(2), tags="brush_live"
        )

    def on_press(self, event):
        self.canvas.focus_set()  # so Space (compare) is not swallowed by a focused button
        if self.source_rgba is None:
            self.open_dialog()
            return
        if self._comparing or self._busy:
            return
        mode = self.opt_mode.get()
        if mode == "color":
            self.pick_background(event)
            return
        if mode == "brush":
            p = self._canvas_to_source(event.x, event.y, clamp=True)
            if p is None:
                return
            self.canvas.delete("brush_cursor")
            self._active_stroke = [(event.x, event.y)]
            self._Active_stroke = [p]
            self._draw_brush_live(event.x, event.y, self._active_stroke)
            return
        self._drag = [(event.x, event.y)]
        self._drag_far = False

    def on_motion(self, event):
        if self.opt_mode.get() == "brush":
            if not self._active_stroke:
                return
            p = self._canvas_to_source(event.x, event.y, clamp=True)
            if p is not None:
                if not self._Active_stroke or np.hypot(p[0] - self._Active_stroke[-1][0], p[1] - self._Active_stroke[-1][1]) >= 1.0:
                    self._Active_stroke.append(p)
            lx, ly = self._active_stroke[-1]
            if np.hypot(event.x - lx, event.y - ly) >= 2:
                self._active_stroke.append((event.x, event.y))
            self._draw_brush_live(event.x, event.y, self._active_stroke)
            return
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
        """Escape: abandon an outline or brush stroke that is being drawn."""
        if self._drag:
            self._drag, self._drag_far = None, False
            self.canvas.delete("lasso_live")
            self.set_msg("Outline cancelled.")
        if self._active_stroke:
            self._active_stroke = None
            self._Active_stroke = None
            self.canvas.delete("brush_live")
            self.set_msg("Brush stroke cancelled.")

    def on_release(self, event):
        if self.opt_mode.get() == "brush":
            src_pts = self._Active_stroke
            self._active_stroke = None
            self._Active_stroke = None
            self.canvas.delete("brush_live")
            if not src_pts or self.source_rgba is None:
                return
            p = self._canvas_to_source(event.x, event.y, clamp=True)
            if p is not None and (not src_pts or np.hypot(p[0] - src_pts[-1][0], p[1] - src_pts[-1][1]) >= 1.0):
                src_pts.append(p)
            self._push_history()
            radius_src = float(self._get_brush_size())
            self.brush_strokes.append((self.opt_brush_mode.get(), radius_src, list(src_pts), self._get_brush_soft()))
            self._update_mode_widgets()
            self.update_live_preview()
            self._draw_brush_cursor(event.x, event.y)
            return
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
        self.seeds.append((ix, iy, int(self.opt_tol.get())))  # the tolerance sticks with this click
        self._update_mode_widgets()
        self.update_live_preview()

    # --- options ------------------------------------------------------------------------------

    def _on_mode_change(self):
        self.cancel_drag()
        self.canvas.delete("lasso", "brush_live")
        self._update_mode_widgets()
        self.update_live_preview()

    def _active_transp_state(self):
        """Return (do_transp, seeds, lasso, use_border) for the currently selected mode."""
        seeds = list(self.seeds) or None
        lasso = list(self.lasso) or None
        return bool(seeds or lasso), seeds, lasso, False

    def _has_mask_refinements(self):
        return bool(self.brush_strokes or self._get_edge_choke() != 0 or self._get_edge_feather() > 0)

    def _on_tol_drag(self, value):
        v = int(round(float(value)))
        self.tol_value.configure(text=str(v))
        if v != self.opt_tol.get():
            self.opt_tol.set(v)
            if self.seeds:
                # Each click keeps the tolerance it was made with; the slider is for the next click
                self.set_msg(f"Tolerance {v} — used for your next click. Earlier clicks keep their own.")
            else:
                self.schedule_live_preview()

    def _set_tolerance(self, v):
        self.opt_tol.set(int(v))
        self.tol_scale.set(int(v))
        self.tol_value.configure(text=str(int(v)))

    def _update_mode_widgets(self):
        """Update hints and grey out controls that do not apply to the active mode."""
        mode = self.opt_mode.get()
        is_outline = mode == "outline"
        is_brush = mode == "brush"
        self.tol_scale.state(["disabled"] if (is_outline or is_brush) else ["!disabled"])
        self.tol_label.configure(style="Muted.TLabel" if (is_outline or is_brush) else "TLabel")
        for w in (self.rad_brush_erase, self.rad_brush_restore, self.spin_brush_size, self.spin_brush_soft):
            w.state(["!disabled"] if is_brush else ["disabled"])
        if not is_brush:
            self.canvas.delete("brush_cursor")
        else:
            self._redraw_brush_cursor()
        has_transp = bool(self.lasso or self.seeds)
        self.chk_h.state(["!disabled"] if has_transp else ["disabled"])
        if is_outline:
            self.mode_hint.configure(
                text="Drag a loop around an object to cut it out. "
                "Ctrl+Z undoes, right-click clears."
            )
        elif is_brush:
            self.mode_hint.configure(
                text="Drag over the image to erase to transparent or restore from original. "
                "[ and ] change the size. Ctrl+Z undoes, right-click clears."
            )
        else:
            self.mode_hint.configure(
                text="Click any color in the preview to delete it (click more colors to remove them too). "
                "Each click uses the tolerance set when you click. Ctrl+Z undoes, right-click clears."
            )

    def _update_compare_state(self):
        can = self.source_rgba is not None and self.last_preview is not self.source_rgba
        self.btn_compare.state(["!disabled"] if can else ["disabled"])
        has_img = self.source_rgba is not None
        if hasattr(self, "btn_copy"):
            self.btn_copy.state(["!disabled"] if has_img else ["disabled"])

    def _on_color_count_change(self, *_args):
        self.hidden_colors.clear()
        self.color_overrides.clear()
        self._update_filament_widgets()
        self.schedule_live_preview()

    def toggle_swatch(self, idx):
        if idx in self.hidden_colors:
            self.hidden_colors.remove(idx)
        else:
            self.hidden_colors.add(idx)
        self.update_live_preview()

    def recolor_swatch(self, idx, rgb=None):
        if rgb is None:
            cur = self._shown_swatch_color(idx)
            init_hex = f"#{cur[0]:02x}{cur[1]:02x}{cur[2]:02x}" if cur else "#ffffff"
            chosen_rgb, _ = colorchooser.askcolor(color=init_hex, parent=self.root, title=f"Recolor bin #{idx + 1}")
            if not chosen_rgb:
                return
            rgb = tuple(int(round(c)) for c in chosen_rgb[:3])
        else:
            rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self.color_overrides[idx] = rgb
        self.update_live_preview()

    def _rebuild_swatches(self, raw_palette=None):
        for child in self.swatch_frame.winfo_children():
            child.destroy()
        if not self.opt_bin.get() or not raw_palette:
            self._last_raw_palette = []
            return
        self._last_raw_palette = list(raw_palette)
        P = self.px
        fil = self._active_filaments()
        fil_map = filament_overrides(raw_palette, fil) if fil else {}
        for idx, raw_rgb in enumerate(raw_palette):
            rgb = self.color_overrides.get(idx, fil_map.get(idx, raw_rgb))
            hex_col = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            is_hidden = idx in self.hidden_colors
            sw = tkinter.Canvas(
                self.swatch_frame,
                width=P(22),
                height=P(22),
                highlightthickness=1,
                highlightbackground=self.palette["muted"],
                background=self.palette["canvas"] if is_hidden else hex_col,
                cursor="hand2",
            )
            sw.grid(row=idx // 8, column=idx % 8, padx=(0, P(4)), pady=(0, P(4)))
            if is_hidden:
                sw.create_line(2, 2, P(20), P(20), fill="#ff5c5c", width=2)
                sw.create_line(2, P(20), P(20), 2, fill="#ff5c5c", width=2)
            sw.bind("<Button-1>", lambda _e, i=idx: self.toggle_swatch(i))
            sw.bind("<Shift-Button-1>", lambda _e, i=idx: self.recolor_swatch(i))
            sw.bind("<Button-3>", lambda _e, i=idx: self.recolor_swatch(i))
            sw.bind("<Enter>", lambda _e, i=idx: self._swatch_hover(i))
            sw.bind("<Leave>", lambda _e: self.set_msg(self.last_label or ""))

    def _shown_swatch_color(self, idx):
        """The color a swatch currently stands for: hand recolor, else filament, else binned."""
        if idx in self.color_overrides:
            return tuple(self.color_overrides[idx])
        fil = self._active_filaments()
        if fil and idx < len(self._last_raw_palette):
            return filament_overrides(self._last_raw_palette, fil).get(idx, self._last_raw_palette[idx])
        return tuple(self._last_raw_palette[idx]) if idx < len(self._last_raw_palette) else None

    def _swatch_hover(self, idx):
        rgb = self._shown_swatch_color(idx)
        if rgb is None:
            return
        hex_col = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        name = None
        if idx not in self.color_overrides:
            name = next((filament_label(f) for f in self._filament_candidates() if tuple(f["rgb"]) == rgb), None)
        hidden = " (hidden)" if idx in self.hidden_colors else ""
        self.set_msg(f"{name + ' ' if name else ''}{hex_col}{hidden} — click to hide or show, right-click to recolor.")

    def _update_filament_widgets(self):
        """Show the material box (Auto) or the Pick button (Manual); grey out when filaments are off."""
        if not hasattr(self, "fil_auto_frame"):
            return
        manual = self.opt_fil_mode.get() == "manual"
        (self.fil_manual_frame if manual else self.fil_auto_frame).grid()
        (self.fil_auto_frame if manual else self.fil_manual_frame).grid_remove()
        on = ["!disabled"] if self.opt_filaments.get() else ["disabled"]
        for w in (self.rad_fil_auto, self.rad_fil_manual, self.btn_fil):
            w.state(on)
        self.cmb_fil_material.state(["readonly"] + on)
        n, want = len(self.filaments), self._get_color_count()[0]
        if not n:
            text = f"pick {want}"
        elif n == want:
            text = f"{n} picked"
        else:
            text = f"{n} picked, color limit {want}"
        self.lbl_fil_picked.configure(text=text)

    def _filaments_changed(self):
        self.hidden_colors.clear()
        self.color_overrides.clear()
        self._update_filament_widgets()
        self.schedule_live_preview()

    def _on_filaments_toggle(self):
        if self.opt_filaments.get():
            self.opt_bin.set(True)
            if self.opt_fil_mode.get() == "manual" and not self.filaments:
                self.pick_filaments()
                return
        self._filaments_changed()

    def _auto_chosen_filaments(self):
        """The filaments Auto is using for the current preview, in swatch order (no repeats)."""
        if not self._last_raw_palette:
            return []
        cands = material_filaments(self.opt_fil_material.get())
        by_rgb = {tuple(f["rgb"]): f for f in cands}
        mapping = filament_overrides(self._last_raw_palette, list(by_rgb))
        chosen = []
        for idx in range(len(self._last_raw_palette)):
            f = by_rgb.get(mapping.get(idx))
            if f is not None and f not in chosen:
                chosen.append(f)
        return chosen

    def _on_fil_mode_change(self):
        if self.opt_fil_mode.get() == "manual":
            # Start Manual from what Auto chose, so changing one filament is a single click.
            # Auto's choice is the best match among exactly these filaments, so the preview and
            # any hidden or recolored swatches stay the same.
            auto = self._auto_chosen_filaments() if self.opt_filaments.get() else []
            if auto:
                self.filaments = [dict(f) for f in auto]
                self._update_filament_widgets()
                self.update_live_preview()  # now, so its status line does not replace the message below
                names = ", ".join(filament_label(f) for f in self.filaments)
                self.set_msg(f"Manual, starting from Auto's filaments: {names}. Click Pick… to change one.")
                return
            if not self.filaments:
                self.pick_filaments()
                return
        self._filaments_changed()

    def _on_fil_material_change(self):
        self.cmb_fil_material.selection_clear()
        self._filaments_changed()

    def pick_filaments(self):
        """Manual mode: pick as many Bambu filaments as the color limit; each binned color becomes one."""
        count = self._get_color_count()[0]
        dlg = FilamentPicker(
            self.root, self.palette, self.px, count=count, selected=self.filaments, material=self.opt_fil_material.get()
        )
        self.root.wait_window(dlg.top)
        if dlg.result is None:
            if not self.filaments:  # nothing to use manually: fall back to Auto
                self.opt_fil_mode.set("auto")
            self._filaments_changed()
            return
        self.filaments = dlg.result
        self.opt_fil_mode.set("manual")
        self.opt_filaments.set(True)
        self.opt_bin.set(True)
        self._filaments_changed()
        names = ", ".join(filament_label(f) for f in self.filaments)
        self.set_msg(f"Filaments: {names}.")

    def clear_picks(self):
        if not self.seeds and not self.lasso and not self.brush_strokes:
            return
        self._push_history()
        self.seeds = []
        self.lasso = []
        self.brush_strokes = []
        self.canvas.delete("lasso", "brush_live")
        self._update_mode_widgets()
        self.update_live_preview()

    def reset_all(self):
        """Back to the original picture: no outline or picks, all options off, defaults restored."""
        self.seeds = []
        self.lasso = []
        self.brush_strokes = []
        self._history = []
        self._redo_stack = []
        self.hidden_colors.clear()
        self.color_overrides.clear()
        self._last_raw_palette = []
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None
        self.canvas.delete("lasso", "brush_live")
        self.opt_mode.set("color")
        for var in (self.opt_holes, self.opt_crop, self.opt_square, self.opt_bin):
            var.set(False)
        self.opt_pad.set(0)
        self.opt_choke.set(0)
        self.opt_feather.set(0.0)
        self.opt_svg_eps.set(0.85)
        self.opt_filaments.set(False)  # the picked filaments are kept for next time
        self._update_filament_widgets()
        self.opt_out_size.set(0)
        self.opt_brush_soft.set(0)
        self._set_tolerance(BG_TOL)
        self.opt_colors.set(DEFAULT_SVG_COLORS)  # its trace schedules a preview; cancelled below
        self._rebuild_swatches(None)
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
        choke = self._get_edge_choke()
        feather = self._get_edge_feather()
        has_refine = bool(self.brush_strokes or choke != 0 or feather > 0)
        if not (do_t or do_c or crop_pad > 0 or square_canvas or do_b or has_refine):
            self._rebuild_swatches(None)
            self._crop_box = (0, 0, self.source_rgba.width, self.source_rgba.height)
            self._is_icon_preview = False
            self.last_preview = self.source_rgba
            mode = self.opt_mode.get()
            if mode == "outline":
                self.last_label = "Original image. Drag a loop around an object to cut it out."
            elif mode == "brush":
                self.last_label = "Original image. Drag with the touch-up brush to erase or restore."
            else:
                self.last_label = "Original image. Click a color in the image to delete it."
            self.show_image(self.source_rgba, self.last_label)
            self._update_compare_state()
            return
        s = self.preview_scale
        seeds = [(x * s, y * s, *rest) for x, y, *rest in raw_seeds] if raw_seeds else None
        lasso = [(x * s, y * s) for x, y in raw_lasso] if raw_lasso else None
        scaled_strokes = (
            [(m, max(1.0, r * s), [(x * s, y * s) for x, y in pts], *rest) for m, r, pts, *rest in self.brush_strokes]
            if self.brush_strokes
            else None
        )
        try:
            full_im, preview, notes, crop_box = render_preview(
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
                edge_choke=int(round(choke * s)) if choke else 0,
                edge_feather=feather * s,
                brush_strokes=scaled_strokes,
                hidden_indices=self.hidden_colors or None,
                color_overrides=self.color_overrides or None,
                filaments=self._active_filaments() if do_b else None,
            )
        except Exception as exc:  # pythonw has no console: surface the error in the UI
            self.set_msg(f"Preview error: {exc}")
            return
        raw_pal = full_im.info.get("raw_palette") or preview.info.get("raw_palette")
        self._rebuild_swatches(raw_pal if do_b else None)
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
        if raw_seeds:
            tols = sorted({int(sd[2]) if len(sd) > 2 else tol for sd in raw_seeds})
            notes.append(("tolerance " if len(tols) == 1 else "tolerances ") + ", ".join(map(str, tols)))
        elif do_t and not raw_lasso:
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
        choke = self._get_edge_choke()
        feather = self._get_edge_feather()
        has_refine = bool(self.brush_strokes or choke != 0 or feather > 0)
        try:
            if do_t or do_c or crop_pad > 0 or square_canvas or do_b or has_refine:
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
                    edge_choke=choke,
                    edge_feather=feather,
                    brush_strokes=list(self.brush_strokes) or None,
                    hidden_indices=self.hidden_colors or None,
                    color_overrides=self.color_overrides or None,
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
        self.brush_strokes = []
        self._history = []
        self._redo_stack = []
        self.hidden_colors.clear()
        self.color_overrides.clear()
        self._last_raw_palette = []
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None
        self._comparing = False
        self._rebuild_swatches(None)
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
        elif self.opt_mode.get() == "brush":
            self.last_label = "Original image. Drag with the touch-up brush to erase or restore."
        else:
            self.last_label = "Original image. Click a color in the image to delete it."
        self.show_image(rgba, self.last_label)
        if (
            self.opt_crop.get()
            or self._get_crop_pad() > 0
            or self.opt_square.get()
            or self.opt_bin.get()
            or self._has_mask_refinements()
        ):
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
        # Load the first valid image into the editor
        path = rgba = None
        failed = []
        valid_images = []
        for p in images:
            try:
                im = load_rgba(p)
                valid_images.append(Path(p))
                if path is None:
                    rgba = im
                    path = Path(p)
            except Exception:
                failed.append(Path(p).name)
        if path is None:
            self._restore_view()
            self.set_msg(f"Could not open {', '.join(failed)}.")
            return COPY
        self.load_picture(path, rgba)
        if len(valid_images) > 1:
            self._run_batch(valid_images, from_drop=True)
        return COPY

    # --- exporting & batch processing -----------------------------------------------------------

    def _set_busy(self, busy):
        self._busy = busy
        for b in (self.btn, self.btn_webp, self.btn_ico, self.btn_svg, self.btn_batch, self.btn_open, self.btn_paste):
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

    def batch_dialog(self):
        if self._busy:
            return
        pattern = " ".join(f"*{e}" for e in sorted(IMG_EXT))
        chosen = filedialog.askopenfilenames(
            parent=self.root,
            title="Batch process images",
            filetypes=[("Images", pattern), ("All files", "*.*")],
            initialdir=str(self.current_path.parent) if self.current_path else None,
        )
        if not chosen:
            return
        paths = [Path(p) for p in chosen if is_image_file(p)]
        if not paths:
            self.set_msg("No valid image files selected for batch processing.")
            return
        self._run_batch(paths, from_drop=False)

    def _batch_settings_summary(self):
        parts = []
        if self.opt_crop.get():
            parts.append("crop to content")
        if self._get_crop_pad() > 0:
            parts.append(f"pad {self._get_crop_pad()} px")
        if self.opt_square.get():
            parts.append("square")
        if self.opt_bin.get():
            n = self._get_color_count()[0]
            if not self._active_filaments():
                parts.append(f"{n} colors")
            elif self.opt_fil_mode.get() == "auto":
                parts.append(f"{n} best-matching Bambu {self.opt_fil_material.get()} filaments")
            else:
                parts.append(f"{len(self.filaments)} picked Bambu filaments")
        if self._get_edge_choke():
            parts.append(f"edge {self._get_edge_choke():+d} px")
        if self._get_edge_feather() > 0:
            parts.append(f"feather {self._get_edge_feather():g} px")
        if self._get_out_size():
            parts.append(f"longest side {self._get_out_size()} px")
        parts.append(f"tolerance {self.opt_tol.get()}")
        return ", ".join(parts)

    def _ask_batch_options(self, paths):
        initial = str(Path(paths[0]).parent) if paths else None
        dlg = BatchOptions(
            self.root, self.px, len(paths), self._batch_settings_summary(), options=self._batch_opts, initial_dir=initial
        )
        self.root.wait_window(dlg.top)
        if dlg.result is not None:
            self._batch_opts = dlg.result
        return dlg.result

    def _run_batch(self, paths, from_drop=False, options=None):
        if self._busy:
            return
        if options is None:
            options = self._ask_batch_options(paths)
            if options is None:
                self.set_msg("Batch cancelled.")
                return
        fmt = options["format"]
        n_colors, _ = self._get_color_count()
        do_c = self.opt_crop.get()
        do_b = self.opt_bin.get()
        crop_pad = self._get_crop_pad()
        square_canvas = self.opt_square.get()
        choke = self._get_edge_choke()
        feather = self._get_edge_feather()
        svg_eps = self._get_svg_eps()
        tol = self.opt_tol.get()
        remove_holes = self.opt_holes.get()
        # Batch files have no clicked colors: background removal uses the colors at the image border
        do_t = bool(options["remove_bg"])
        kwargs = dict(
            do_transp=do_t,
            do_crop=do_c,
            do_icon=fmt == "ico",
            tol=tol,
            remove_holes=remove_holes,
            do_svg=fmt == "svg",
            bin_colors=do_b or fmt == "svg",
            n_colors=n_colors,
            use_border=do_t,
            do_webp=fmt == "webp",
            crop_pad=crop_pad,
            square_canvas=square_canvas,
            edge_choke=choke,
            edge_feather=feather,
            svg_eps=svg_eps,
            out_size=self._get_out_size(),
            filaments=self._active_filaments(),
            out_dir=options["out_dir"] if options.get("use_folder") else None,
            suffix=options["suffix"],
        )
        results = queue.Queue()

        def work():
            try:
                results.put(("ok", batch_process(paths, **kwargs)))
            except Exception as exc:
                results.put(("error", exc))

        self._comparing = False
        self._set_busy(True)
        self.set_msg(f"Batch processing {len(paths)} files…")
        threading.Thread(target=work, name="quickimage-batch", daemon=True).start()
        self.root.after(50, self._poll_batch, results, from_drop)

    def _poll_batch(self, results, from_drop):
        try:
            kind, value = results.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_batch, results, from_drop)
            return
        self._set_busy(False)
        if kind == "error":
            self.set_msg(f"Batch failed: {value}")
            return
        outputs, errors = value
        if outputs:
            self.last_output = outputs[-1][1]
            self.link.grid()
        err_part = ""
        if errors:
            first = errors[0]
            err_part = f" ({len(errors)} failed — {Path(first[0]).name}: {first[1]})"
        if from_drop:
            self.set_msg(f"Opened first image and batch-saved {len(outputs)} files{err_part}.")
        else:
            self.set_msg(f"Batch-saved {len(outputs)} file(s){err_part}.")

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
            edge_choke=self._get_edge_choke(),
            edge_feather=self._get_edge_feather(),
            brush_strokes=list(self.brush_strokes) or None,
            hidden_indices=set(self.hidden_colors) if self.hidden_colors else None,
            color_overrides=dict(self.color_overrides) if self.color_overrides else None,
            svg_eps=self._get_svg_eps(),
            out_size=self._get_out_size(),
            filaments=self._active_filaments(),
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
        has_refine = self._has_mask_refinements()
        if not (do_t or do_c or has_crop_opts or do_b or has_refine or getattr(self, "_save_as_next", False)):
            if self.opt_mode.get() == "outline":
                self.set_msg("Nothing to save yet — drag an outline around an object or choose an option on the right.")
            elif self.opt_mode.get() == "brush":
                self.set_msg("Nothing to save yet — paint with the touch-up brush or choose an option on the right.")
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
