"""Dialog for batch output choices: format, background removal, where to save, and file names."""

import tkinter
from pathlib import Path
from tkinter import filedialog, ttk

from ..core.io import DEFAULT_SUFFIX

FORMATS = (("png", "PNG"), ("webp", "WebP"), ("ico", "Icon (.ico)"), ("svg", "SVG"))
DEFAULT_OPTIONS = {"format": "png", "remove_bg": True, "out_dir": "", "use_folder": False, "suffix": DEFAULT_SUFFIX}


class BatchOptions:
    """Ask how to process `count` files. `self.result` is an options dict, or None if cancelled."""

    def __init__(self, parent, px, count, settings_summary, options=None, initial_dir=None):
        opts = dict(DEFAULT_OPTIONS, **(options or {}))
        self.result = None
        self.initial_dir = initial_dir
        P = px
        self.top = top = tkinter.Toplevel(parent)
        top.title(f"Batch process {count} file{'s' if count != 1 else ''}")
        top.transient(parent)
        top.resizable(False, False)
        frame = ttk.Frame(top, padding=P(16))
        frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)

        self.opt_format = tkinter.StringVar(value=opts["format"])
        self.opt_remove_bg = tkinter.BooleanVar(value=opts["remove_bg"])
        self.opt_use_folder = tkinter.BooleanVar(value=opts["use_folder"])
        self.opt_out_dir = tkinter.StringVar(value=opts["out_dir"])
        self.opt_suffix = tkinter.StringVar(value=opts["suffix"])

        r = 0
        ttk.Label(frame, text="Save as").grid(row=r, column=0, sticky="w", padx=(0, P(12)))
        fmt = ttk.Frame(frame)
        fmt.grid(row=r, column=1, sticky="w")
        for i, (val, label) in enumerate(FORMATS):
            ttk.Radiobutton(fmt, text=label, value=val, variable=self.opt_format).grid(
                row=0, column=i, padx=(0, P(10))
            )
        r += 1
        ttk.Checkbutton(
            frame, text="Remove background (colors matching the image border)", variable=self.opt_remove_bg
        ).grid(row=r, column=1, sticky="w", pady=(P(8), 0))
        r += 1

        ttk.Label(frame, text="Save to").grid(row=r, column=0, sticky="nw", padx=(0, P(12)), pady=(P(12), 0))
        where = ttk.Frame(frame)
        where.grid(row=r, column=1, sticky="ew", pady=(P(12), 0))
        where.columnconfigure(1, weight=1)
        ttk.Radiobutton(
            where, text="Next to each original", value=False, variable=self.opt_use_folder, command=self._sync
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(where, text="Folder:", value=True, variable=self.opt_use_folder, command=self._sync).grid(
            row=1, column=0, sticky="w", pady=(P(4), 0)
        )
        self.ent_dir = ttk.Entry(where, textvariable=self.opt_out_dir, width=40)
        self.ent_dir.grid(row=1, column=1, sticky="ew", padx=(P(6), P(6)), pady=(P(4), 0))
        self.btn_browse = ttk.Button(where, text="Browse…", command=self._browse)
        self.btn_browse.grid(row=1, column=2, pady=(P(4), 0))
        r += 1

        ttk.Label(frame, text="Name suffix").grid(row=r, column=0, sticky="w", padx=(0, P(12)), pady=(P(12), 0))
        suf = ttk.Frame(frame)
        suf.grid(row=r, column=1, sticky="w", pady=(P(12), 0))
        ttk.Entry(suf, textvariable=self.opt_suffix, width=14).grid(row=0, column=0)
        self.example = ttk.Label(suf, text="", style="Hint.TLabel")
        self.example.grid(row=0, column=1, padx=(P(8), 0))
        r += 1

        ttk.Label(
            frame,
            text=f"Also uses the sidebar settings: {settings_summary}. Existing files are never overwritten.",
            style="Hint.TLabel",
            wraplength=P(460),
            justify="left",
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(P(14), 0))
        r += 1
        self.error = ttk.Label(frame, text="", foreground="#D64545")
        self.error.grid(row=r, column=0, columnspan=2, sticky="w", pady=(P(6), 0))
        r += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=r, column=0, columnspan=2, sticky="e", pady=(P(8), 0))
        ttk.Button(buttons, text="Process", style="Accent.TButton", command=self._ok).grid(
            row=0, column=0, padx=(0, P(8))
        )
        ttk.Button(buttons, text="Cancel", command=top.destroy).grid(row=0, column=1)
        top.bind("<Escape>", lambda _e: top.destroy())
        top.bind("<Return>", lambda _e: self._ok())
        for var in (self.opt_format, self.opt_suffix):
            var.trace_add("write", lambda *_a: self._update_example())
        self._sync()
        self._update_example()
        top.grab_set()

    def _sync(self):
        state = ["!disabled"] if self.opt_use_folder.get() else ["disabled"]
        self.ent_dir.state(state)
        self.btn_browse.state(state)

    def _update_example(self):
        self.example.configure(text=f"photo.jpg → photo{self.opt_suffix.get()}.{self.opt_format.get()}")

    def _browse(self):
        chosen = filedialog.askdirectory(
            parent=self.top, title="Save batch output to", initialdir=self.opt_out_dir.get() or self.initial_dir
        )
        if chosen:
            self.opt_out_dir.set(chosen)
            self.opt_use_folder.set(True)
            self._sync()

    def _ok(self):
        suffix = self.opt_suffix.get()
        if any(c in suffix for c in '\\/:*?"<>|'):
            self.error.configure(text='The suffix cannot contain \\ / : * ? " < > |')
            return
        out_dir = self.opt_out_dir.get().strip()
        if self.opt_use_folder.get():
            if not out_dir:
                self.error.configure(text="Choose a folder, or save next to each original.")
                return
            if Path(out_dir).exists() and not Path(out_dir).is_dir():
                self.error.configure(text="That path is a file, not a folder.")
                return
        self.result = {
            "format": self.opt_format.get(),
            "remove_bg": self.opt_remove_bg.get(),
            "use_folder": self.opt_use_folder.get(),
            "out_dir": out_dir,
            "suffix": suffix,
        }
        self.top.destroy()
