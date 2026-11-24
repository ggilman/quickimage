"""Dialog for choosing the Bambu Lab filaments that binned colors are mapped onto."""

import tkinter
from tkinter import ttk

from ..core.filaments import DEFAULT_MATERIAL, load_catalog, materials

GRID_COLUMNS = 6
MAX_FILAMENTS = 16  # same limit as the color count


def filament_label(f):
    return f"{f['material']} {f['name']}"


class FilamentPicker:
    """Pick `count` filaments (the color limit) from Bambu's palette. `self.result` is the list,
    or None if cancelled.

    Picks persist while switching material, so a set can mix e.g. PLA Basic and PLA Matte.
    """

    def __init__(self, parent, palette, px, count=4, selected=None, material=None):
        self.palette = palette
        self.px = px
        self.catalog = load_catalog()
        self.count = max(1, min(MAX_FILAMENTS, int(count)))
        self.selected = [dict(f) for f in (selected or [])][: self.count]
        self.result = None
        P = px

        self.top = top = tkinter.Toplevel(parent)
        top.title("Bambu filaments")
        top.transient(parent)
        top.resizable(False, False)
        frame = ttk.Frame(top, padding=P(16))
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)

        head = ttk.Frame(frame)
        head.grid(row=0, column=0, sticky="ew")
        ttk.Label(head, text="Material").grid(row=0, column=0, sticky="w")
        names = materials(self.catalog)
        self.opt_material = tkinter.StringVar(value=material if material in names else DEFAULT_MATERIAL)
        combo = ttk.Combobox(head, textvariable=self.opt_material, values=names, state="readonly", width=24)
        combo.grid(row=0, column=1, padx=(P(8), P(18)))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._build_grid())

        ttk.Label(
            frame,
            text=f"Pick the {self.count} filament{'s' if self.count != 1 else ''} you will print with "
            f"(the color limit is {self.count}); click again to remove. "
            "Each color in the image becomes one of these filaments.",
            style="Hint.TLabel",
            wraplength=P(560),
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(P(8), P(8)))

        self.grid_frame = ttk.Frame(frame)
        self.grid_frame.grid(row=2, column=0, sticky="nw")

        ttk.Separator(frame).grid(row=3, column=0, sticky="ew", pady=(P(10), P(8)))
        self.picked_label = ttk.Label(frame, text="")
        self.picked_label.grid(row=4, column=0, sticky="w")
        self.picked_frame = ttk.Frame(frame)
        self.picked_frame.grid(row=5, column=0, sticky="w", pady=(P(4), 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, sticky="e", pady=(P(12), 0))
        ttk.Button(buttons, text="Clear", command=self._clear).grid(row=0, column=0, padx=(0, P(8)))
        self.btn_ok = ttk.Button(buttons, text="Use filaments", style="Accent.TButton", command=self._ok)
        self.btn_ok.grid(row=0, column=1, padx=(0, P(8)))
        ttk.Button(buttons, text="Cancel", command=self.top.destroy).grid(row=0, column=2)
        top.bind("<Escape>", lambda _e: top.destroy())
        top.bind("<Return>", lambda _e: self._ok())

        self._build_grid()
        self._refresh_picked()
        top.grab_set()

    def _count(self):
        return self.count

    def _is_selected(self, f):
        return any(s["material"] == f["material"] and s["name"] == f["name"] for s in self.selected)

    def toggle(self, f):
        if self._is_selected(f):
            self.selected = [s for s in self.selected if not (s["material"] == f["material"] and s["name"] == f["name"])]
        elif len(self.selected) < self._count():
            self.selected.append(dict(f))
        else:
            self.picked_label.configure(
                text=f"Already {self._count()} picked — remove one first, or raise the color limit."
            )
            return
        self._build_grid()
        self._refresh_picked()

    def _swatch(self, parent, f, size, selected):
        P = self.px
        sw = tkinter.Canvas(
            parent,
            width=size,
            height=size,
            background=f["hex"],
            highlightthickness=P(3) if selected else 1,
            highlightbackground=self.palette["accent"] if selected else self.palette["muted"],
            cursor="hand2",
        )
        if selected:  # check mark in a contrasting color
            r, g, b = f["rgb"]
            mark = "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#FFFFFF"
            s = size
            sw.create_line(s * 0.25, s * 0.52, s * 0.43, s * 0.7, s * 0.76, s * 0.3, fill=mark, width=P(3))
        return sw

    def _build_grid(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        P = self.px
        mat = self.opt_material.get()
        items = [f for f in self.catalog if f["material"] == mat]
        for i, f in enumerate(items):
            cell = ttk.Frame(self.grid_frame, width=P(92))
            cell.grid(row=i // GRID_COLUMNS, column=i % GRID_COLUMNS, padx=P(3), pady=P(3), sticky="n")
            sw = self._swatch(cell, f, P(34), self._is_selected(f))
            sw.grid(row=0, column=0)
            lbl = ttk.Label(cell, text=f["name"], style="Hint.TLabel", wraplength=P(88), justify="center", cursor="hand2")
            lbl.grid(row=1, column=0)
            for w in (sw, lbl):
                w.bind("<Button-1>", lambda _e, f=f: self.toggle(f))

    def _refresh_picked(self):
        for child in self.picked_frame.winfo_children():
            child.destroy()
        n, want = len(self.selected), self._count()
        self.picked_label.configure(text=f"Picked {n} of {want}" + (":" if n else " — click colors above."))
        P = self.px
        for i, f in enumerate(self.selected):
            chip = ttk.Frame(self.picked_frame)
            chip.grid(row=i // 4, column=i % 4, sticky="w", padx=(0, P(12)), pady=(0, P(4)))
            sw = self._swatch(chip, f, P(18), False)
            sw.grid(row=0, column=0)
            ttk.Label(chip, text=filament_label(f)).grid(row=0, column=1, padx=(P(4), 0))
            sw.bind("<Button-1>", lambda _e, f=f: self.toggle(f))
        self.btn_ok.state(["!disabled"] if n else ["disabled"])

    def _clear(self):
        self.selected = []
        self._build_grid()
        self._refresh_picked()

    def _ok(self):
        if not self.selected:
            return
        self.result = list(self.selected)
        self.top.destroy()
