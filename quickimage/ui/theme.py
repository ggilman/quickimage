"""Light / dark theme and the colors the app draws itself."""

from tkinter import ttk


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
