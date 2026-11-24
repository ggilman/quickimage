"""Operating-system integration: clipboard, dark title bar, revealing files."""

import io
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageGrab, ImageOps

from ..core.io import is_image_file, load_rgba


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
